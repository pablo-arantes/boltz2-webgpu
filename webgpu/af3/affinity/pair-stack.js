/**
 * Boltz-2's affinity pairformer on the GPU: a pair-only stack over the kernels
 * this repository already ships.
 *
 * 🔴 A DRIVER, NOT NEW KERNELS, AND THAT IS A MEASURED DECISION.
 * tools/check-affinity-kernel-contract.mjs holds AF3's triangle multiplication,
 * grid attention (both directions) and transition to the float64 affinity
 * reference with Boltz's own weights remapped through
 * shared/af3/weights/affinity-weights.js, and all twelve layers of both
 * ensemble members agree at ~8e-8. So the arithmetic is already here; what was
 * missing is a loop that runs it without a sequence track.
 *
 * 🔴 WHY NOT Af3PairformerStackGpu. It reads
 * `blocks[0].singleAttention.heads` unconditionally - the single track is
 * intrinsic to it, not optional - so threading a `noSingle` flag through would
 * change the stack the page folds with in order to serve a stack it does not
 * run. PairformerNoSeqModule genuinely has no sequence track: four pair ops and
 * a transition, all residual, and nothing else.
 *
 * 🔴 AND THE PIPELINE CACHE KEY CARRIES ITS OWN PREFIX. `ComputePipelineCache`
 * indexes by `entryPoint + source` as well as by key, so a collision is refused
 * and named rather than silently served - but CLAUDE.md records two real
 * collisions from keys that did not name constants their shaders embed, and
 * this stack is 128 channels with 4 heads of 32, which is in the same family as
 * the trunk's shapes. The kernels build their own keys; what this adds is a
 * distinct `variant` so two stacks over one device cannot share a packed weight
 * buffer. `npm run test:cache` is extended with a boltz2-structure-then-affinity
 * arm in BOTH orders, because a collision is only reported by the second
 * compile.
 *
 * 🔴 THE RESIDUALS ARE ADDED ON THE HOST, AND THAT IS DELIBERATE FOR NOW. Each
 * kernel here returns a Float32Array, so a layer costs five readbacks. That is
 * the right shape for a DIFFERENTIAL - every seam is observable, which is what
 * G3 compares - and the wrong shape for production, where the pair should stay
 * resident and the adds should be `addInPlace` dispatches. Keeping the pair on
 * the device is a performance change over a verified baseline, not a
 * correctness one, and it is deliberately not bundled with establishing the
 * baseline. `seams` is what makes the difference visible either way.
 */
// 🔴 THE LOW-LEVEL CLASSES, NOT `af3TriangleMultiplication`. That wrapper maps
// from AF3's CHECKPOINT names and calls `splitInterleaved(weights.projection)`,
// because AF3 stores the a and b projections INTERLEAVED in one tensor. Boltz
// CONCATENATES them - `p_in` is [2*cz, cz] with rows 0..cz-1 the a half - so
// `af3TriangleContract` has already split them and routing through the wrapper
// would split an already-split tensor. These classes take the sixteen-name
// contract directly, which is the same path
// tools/check-affinity-kernel-contract.mjs verified against the host reference.
import { TriangleMultiplicationOutgoingGpu, TriangleMultiplicationIncomingGpu }
  from "../../kernels/triangle/webgpu.js";
import { Af3GridSelfAttentionGpu } from "../trunk/grid-attention.js";
import { Af3TransitionGpu } from "../trunk/transition.js";
import { af3TriangleContract, af3GridContract, af3TransitionContract }
  from "../../../shared/af3/weights/affinity-weights.js";

/**
 * 🔴 BOLTZ-2'S OWN DIALECT FLAG, NOT A NEW ONE. `gridSelfAttention` takes the
 * bias from the transposed activation only when `swapTransposedBias` is set,
 * and BOLTZ2 in shared/af3/dialect.js already sets it - because upstream's
 * `TriangleAttention` runs `self.linear(x)` AFTER its transpose, so the bias is
 * the transposed pair's by construction. The flag that the trunk needed for
 * boltz2 is the flag this needs, which is why it is read rather than restated.
 */
const AFFINITY_DIALECT = Object.freeze({ swapTransposedBias: true });

/**
 * One `precision` request, in each kernel's own option name.
 *
 * 🔴 THE THREE KERNELS SPELL THE SAME CONCEPT THREE WAYS, AND PASSING ONE NAME
 * TO ALL OF THEM SILENTLY RAN TWO OF THEM IN f32 IN BOTH ARMS.
 *
 *     webgpu/kernels/triangle/webgpu.js   options.precision
 *     webgpu/af3/trunk/transition.js      options.weightPrecision
 *     webgpu/af3/trunk/grid-attention.js  options.stagedPrecision
 *
 * The first version of this driver forwarded `{ stagedPrecision }` to all
 * three, so G3's f16 arm exercised the grid attention's half path and left the
 * triangle's and the transition's untested while every number still looked
 * reasonable. CLAUDE.md names the shape: "A control arm that cannot vary the
 * thing under test is not a control - check the arm reached the code path
 * before reading its result." `--precision` now moves all three, and
 * tools/gpu/probe-affinity-steps.js prints each step at both precisions so
 * that the arm is visibly reaching every kernel.
 */
function precisionOptions(options) {
  const precision = options.precision ?? options.stagedPrecision ?? "f32";
  return {
    triangle: { ...options, precision },
    transition: { ...options, weightPrecision: precision },
    grid: { ...options, stagedPrecision: precision },
    precision,
  };
}

function addInto(target, update) {
  for (let i = 0; i < target.length; i += 1) target[i] += update[i];
  return target;
}

export class AffinityPairStackGpu {
  // 🔴 THE PER-LAYER SEAMS ARE DIAGNOSTICS AND COST ~0.6 GB A LIGAND. At 256
  // tokens one pair tensor is 33.55 MiB; this kept five distinct seams per
  // layer plus every layer's output, which for affinity_module1's eight layers
  // is ~436 MiB live at the end of a member - and web/affinity-passes.js
  // discards all of it on return, so the page never reads a byte. The gates
  // DO read it, so it is an option rather than a deletion, defaulting off.

  #grid;
  #transition;
  #triangle;

  constructor(device) {
    this.#grid = new Af3GridSelfAttentionGpu(device);
    this.#transition = new Af3TransitionGpu(device);
    this.#triangle = {
      outgoing: new TriangleMultiplicationOutgoingGpu(device),
      incoming: new TriangleMultiplicationIncomingGpu(device),
    };
  }

  /**
   * One PairformerNoSeqLayer, in upstream's order, every step residual.
   *
   * @param {Float32Array} pair n*n*channels, consumed and not mutated
   * @param {Float32Array} mask n*n, the cross-pair mask
   * @param {{n: number, channels: number, heads: number, dimension: number}} shape
   * @param {object} weights one entry of `affinityWeights(...).pairformer`
   * @param {{precision?: "f32"|"f16", variance?: string}} [options]
   *   one `precision`, translated into each kernel's own option name
   * @returns {Promise<{pair: Float32Array, seams: Record<string, Float32Array>}>}
   */
  async layer(pair, mask, shape, weights, options = {}) {
    const { n, channels, heads, dimension } = shape;
    if (pair.length !== n * n * channels) {
      throw new Error(`pair has ${pair.length} elements;`
        + ` expected ${n * n * channels}`);
    }
    if (mask.length !== n * n) {
      throw new Error(`mask has ${mask.length} elements; expected ${n * n}`);
    }
    const seams = {};
    const precision = precisionOptions(options);
    let current = Float32Array.from(pair);

    for (const [tag, direction, key] of [
      ["tri_mul_out", "outgoing", "triMulOut"],
      ["tri_mul_in", "incoming", "triMulIn"],
    ]) {
      const run = await this.#triangle[direction].run({
        shape: { length: n, cZ: channels, cHidden: channels },
        epsilon: 1e-5,
        z: current,
        mask,
        weights: af3TriangleContract(weights[key], channels, direction),
      }, precision.triangle);
      const update = run.output ?? run;
      current = addInto(Float32Array.from(current), update);
      seams[tag] = current;
    }

    for (const [tag, key, starting] of [
      ["tri_att_start", "triAttStart", true],
      ["tri_att_end", "triAttEnd", false],
    ]) {
      const run = await this.#grid.run(
        current, mask, { n, channels, transpose: !starting },
        af3GridContract(weights[key], channels, heads, dimension),
        AFFINITY_DIALECT, precision.grid);
      current = addInto(Float32Array.from(current), run.output);
      seams[tag] = current;
    }

    const wide = await this.#transition.run(
      current, { rows: n * n, channels, factor: 4 },
      af3TransitionContract(weights.transitionZ, channels, channels * 4),
      precision.transition);
    current = addInto(Float32Array.from(current), wide.output);
    seams.transition_z = current;

    return { pair: current, seams };
  }

  /**
   * The whole stack. `seams.pairformerNN` is each layer's output, which is what
   * localises a fault - a single end-to-end number cannot.
   */
  async run(pair, mask, shape, layers, options = {}) {
    let current = Float32Array.from(pair);
    const seams = {};
    for (let index = 0; index < layers.length; index += 1) {
      const step = await this.layer(current, mask, shape, layers[index], options);
      current = step.pair;
      seams[`pairformer${String(index).padStart(2, "0")}`] = current;
      if (index === 0) {
        for (const [tag, value] of Object.entries(step.seams)) {
          seams[`layer00/${tag}`] = value;
        }
      }
    }
    return { pair: current, seams };
  }
}
