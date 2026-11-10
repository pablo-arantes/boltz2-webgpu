/**
 * Boltz-2's affinity module, end to end, with the pairformer on the GPU.
 *
 * 🔴 THE GLUE IS WRITTEN AGAIN HERE RATHER THAN IMPORTED FROM cpu/af3/affinity/.
 * Layering permits it - shared <- cpu <- webgpu - and it would be wrong anyway:
 * cpu/ is the REFERENCE this path is differentially tested against, so sharing
 * its LayerNorm, its projections or its pooling would make those stages compare
 * themselves. docs/ records three checkers here that had to be rescued from
 * exactly that, including one that reported 1622/1622 while every molecule was
 * the wrong enantiomer. These are float32 and independent on purpose.
 *
 * 🔴 WHAT IS ON THE DEVICE, AND A CORRECTION TO WHY. The first version of this
 * file kept all the non-pairformer work on the host and justified it with an
 * O(tokens^2) against O(tokens^3) argument. THE MEASUREMENT SAID THE OPPOSITE:
 * the GPU's share FELL from 32% at 35 tokens to 10% at 256, where the host glue
 * was 16.5 s against the pairformer's 1.8.
 *
 * The O() reasoning ignored the channel constant. A pair projection is
 * `tokens^2 * C_in * C_out`, which at C = 128 is 2.1e9 multiply-adds for ONE
 * projection at 256 tokens - and the conditioner runs five of them (its init
 * projection plus three per transition, twice). Broken down at 256 tokens:
 *
 *     conditioner  15451 ms    <- 94% of the host time
 *     z_linear       955 ms
 *     z_norm          29 ms    s_to_z 11 ms    distogram 13 ms
 *     GPU pairformer 1837 ms
 *
 * So the conditioner's two transitions now run on `Af3TransitionGpu`, whose
 * contract for factor 2 is already held to the float64 reference at 6.6e-8 by
 * tools/check-affinity-kernel-contract.mjs. `timings.hostBreakdown` keeps
 * reporting the split, because the remaining host stages - the init projection
 * and z_linear - are named, measured work rather than a deferral justified by
 * an argument.
 */
import { AffinityPairStackGpu } from "./pair-stack.js";
import { Af3TransitionGpu } from "../trunk/transition.js";
import { AffinityPairProjectGpu } from "./pair-project.js";
import { affinityBoundaries, af3TransitionContract, AFFINITY_HEADS,
  AFFINITY_HEAD_DIM } from "../../../shared/af3/weights/affinity-weights.js";

const EPS_LAYER_NORM = 1e-5;
const EPS_POOL = 1e-7;
const NONPOLYMER = 3;

const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/** y[r, o] = sum_i x[r, i] * w[o, i] (+ b[o]) - torch's [out, in] layout. */
function project(x, rows, inDim, w, outDim, b = null) {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r += 1) {
    const xr = r * inDim;
    const or = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      let sum = b === null ? 0 : b[o];
      const wr = o * inDim;
      for (let i = 0; i < inDim; i += 1) sum += x[xr + i] * w[wr + i];
      out[or + o] = sum;
    }
  }
  return out;
}

/**
 * The cross-pair mask.
 * @param {boolean} excludeDiagonal the HEADS' mask does; the pairformer's does not
 */
export function crossPairMaskF32(molType, affinityMask, padMask, n,
                                 excludeDiagonal = false) {
  const mask = new Float32Array(n * n);
  const receptor = new Float32Array(n);
  const ligand = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    receptor[i] = (molType[i] === 0 ? 1 : 0) * (padMask[i] > 0 ? 1 : 0);
    ligand[i] = (affinityMask[i] > 0 ? 1 : 0) * (padMask[i] > 0 ? 1 : 0);
  }
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      let value = ligand[i] * receptor[j] + receptor[i] * ligand[j]
        + ligand[i] * ligand[j];
      if (excludeDiagonal && i === j) value = 0;
      mask[i * n + j] = value;
    }
  }
  return mask;
}

/**
 * The whole module for one ensemble member.
 *
 * @param {GPUDevice} device
 * @param {object} input `{ sInputs, zAffinity, positions, molType,
 *   affinityMask, padMask, tokens, tokenS, tokenZ }` - Float32Array/typed
 * @param {object} weights from `affinityWeights(read, scope, layers)`
 * @param {{precision?: "f32"|"f16", stack?: AffinityPairStackGpu}} [options]
 */
export async function affinityForwardGpu(device, input, weights, options = {}) {
  const n = input.tokens;
  const cz = input.tokenZ;
  const cs = input.tokenS;
  const pairs = n * n;
  const timings = {};
  const clock = async (name, work) => {
    const started = performance.now();
    const value = await work();
    timings[name] = Number((performance.now() - started).toFixed(2));
    return value;
  };

  const sub = {};
  const transitionRunner = options.transitionRunner ?? new Af3TransitionGpu(device);
  const projector = options.projector ?? new AffinityPairProjectGpu(device);
  const subClock = (name, work) => {
    const started = performance.now();
    const value = work();
    sub[name] = Number((performance.now() - started).toFixed(1));
    return value;
  };
  const host = await clock("host.pre", async () => {
    // z = z_linear(z_norm(z))
    // 🔴 FUSED AND ON THE DEVICE. `z_linear(z_norm(z))` has no intermediate
    // anyone reads, and on the host it was 954 ms at 256 tokens.
    const zStart = performance.now();
    const zLinear = (await projector.run(input.zAffinity,
      { rows: pairs, inChannels: cz, outChannels: cz },
      { normWeight: weights.zNormWeight, normBias: weights.zNormBias,
        projection: weights.zLinear },
      { epsilon: EPS_LAYER_NORM })).output;
    sub["gpu.zNormLinear"] = Number((performance.now() - zStart).toFixed(1));

    // 🔴 A SUM over the two axes, despite `prod` in upstream's name.
    const s1 = subClock("s_to_z", () => project(input.sInputs, n, cs, weights.sToZ1, cz));
    const s2 = project(input.sInputs, n, cs, weights.sToZ2, cz);
    const summed = new Float32Array(pairs * cz);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const at = (i * n + j) * cz;
        for (let c = 0; c < cz; c += 1) {
          summed[at + c] = zLinear[at + c] + s1[i * cz + c] + s2[j * cz + c];
        }
      }
    }

    // the distogram: representative-atom distances, bucketed, embedded.
    // 🔴 DISTANCES COMPUTED DIRECTLY, NOT BY A MATMUL EXPANSION - upstream's
    // torch.cdist reads 4.77e-07 on a diagonal that is exactly zero.
    const boundaries = affinityBoundaries();
    const distoStart = performance.now();
    // 🔴 THE LENGTH IS CHECKED HERE BECAUSE IT WAS NOT, AND A WIRING BUG
    // SURVIVED EVERY GATE. `positions` must be ONE COORDINATE PER TOKEN; the
    // page passed the dense per-ATOM array (`tokens * dense * 3`) and this
    // loop happily indexed its first n entries, producing a distogram of the
    // first few tokens' atom positions. The float64 reference asserted the
    // same contract and threw; this path did not, so every oracle gate - which
    // feeds upstream's own per-token coordinates - stayed green.
    if (input.positions.length !== n * 3) {
      throw new Error(`affinity positions must be one coordinate per token:`
        + ` expected ${n * 3} values for ${n} tokens, got`
        + ` ${input.positions.length}. The dense per-atom array is`
        + ` tokens * dense * 3 - pass the pseudo-beta / disto gather instead.`);
    }
    const embedded = new Float32Array(pairs * cz);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const dx = input.positions[i * 3] - input.positions[j * 3];
        const dy = input.positions[i * 3 + 1] - input.positions[j * 3 + 1];
        const dz = input.positions[i * 3 + 2] - input.positions[j * 3 + 2];
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let bin = 0;
        // a STRICT `>` against 63 edges, giving 64 bins
        for (let e = 0; e < boundaries.length; e += 1) if (d > boundaries[e]) bin += 1;
        const from = bin * cz;
        const at = (i * n + j) * cz;
        for (let c = 0; c < cz; c += 1) embedded[at + c] = weights.binEmbed[from + c];
      }
    }

    sub.distogram = Number((performance.now() - distoStart).toFixed(1));
    // PairwiseConditioning, then ADDED to its own input at the call site
    const condStart = performance.now();
    const joined = new Float32Array(pairs * 2 * cz);
    for (let pair = 0; pair < pairs; pair += 1) {
      const to = pair * 2 * cz;
      for (let c = 0; c < cz; c += 1) {
        joined[to + c] = summed[pair * cz + c];
        joined[to + cz + c] = embedded[pair * cz + c];
      }
    }
    const conditioner = weights.conditioner;
    // 🔴 THE SAME KERNEL AT 256 -> 128, which was 3718 ms on the host at 256
    // tokens - the single largest remaining cost before this.
    let conditioned = (await projector.run(joined,
      { rows: pairs, inChannels: 2 * cz, outChannels: cz },
      { normWeight: conditioner.initNormWeight,
        normBias: conditioner.initNormBias,
        projection: conditioner.initProject },
      { epsilon: EPS_LAYER_NORM })).output;
    sub["gpu.conditionerInit"] = Number((performance.now() - condStart).toFixed(1));
    // 🔴 THE TWO TRANSITIONS ON THE DEVICE - 94% of the host time, measured.
    // Factor 2, because the conditioner's hidden width is 2*token_z where the
    // pairformer's transition_z is 4*. The contract gate holds both.
    const transitionStart = performance.now();
    for (const t of conditioner.transitions) {
      const run = await transitionRunner.run(
        conditioned, { rows: pairs, channels: cz, factor: 2 },
        af3TransitionContract(t, cz, cz * 2),
        { weightPrecision: options.precision ?? "f32" });
      const update = run.output;
      const next = new Float32Array(conditioned.length);
      for (let i = 0; i < next.length; i += 1) next[i] = conditioned[i] + update[i];
      conditioned = next;
    }
    sub["gpu.conditionerTransitions"] =
      Number((performance.now() - transitionStart).toFixed(1));
    sub.conditioner = Number((performance.now() - condStart).toFixed(1));
    const pair = new Float32Array(pairs * cz);
    for (let i = 0; i < pair.length; i += 1) pair[i] = summed[i] + conditioned[i];
    return pair;
  });

  const mask = crossPairMaskF32(input.molType, input.affinityMask,
                                input.padMask, n, false);
  const stack = options.stack ?? new AffinityPairStackGpu(device);
  const run = await clock("gpu.pairformer", () => stack.run(
    host, mask, { n, channels: cz, heads: AFFINITY_HEADS,
      dimension: AFFINITY_HEAD_DIM },
    weights.pairformer, { precision: options.precision ?? "f32" }));

  const out = await clock("host.heads", async () => {
    // 🔴 THE HEADS' MASK EXCLUDES THE DIAGONAL; the pairformer's does not.
    const poolMask = crossPairMaskF32(input.molType, input.affinityMask,
                                      input.padMask, n, true);
    let denominator = 0;
    const pooled = new Float32Array(cz);
    for (let pair = 0; pair < pairs; pair += 1) {
      const m = poolMask[pair];
      if (m === 0) continue;
      denominator += m;
      const at = pair * cz;
      for (let c = 0; c < cz; c += 1) pooled[c] += run.pair[at + c] * m;
    }
    for (let c = 0; c < cz; c += 1) pooled[c] /= denominator + EPS_POOL;

    const w = weights.heads2;
    const relu = (a) => { for (let i = 0; i < a.length; i += 1) if (a[i] < 0) a[i] = 0; return a; };
    const h1 = relu(project(pooled, 1, cz, w.outMlp0Weight, cz, w.outMlp0Bias));
    const g = relu(project(h1, 1, cz, w.outMlp2Weight, cs, w.outMlp2Bias));
    const head = (prefix) => {
      const a = relu(project(g, 1, cs, w[`${prefix}0Weight`], cs, w[`${prefix}0Bias`]));
      const b = relu(project(a, 1, cs, w[`${prefix}2Weight`], cs, w[`${prefix}2Bias`]));
      return project(b, 1, cs, w[`${prefix}4Weight`], 1, w[`${prefix}4Bias`])[0];
    };
    const predValue = head("predValue");
    const predScore = head("predScore");
    // 🔴 the binary logit is a 1->1 linear on pred_SCORE, not on pred_VALUE
    const logitsBinary = predScore * w.logitsWeight[0] + w.logitsBias[0];
    return { pooled, g, predValue, predScore, logitsBinary,
      probabilityBinary: sigmoid(logitsBinary) };
  });

  timings.hostBreakdown = sub;
  return { ...out, seams: run.seams, pair: run.pair, timings,
    gpuShare: timings["gpu.pairformer"]
      / (timings["gpu.pairformer"] + timings["host.pre"] + timings["host.heads"]) };
}

export { NONPOLYMER };
