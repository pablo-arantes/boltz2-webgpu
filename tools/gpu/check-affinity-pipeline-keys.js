/**
 * G6 - does the affinity stack collide with the trunk in ONE pipeline cache?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-pipeline-keys.js
 *
 * 🔴 THE SHAPES ARE IDENTICAL, WHICH IS WHY THIS IS A REAL CASE AND NOT A
 * FORMALITY. AF3's trunk grid attention is 128 channels, 4 heads of dimension
 * 32 (check-af3-grid-attention.js's own constants) and so is the affinity
 * pairformer, measured off the checkpoint: `tri_att_*.linear [4, 128]` with
 * `mha.linear_q [128, 128]`. The two stacks therefore ask the SAME kernels for
 * the SAME widths, and the only things separating their keys are
 * `swapTransposedBias` - true for boltz2 and the affinity module, false for
 * stock AF3 - and the staged precision. CLAUDE.md records two collisions that
 * were exactly this: a key that did not name a constant its shader embeds.
 *
 * 🔴 AND IT IS BUNDLE-FREE ON PURPOSE. `check-pipeline-key-collisions.js`
 * refuses with fewer than two model bundles - correctly, since with one it
 * "can never see a collision and would pass by finding nothing" - and a box
 * with only the affinity export cannot satisfy it. A collision is a property of
 * KEYS and SOURCES, not of weights being real, so this drives the three shared
 * kernels directly across every configuration the two stacks ask for, in one
 * device and one cache.
 *
 * WHAT WOULD FAIL. `ComputePipelineCache` indexes by `entryPoint + source` as
 * well as by key, so a key reused for different WGSL is REFUSED and named
 * rather than silently served - which is what makes this gate's assertion
 * simply "nothing threw", plus a count so that a silent MERGE (two
 * configurations sharing one pipeline because the key collapsed them) is
 * visible rather than inferred.
 *
 * 🔴 BOTH ORDERS. A collision is only reported by the SECOND compile, so a
 * sequence that is clean one way round is not evidence.
 */
import { pipelineCacheForDevice } from "../../webgpu/runtime/pipeline-cache.js";
import { Af3GridSelfAttentionGpu } from "../../webgpu/af3/trunk/grid-attention.js";
import { Af3TransitionGpu } from "../../webgpu/af3/trunk/transition.js";
import { TriangleMultiplicationOutgoingGpu, TriangleMultiplicationIncomingGpu }
  from "../../webgpu/kernels/triangle/webgpu.js";

const CHANNELS = 128;
const HEADS = 4;
const DIMENSION = 32;
const N = 10;

function deterministic(length, seed) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) & 0x7fffffff;
    out[i] = (state / 0x7fffffff - 0.5) * 0.8;
  }
  return out;
}

export async function main(device, args = []) {
  const cache = pipelineCacheForDevice(device);
  const pair = deterministic(N * N * CHANNELS, 31337);
  const mask = new Float32Array(N * N).fill(1);
  // a ragged mask for one arm, since the mask reaches the shader as a binding
  const ragged = Float32Array.from(mask, (_, i) => (i % 7 === 0 ? 0 : 1));

  const triWeights = () => {
    const c2 = CHANNELS * CHANNELS;
    const z = () => new Float32Array(CHANNELS);
    const m = (seed) => deterministic(c2, seed);
    return {
      layerNormInWeight: new Float32Array(CHANNELS).fill(1),
      layerNormInBias: z(),
      linearAPWeight: m(11), linearAPBias: z(),
      linearBPWeight: m(12), linearBPBias: z(),
      linearAGWeight: m(13), linearAGBias: z(),
      linearBGWeight: m(14), linearBGBias: z(),
      layerNormOutWeight: new Float32Array(CHANNELS).fill(1),
      layerNormOutBias: z(),
      linearZWeight: m(15), linearZBias: z(),
      linearGWeight: m(16), linearGBias: z(),
    };
  };
  const gridWeights = () => ({
    heads: HEADS, dimension: DIMENSION,
    actNormScale: new Float32Array(CHANNELS).fill(1),
    actNormOffset: new Float32Array(CHANNELS),
    pairBiasProjection: deterministic(CHANNELS * HEADS, 21),
    qProjection: deterministic(CHANNELS * HEADS * DIMENSION, 22),
    kProjection: deterministic(CHANNELS * HEADS * DIMENSION, 23),
    vProjection: deterministic(CHANNELS * HEADS * DIMENSION, 24),
    gatingQuery: deterministic(CHANNELS * HEADS * DIMENSION, 25),
    outputProjection: deterministic(CHANNELS * HEADS * DIMENSION, 26),
  });
  const transitionWeights = (factor) => ({
    inputLayerNormScale: new Float32Array(CHANNELS).fill(1),
    inputLayerNormOffset: new Float32Array(CHANNELS),
    transition1: deterministic(CHANNELS * CHANNELS * factor * 2, 31),
    transition2: deterministic(CHANNELS * factor * CHANNELS, 32),
  });

  const grid = new Af3GridSelfAttentionGpu(device);
  const transitionRunner = new Af3TransitionGpu(device);
  const outgoing = new TriangleMultiplicationOutgoingGpu(device);
  const incoming = new TriangleMultiplicationIncomingGpu(device);
  const haveF16 = device.features.has("shader-f16");
  const precisions = haveF16 ? ["f32", "f16"] : ["f32"];

  // Every configuration the two stacks ask for. The affinity rows and the
  // stock-AF3 rows differ ONLY in swapTransposedBias at identical widths,
  // which is the collision this gate exists for.
  const arms = [];
  for (const precision of precisions) {
    for (const swapTransposedBias of [true, false]) {
      for (const transpose of [false, true]) {
        arms.push({
          label: `grid:swap=${swapTransposedBias}:transpose=${transpose}:${precision}`,
          run: () => grid.run(pair, transpose ? ragged : mask,
            { n: N, channels: CHANNELS, transpose }, gridWeights(),
            { swapTransposedBias }, { stagedPrecision: precision }),
        });
      }
    }
    for (const [label, runner] of [["outgoing", outgoing], ["incoming", incoming]]) {
      arms.push({
        label: `triangle:${label}:${precision}`,
        run: () => runner.run({
          shape: { length: N, cZ: CHANNELS, cHidden: CHANNELS },
          epsilon: 1e-5, z: pair, mask, weights: triWeights(),
        }, { precision }),
      });
    }
    // factor 4 is the pairformer's transition_z; factor 2 is the conditioner's
    for (const factor of [4, 2]) {
      arms.push({
        label: `transition:factor=${factor}:${precision}`,
        run: () => transitionRunner.run(pair,
          { rows: N * N, channels: CHANNELS, factor },
          transitionWeights(factor), { weightPrecision: precision }),
      });
    }
  }

  const results = {};
  const problems = [];
  let before = cache.size;
  for (const order of ["forwards", "backwards"]) {
    const sequence = order === "forwards" ? arms : [...arms].reverse();
    const sizeAtStart = cache.size;
    for (const arm of sequence) {
      try {
        await arm.run();
      } catch (error) {
        const message = String(error?.message ?? error);
        // A collision is what this gate is for; anything else is a bug in the
        // arm and must not be reported as a pass.
        problems.push({ order, arm: arm.label, message });
      }
    }
    results[order] = { armsRun: sequence.length,
      pipelinesAfter: cache.size, newPipelines: cache.size - sizeAtStart };
    console.log(`${order}\t${sequence.length} arms`
      + `\tcache ${sizeAtStart} -> ${cache.size}`
      + `\t+${cache.size - sizeAtStart}`);
  }

  for (const problem of problems) {
    console.log(`PROBLEM\t${problem.order}\t${problem.arm}\t${problem.message}`);
  }

  // 🔴 THE SECOND PASS MUST ADD NOTHING. Every key it asks for was compiled by
  // the first, so a non-zero count means a key depends on something other than
  // the configuration - which is the same defect as a collision seen from the
  // other side.
  const reusedCleanly = results.backwards.newPipelines === 0;
  if (!reusedCleanly) {
    console.log(`PROBLEM\tbackwards added ${results.backwards.newPipelines}`
      + " pipelines where it should have reused every one");
  }

  // ...and the distinct count must match the distinct CONFIGURATIONS, or two
  // arms collapsed onto one pipeline and the gate would never know.
  const distinctArms = new Set(arms.map((a) => a.label)).size;
  console.log(`arms ${arms.length}, distinct labels ${distinctArms},`
    + ` pipelines compiled ${results.forwards.pipelinesAfter - before}`);

  // 🔴 THE SENSITIVITY CONTROL: "the second pass added nothing" is only
  // evidence if the counter CAN move. A genuinely new configuration - the same
  // kernels at a different token count - must add pipelines; if it does not,
  // the counter is blind and the backwards result above means nothing. This is
  // the same discipline as the separation controls on G1, G2 and G3.
  const sizeBeforeNew = cache.size;
  const biggerN = N + 6;
  const biggerPair = deterministic(biggerN * biggerN * CHANNELS, 777);
  const biggerMask = new Float32Array(biggerN * biggerN).fill(1);
  await grid.run(biggerPair, biggerMask,
    { n: biggerN, channels: CHANNELS, transpose: false }, gridWeights(),
    { swapTransposedBias: true }, { stagedPrecision: "f32" });
  const sensitivity = cache.size - sizeBeforeNew;
  const canMove = sensitivity > 0;
  console.log(`sensitivity	n=${biggerN} added ${sensitivity} pipeline(s)`
    + `	${canMove ? "COUNTER MOVES" : "COUNTER BLIND"}`);
  if (!canMove) {
    problems.push({ order: "sensitivity", arm: `n=${biggerN}`,
      message: "a new configuration added no pipelines, so the reuse check"
        + " above cannot distinguish reuse from a blind counter" });
  }
  results.sensitivity = { tokens: biggerN, added: sensitivity, canMove };

  if (problems.length > 0 || !reusedCleanly) {
    throw new Error(`${problems.length} pipeline problem(s);`
      + ` backwards added ${results.backwards.newPipelines}`);
  }
  return {
    channels: CHANNELS, heads: HEADS, dimension: DIMENSION, tokens: N,
    shaderF16: haveF16, arms: arms.length, results,
    note: "affinity and stock-AF3 rows differ ONLY in swapTransposedBias at"
      + " identical widths - see this tool's header",
  };
}
