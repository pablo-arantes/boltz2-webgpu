/**
 * Does the fused pair LayerNorm+projection compute the host's answer?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-pair-project.js
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-pair-project.js --falsify
 *
 * AGENTS.md: every GPU kernel must have a deterministic differential test
 * against an independent reference. The reference here is written in this file
 * rather than imported, because the two host implementations that already exist
 * - cpu/af3/affinity/affinity.js and the f32 glue in
 * webgpu/af3/affinity/affinity.js - are the things this kernel is meant to
 * REPLACE, and importing either would make the check compare the kernel with
 * the code it is replacing via that code's own helpers.
 *
 * 🔴 BOTH REAL SHAPES, NOT ONE. The two callers are
 * `LayerNorm(128) + 128 -> 128` (z_linear) and
 * `LayerNorm(256) + 256 -> 128` (the conditioner's init), and the second is the
 * one that stages 256 channels in workgroup memory and loops its output lanes
 * twice. A kernel checked only at 128 would leave the expensive caller's path
 * unexercised - which is the configuration-not-checked shape CLAUDE.md records
 * for LOCALFOLD_STOCK_FLAGS and the pipeline cache.
 *
 * 🔴 AND A ROW COUNT THAT IS NOT A MULTIPLE OF THE LANE COUNT. The channel
 * loops stride by 64, so `IN` and `OUT` that are not multiples of 64 exercise
 * the ragged tail; 200 and 35 tokens give row counts that are not either.
 *
 * `--falsify` perturbs the normalisation (unbiased variance, eps outside the
 * sqrt, the bias dropped) and the projection layout (transposed), and each must
 * exceed the bound - because a bound that cannot tell a correct kernel from one
 * of those is not a bound.
 *
 * 🔴 AND ITS INPUT CARRIES LOW-VARIANCE ROWS, BECAUSE ONE CONVENTION IS ONLY
 * EXPRESSIBLE ON THEM. With variance ~1, `1/sqrt(v + eps)` and
 * `1/(sqrt(v) + eps)` agree to ~5e-6 and the control reported a MISS; at v = 0
 * they are 316 against 100000. Masked pair rows are near-constant - G2 and G3
 * both measured that - and the conditioner's init runs over the whole pair
 * including the masked region, so these rows are what the kernel really sees.
 */
import { AffinityPairProjectGpu }
  from "../../webgpu/af3/affinity/pair-project.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

/** The reference: torch's LayerNorm then y = x @ W.T, in float64. */
function reference(input, rows, inChannels, outChannels, weights, epsilon,
                   variant = {}) {
  const out = new Float64Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const at = row * inChannels;
    let mean = 0;
    for (let c = 0; c < inChannels; c += 1) mean += input[at + c];
    mean /= inChannels;
    let variance = 0;
    for (let c = 0; c < inChannels; c += 1) {
      const d = input[at + c] - mean;
      variance += d * d;
    }
    // torch's LayerNorm is the BIASED variance; `--falsify` asks for the other
    variance /= variant.unbiased ? (inChannels - 1) : inChannels;
    // ...and eps is INSIDE the sqrt's argument
    const inverse = variant.epsOutside
      ? 1 / (Math.sqrt(variance) + epsilon)
      : 1 / Math.sqrt(variance + epsilon);
    const normed = new Float64Array(inChannels);
    for (let c = 0; c < inChannels; c += 1) {
      normed[c] = (input[at + c] - mean) * inverse * weights.normWeight[c]
        + (variant.noBias ? 0 : weights.normBias[c]);
    }
    for (let o = 0; o < outChannels; o += 1) {
      let total = 0;
      for (let k = 0; k < inChannels; k += 1) {
        total += normed[k] * (variant.transposed
          ? weights.projection[k * outChannels + o]
          : weights.projection[o * inChannels + k]);
      }
      out[row * outChannels + o] = total;
    }
  }
  return out;
}

function noise(length, seed, scale = 1) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) & 0x7fffffff;
    out[i] = (state / 0x7fffffff - 0.5) * scale;
  }
  return out;
}

function relRms(mine, theirs) {
  if (mine.length === 0 || mine.length !== theirs.length) {
    throw new RangeError(`not comparable: ${mine.length} vs ${theirs.length}`);
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < mine.length; i += 1) {
    const d = mine[i] - theirs[i];
    num += d * d;
    den += theirs[i] * theirs[i];
  }
  const value = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / mine.length);
  if (!Number.isFinite(value)) throw new Error(`residual is ${value}`);
  return value;
}

export async function main(device, args = []) {
  const falsify = args.includes("--falsify");
  const epsilon = 1e-5;
  // f32 kernel against an f64 reference over a 128- or 256-long dot product.
  const BOUND = 1e-5;
  const runner = new AffinityPairProjectGpu(device);

  // the two real callers, at token counts whose row counts are ragged
  const cases = [
    { label: "z_linear 128->128", tokens: 35, inChannels: 128, outChannels: 128 },
    { label: "conditioner 256->128", tokens: 35, inChannels: 256, outChannels: 128 },
    { label: "z_linear 128->128 @200", tokens: 200, inChannels: 128, outChannels: 128 },
    { label: "conditioner 256->128 @200", tokens: 200, inChannels: 256, outChannels: 128 },
    // a non-multiple-of-64 output width, for the lane tail
    { label: "ragged 130->100", tokens: 24, inChannels: 130, outChannels: 100 },
  ];

  const results = {};
  let failed = 0;
  let worst = 0;
  for (const testCase of cases) {
    const rows = testCase.tokens * testCase.tokens;
    const { inChannels, outChannels } = testCase;
    const input = noise(rows * inChannels, 7919 + inChannels + rows, 2.5);
    const weights = {
      normWeight: noise(inChannels, 11, 0.4).map((v) => v + 1),
      normBias: noise(inChannels, 12, 0.2),
      projection: noise(inChannels * outChannels, 13, 0.3),
    };
    const run = await runner.run(input, { rows, inChannels, outChannels },
                                 weights, { epsilon });
    const expected = reference(Float64Array.from(input), rows, inChannels,
                               outChannels, weights, epsilon);
    const value = relRms(Float64Array.from(run.output), expected);
    const ok = value <= BOUND;
    failed += ok ? 0 : 1;
    worst = Math.max(worst, value);
    results[testCase.label] = { rows, value, ok,
      ms: Number(run.elapsedMilliseconds.toFixed(2)),
      peakMiB: Number((run.memory.peakBytes / 2 ** 20).toFixed(2)) };
    console.log(`${ok ? "OK  " : "FAIL"} ${testCase.label.padEnd(26)}`
      + ` ${String(rows).padStart(6)} rows  ${value.toExponential(2)}`
      + `  ${run.elapsedMilliseconds.toFixed(1)} ms`);
  }

  if (falsify) {
    console.log();
    console.log("=== separation control: one convention broken at a time ===");
    const rows = 35 * 35;
    const inChannels = 256;
    const outChannels = 128;
    const input = noise(rows * inChannels, 4242, 2.5);
    // 🔴 LOW-VARIANCE ROWS, BECAUSE THE eps CONVENTION IS ONLY EXPRESSIBLE ON
    // THEM. The first version of this control reported
    // "MISSED epsOutside 4.23e-6" - inside the bound - and that is not a weak
    // gate, it is arithmetic: at variance ~1, `1/sqrt(v + 1e-5)` and
    // `1/(sqrt(v) + 1e-5)` agree to about 5e-6. The two only part company where
    // eps DOMINATES, i.e. where the variance approaches zero: at v = 0 they are
    // 316 against 100000.
    //
    // Which is not a contrived input. G2 and G3 both measured that a masked
    // pair row is near-constant, and the conditioner's init runs over the WHOLE
    // pair including the masked region - so low-variance rows are what this
    // kernel actually sees. Every eighth row is made near-constant here.
    //
    // 🔴 AND THE VARIANCE HAS TO BE COMPARABLE TO eps, NOT ZERO. A perfectly
    // constant row was the second attempt and it MISSED too, at the same
    // 4.23e-6: with every channel equal, `x - mean` is exactly 0, so whatever
    // the inverse scale is it multiplies zero and both conventions return the
    // bias. What separates them is a row whose variance is the SIZE of eps -
    // at v = 1e-5 the two inverse scales are 224 against 315, a 41% difference
    // on a non-zero deviation. A tiny jitter on a flat row is that.
    const jitter = noise(inChannels, 98765, 2 * Math.sqrt(3) * 3e-3);
    for (let row = 0; row < rows; row += 8) {
      const at = row * inChannels;
      const level = (row % 32) * 0.01;
      for (let c = 0; c < inChannels; c += 1) input[at + c] = level + jitter[c];
    }
    const weights = {
      normWeight: noise(inChannels, 11, 0.4).map((v) => v + 1),
      normBias: noise(inChannels, 12, 0.2),
      projection: noise(inChannels * outChannels, 13, 0.3),
    };
    const run = await runner.run(input, { rows, inChannels, outChannels },
                                 weights, { epsilon });
    let caught = 0;
    const variants = { unbiased: {}, epsOutside: {}, noBias: {}, transposed: {} };
    for (const name of Object.keys(variants)) {
      const wrong = reference(Float64Array.from(input), rows, inChannels,
                              outChannels, weights, epsilon, { [name]: true });
      const value = relRms(Float64Array.from(run.output), wrong);
      const sees = value > BOUND * 100;
      caught += sees ? 1 : 0;
      console.log(`  ${sees ? "CAUGHT " : "MISSED "} ${name.padEnd(14)}`
        + ` ${value.toExponential(2)}`);
    }
    console.log();
    const all = Object.keys(variants).length;
    if (caught === all) {
      console.log(`SEPARATION OK: all ${all} conventions exceed the bound`);
    } else {
      console.log(`SEPARATION FAILED: ${caught} of ${all}`);
      failed += 1;
    }
  }

  if (failed > 0) throw new Error(`${failed} pair-project case(s) failed`);
  return { worst, bound: BOUND, results };
}
