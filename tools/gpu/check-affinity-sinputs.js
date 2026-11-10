/**
 * Does the `affinity=True` substitution reach `s_inputs`, and only through the
 * profile?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-sinputs.js \
 *     --model=/model-boltz2-int5/manifest.json
 *
 * 🔴 THE CLAIM UNDER TEST IS NARROW AND STATED. `targetFeat` is Boltz's
 * `s_inputs`: `buildTargetFeat` sums an atom-encoder output, a restype one-hot,
 * the MSA profile and the deletion mean at 384 channels - `token_s` - plus
 * boltz2's own `targetFeatSum` summands. upstream's `affinity=True` changes
 * exactly two of those inputs, `profile` and `deletion_mean`, for their depth-1
 * variants. So this asserts three things and nothing more:
 *
 *   1. the substitution MOVES `s_inputs` - otherwise the affinity module is
 *      being handed the structure pass's input and nothing would say so;
 *   2. it moves it ONLY where the profile differs - a token whose deep profile
 *      is already a one-hot must come out unchanged, which is what shows the
 *      change is going through the profile term and not corrupting the rest;
 *   3. it is REPRODUCIBLE and equals the same batch featurised with no
 *      alignment at all, which is what upstream's depth-1 crop is.
 *
 * 🔴 WHAT THIS DOES NOT ESTABLISH, SAID HERE RATHER THAN IMPLIED. It does not
 * compare against upstream's own `input_embedder(feats, affinity=True)`
 * elementwise. That needs an oracle dump of upstream's embedder on a fully
 * featurised input, which means standing up its whole featuriser (Tokenized +
 * MSA + Structure) - a bigger build than the cropper's. docs/ records
 * `target_feat` at 1e-8 against the reference's own batch for every AF3-lineage
 * model, so the machinery either side of the substitution is verified; the
 * substitution itself is verified here; the composition of the two is NOT, and
 * is the open item.
 */
import { buildTargetFeat } from "../../webgpu/af3/fold.js";
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { featuriserDialect } from "../../shared/af3/dialect.js";
import { affinityProfile, RESTYPES }
  from "../../shared/af3/affinity/features.js";
import { openAf3Store } from "../../shared/af3/weights/weights.js";
import { targetFeatureWeights } from "../../shared/af3/weights/diffusion-weights.js";
import { dialectFor } from "../../shared/af3/dialect.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

const SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ";
const DEEP_A3M = [
  `>101\n${SEQUENCE}`,
  `>h1\n${SEQUENCE.replace(/A/g, "G").replace(/K/g, "R")}`,
  `>h2\n${SEQUENCE.replace(/I/g, "L").replace(/S/g, "T")}`,
  `>h3\n${SEQUENCE.replace(/Q/g, "E").replace(/Y/g, "F")}`,
].join("\n") + "\n";

function relRms(a, b) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  const value = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / a.length);
  if (!Number.isFinite(value)) throw new Error(`residual is ${value}`);
  return value;
}

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const store = await openAf3Store(manifest);
  const weights = await targetFeatureWeights(store);
  const dialect = featuriserDialect(dialectFor("boltz2"));

  const deep = af3BatchFromA3m(SEQUENCE, DEEP_A3M, dialect).batch;
  const single = af3BatchFromA3m(SEQUENCE, null, dialect).batch;
  const tokens = deep.tokens;

  // which tokens the deep profile actually differs on - everything below is
  // relative to this, because a token whose deep profile is already a one-hot
  // CANNOT move and counting it as unchanged would be no evidence at all
  const mixed = [];
  for (let token = 0; token < tokens; token += 1) {
    let nonZero = 0;
    for (let c = 0; c < RESTYPES; c += 1) {
      if (deep.profile[token * RESTYPES + c] > 0) nonZero += 1;
    }
    if (nonZero > 1) mixed.push(token);
  }
  if (mixed.length < 5) {
    throw new Error(`only ${mixed.length} tokens have a mixed deep profile;`
      + " the separation below would prove nothing");
  }

  const structureS = await buildTargetFeat(deep, weights, device);
  const { profile, deletionMean } = affinityProfile(deep);
  const affinityS = await buildTargetFeat(
    { ...deep, profile, deletionMean }, weights, device);
  const noAlignmentS = await buildTargetFeat(single, weights, device);

  const channels = structureS.length / tokens;
  const perToken = (a, b, token) => {
    let num = 0;
    let den = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = a[token * channels + c] - b[token * channels + c];
      num += d * d;
      den += b[token * channels + c] ** 2;
    }
    return den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / channels);
  };

  // 1. it moves at all
  const moved = relRms(affinityS, structureS);
  // 2. ...only where the profile differs
  const mixedSet = new Set(mixed);
  let movedMixed = 0;
  let movedUnchanged = 0;
  let worstUnchanged = 0;
  for (let token = 0; token < tokens; token += 1) {
    const delta = perToken(affinityS, structureS, token);
    if (mixedSet.has(token)) { if (delta > 1e-6) movedMixed += 1; }
    else if (delta > 1e-6) { movedUnchanged += 1; worstUnchanged = Math.max(worstUnchanged, delta); }
  }
  // 3. ...and it is the no-alignment batch's
  const againstNoAlignment = relRms(affinityS, noAlignmentS);

  const checks = {
    "substitution moves s_inputs": { value: moved, want: "> 1e-3",
      ok: moved > 1e-3 },
    "every mixed-profile token moved": { value: movedMixed,
      want: `= ${mixed.length}`, ok: movedMixed === mixed.length },
    "no one-hot token moved": { value: movedUnchanged, want: "= 0",
      ok: movedUnchanged === 0 },
    "equals the no-alignment s_inputs": { value: againstNoAlignment,
      want: "<= 1e-6", ok: againstNoAlignment <= 1e-6 },
  };

  let failed = 0;
  for (const [label, check] of Object.entries(checks)) {
    failed += check.ok ? 0 : 1;
    const shown = typeof check.value === "number" && check.value < 1
      ? check.value.toExponential(2) : String(check.value);
    console.log(`${check.ok ? "OK  " : "FAIL"} ${label.padEnd(34)}`
      + ` ${shown.padStart(10)}  want ${check.want}`);
  }
  console.log(`tokens ${tokens}, channels ${channels} (token_s),`
    + ` mixed-profile tokens ${mixed.length}`);
  console.log(`worst unchanged-token delta ${worstUnchanged.toExponential(2)}`);

  if (failed > 0) throw new Error(`${failed} s_inputs check(s) failed`);
  return { tokens, channels, mixedTokens: mixed.length,
    moved, againstNoAlignment, worstUnchanged,
    note: "verifies the affinity=True substitution reaches s_inputs through the"
      + " profile only; does NOT compare against upstream's own embedder - see"
      + " this tool's header" };
}
