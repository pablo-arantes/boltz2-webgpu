/**
 * G3 - does the GPU affinity pairformer compute the float64 reference's answer?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-pair-stack.js
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-pair-stack.js --precision=f32,f16
 *
 * AGENTS.md requires every GPU kernel to have a deterministic differential test
 * against an independent reference. The reference is cpu/af3/affinity/, which
 * G2 holds to upstream PyTorch seam by seam; this holds the GPU to it.
 *
 * 🔴 PAIR SEAMS ARE COMPARED ONLY WHERE BOTH TOKENS ARE LIVE, AND THAT IS NOT
 * A CONVENIENCE. A fully masked attention row has `inf * (mask - 1) = -1e9` on
 * every key, so every logit is `-1e9 + O(1)`. Float32 spacing near 1e9 is 64,
 * so the O(1) term is ERASED and softmax returns a uniform 1/n; the float64
 * reference keeps it and returns a structured distribution. The GPU runs f32
 * and will agree with upstream, not with the reference - so a gate that
 * compared padded rows would FAIL A CORRECT KERNEL. G2 measured the same split:
 * live tokens 4e-7, padded rows up to 2.6e-1.
 *
 * 🔴 AND THE MASK IS RAGGED ON PURPOSE. The cross-pair mask is zero over every
 * receptor-receptor pair, which is most of the tensor, so an all-ones mask
 * would leave the masking convention - one of the three this port had to get
 * right - unexercised while every number still looked fine.
 *
 * 🔴 AND `--precision` HAS TO REACH ALL THREE KERNELS, WHICH IT DID NOT AT
 * FIRST. The triangle reads `options.precision`, the transition
 * `options.weightPrecision` and the grid attention `options.stagedPrecision` -
 * three names for one concept - so the driver's first version forwarded
 * `stagedPrecision` to all three and the f16 arm exercised one kernel of
 * three while every number still looked reasonable. `precisionOptions()` in
 * the driver translates, and probe-affinity-steps.js prints each step at both
 * precisions so that an arm which is not reaching a kernel SHOWS.
 *
 * 🔴 TWO PRECISIONS, TWO BOUNDS. The grid attention stages its key and value
 * in f16 wherever the device has shader-f16, which is eleven significant bits;
 * check-af3-grid-attention.js records 3e-2 as what that costs and 1e-5 as what
 * the f32 path reaches. Widening one bound to cover both is how the f32 path
 * stops being checked at all.
 */
import { AffinityPairStackGpu } from "../../webgpu/af3/affinity/pair-stack.js";
import { pairformerNoSeqLayer, crossPairMask } from "../../cpu/af3/affinity/affinity.js";
import { affinityWeights, AFFINITY_TOKEN_Z, AFFINITY_HEADS, AFFINITY_HEAD_DIM }
  from "../../shared/af3/weights/affinity-weights.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

/** relRMS over the pairs where BOTH tokens are live, and the padded rest. */
function split(mine, theirs, channels, live) {
  if (!(mine instanceof Float32Array) && !(mine instanceof Float64Array)) {
    throw new TypeError("relRMS over a non-array returns 0 - refusing");
  }
  if (mine.length === 0 || mine.length !== theirs.length) {
    throw new RangeError(`not comparable: ${mine.length} vs ${theirs.length}`);
  }
  const n = live.length;
  let ln = 0; let ld = 0; let pn = 0; let pd = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const isLive = live[i] && live[j];
      const at = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) {
        const d = mine[at + c] - theirs[at + c];
        if (isLive) { ln += d * d; ld += theirs[at + c] * theirs[at + c]; }
        else { pn += d * d; pd += theirs[at + c] * theirs[at + c]; }
      }
    }
  }
  const ratio = (num, den) => (den > 0 ? Math.sqrt(num / den) : Math.sqrt(num));
  const value = ratio(ln, ld);
  if (!Number.isFinite(value)) throw new Error(`residual is ${value}`);
  return { live: value, padded: ratio(pn, pd) };
}

function deterministic(length, seed) {
  const out = new Float32Array(length);
  let state = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1103515245) + 12345) & 0x7fffffff;
    out[i] = (state / 0x7fffffff - 0.5) * 1.5;
  }
  return out;
}

export async function main(device, args = []) {
  const exportDir = option(args, "f32", "/../refs/export/boltz2-affinity-f32");
  const protein = Number(option(args, "protein", "16"));
  const ligand = Number(option(args, "ligand", "6"));
  const padded = Number(option(args, "padded", "3"));
  const layersToRun = Number(option(args, "layers", "2"));
  const precisions = option(args, "precision", "f32,f16").split(",");
  const bounds = { f32: 1e-5, f16: 3e-2 };

  // The export is read over the dev server gpu-chrome.mjs is serving.
  const base = exportDir.startsWith("/") ? exportDir : `/${exportDir}`;
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const shards = new Map();
  async function weight(name) {
    const record = manifest.tensors[name];
    if (record === undefined) return null;
    if (!shards.has(record.file)) {
      shards.set(record.file,
        await (await fetch(`${base}/${record.file}`)).arrayBuffer());
    }
    const count = record.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(
      new Float32Array(shards.get(record.file), record.byteOffset, count));
  }
  // affinityWeights wants a synchronous reader, so every tensor is fetched up
  // front - one shard, so this is one request.
  const cache = new Map();
  for (const name of Object.keys(manifest.tensors)) {
    cache.set(name, await weight(name));
  }
  const read = (name) => cache.get(name) ?? null;

  const channels = AFFINITY_TOKEN_Z;
  const n = protein + ligand + padded;
  const real = protein + ligand;
  const molType = new Float64Array(n);
  const affinityMask = new Float64Array(n);
  const padMask = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    molType[i] = i >= protein && i < real ? 3 : 0;
    affinityMask[i] = i >= protein && i < real ? 1 : 0;
    padMask[i] = i < real ? 1 : 0;
  }
  const mask64 = crossPairMask(molType, affinityMask, padMask, n, false);
  const mask = Float32Array.from(mask64);
  const live = Array.from(padMask, (v) => v > 0);
  const livePairs = mask.reduce((a, b) => a + (b > 0 ? 1 : 0), 0);

  const stack = new AffinityPairStackGpu(device);
  const results = {};
  let failed = 0;
  let worstLive = 0;
  let worstPadded = 0;

  for (const member of ["affinity_module1", "affinity_module2"]) {
    const weights = affinityWeights(read, member, member.endsWith("1") ? 8 : 4);
    const layers = weights.pairformer.slice(0, layersToRun);
    const pair32 = deterministic(n * n * channels, 4177 + n);
    const pair64 = Float64Array.from(pair32);

    // the reference, layer by layer
    const expected = [];
    let reference = pair64;
    for (const layer of layers) {
      reference = pairformerNoSeqLayer(reference, n, channels, layer, mask64,
                                       AFFINITY_HEADS, AFFINITY_HEAD_DIM);
      expected.push(reference);
    }

    for (const stagedPrecision of precisions) {
      if (stagedPrecision === "f16" && !device.features.has("shader-f16")) continue;
      // 🔴 ONE `precision`, WHICH THE DRIVER TRANSLATES INTO EACH KERNEL'S OWN
      // OPTION NAME. Forwarding `stagedPrecision` straight through reached the
      // grid attention only and left the triangle and the transition in f32 in
      // BOTH arms - see precisionOptions() in the driver, and
      // probe-affinity-steps.js, which prints each step at both precisions so
      // that an arm which is not reaching a kernel is visible.
      const run = await stack.run(
        pair32, mask, { n, channels, heads: AFFINITY_HEADS,
          dimension: AFFINITY_HEAD_DIM },
        layers, { precision: stagedPrecision });
      const bound = bounds[stagedPrecision];
      for (let index = 0; index < layers.length; index += 1) {
        const tag = `pairformer${String(index).padStart(2, "0")}`;
        const got = run.seams[tag];
        const { live: liveRms, padded } = split(got, expected[index], channels, live);
        const ok = liveRms <= bound;
        failed += ok ? 0 : 1;
        worstLive = Math.max(worstLive, liveRms);
        worstPadded = Math.max(worstPadded, padded);
        results[`${member}/${stagedPrecision}/${tag}`] = {
          live: liveRms, padded, bound, ok };
        console.log(`${member}\t${stagedPrecision}\t${tag}`
          + `\tlive ${liveRms.toExponential(2)}\tbound ${bound.toExponential(0)}`
          + `\t[padded ${padded.toExponential(2)}]\t${ok ? "OK" : "FAIL"}`);
      }
    }
  }

  // 🔴 THE SEPARATION CONTROL: the bound has to be able to tell a correct
  // kernel from a wrong remap. Without it the gate says only that two things
  // agree, and docs/ records three checkers here that could not fail.
  const weights = affinityWeights(read, "affinity_module1", 8);
  const broken = structuredClone(weights.pairformer[0]);
  const swap = broken.triMulOut;
  broken.triMulOut = broken.triMulIn;
  broken.triMulIn = swap;
  const pair32 = deterministic(n * n * channels, 4177 + n);
  const control = await stack.run(
    pair32, mask, { n, channels, heads: AFFINITY_HEADS, dimension: AFFINITY_HEAD_DIM },
    [broken], { precision: "f32" });
  const reference = pairformerNoSeqLayer(
    Float64Array.from(pair32), n, channels, weights.pairformer[0], mask64,
    AFFINITY_HEADS, AFFINITY_HEAD_DIM);
  const controlRms = split(control.seams.pairformer00, reference, channels, live).live;
  const discriminates = controlRms > bounds.f32 * 100;
  if (!discriminates) failed += 1;
  console.log(`control\tf32\ttri_mul out/in swapped`
    + `\tlive ${controlRms.toExponential(2)}`
    + `\t${discriminates ? "DISCRIMINATES" : "TOO CLOSE"}`);

  if (failed > 0) throw new Error(`${failed} affinity pair-stack seam(s) outside tolerance`);
  return {
    tokens: n, protein, ligand, padded, livePairs,
    layersCompared: layersToRun,
    worstLive, worstPadded, controlRms, bounds, results,
    note: "padded rows are REPORTED, never asserted: f32 erases a fully masked"
      + " row's logits where f64 keeps them - see this tool's header",
  };
}
