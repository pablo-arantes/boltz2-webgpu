#!/usr/bin/env node
/**
 * G2 - does the float64 affinity reference compute upstream PyTorch's function?
 *
 *   OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
 *     python3 tools/oracle/dump_boltz2_affinity.py --boltz=<checkout> \
 *       --f32=<export> --out=oracle-dumps/boltz2-affinity
 *   node tools/check-affinity-oracle.mjs --dump=oracle-dumps/boltz2-affinity \
 *     --f32=<export>
 *   node tools/check-affinity-oracle.mjs ... --falsify
 *
 * EVERY SEAM, NOT THE ANSWER. A single end-to-end number cannot localise a
 * fault, and this module is a chain of eleven stages where several of them
 * return a plausible tensor when they are wrong: the s_inputs term is a SUM
 * whose product version has the same shape, the two triangle multiplications
 * differ only in which index is summed, and the two masks differ by a
 * diagonal. So each of z_norm, z_linear, the two s projections, the bins, the
 * distogram embedding, the conditioner, every pairformer layer, the pooled
 * vector, the head MLP and all three outputs is compared on its own.
 *
 * 🔴 THE ORACLE IS FLOAT32 AND THE REFERENCE IS FLOAT64, SO THE BOUND IS A
 * MEASUREMENT AND NOT A HOPE. Upstream's triangle multiplications both do
 * `torch.chunk(x.float(), 2, dim=-1)` before the O(L^3) einsum - a deliberate
 * precision decision - so the module cannot be run in double at all and there
 * is no tight arm to be had. docs/ records the identical shape for the AF3
 * denoise checker, which "holds the GPU to the ORACLE and the CPU to the GPU"
 * for exactly this reason.
 *
 * What makes a loose bound meaningful is the SEPARATION CONTROL: `--falsify`
 * runs eight single-convention mutations - the sum as a product, the two
 * triangle einsums swapped, `>=` for the bins, the diagonal left in the
 * pooling mask, the attention bias transposed, the gate dropped, the two
 * conditioner residuals collapsed - and each must push its seam well past the
 * bound. A bound that cannot tell a correct port from a broken one is not a
 * bound, and nine of these would otherwise be invisible.
 *
 * 🔴 PAIR SEAMS ARE COMPARED ONLY WHERE BOTH TOKENS ARE LIVE, AND THE PADDED
 * REGION IS REPORTED RATHER THAN ASSERTED. This is the fourth time this
 * repository has had to learn the rule - docs/ records `ref_element` differing
 * in one slot of 1632 reading as a defect in four shipped models, where "no
 * gather on either side reads that slot live".
 *
 * Measured here: on live tokens the eight pairformer layers agree at 4.0e-7
 * growing to 1.0e-6, and the padded rows and columns diverge from 4.1e-2 to
 * 2.6e-1. The first version of this gate compared the whole pair and reported
 * eight FAILs while `pooled`, `out_mlp` and all three outputs agreed at 1e-7 -
 * two residuals that cannot both be true, which docs/ names as the fastest
 * route to the fault.
 *
 * THE MECHANISM, because it is not rounding and it matters for G3. A fully
 * masked attention row has `mask_bias = inf * (mask - 1) = -1e9` on EVERY key,
 * so every logit is `-1e9 + O(1)`. Float32 spacing near 1e9 is 64, so the O(1)
 * term is erased entirely and softmax returns a uniform 1/n; float64 keeps it
 * and returns a structured distribution. Both are garbage - the row is masked
 * out of everything downstream - but they are DIFFERENT garbage, and no amount
 * of correctness in the port will reconcile them.
 *
 * 🔴 SO G3 MUST USE THIS SAME LIVE-ONLY COMPARISON. The GPU runs f32 and will
 * match upstream's uniform rows; this f64 reference will not. A GPU-against-CPU
 * gate that compares padded rows would fail on a correct kernel.
 *
 * 🔴 AND THE BIN COMPARISON IS EXACT, BY CONSTRUCTION. The oracle reaches the
 * distogram through `torch.cdist`, whose |a-b|^2 expansion reads 4.77e-07 on a
 * diagonal that is exactly zero; this reference subtracts coordinates
 * directly. The dumper's input keeps every distance at least 8.7e-4 A from a
 * bin edge - ~360x the f32 error - so the two bucketings must agree EXACTLY,
 * and that is asserted rather than assumed.
 */
import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { affinityModule } from "../cpu/af3/affinity/affinity.js";
import { affinityWeights, affinityBoundaries, AFFINITY_MEMBERS, AFFINITY_TOKEN_S,
  AFFINITY_TOKEN_Z } from "../shared/af3/weights/affinity-weights.js";

const option = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const dumpDir = option("dump", "oracle-dumps/boltz2-affinity");
const exportDir = option("f32", "../refs/export/boltz2-affinity-f32");
const falsify = argv.includes("--falsify");
// 🔴 SET FROM THE MEASUREMENT. A correct port's worst LIVE residual is 5.5e-6
// (on affinity_pred_value, where an f32 oracle's accumulated rounding lands
// after eight triangle layers), so this leaves roughly an order of magnitude.
// The first version used 3e-4, which the separation control below shows is far
// looser than needed - every mutation exceeds it by three orders or more.
const BOUND = 5e-5;

const index = JSON.parse(readFileSync(`${dumpDir}/index.json`, "utf8"));
const dumpCache = new Map();
function oracle(name) {
  if (!dumpCache.has(name)) {
    const record = index.tensors[name];
    if (record === undefined) throw new Error(`no oracle seam ${name}`);
    const bytes = readFileSync(`${dumpDir}/${record.file}`);
    dumpCache.set(name, new Float64Array(
      bytes.buffer, bytes.byteOffset, bytes.byteLength / 8));
  }
  return dumpCache.get(name);
}

const manifest = JSON.parse(readFileSync(`${exportDir}/manifest.json`, "utf8"));
const shards = new Map();
function weight(name) {
  const record = manifest.tensors[name];
  if (record === undefined) return null;
  if (!shards.has(record.file)) {
    shards.set(record.file, readFileSync(`${exportDir}/${record.file}`));
  }
  const bytes = shards.get(record.file);
  const count = record.shape.reduce((a, b) => a * b, 1);
  const view = new Float32Array(bytes.buffer,
    bytes.byteOffset + record.byteOffset, count);
  return Float64Array.from(view);
}

/**
 * relRMS over the pairs where BOTH tokens are live, and over the padded ones
 * separately. `pairs` is null for a seam that is not pair-shaped.
 */
function relativeRmsMasked(a, b, channels, live, what) {
  if (!(a instanceof Float64Array) || !(b instanceof Float64Array)) {
    throw new TypeError(`${what}: relRMS over a non-array returns 0 - refusing`);
  }
  if (a.length === 0 || a.length !== b.length) {
    throw new RangeError(`${what}: ${a.length} against ${b.length}`);
  }
  const n = live.length;
  const acc = { liveNum: 0, liveDen: 0, padNum: 0, padDen: 0, livePairs: 0 };
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const isLive = live[i] && live[j];
      if (isLive) acc.livePairs += 1;
      const at = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) {
        const d = a[at + c] - b[at + c];
        if (isLive) { acc.liveNum += d * d; acc.liveDen += b[at + c] * b[at + c]; }
        else { acc.padNum += d * d; acc.padDen += b[at + c] * b[at + c]; }
      }
    }
  }
  const ratio = (num, den, count) =>
    (den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / Math.max(count, 1)));
  const value = ratio(acc.liveNum, acc.liveDen, acc.livePairs * channels);
  if (!Number.isFinite(value)) throw new Error(`${what}: residual is ${value}`);
  return { live: value, padded: ratio(acc.padNum, acc.padDen, 1),
    livePairs: acc.livePairs };
}

/** relRMS that REFUSES rather than scoring a comparison it cannot make. */
function relativeRms(a, b, what) {
  if (!(a instanceof Float64Array) || !(b instanceof Float64Array)) {
    throw new TypeError(`${what}: relRMS over a non-array returns 0 - refusing`);
  }
  if (a.length === 0 || a.length !== b.length) {
    throw new RangeError(`${what}: ${a.length} against ${b.length}`);
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  const value = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / a.length);
  // 🔴 `value > BOUND` is FALSE for NaN, which is how a checker passes without
  // computing anything. Asserted, not assumed.
  if (!Number.isFinite(value)) throw new Error(`${what}: residual is ${value}`);
  return value;
}

/**
 * One convention broken per mutation, expressed as a WEIGHT change.
 *
 * 🔴 THE FIRST VERSION OF THIS SET SHOWED NOTHING, AND LOOKED LIKE IT DID.
 * Each entry assigned a flag (`w.mutateProduct = true`) that the reference
 * never reads, so every arm computed the correct answer and the control would
 * have reported eight clean MISSES as the gate's own verdict. docs/ records
 * this exact shape three times: "its first version could not fail... the third
 * chirality check here to need rescuing from comparing something with itself."
 *
 * Expressing them as weight swaps keeps the falsification real without putting
 * a mutation hook into the module the page runs. Each one names the convention
 * it probes, and a swap that moves NOTHING is itself a finding - it would mean
 * the two roles are interchangeable and the port cannot be getting them wrong.
 */
const swap = (object, a, b) => { const v = object[a]; object[a] = object[b]; object[b] = v; };

const MUTATIONS = {
  // which axis each s_inputs projection broadcasts along
  "s_to_z_1 and _2 swapped": (w) => swap(w, "sToZ1", "sToZ2"),
  // bikd,bjkd vs bkid,bkjd - the two einsums
  "triangle mul out/in swapped": (w) => {
    for (const layer of w.pairformer) swap(layer, "triMulOut", "triMulIn");
  },
  // the ending node is the starting node on a transposed pair
  "triangle att start/end swapped": (w) => {
    for (const layer of w.pairformer) swap(layer, "triAttStart", "triAttEnd");
  },
  // the distogram's bin indexing: roll the table by one row
  "bin embedding rolled by one": (w) => {
    const cz = AFFINITY_TOKEN_Z;
    const rolled = new Float64Array(w.binEmbed.length);
    for (let bin = 0; bin < w.binEmbed.length / cz; bin += 1) {
      const from = ((bin + 1) % (w.binEmbed.length / cz)) * cz;
      rolled.set(w.binEmbed.subarray(from, from + cz), bin * cz);
    }
    w.binEmbed = rolled;
  },
  // q is scaled and k is not, so the two are not interchangeable
  "attention q and k swapped": (w) => {
    for (const layer of w.pairformer) {
      swap(layer.triAttStart, "q", "k"); swap(layer.triAttEnd, "q", "k");
    }
  },
  // the output gate: sigmoid(linear_g(q_x)) multiplying the gathered value
  "attention gate zeroed": (w) => {
    for (const layer of w.pairformer) {
      for (const att of [layer.triAttStart, layer.triAttEnd]) {
        att.g = new Float64Array(att.g.length);
      }
    }
  },
  // the conditioner's two transitions are applied in order
  "conditioner transitions swapped": (w) => {
    const list = w.conditioner.transitions;
    w.conditioner.transitions = [list[1], list[0]];
  },
  // 🔴 the binary logit is a 1->1 linear on pred_SCORE, not on pred_VALUE
  "pred_value/pred_score heads swapped": (w) => {
    for (const n of ["0Weight", "0Bias", "2Weight", "2Bias", "4Weight", "4Bias"]) {
      swap(w.heads2, `predValue${n}`, `predScore${n}`);
    }
  },
  "z_norm bias zeroed": (w) => {
    w.zNormBias = new Float64Array(AFFINITY_TOKEN_Z);
  },
};

const tokens = index.tokens;
const input = {
  tokens,
  tokenS: AFFINITY_TOKEN_S,
  tokenZ: AFFINITY_TOKEN_Z,
  boundaries: affinityBoundaries(index.numDistBins, index.maxDist),
  sInputs: oracle("input/s_inputs"),
  zAffinity: oracle("input/z_affinity"),
  positions: oracle("input/x"),
  molType: oracle("input/mol_type"),
  affinityMask: oracle("input/affinity_token_mask"),
  padMask: oracle("input/token_pad_mask"),
};

console.log(`oracle    : ${index.source}`);
console.log(`precision : ${index.precision ?? "?"}`);
console.log(`input     : ${index.protein} protein + ${index.ligand} ligand`
  + ` + ${index.padded} padded = ${tokens} tokens`);
console.log(`distogram : ${index.binsCovered}/64 bins covered,`
  + ` closest distance to an edge ${index.binEdgeMargin.toExponential(2)} A`);
console.log(`bound     : ${BOUND.toExponential(0)} (f64 reference against an f32 oracle)`);
console.log();

let failures = 0;
let worst = 0;
let worstName = "";
let worstPadded = 0;
// Which tokens the model actually reads. Everything downstream of the
// pairformer multiplies by a mask built from this, so a padded row's value is
// read by nothing - see the header.
const live = Array.from(input.padMask, (v) => v > 0);
console.log(`live tokens: ${live.filter(Boolean).length} of ${tokens}`
  + ` (${tokens - live.filter(Boolean).length} padded)`);
console.log();

for (const { scope, layers } of AFFINITY_MEMBERS) {
  const weights = affinityWeights(weight, scope, layers);
  const { seams, out } = affinityModule(input, weights);
  console.log(`=== ${scope} (${layers} pairformer layers) ===`);

  const compare = [
    ["z_norm", seams.z_norm], ["z_linear", seams.z_linear],
    ["s_to_z_1", seams.s_to_z_1], ["s_to_z_2", seams.s_to_z_2],
    ["conditioner/z_trunk", seams.z_plus_s],
    ["distogram_embed", seams.distogram_embed],
    ["conditioner_out", seams.conditioner_out],
    ...Array.from({ length: layers }, (_, i) =>
      [`pairformer${String(i).padStart(2, "0")}`,
        seams[`pairformer${String(i).padStart(2, "0")}`]]),
    ["pooled/in0", seams.pooled], ["out_mlp", seams.out_mlp],
  ];

  // the bins, exactly - see the header
  const oracleBins = oracle("input/bins");
  let binsDiffer = 0;
  for (let i = 0; i < oracleBins.length; i += 1) {
    if (oracleBins[i] !== seams.bins[i]) binsDiffer += 1;
  }
  const binsOk = binsDiffer === 0;
  failures += binsOk ? 0 : 1;
  console.log(`  ${binsOk ? "OK  " : "FAIL"} ${"distogram bins (exact)".padEnd(26)}`
    + ` ${binsDiffer} of ${oracleBins.length} differ`);

  for (const [name, mine] of compare) {
    const theirs = oracle(`${scope}/${name}`);
    const pairShaped = mine.length === tokens * tokens * AFFINITY_TOKEN_Z;
    if (!pairShaped) {
      const value = relativeRms(mine, theirs, `${scope}/${name}`);
      const ok = value <= BOUND;
      failures += ok ? 0 : 1;
      if (value > worst) { worst = value; worstName = `${scope}/${name}`; }
      console.log(`  ${ok ? "OK  " : "FAIL"} ${name.padEnd(26)}`
        + ` ${value.toExponential(3)}`);
      continue;
    }
    const split = relativeRmsMasked(mine, theirs, AFFINITY_TOKEN_Z, live,
                                    `${scope}/${name}`);
    const ok = split.live <= BOUND;
    failures += ok ? 0 : 1;
    if (split.live > worst) { worst = split.live; worstName = `${scope}/${name}`; }
    if (split.padded > worstPadded) worstPadded = split.padded;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${name.padEnd(26)}`
      + ` ${split.live.toExponential(3)}`
      + `   [padded ${split.padded.toExponential(2)}, reported]`);
  }

  for (const [name, mine] of [["affinity_pred_value", out.predValue],
    ["affinity_logits_binary", out.logitsBinary],
    ["affinity_probability_binary", out.probabilityBinary]]) {
    const theirs = oracle(`${scope}/out/${name}`)[0];
    const delta = Math.abs(mine - theirs);
    const scale = Math.max(Math.abs(theirs), 1e-6);
    const value = delta / scale;
    const ok = value <= BOUND;
    failures += ok ? 0 : 1;
    if (value > worst) { worst = value; worstName = `${scope}/${name}`; }
    console.log(`  ${ok ? "OK  " : "FAIL"} ${name.padEnd(26)} ${value.toExponential(3)}`
      + `   ours ${mine.toFixed(6)} oracle ${theirs.toFixed(6)}`);
  }
  console.log();
}

console.log(`worst LIVE residual  : ${worst.toExponential(3)} on ${worstName}`);
console.log(`worst PADDED residual: ${worstPadded.toExponential(3)}`
  + "  (reported, never asserted - f32 erases a fully masked row's logits)");
console.log();

if (falsify) {
  console.log("=== separation control: one convention broken at a time ===");
  const { scope, layers } = AFFINITY_MEMBERS[0];
  let separated = 0;
  for (const [label, mutate] of Object.entries(MUTATIONS)) {
    const weights = affinityWeights(weight, scope, layers);
    mutate(weights);
    let value = Infinity;
    try {
      const { seams, out } = affinityModule(input, weights);
      const theirs = oracle(`${scope}/out/affinity_pred_value`)[0];
      const onValue = Math.abs(out.predValue - theirs)
        / Math.max(Math.abs(theirs), 1e-6);
      // 🔴 AND ON A PAIR SEAM TOO, over LIVE pairs. A mutation inside the pair
      // track could in principle leave the pooled scalar close by coincidence;
      // taking the larger of the two means a convention has to be invisible at
      // BOTH places to be missed.
      const onPair = relativeRmsMasked(
        seams.pairformer00, oracle(`${scope}/pairformer00`),
        AFFINITY_TOKEN_Z, live, "falsify/pairformer00").live;
      value = Math.max(onValue, onPair);
    } catch (error) {
      value = Infinity;                 // a mutation that throws is also caught
    }
    const caught = !(value <= BOUND);
    separated += caught ? 1 : 0;
    console.log(`  ${caught ? "CAUGHT " : "MISSED "} ${label.padEnd(32)}`
      + ` ${Number.isFinite(value) ? value.toExponential(3) : "threw"}`);
  }
  console.log();
  const all = Object.keys(MUTATIONS).length;
  console.log(separated === all
    ? `SEPARATION OK: ${separated} of ${all} mutations exceed the bound`
    : `SEPARATION FAILED: ${separated} of ${all} - the bound cannot see`
      + " every convention it covers");
  if (separated !== all) failures += 1;
}

console.log(failures === 0
  ? "G2 PASS: the float64 reference computes upstream's function at every seam"
  : `G2 FAIL: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
