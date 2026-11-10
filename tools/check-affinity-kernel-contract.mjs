#!/usr/bin/env node
/**
 * Does LocalFold's AF3 triangle kernel accept Boltz-2's affinity weights?
 *
 *   node tools/check-affinity-kernel-contract.mjs [--f32=<export>]
 *
 * 🔴 THE WHOLE GPU PORT RESTS ON THIS ANSWER, SO IT IS MEASURED. The Phase 2
 * plan proposed driving the affinity pairformer with the kernels already here -
 * triangle multiplication, grid attention, transition - behind a pair-only
 * driver. That is only sound if the kernels' WEIGHT CONTRACTS mean the same
 * thing Boltz's do, and the two checkpoints lay the same arithmetic out
 * differently:
 *
 *   AF3 splits the two projections into FOUR tensors - linearAP, linearAG,
 *   linearBP, linearBG - where Boltz carries `p_in [2*cz, cz]` and
 *   `g_in [2*cz, cz]` and splits them with `torch.chunk(x, 2, dim=-1)` AFTER
 *   projecting. Rows 0..cz-1 are `a` and cz..2cz-1 are `b`.
 *
 * Reading both implementations says they agree on the three conventions that
 * could have differed, and this gate is what turns that reading into a number:
 *
 *   * THE OUTPUT GATE READS THE NORMALISED INPUT, NOT THE RAW PAIR.
 *     cpu-reference.js reassigns `z` to the normalised tensor on its first
 *     line and then gates with `project(z, ...)`; Boltz keeps `x_in =
 *     self.norm_in(x)` and gates with `g_out(x_in)`. Had either used the raw
 *     pair this would be a different function with the same shape.
 *   * BOTH HALVES ARE MASKED. AF3 multiplies the mask into `ap` and `bp`
 *     separately; Boltz masks the concatenated 2*cz tensor, which is the same
 *     thing one step earlier.
 *   * THE CONTRACTION SUMS THE SECOND PAIR INDEX for outgoing -
 *     `ap[i*n+k] * bp[j*n+k]`, which is Boltz's `bikd,bjkd->bijd`.
 *
 * 🔴 AND THE BIASES ARE ZEROS BECAUSE BOLTZ HAS NONE. Every projection in the
 * affinity pairformer is a `LinearNoBias`, so the AF3 contract's six bias
 * slots are filled with zeros rather than left undefined - `weight-sources.js`
 * records what an absent weight reaching a consumer costs.
 *
 * 🔴 ONLY THE OUTGOING DIRECTION IS COVERED HERE, because that is all the host
 * reference implements (`grep -c Incoming` is 0 in cpu-reference.js). The
 * incoming direction's contract is therefore NOT established by this gate and
 * is left to the GPU differential against the float64 reference, which covers
 * both. Saying so is the point: an untested direction that looks tested is
 * worse than a gap.
 */
import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { triangleMultiplicationOutgoingReference }
  from "../webgpu/kernels/triangle/cpu-reference.js";
import { gridSelfAttention, transition as af3Transition,
  triangleMultiplication as af3TriangleMultiplication }
  from "../cpu/af3/trunk/pairformer.js";
import { triangleMultiplication, triangleAttention, transition }
  from "../cpu/af3/affinity/affinity.js";
import { affinityWeights, AFFINITY_TOKEN_Z, af3TriangleContract,
  af3GridContract, af3TransitionContract, transposeOutIn,
  af3InterleavedTriangleContract }
  from "../shared/af3/weights/affinity-weights.js";

const option = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const exportDir = option("f32", "../refs/export/boltz2-affinity-f32");
// f32 host reference against an f64 one: agreement is f32 rounding.
const BOUND = 1e-5;

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
  return Float64Array.from(new Float32Array(
    bytes.buffer, bytes.byteOffset + record.byteOffset, count));
}



function relRms(a, b) {
  if (a.length === 0 || a.length !== b.length) {
    throw new RangeError(`not comparable: ${a.length} against ${b.length}`);
  }
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

const cz = AFFINITY_TOKEN_Z;

// A deterministic input with a ragged mask, since the mask is one of the three
// conventions under test and an all-ones draw cannot see it.
const n = 12;
let state = 7;
const next = () => {
  state = (state * 1103515245 + 12345) & 0x7fffffff;
  return state / 0x7fffffff - 0.5;
};
const pair = new Float64Array(n * n * cz);
for (let i = 0; i < pair.length; i += 1) pair[i] = next() * 1.5;
const mask = new Float64Array(n * n);
for (let i = 0; i < n * n; i += 1) mask[i] = next() > -0.25 ? 1 : 0;

console.log(`export    : ${exportDir}`);
console.log(`input     : ${n} tokens, ${cz} channels,`
  + ` ${mask.reduce((a, b) => a + b, 0)} of ${n * n} pairs live`);
console.log(`bound     : ${BOUND.toExponential(0)} (f32 host reference against f64)`);
console.log();

let failures = 0;
for (const member of ["affinity_module1", "affinity_module2"]) {
  const weights = affinityWeights(weight, member, member.endsWith("1") ? 8 : 4);
  for (let layer = 0; layer < weights.pairformer.length; layer += 1) {
    const tri = weights.pairformer[layer].triMulOut;
    const mine = triangleMultiplication(pair, n, cz, tri, mask, true);
    const theirs = triangleMultiplicationOutgoingReference({
      shape: { length: n, cZ: cz, cHidden: cz }, epsilon: 1e-5,
      z: Float32Array.from(pair), mask: Float32Array.from(mask),
      weights: af3TriangleContract(tri, cz),
    });
    const value = relRms(Float64Array.from(theirs), mine);
    const ok = value <= BOUND;
    failures += ok ? 0 : 1;
    if (layer === 0 || !ok) {
      console.log(`  ${ok ? "OK  " : "FAIL"} ${member} layer ${layer}`
        + ` tri_mul_out   ${value.toExponential(3)}`);
    }
  }
}

// --- triangle multiplication, BOTH directions -------------------------------
// 🔴 THIS ARM EXISTS BECAUSE ITS ABSENCE LET A REAL BUG THROUGH. The gate above
// uses webgpu/kernels/triangle/cpu-reference.js, which implements OUTGOING
// only, and this header used to say so and stop there. `af3TriangleContract`
// shipped with a and b the wrong way round for INCOMING - AF3 contracts
// `a[k][j] * b[k][i]` where Boltz contracts `a[k][i] * b[k][j]` - and
// tools/gpu/check-affinity-pair-stack.js found it at 0.304 while tri_mul_out
// was exact. A declared gap is still a gap.
//
// cpu/af3/trunk/pairformer.js takes a `direction`, so the host CAN check both;
// it wants AF3's interleaved layout, which af3InterleavedTriangleContract
// builds.
for (const member of ["affinity_module1", "affinity_module2"]) {
  const weights = affinityWeights(weight, member, member.endsWith("1") ? 8 : 4);
  for (const [label, key, direction] of [
    ["tri_mul_out (interleaved)", "triMulOut", "outgoing"],
    ["tri_mul_in  (interleaved)", "triMulIn", "incoming"]]) {
    const tri = weights.pairformer[0][key];
    const mine = triangleMultiplication(pair, n, cz, tri, mask,
                                        direction === "outgoing");
    const theirs = af3TriangleMultiplication(
      Float32Array.from(pair), Float32Array.from(mask), n, cz, direction,
      af3InterleavedTriangleContract(tri, cz, direction), {});
    const value = relRms(Float64Array.from(theirs), mine);
    const ok = value <= BOUND;
    failures += ok ? 0 : 1;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${member} ${label}  ${value.toExponential(3)}`);
  }
}

// 🔴 AND THE DIRECTION SWAP IS PINNED: feeding incoming the OUTGOING a/b order
// must disagree, or the swap above is decoration.
{
  const weights = affinityWeights(weight, "affinity_module1", 8);
  const tri = weights.pairformer[0].triMulIn;
  const mine = triangleMultiplication(pair, n, cz, tri, mask, false);
  const theirs = af3TriangleMultiplication(
    Float32Array.from(pair), Float32Array.from(mask), n, cz, "incoming",
    af3InterleavedTriangleContract(tri, cz, "outgoing"), {});
  const value = relRms(Float64Array.from(theirs), mine);
  const pinned = value > BOUND * 100;
  failures += pinned ? 0 : 1;
  console.log(`  ${pinned ? "PINNED " : "AMBIG  "} `
    + `${"incoming fed outgoing a/b order".padEnd(30)} ${value.toExponential(3)}`);
}

// --- grid attention, both directions ---------------------------------------
// 🔴 BOLTZ2's DIALECT SETS `swapTransposedBias: true`, which is what AF3's
// reference needs to take the bias from the transposed activation - which is
// where Boltz's own `linear(x)` takes it, since it runs AFTER the transpose.
const boltz2Dialect = { swapTransposedBias: true };
for (const member of ["affinity_module1", "affinity_module2"]) {
  const weights = affinityWeights(weight, member, member.endsWith("1") ? 8 : 4);
  for (const [label, key, starting] of [
    ["tri_att_start", "triAttStart", true], ["tri_att_end", "triAttEnd", false]]) {
    const att = weights.pairformer[0][key];
    const mine = triangleAttention(pair, n, cz, att, mask,
                                   weights.heads, weights.headDim, starting);
    const theirs = gridSelfAttention(
      Float32Array.from(pair), Float32Array.from(mask), n, cz, !starting,
      af3GridContract(att, cz, weights.heads, weights.headDim), boltz2Dialect);
    const value = relRms(Float64Array.from(theirs), mine);
    const ok = value <= BOUND;
    failures += ok ? 0 : 1;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${member} layer 0 ${label}`
      + `${" ".repeat(Math.max(0, 14 - label.length))}${value.toExponential(3)}`);
  }
}

// --- transition -------------------------------------------------------------
for (const member of ["affinity_module1", "affinity_module2"]) {
  const weights = affinityWeights(weight, member, member.endsWith("1") ? 8 : 4);
  const cases = [
    ["transition_z (factor 4)", weights.pairformer[0].transitionZ, cz * 4, 4],
    ["conditioner t0 (factor 2)", weights.conditioner.transitions[0], cz * 2, 2],
  ];
  for (const [label, t, hidden, factor] of cases) {
    const mine = transition(pair, n * n, cz, hidden, t);
    const theirs = af3Transition(Float32Array.from(pair), n * n, cz,
                                 af3TransitionContract(t, cz, hidden), factor);
    const value = relRms(Float64Array.from(theirs), mine);
    const ok = value <= BOUND;
    failures += ok ? 0 : 1;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${member} ${label}`
      + `${" ".repeat(Math.max(1, 26 - label.length))}${value.toExponential(3)}`);
  }
}

// --- layout discrimination --------------------------------------------------
// 🔴 THREE TENSORS' ORIENTATIONS WERE READ OFF THE SOURCE RATHER THAN THE
// SHAPES, BECAUSE ALL THREE ARE SQUARE OR SQUARE-ISH AND A WRONG ORIENTATION IS
// NOT A SHAPE ERROR. Each is re-run with the OTHER layout and must disagree,
// so the choice is pinned by a measurement instead of by a comment.
console.log();
{
  const weights = affinityWeights(weight, "affinity_module1", 8);
  const att = weights.pairformer[0].triAttStart;
  const good = af3GridContract(att, cz, weights.heads, weights.headDim);
  const mine = triangleAttention(pair, n, cz, att, mask,
                                 weights.heads, weights.headDim, true);
  const arms = {
    "v NOT transposed": { ...good, vProjection: Float32Array.from(att.v) },
    "output NOT transposed": { ...good, outputProjection: Float32Array.from(att.o) },
    "q transposed (should not be)": { ...good,
      qProjection: transposeOutIn(Float32Array.from(att.q),
                                  weights.heads * weights.headDim, cz) },
  };
  for (const [label, armWeights] of Object.entries(arms)) {
    const theirs = gridSelfAttention(
      Float32Array.from(pair), Float32Array.from(mask), n, cz, false,
      armWeights, boltz2Dialect);
    const value = relRms(Float64Array.from(theirs), mine);
    const discriminates = value > BOUND * 100;
    failures += discriminates ? 0 : 1;
    console.log(`  ${discriminates ? "PINNED " : "AMBIG  "} ${label.padEnd(30)}`
      + ` ${value.toExponential(3)}`);
  }
  const t = weights.pairformer[0].transitionZ;
  const goodT = af3TransitionContract(t, cz, cz * 4);
  const fc1 = Float32Array.from(t.fc1);
  const fc2 = Float32Array.from(t.fc2);
  const swappedPacked = new Float32Array(fc1.length + fc2.length);
  swappedPacked.set(fc2, 0);
  swappedPacked.set(fc1, fc2.length);
  const mineT = transition(pair, n * n, cz, cz * 4, t);
  const theirsT = af3Transition(Float32Array.from(pair), n * n, cz,
    { ...goodT, transition1: transposeOutIn(swappedPacked, cz * 8, cz) }, 4);
  const valueT = relRms(Float64Array.from(theirsT), mineT);
  const disc = valueT > BOUND * 100;
  failures += disc ? 0 : 1;
  console.log(`  ${disc ? "PINNED " : "AMBIG  "} ${"fc1/fc2 halves swapped".padEnd(30)}`
    + ` ${valueT.toExponential(3)}`);
}

// 🔴 THE CONTROL: a remap that is wrong in one role must NOT agree. Without
// this the gate says only that two implementations of something agree.
const control = (() => {
  const weights = affinityWeights(weight, "affinity_module1", 8);
  const tri = weights.pairformer[0].triMulOut;
  const good = af3TriangleContract(tri, cz);
  const swapped = { ...good,
    linearAPWeight: good.linearBPWeight, linearBPWeight: good.linearAPWeight };
  const mine = triangleMultiplication(pair, n, cz, tri, mask, true);
  const theirs = triangleMultiplicationOutgoingReference({
    shape: { length: n, cZ: cz, cHidden: cz }, epsilon: 1e-5,
    z: Float32Array.from(pair), mask: Float32Array.from(mask), weights: swapped,
  });
  return relRms(Float64Array.from(theirs), mine);
})();
console.log();
console.log(`  control (a/b projection rows swapped): ${control.toExponential(3)}`
  + (control > BOUND * 100 ? "  <- the bound discriminates" : "  <- TOO CLOSE"));
if (!(control > BOUND * 100)) failures += 1;

console.log();
console.log(failures === 0
  ? "CONTRACT PASS: AF3's triangle kernel accepts Boltz's affinity weights"
    + " - triangle mul, grid attention both ways, and the transition"
  : `CONTRACT FAIL: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
