/** Which of the five steps in an affinity pairformer layer disagrees? */
import { AffinityPairStackGpu } from "../../webgpu/af3/affinity/pair-stack.js";
import { triangleMultiplication, triangleAttention, transition, crossPairMask }
  from "../../cpu/af3/affinity/affinity.js";
import { affinityWeights, AFFINITY_TOKEN_Z, AFFINITY_HEADS, AFFINITY_HEAD_DIM }
  from "../../shared/af3/weights/affinity-weights.js";

function liveRms(mine, theirs, channels, live) {
  const n = live.length;
  let num = 0; let den = 0;
  for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
    if (!(live[i] && live[j])) continue;
    const at = (i * n + j) * channels;
    for (let c = 0; c < channels; c += 1) {
      const d = mine[at + c] - theirs[at + c];
      num += d * d; den += theirs[at + c] * theirs[at + c];
    }
  }
  return Math.sqrt(num / den);
}
function deterministic(length, seed) {
  const out = new Float32Array(length); let s = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    out[i] = (s / 0x7fffffff - 0.5) * 1.5;
  }
  return out;
}

export async function main(device, args = []) {
  const base = "/model-boltz2-affinity-f32";
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const buf = await (await fetch(`${base}/weights-00.f32.bin`)).arrayBuffer();
  const read = (name) => {
    const r = manifest.tensors[name];
    if (!r) return null;
    const c = r.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(new Float32Array(buf, r.byteOffset, c));
  };
  const cz = AFFINITY_TOKEN_Z;
  const protein = 16; const ligand = 6; const padded = 3;
  const n = protein + ligand + padded; const real = protein + ligand;
  const molType = new Float64Array(n); const aff = new Float64Array(n);
  const pad = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    molType[i] = i >= protein && i < real ? 3 : 0;
    aff[i] = i >= protein && i < real ? 1 : 0;
    pad[i] = i < real ? 1 : 0;
  }
  const mask64 = crossPairMask(molType, aff, pad, n, false);
  const mask = Float32Array.from(mask64);
  const live = Array.from(pad, (v) => v > 0);
  const w = affinityWeights(read, "affinity_module1", 8);
  const layer = w.pairformer[0];
  const pair32 = deterministic(n * n * cz, 4177 + n);
  const pair64 = Float64Array.from(pair32);

  const stack = new AffinityPairStackGpu(device);
  const shape = { n, channels: cz, heads: AFFINITY_HEADS,
                  dimension: AFFINITY_HEAD_DIM };
  // 🔴 BOTH PRECISIONS, SO "THE ARM REACHED THE KERNEL" IS VISIBLE. Each of the
  // three kernels spells its precision option differently; forwarding one name
  // to all of them left two running f32 in both arms. A step whose residual
  // does NOT move between these columns is a step the f16 arm is not reaching.
  const runs = {};
  for (const precision of ["f32", "f16"]) {
    if (precision === "f16" && !device.features.has("shader-f16")) continue;
    runs[precision] = await stack.layer(pair32, mask, shape, layer, { precision });
  }

  // the reference's own five steps, each residual
  const add = (a, b) => { const o = Float64Array.from(a); for (let i = 0; i < o.length; i += 1) o[i] += b[i]; return o; };
  let r = pair64;
  const steps = {};
  r = add(r, triangleMultiplication(r, n, cz, layer.triMulOut, mask64, true));
  steps.tri_mul_out = r;
  r = add(r, triangleMultiplication(r, n, cz, layer.triMulIn, mask64, false));
  steps.tri_mul_in = r;
  r = add(r, triangleAttention(r, n, cz, layer.triAttStart, mask64, AFFINITY_HEADS, AFFINITY_HEAD_DIM, true));
  steps.tri_att_start = r;
  r = add(r, triangleAttention(r, n, cz, layer.triAttEnd, mask64, AFFINITY_HEADS, AFFINITY_HEAD_DIM, false));
  steps.tri_att_end = r;
  r = add(r, transition(r, n * n, cz, cz * 4, layer.transitionZ));
  steps.transition_z = r;

  const out = {};
  console.log("step\tf32\tf16\tf16 moved?");
  for (const tag of Object.keys(steps)) {
    const row = {};
    for (const precision of Object.keys(runs)) {
      row[precision] = liveRms(runs[precision].seams[tag], steps[tag], cz, live);
    }
    const moved = row.f16 !== undefined && row.f16 > row.f32 * 2;
    out[tag] = { f32: Number(row.f32.toExponential(3)),
      f16: row.f16 === undefined ? null : Number(row.f16.toExponential(3)),
      f16Reached: moved };
    console.log(`${tag}\t${row.f32.toExponential(2)}`
      + `\t${row.f16 === undefined ? "-" : row.f16.toExponential(2)}`
      + `\t${moved ? "YES" : "no"}`);
  }
  return { tokens: n, steps: out };
}
