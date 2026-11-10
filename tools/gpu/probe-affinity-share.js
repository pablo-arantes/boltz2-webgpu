/** How the host/GPU split moves with the token count - O(L^2) glue vs O(L^3) stack. */
import { affinityForwardGpu } from "../../webgpu/af3/affinity/affinity.js";
import { affinityWeights, AFFINITY_TOKEN_S, AFFINITY_TOKEN_Z }
  from "../../shared/af3/weights/affinity-weights.js";

function noise(length, seed) {
  const out = new Float32Array(length);
  let s = seed >>> 0;
  for (let i = 0; i < length; i += 1) {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    out[i] = (s / 0x7fffffff - 0.5) * 0.8;
  }
  return out;
}

export async function main(device, args = []) {
  const base = "/model-boltz2-affinity-f32";
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const shard = await (await fetch(`${base}/weights-00.f32.bin`)).arrayBuffer();
  const read = (name) => {
    const r = manifest.tensors[name];
    if (!r) return null;
    const c = r.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(new Float32Array(shard, r.byteOffset, c));
  };
  const weights = affinityWeights(read, "affinity_module1", 8);
  const cz = AFFINITY_TOKEN_Z;
  const rows = [];
  for (const n of [35, 64, 128, 200, 256]) {
    const protein = Math.max(1, n - 8);
    const molType = new Float64Array(n);
    const affinityMask = new Float64Array(n);
    const padMask = new Float64Array(n).fill(1);
    for (let i = protein; i < n; i += 1) { molType[i] = 3; affinityMask[i] = 1; }
    const out = await affinityForwardGpu(device, {
      tokens: n, tokenS: AFFINITY_TOKEN_S, tokenZ: cz,
      sInputs: noise(n * AFFINITY_TOKEN_S, 11 + n),
      zAffinity: noise(n * n * cz, 22 + n),
      positions: noise(n * 3, 33 + n).map((v) => v * 30),
      molType, affinityMask, padMask,
    }, weights, { precision: "f32" });
    const total = out.timings["host.pre"] + out.timings["gpu.pairformer"]
      + out.timings["host.heads"];
    rows.push({ tokens: n, ...out.timings, totalMs: Number(total.toFixed(1)),
      gpuShare: Number((out.gpuShare * 100).toFixed(1)) });
    console.log(`${n}\thost.pre ${out.timings["host.pre"]}`
      + `\tgpu ${out.timings["gpu.pairformer"]}`
      + `\theads ${out.timings["host.heads"]}`
      + `\tgpuShare ${(out.gpuShare * 100).toFixed(1)}%`);
  }
  return { rows,
    note: "the glue is O(tokens^2) and the pairformer O(tokens^3), so the share"
      + " at 35 tokens says nothing about the share at a real crop of 256" };
}
