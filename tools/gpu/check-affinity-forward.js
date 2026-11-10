/**
 * Does the GPU affinity module compute the whole function, outputs included?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-forward.js \
 *     --dump=/oracle-dumps/boltz2-affinity --f32=/model-boltz2-affinity-f32
 *
 * G3 holds the PAIRFORMER to the float64 reference. This holds the WHOLE
 * module - the two pair LayerNorms, the projections, the broadcast sum, the
 * distogram, the conditioner, the stack, the pooling and all three heads - to
 * upstream's own dump, so the number a reader ranks on is compared against
 * PyTorch rather than against another of our own implementations.
 *
 * 🔴 IT USES G2's ORACLE, WHICH IS WHAT MAKES IT A CHECK AND NOT A
 * SELF-COMPARISON. `tools/oracle/dump_boltz2_affinity.py` recorded every seam
 * of upstream's real module under forward hooks; the GPU path reads the same
 * inputs out of that dump and is compared against the same seams. The float32
 * glue in webgpu/af3/affinity/affinity.js is written independently of
 * cpu/af3/affinity/ for the same reason - sharing it would compare those
 * stages with themselves, which docs/ records three checkers here needing
 * rescue from.
 *
 * 🔴 PAIR SEAMS LIVE-ONLY, OUTPUTS WHOLE. A fully masked attention row is
 * -1e9 on every key and f32 spacing there is 64, so the logits collapse to
 * uniform where f64 keeps them - different garbage in a region nothing reads.
 * The OUTPUTS have no such caveat: they are pooled over the cross-pair mask,
 * which excludes padded tokens, so they are compared outright.
 */
import { affinityForwardGpu } from "../../webgpu/af3/affinity/affinity.js";
import { affinityWeights, AFFINITY_TOKEN_S, AFFINITY_TOKEN_Z }
  from "../../shared/af3/weights/affinity-weights.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

function relRms(mine, theirs, what) {
  if (!(mine instanceof Float32Array) && !(mine instanceof Float64Array)) {
    throw new TypeError(`${what}: relRMS over a non-array returns 0 - refusing`);
  }
  if (mine.length === 0 || mine.length !== theirs.length) {
    throw new RangeError(`${what}: ${mine.length} against ${theirs.length}`);
  }
  let num = 0;
  let den = 0;
  for (let i = 0; i < mine.length; i += 1) {
    const d = mine[i] - theirs[i];
    num += d * d;
    den += theirs[i] * theirs[i];
  }
  const value = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num / mine.length);
  if (!Number.isFinite(value)) throw new Error(`${what}: residual is ${value}`);
  return value;
}

function liveOnly(mine, theirs, channels, live, what) {
  const n = live.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      if (!(live[i] && live[j])) continue;
      const at = (i * n + j) * channels;
      for (let c = 0; c < channels; c += 1) {
        const d = mine[at + c] - theirs[at + c];
        num += d * d;
        den += theirs[at + c] * theirs[at + c];
      }
    }
  }
  const value = den > 0 ? Math.sqrt(num / den) : Math.sqrt(num);
  if (!Number.isFinite(value)) throw new Error(`${what}: residual is ${value}`);
  return value;
}

export async function main(device, args = []) {
  const dumpBase = option(args, "dump", "/oracle-dumps/boltz2-affinity");
  const exportBase = option(args, "f32", "/model-boltz2-affinity-f32");
  const precisions = option(args, "precision", "f32,f16").split(",");

  const index = await (await fetch(`${dumpBase}/index.json`)).json();
  const seamCache = new Map();
  async function oracle(name) {
    if (!seamCache.has(name)) {
      const record = index.tensors[name];
      if (record === undefined) throw new Error(`no oracle seam ${name}`);
      const buffer = await (await fetch(`${dumpBase}/${record.file}`)).arrayBuffer();
      seamCache.set(name, new Float64Array(buffer));
    }
    return seamCache.get(name);
  }

  const manifest = await (await fetch(`${exportBase}/manifest.json`)).json();
  const shard = await (await fetch(`${exportBase}/weights-00.f32.bin`)).arrayBuffer();
  const read = (name) => {
    const record = manifest.tensors[name];
    if (record === undefined) return null;
    const count = record.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(new Float32Array(shard, record.byteOffset, count));
  };

  const n = index.tokens;
  const cz = AFFINITY_TOKEN_Z;
  const toF32 = (a) => Float32Array.from(a);
  const input = {
    tokens: n, tokenS: AFFINITY_TOKEN_S, tokenZ: cz,
    sInputs: toF32(await oracle("input/s_inputs")),
    zAffinity: toF32(await oracle("input/z_affinity")),
    positions: toF32(await oracle("input/x")),
    molType: await oracle("input/mol_type"),
    affinityMask: await oracle("input/affinity_token_mask"),
    padMask: await oracle("input/token_pad_mask"),
  };
  const live = Array.from(input.padMask, (v) => v > 0);
  const bounds = { f32: 1e-5, f16: 3e-2 };

  console.log(`oracle    : ${index.source}`);
  console.log(`precision : ${index.precision ?? "?"}`);
  console.log(`input     : ${n} tokens`
    + ` (${live.filter(Boolean).length} live, ${n - live.filter(Boolean).length} padded)`);
  console.log();

  const results = {};
  let failed = 0;
  let worst = 0;

  for (const member of ["affinity_module1", "affinity_module2"]) {
    const layers = member.endsWith("1") ? 8 : 4;
    const weights = affinityWeights(read, member, layers);
    for (const precision of precisions) {
      if (precision === "f16" && !device.features.has("shader-f16")) continue;
      const bound = bounds[precision];
      const out = await affinityForwardGpu(device, input, weights, { precision });

      // the deepest pair seam, live-only
      const deepest = `pairformer${String(layers - 1).padStart(2, "0")}`;
      const pairRms = liveOnly(out.pair, await oracle(`${member}/${deepest}`),
                               cz, live, `${member}/${deepest}`);

      // the pooled vector and the head MLP output, whole
      const pooledRms = relRms(out.pooled, await oracle(`${member}/pooled/in0`),
                               `${member}/pooled`);
      const mlpRms = relRms(out.g, await oracle(`${member}/out_mlp`),
                            `${member}/out_mlp`);

      const scalars = {};
      for (const [tag, mine] of [
        ["affinity_pred_value", out.predValue],
        ["affinity_logits_binary", out.logitsBinary],
        ["affinity_probability_binary", out.probabilityBinary]]) {
        const theirs = (await oracle(`${member}/out/${tag}`))[0];
        scalars[tag] = { ours: mine, oracle: theirs,
          rel: Math.abs(mine - theirs) / Math.max(Math.abs(theirs), 1e-6) };
      }

      const checks = { pair: pairRms, pooled: pooledRms, outMlp: mlpRms,
        ...Object.fromEntries(Object.entries(scalars).map(([k, v]) => [k, v.rel])) };
      for (const [tag, value] of Object.entries(checks)) {
        const ok = value <= bound;
        failed += ok ? 0 : 1;
        worst = Math.max(worst, value);
        console.log(`${member}\t${precision}\t${tag.padEnd(28)}`
          + `\t${value.toExponential(2)}\tbound ${bound.toExponential(0)}`
          + `\t${ok ? "OK" : "FAIL"}`);
      }
      console.log(`${member}\t${precision}\tpred_value ours`
        + ` ${scalars.affinity_pred_value.ours.toFixed(6)}`
        + ` oracle ${scalars.affinity_pred_value.oracle.toFixed(6)}`
        + `\tprobability ${scalars.affinity_probability_binary.ours.toFixed(6)}`);
      console.log(`${member}\t${precision}\ttimings ${JSON.stringify(out.timings)}`
        + `\tgpuShare ${(out.gpuShare * 100).toFixed(0)}%`);
      results[`${member}/${precision}`] = { checks, scalars,
        timings: out.timings, gpuShare: out.gpuShare };
    }
  }

  if (failed > 0) throw new Error(`${failed} affinity forward seam(s) outside tolerance`);
  return { tokens: n, worst, results,
    note: "pair seams are live-only (f32 erases a fully masked row's logits);"
      + " the outputs pool over a mask that excludes padded tokens and are"
      + " compared whole" };
}
