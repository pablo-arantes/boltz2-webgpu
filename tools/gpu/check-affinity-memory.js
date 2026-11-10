/**
 * G7: does the screening loop leak GPU memory across ligands?
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-memory.js
 *   node tools/gpu-chrome.mjs tools/gpu/check-affinity-memory.js --free=off
 *
 * 🔴 THE FAILURE THIS IS WRITTEN FOR IS A MEASURED ONE, FROM THE OTHER SIDE OF
 * THE SAME TREE. docs/ records `npm run test:switch`: ten models in one page
 * went 677 -> 1281 -> 2171 -> 2874 MiB and the fourth fold died in validation,
 * because residency kept between folds was kept across a CHANGE of model. A
 * screen is the same shape with one difference that matters - every ligand is
 * the SAME family, so `releaseAllWeights` would be exactly wrong here (365 MiB
 * re-packed per ligand for nothing) and what must come back per ligand is the
 * RUN's own buffers. web/affinity-run.js says so in its header and frees them
 * through the caller's `freeRunBuffers`; nothing was passing one.
 *
 * 🔴 AND "MEMORY GREW" IS NOT A LEAK WHEN THE LIGANDS DIFFER IN SIZE. A
 * 10-atom ligand crops to 43 tokens where a 3-atom one crops to 36, so every
 * pair tensor is larger and a monotone rise across a varied list is the correct
 * behaviour of a correct loop. So there are two phases and only the first can
 * fail on growth:
 *
 *   repeat  ONE ligand N times. Every shape is identical, so any growth at all
 *           is the loop holding something, and the bound is tight.
 *   varied  N ligands of different sizes. The bound is one ligand's worth,
 *           because the pool keys on exact byte length and a new size brings
 *           its own buffers.
 *
 * 🔴 MEASURED ON AN M2 (metal-3), 33-residue target, 4 sampler steps, 1
 * diffusion sample, three repeats:
 *
 *   arm                resident        peak   host-pack/ligand   ms/ligand
 *   shipped          19 MiB flat    727 MiB         71 MiB         1094
 *   --keep=on       802 MiB flat    818 MiB          9 MiB          998
 *   --pooling=off  512->1004->1497 1839 MiB         71 MiB         1252
 *
 * 🔴 `--free=off` MOVES NOTHING, AND THAT IS THE REPORT RATHER THAN A PASS.
 * The drain has nothing per-ligand to reclaim here: every allocator on the AF3
 * path but `WebGpuExecution`'s is non-pooling, so a released buffer is
 * destroyed at release and `settleReleasedMemory` only tells the driver what it
 * has already been told. So that arm is NOT this gate's control - it cannot
 * vary the thing under test, which docs/ names as the trap it is.
 * `--pooling=off` is the control: the allocator's own switch, whose comment
 * says a released buffer is then "kept alive and never handed back", and it
 * reproduces test:switch's failure shape inside this loop - 492 MiB a ligand,
 * 4131 MiB after four varied ones, against a flat 19.
 *
 * 🔴 AND THE WEIGHT-RESIDENCY TRADE IS PRICED AND DECLINED. `keepTrunkWeights`
 * alone buys EXACTLY NOTHING - identical pack calls, identical runtime, for
 * 239 MiB - because what this loop re-packs every fold is the diffusion
 * CONDITIONER's (`cond.single-transition-w0`/`-w1`, 13.5 MiB a call), which
 * `keepSamplerWeights` governs. With both: host packing 71 -> 9 MiB a ligand
 * and 1094 -> 998 ms, which is 8.8% at 4 steps, **3.9% at 50 steps**
 * (3161 -> 3039) and under 1% at upstream's 200, for 783 MiB of residency. So
 * the shipped rule - give them back every fold - is right for a screen too,
 * and that is now a measurement rather than an assumption.
 */
import { openAf3Store, trunkDepths, trunkWeights, confidenceWeights }
  from "../../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../shared/af3/weights/diffusion-weights.js";
import { makeFoldStructure, makeFoldAffinityPass, makeAffinityModule }
  from "../../web/affinity-passes.js";
import { runAffinityScreen } from "../../web/affinity-run.js";
import { parseSmiles } from "../../shared/chem/smiles.js";
import { memorySnapshot } from "../../webgpu/runtime/device-memory.js";
import { settleReleasedMemory, setBufferPooling }
  from "../../webgpu/runtime/allocator.js";
import { residentPackStats } from "../../webgpu/runtime/resident.js";
import { setDeviceTuning } from "../../webgpu/runtime/device-profile.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const mib = (bytes) => Math.round(bytes / 1048576);

/** One ligand's worth of growth, and a tight bound for identical shapes. */
const REPEAT_BOUND_MIB = 24;
const VARIED_BOUND_MIB = 256;

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const affinityBase = option(args, "affinity", "/model-boltz2-affinity-f32");
  const sequence = option(args, "sequence",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ");
  const steps = Number(option(args, "steps", "4"));
  const samples = Number(option(args, "samples", "1"));
  const repeats = Number(option(args, "repeats", "4"));
  const freeing = option(args, "free", "on") !== "off";
  // 🔴 THE SHIPPED RULE GIVES THE TRUNK WEIGHTS BACK EVERY FOLD, AND A SCREEN
  // IS NOT ONE FOLD. `keepTrunkWeights` is null everywhere but the ampere
  // prior - "give them back every fold" - which is the right trade for a page
  // that folds once and the wrong arithmetic for a loop that folds
  // `1 + diffusionSamples` times a ligand, hundreds of ligands deep. This arm
  // is what prices it rather than arguing it; `--keep=on` holds them.
  const keeping = option(args, "keep", "off") === "on";
  // 🔴 THE ARM THAT ACTUALLY VARIES THE THING UNDER TEST. `--free=off` does
  // not (see the report): there is nothing per-ligand for a drain to reclaim
  // on this path, so dropping the drain moves nothing and an unchanged number
  // proves nothing - docs/ calls that "a control arm that cannot vary the
  // thing under test is not a control". `setBufferPooling(false)` is a switch
  // the shipped allocator already has, and its own comment says what it does:
  // a released buffer is "kept alive and never handed back". That is a real
  // leak, in real code, and it is what this bound exists to catch.
  const leaking = option(args, "pooling", "on") === "off";
  if (leaking) setBufferPooling(false);
  // 🔴 AND IT IS `keepSamplerWeights` THAT MATTERS HERE, NOT `keepTrunkWeights`.
  // Measured: with the trunk's kept alone the host-pack calls and the runtime
  // are IDENTICAL (212/408/604 calls, 1049 ms both ways) for 239 MiB of extra
  // residency - because what this loop re-packs every fold is the DIFFUSION
  // CONDITIONER's, named by the per-label table: `cond.single-transition-w0`
  // and `-w1` at 13.5 MiB a call. Both are set, and the table says which paid.
  if (keeping) {
    setDeviceTuning(device,
      { keepTrunkWeights: true, keepSamplerWeights: true });
  }
  // 🔴 SIZES THAT DIFFER, because the pool keys on EXACT byte length: a list of
  // same-size ligands cannot show pool accumulation at all.
  const varied = option(args, "ligands",
    "CCO,CCCc1ccccc1O,CC(=O)Oc1ccccc1C(=O)O,CN1CCC[C@H]1c1cccnc1").split(",");

  const store = await openAf3Store(manifest);
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks,
                              { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
    confidence: await confidenceWeights(store),
  };

  const affinityManifest = await (await fetch(`${affinityBase}/manifest.json`)).json();
  const shard = await (await fetch(`${affinityBase}/weights-00.f32.bin`)).arrayBuffer();
  const readAffinity = (name) => {
    const record = affinityManifest.tensors[name];
    if (record === undefined) return null;
    const count = record.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(new Float32Array(shard, record.byteOffset, count));
  };

  const samplesSeen = [];
  const deps = {
    parseLigand: async (smiles) => parseSmiles(smiles),
    foldStructure: makeFoldStructure(device, weights,
      { sequence, recycles: 1, steps, seed: 0 }),
    foldAffinityPass: makeFoldAffinityPass(device, weights, { seed: 0 }),
    affinityModule: makeAffinityModule(device, readAffinity, { precision: "f32" }),
    // 🔴 THE TEARDOWN AND THE MEASUREMENT ARE THE SAME HOOK, deliberately: a
    // snapshot taken before the drain measures buffers the driver has already
    // been told to take back, which reads as a leak that settles by itself.
    freeRunBuffers: async () => {
      if (freeing) await settleReleasedMemory(device);
      const snap = memorySnapshot(device);
      samplesSeen.push({ residentBytes: snap.residentBytes,
        peakBytes: snap.peakBytes, buffers: snap.bufferCount,
        // 🔴 WHETHER THE WEIGHTS SURVIVED THE LIGAND, which the residency
        // total alone cannot say: a loop that releases and re-packs 365 MiB
        // every ligand reads as PERFECTLY FLAT here and costs a second each.
        packCalls: residentPackStats.calls, packMib: mib(residentPackStats.bytes),
        packMs: Math.round(residentPackStats.ms),
        packLabels: [...residentPackStats.byLabel.entries()]
          .map(([label, seen]) => ({ label, mib: mib(seen.bytes ?? 0),
            calls: seen.calls ?? 0 }))
          .sort((a, b) => b.mib - a.mib).slice(0, 5),
        largest: snap.currentByLabel.slice(0, 3)
          .map((row) => `${row.label} ${mib(row.bytes)}`) });
    },
  };

  const phases = {};
  for (const [phase, jobs] of [
    ["repeat", Array.from({ length: repeats }, (_, i) => ({
      id: `rep${i + 1}`, smiles: varied[0] })) ],
    ["varied", varied.map((smiles, i) => ({ id: `var${i + 1}`, smiles }))],
  ]) {
    samplesSeen.length = 0;
    const run = await runAffinityScreen(jobs, deps, {
      recyclingSteps: 1, samplingSteps: steps, diffusionSamples: samples,
      mwCorrection: false,
    });
    if (run.failures.length > 0) {
      throw new Error(`${phase}: ${run.failures[0].id} failed: `
        + run.failures[0].error);
    }
    const seen = samplesSeen.map((s) => ({ ...s }));
    const residents = seen.map((s) => s.residentBytes);
    phases[phase] = {
      ligands: jobs.length,
      residentMib: residents.map(mib),
      peakMib: mib(Math.max(...seen.map((s) => s.peakBytes))),
      buffers: seen.map((s) => s.buffers),
      // the growth AFTER the first ligand: the first one is what the loop
      // costs, the rest is what it keeps
      growthMib: mib(residents[residents.length - 1] - residents[0]),
      packCalls: seen.map((s) => s.packCalls),
      packMib: seen.map((s) => s.packMib),
      largestAtEnd: seen[seen.length - 1].largest,
      runtimeMs: run.results.map((r) => r.runtimeMs),
      packLabelsAtEnd: seen[seen.length - 1].packLabels,
    };
    console.log(`${phase.padEnd(7)} resident `
      + phases[phase].residentMib.map((v) => `${v}`).join(" -> ")
      + ` MiB   growth ${phases[phase].growthMib}`
      + `   peak ${phases[phase].peakMib}`
      + `   buffers ${phases[phase].buffers.join("/")}`
      + `\n        host-pack calls ${phases[phase].packCalls.join("/")}`
      + `  cumulative MiB ${phases[phase].packMib.join("/")}`
      + `\n        runtime ms ${phases[phase].runtimeMs.join("/")}`);
  }

  const bounds = { repeat: REPEAT_BOUND_MIB, varied: VARIED_BOUND_MIB };
  const failures = [];
  for (const [phase, record] of Object.entries(phases)) {
    // 🔴 `if (x > bound) throw` NEVER FIRES ON NaN - docs/ records a checker
    // that passed for as long as a bundle existed without computing anything.
    if (!Number.isFinite(record.growthMib)) {
      failures.push(`${phase}: growth is not a number`);
    } else if (record.growthMib > bounds[phase]) {
      failures.push(`${phase}: grew ${record.growthMib} MiB across`
        + ` ${record.ligands} ligands, bound ${bounds[phase]}`);
    }
  }
  if (leaking) {
    console.log("--pooling=off: the falsification arm, EXPECTED to exceed the"
      + " repeat bound");
    return { freeing, keeping, leaking, phases, bounds, failures,
      falsified: failures.length > 0 };
  }
  if (!freeing) {
    console.log("--free=off: this arm is the falsification and is EXPECTED"
      + " to exceed the repeat bound");
    return { freeing, keeping, phases, bounds, failures,
      falsified: failures.length > 0 };
  }
  if (failures.length > 0) throw new Error(failures.join("; "));
  return { freeing, keeping, phases, bounds, pass: true };
}
