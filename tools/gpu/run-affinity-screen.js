/**
 * The whole thing: one target, several ligands, both passes, affinity per ligand.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/run-affinity-screen.js \
 *     --model=/model-boltz2-int5/manifest.json \
 *     --affinity=/model-boltz2-affinity-f32 \
 *     --sequence=<one-letter> --ligands=CCO,CCCc1ccccc1O
 *
 * Every piece this drives is separately gated - the crop, the centre atoms,
 * featuriseCrop, the depth-1 profile, s_inputs, the GPU pairformer, the whole
 * affinity forward, the aggregation. What has never run is the COMPOSITION, and
 * that is what this is for: option names, tensor shapes at the seams, and the
 * pre-masking of the trunk pair are the kind of thing only running finds.
 *
 * 🔴 THE SETTINGS DEFAULT LOW AND SAY SO. upstream's affinity pass is 5
 * recycles, 200 sampler steps and 5 diffusion samples - 1,000 denoiser calls a
 * ligand. That is the number to quote for accuracy and the wrong one for a
 * first integration run, so `--steps` and `--samples` default low here and the
 * output prints what it used. docs/ records the same trap from the other side:
 * "READ THE STATUS LINE, NOT THE pLDDT... these are the cheapest settings the
 * page has, so the numbers say nothing about quality".
 */
import { openAf3Store, trunkDepths, trunkWeights, confidenceWeights }
  from "../../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../shared/af3/weights/diffusion-weights.js";
import { makeFoldStructure, makeFoldAffinityPass, makeAffinityModule }
  from "../../web/affinity-passes.js";
import { runAffinityScreen } from "../../web/affinity-run.js";
import { parseSmiles } from "../../shared/chem/smiles.js";
import { dialectFor } from "../../shared/af3/dialect.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const affinityBase = option(args, "affinity", "/model-boltz2-affinity-f32");
  const sequence = option(args, "sequence",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ");
  const ligands = option(args, "ligands", "CCO,CCCc1ccccc1O").split(",");
  const steps = Number(option(args, "steps", "8"));
  const samples = Number(option(args, "samples", "2"));
  const recycles = Number(option(args, "recycles", "1"));
  const seed = Number(option(args, "seed", "0"));
  // 🔴 UPSTREAM SETS THE TWO PASSES SEPARATELY AND THIS TOOL CONFLATED THEM.
  // `boltz predict` has `--recycling_steps`/`--sampling_steps`/
  // `--diffusion_samples` for the STRUCTURE pass and
  // `--sampling_steps_affinity`/`--diffusion_samples_affinity` for the
  // affinity pass, whose recycles it fixes at 5. Driving both from one
  // `--steps` is fine for a smoke run and wrong for tools/g8_compare.py,
  // which cannot claim to match upstream's settings unless it can set them.
  // These default to the shared knobs, so every existing invocation is
  // unchanged.
  const structRecycles = Number(option(args, "struct-recycles", String(recycles)));
  const structSteps = Number(option(args, "struct-steps", String(steps)));
  const affinityRecycles = Number(option(args, "affinity-recycles", String(recycles)));
  const affinitySteps = Number(option(args, "affinity-steps", String(steps)));
  const affinitySamples = Number(option(args, "affinity-samples", String(samples)));
  // 🔴 AN ALIGNMENT, BECAUSE WITHOUT ONE NEITHER IMPLEMENTATION HAS A FOLD TO
  // AGREE ABOUT. Measured by tools/g8_compare.py on a 33-residue
  // single-sequence target: upstream's structure and this port's are 9.43 A CA
  // rms apart with the ligand centroid 16.69 A away, so the affinity numbers
  // differ because the POSES differ and the comparison says nothing about the
  // affinity module. Both sides must be given the SAME a3m for the structures
  // to converge and the affinity gap to mean anything.
  const a3mPath = option(args, "a3m", "");
  const alignment = a3mPath === ""
    ? null : await (await fetch(a3mPath)).text();

  console.log(`sequence  ${sequence.length} residues`);
  console.log(`ligands   ${ligands.length}: ${ligands.join(", ")}`);
  console.log(`settings  pass1 ${structRecycles} recycles / ${structSteps} steps;`
    + ` pass2 ${affinityRecycles} recycles / ${affinitySteps} steps /`
    + ` ${affinitySamples} samples   (upstream: 3/200/1 then 5/200/5)`);
  console.log(`seed      ${seed}`);
  console.log(`msa       ${alignment === null ? "single sequence"
    : `${(alignment.match(/^>/gm) ?? []).length} rows from ${a3mPath}`}`);
  console.log();

  // --- the structure model -------------------------------------------------
  const store = await openAf3Store(manifest);
  // `trunkDepths` is synchronous and its field is `pairformerBlocks`
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks,
                              { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  // 🔴 THE CONFIDENCE HEAD IS NOT OPTIONAL HERE, where a fold tool can skip it.
  // Pass 2 picks the sample the affinity module is given by ipTM, so without a
  // head there is no selection to make - and silently taking sample 0 would be
  // a different algorithm wearing the same name.
  try {
    weights.confidence = await confidenceWeights(store);
  } catch (error) {
    throw new Error("the confidence head would not load, and the affinity pass"
      + " selects its sample by ipTM: " + String(error?.message ?? error));
  }
  console.log(`weights   trunk blocks ${depths.pairformerBlocks},`
    + ` msa blocks ${depths.msaBlocks}, confidence loaded`);

  // --- the affinity weights ------------------------------------------------
  const affinityManifest = await (await fetch(`${affinityBase}/manifest.json`)).json();
  const shard = await (await fetch(`${affinityBase}/weights-00.f32.bin`)).arrayBuffer();
  const readAffinity = (name) => {
    const record = affinityManifest.tensors[name];
    if (record === undefined) return null;
    const count = record.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(new Float32Array(shard, record.byteOffset, count));
  };
  console.log(`affinity  ${Object.keys(affinityManifest.tensors).length} tensors`);
  console.log();

  const deps = {
    parseLigand: async (smiles) => parseSmiles(smiles),
    foldStructure: makeFoldStructure(device, weights,
      { sequence, alignment, recycles: structRecycles, steps: structSteps,
        seed }),
    foldAffinityPass: makeFoldAffinityPass(device, weights, { seed }),
    affinityModule: makeAffinityModule(device, readAffinity, { precision: "f32" }),
  };

  const jobs = ligands.map((smiles, index) => ({
    id: `lig${index + 1}`, smiles,
  }));
  const run = await runAffinityScreen(jobs, deps, {
    recyclingSteps: affinityRecycles, samplingSteps: affinitySteps,
    diffusionSamples: affinitySamples,
    mwCorrection: false,
    onProgress: ({ phase, id, index, total, eta, failure }) => {
      if (phase === "start") {
        console.log(`[${index + 1}/${total}] ${id} ...`
          + (eta.seconds === null ? "" : ` eta ${eta.seconds.toFixed(0)}s`
            + (eta.confident ? "" : " (one sample)")));
      } else if (phase === "failed") {
        console.log(`[${index + 1}/${total}] ${id} FAILED ${failure.error}`);
        if (failure.stack) console.log(failure.stack);
      }
    },
  });

  for (const result of run.results) {
    if (result.skipped) continue;
    console.log(`${result.id}\t${result.smiles}`
      + `\tvalue ${result.affinity_pred_value.toFixed(4)}`
      + `\tpIC50 ${result.pIC50.toFixed(4)}`
      + `\tprob ${result.affinity_probability_binary.toFixed(4)}`
      + `\tcrop ${result.croppedTokens}t/${result.croppedProteinTokens}p`
      + `\tipTM ${result.affinitySampleIptm?.toFixed(3)}`
      + `\t${(result.runtimeMs / 1000).toFixed(1)}s`);
  }
  for (const failure of run.failures) {
    console.log(`FAILED ${failure.id}: ${failure.name}: ${failure.error}`);
  }

  if (run.failures.length > 0) {
    throw new Error(`${run.failures.length} ligand(s) failed`);
  }
  return { sequence: sequence.length, ligands: ligands.length, seed,
    settings: { structRecycles, structSteps, affinityRecycles, affinitySteps,
      affinitySamples },
    results: run.results, elapsedMs: run.elapsedMs,
    throughput: run.throughput };
}
