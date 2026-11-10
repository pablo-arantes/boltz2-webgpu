/**
 * The two-pass screening loop: one target, many ligands, affinity per ligand.
 *
 * 🔴 AFFINITY IS A SECOND FOLD, NOT A HEAD ON THE FIRST. upstream's `main.py`
 * loads `boltz2_aff.ckpt` as a COMPLETE Boltz2 and runs a second
 * `trainer.predict` with its own `recycling_steps: 5`, `sampling_steps: 200`
 * and `diffusion_samples: 3`, on pass 1's PREDICTED structure cropped to a
 * pocket. Reading it as a cheap head understates it by about an order of
 * magnitude: pass 2 is 600 denoiser calls against pass 1's default 25, on <=256
 * tokens instead of the whole protein.
 *
 * 🔴 AND THE TRUNK WEIGHTS ARE THE SAME ONES. G0 compared all 5,019 tensors the
 * two checkpoints share and found them byte-identical, so the structure bundle
 * already on the device serves both passes; only the 8.74 M affinity
 * parameters are new, and they are fetched lazily the first time affinity is
 * asked for.
 *
 * 🔴 THE FOLDS ARE INJECTED, AND THAT IS NOT ONLY FOR TESTING. The loop's job
 * is queueing, aggregation, recovery and honesty about what it measured; the
 * folding belongs to webgpu/af3/fold.js. Injecting it means every branch here -
 * stop and resume, a ligand that throws, the ETA, the molecular-weight
 * correction, partial results - is reachable in a DOM-free test with no weights
 * and no GPU, which is the only way most of them would ever be exercised. A
 * run loop whose failure paths are only reachable by having a real fold fail is
 * a run loop whose failure paths are untested.
 *
 * 🔴 WEIGHTS ARE NOT RELEASED BETWEEN LIGANDS, AND THAT IS DELIBERATE.
 * `releaseAllWeights` exists for a change of FAMILY - docs/ records residency
 * kept across one going 677 -> 2874 MiB and losing the device on the fourth
 * fold. Every ligand here is the same family, so releasing would re-pack 365
 * MiB per ligand for nothing. What is freed per ligand is the run's own
 * buffers, through the caller's `freeRunBuffers`.
 *
 * 🔴 THE MSA IS SEARCHED ONCE, AND THE EXISTING CACHE ALREADY DOES IT. The
 * page's search cache keys on `JSON.stringify(proteinChains)` and the call site
 * passes `chains.filter(kind === "protein")`, so a SMILES ligand is filtered
 * out before the key is built and ligands 2..N hit `reuse: "single"` with no
 * change. What that saves is the repeated MMseqs2 search - tens of seconds
 * each. It does NOT save the trunk: ligand tokens enter the pair and MSA
 * representation, so the trunk runs per ligand, both passes. Nothing here
 * reuses a trunk embedding across ligands and the plan says why.
 */

import { affinityCrop, CROP_MAX_TOKENS, CROP_MAX_ATOMS } from
  "../shared/af3/affinity/crop.js";
import { checkAffinityLigand } from "../shared/chem/affinity-limits.js";
import { molecularWeight } from "../shared/chem/molecular-weight.js";

/** upstream's `predict_affinity_args`, and the CLI defaults behind them. */
export const AFFINITY_PASS = Object.freeze({
  recyclingSteps: 5,
  samplingSteps: 200,
  // 🔴 FIVE, AND THIS FILE SAID THREE. upstream's `predict` function signature
  // reads `diffusion_samples_affinity: int = 3` and its CLICK OPTION reads
  // `default=5` - and click passes its value explicitly, so a plain
  // `boltz predict` runs FIVE. Three is what you get only by importing
  // `predict` and calling it as a Python function, which nobody does.
  //
  // 🔴 AND THE RULE FOR RESOLVING THAT WAS ALREADY WRITTEN DOWN HERE, ONE
  // CONSTANT LOWER. The note on the molecular-weight correction below says the
  // class default is true, the checkpoint's hparam false, main.py's CLI
  // default false, "and we default to off, matching what a plain
  // `boltz predict` does rather than what the class signature says". The same
  // question arose twice and this constant answered it the other way.
  diffusionSamples: 5,
  maxParallelSamples: 1,
  // steering is turned OFF for the affinity pass (main.py)
  steering: false,
});

/**
 * 🔴 THE MOLECULAR-WEIGHT CORRECTION, AND ITS EXACT SCOPE.
 *
 *     value = 1.03525938 * value - 0.59992683 * mw**0.3 + 2.83288489
 *
 * Four things, all load-bearing and none of them obvious:
 *   1. it applies to the ENSEMBLE value only - not the probability, and not
 *      `…1`/`…2`, which are built before it runs. With it on, the aggregate and
 *      the per-member values are on DIFFERENT SCALES.
 *   2. it exists only on the ensemble path; the single-module branch has none.
 *   3. the effective default is OFF: the class default is true, the
 *      checkpoint's stored hparam is false, and main.py's CLI default is false
 *      and passed explicitly. We default to off, matching what a plain
 *      `boltz predict` does rather than what the class signature says.
 *   4. `mw` is RDKit's `Descriptors.MolWt` - see shared/chem/molecular-weight.js
 *      on why the table has to be RDKit's and not IUPAC's.
 */
export const MW_CORRECTION = Object.freeze({
  // 🔴 THE PAPER AND THE CODE DISAGREE ON THE FUNCTIONAL FORM, AND THE CODE
  // WINS. B.5.2 states the correction as
  //     y = C0 * (y1 + y2) + C1 * MW_binder + C2
  // - a LINEAR molecular-weight term - while main.py applies
  //     y = 1.03525938 * mean - 0.59992683 * MW**0.3 + 2.83288489
  // with MW to the 0.3 power. The coefficients were fitted to the form the
  // CODE applies, so reproducing upstream means following the code; the
  // paper's C0 also absorbs a factor of two by taking the sum where the code
  // takes the mean (0.5176 against 1.0353), which is consistent. Recorded
  // because a reader checking this against the paper will find a mismatch and
  // should know which way it was resolved.
  model: 1.03525938, mw: -0.59992683, bias: 2.83288489, exponent: 0.3,
});

export function applyMwCorrection(value, mw) {
  return MW_CORRECTION.model * value
    + MW_CORRECTION.mw * (mw ** MW_CORRECTION.exponent)
    + MW_CORRECTION.bias;
}

/**
 * The two ensemble members, aggregated as upstream does.
 *
 * 🔴 THE PROBABILITY IS THE MEAN OF THE TWO SIGMOIDS, NOT THE SIGMOID OF THE
 * MEAN LOGIT. upstream sigmoids each member and then averages; the two are not
 * equal and the second is the easy mistake.
 */
export function aggregateAffinity(members, { mw = null, mwCorrection = false } = {}) {
  if (!Array.isArray(members) || members.length === 0) {
    throw new TypeError("aggregateAffinity needs at least one member");
  }
  const mean = (pick) => members.reduce((total, m) => total + pick(m), 0)
    / members.length;
  const raw = mean((m) => m.predValue);
  const probability = mean((m) => m.probabilityBinary);
  const out = {
    affinity_pred_value: raw,
    affinity_probability_binary: probability,
  };
  members.forEach((member, index) => {
    // 🔴 NEVER CORRECTED - see MW_CORRECTION. These are the raw per-member
    // values and the CSV header has to say so.
    out[`affinity_pred_value${index + 1}`] = member.predValue;
    out[`affinity_probability_binary${index + 1}`] = member.probabilityBinary;
  });
  // the ensemble path is the only one with a correction, and only with >1 member
  if (mwCorrection && members.length > 1) {
    if (typeof mw !== "number" || !Number.isFinite(mw)) {
      throw new Error("the molecular-weight correction needs a finite"
        + " molecular weight; refusing to apply it to an unknown mw");
    }
    out.affinity_pred_value = applyMwCorrection(raw, mw);
    out.affinity_pred_value_uncorrected = raw;
    out.mwCorrectionApplied = true;
  } else {
    out.mwCorrectionApplied = false;
  }
  // pIC50 = 6 - log10(IC50 in uM), so the axis reads in the conventional
  // direction. Derived, never stored as a third independent number.
  out.pIC50 = 6 - out.affinity_pred_value;
  return out;
}

/**
 * Which diffusion sample the affinity module is given.
 *
 * 🔴 BEST BY ipTM, NOT THE FIRST AND NOT THE MEAN. upstream takes
 * `argsort(iptm, descending)[0]` and hands that one sample's coordinates over.
 * Averaging coordinates across samples would produce a structure that is not a
 * structure.
 */
export function bestSampleByIptm(samples) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new TypeError("bestSampleByIptm needs at least one sample");
  }
  let best = 0;
  for (let i = 1; i < samples.length; i += 1) {
    const value = samples[i].iptm;
    if (!Number.isFinite(value)) continue;
    if (!Number.isFinite(samples[best].iptm) || value > samples[best].iptm) best = i;
  }
  return { index: best, sample: samples[best] };
}

/**
 * An estimate of what is left, from what has actually been measured.
 *
 * 🔴 FROM MEASURED THROUGHPUT, AND IT SAYS WHEN IT DOES NOT KNOW. The first
 * ligand of a run pays for the weight download and the shader compiles, so
 * using it to predict the rest overstates the remaining time by a large
 * factor. The median of the FINISHED ligands is used once there are two, the
 * first one's time is reported as a single sample with `confident: false`, and
 * with none there is no estimate rather than a zero.
 */
export function estimateRemaining(durations, remaining) {
  if (remaining <= 0) return { seconds: 0, confident: true, samples: durations.length };
  if (durations.length === 0) {
    return { seconds: null, confident: false, samples: 0 };
  }
  const sorted = [...durations].sort((a, b) => a - b);
  const median = sorted.length % 2 === 1
    ? sorted[(sorted.length - 1) / 2]
    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  // ...and the first ligand is excluded once there is anything else, because it
  // carries the one-off costs the rest do not.
  const steady = durations.length > 1
    ? [...durations].slice(1).sort((a, b) => a - b) : null;
  const basis = steady && steady.length > 0
    ? (steady.length % 2 === 1 ? steady[(steady.length - 1) / 2]
      : (steady[steady.length / 2 - 1] + steady[steady.length / 2]) / 2)
    : median;
  return {
    seconds: (basis * remaining) / 1000,
    confident: durations.length > 1,
    samples: durations.length,
    excludedFirst: Boolean(steady),
  };
}

/**
 * Run one ligand through both passes.
 *
 * @param {object} job
 * @param {string} job.smiles
 * @param {string} job.id
 * @param {object} graph the parsed SMILES, for the limits and the weight
 * @param {object} deps
 * @param {(input: object) => Promise<object>} deps.foldStructure pass 1
 * @param {(input: object) => Promise<object>} deps.foldAffinityPass pass 2's
 *   trunk + sampler + confidence, returning `{samples, trunk, sInputs, feats}`
 * @param {(input: object) => Promise<object>} deps.affinityModule one member
 * @param {object} options
 */
export async function runOneLigand(job, graph, deps, options = {}) {
  const limits = checkAffinityLigand(graph, job.smiles);
  if (!limits.ok) {
    const error = new Error(limits.reason);
    error.name = "LigandRefused";
    error.heavyAtoms = limits.heavyAtoms;
    throw error;
  }
  const started = Date.now();

  // --- pass 1: the ordinary fold -------------------------------------------
  const structure = await deps.foldStructure({
    smiles: job.smiles, id: job.id, signal: options.signal,
  });

  // --- pass 2: the pocket, re-featurised from pass 1's prediction ----------
  // 🔴 THE CROP MOVES THE LIGAND'S TOKEN INDICES, so `affinity_token_mask` is
  // rebuilt against the cropped order rather than carried over from pass 1.
  const crop = affinityCrop(structure.cropTokens, {
    maxTokens: options.maxTokens ?? CROP_MAX_TOKENS,
    maxAtoms: options.maxAtoms ?? CROP_MAX_ATOMS,
  });

  const pass2 = await deps.foldAffinityPass({
    structure, crop,
    recyclingSteps: options.recyclingSteps ?? AFFINITY_PASS.recyclingSteps,
    samplingSteps: options.samplingSteps ?? AFFINITY_PASS.samplingSteps,
    diffusionSamples: options.diffusionSamples ?? AFFINITY_PASS.diffusionSamples,
    signal: options.signal,
  });

  const best = bestSampleByIptm(pass2.samples);
  const members = [];
  for (const member of deps.affinityMembers ?? ["affinity_module1", "affinity_module2"]) {
    members.push(await deps.affinityModule({
      // 🔴 `z` AND `positions` FROM THE **SAME** SAMPLE. This paired
      // `pass2.z` - which is `samples[0].z` - with the best-ipTM sample's
      // coordinates. Harmless for boltz2 today because its trunk is
      // deterministic (`recycleMcDropout` is null for this dialect, the 0.4 in
      // dialect.js belongs to protenix2), so every sample's `z` is identical.
      // It breaks silently under any dialect with recycle dropout, and
      // tools/gpu/dump-affinity-inputs.js already used the best sample's own
      // `z` - two paths, two answers.
      member, sInputs: pass2.sInputs, z: best.sample.z ?? pass2.z,
      // 🔴 PER-TOKEN COORDINATES, NOT THE DENSE ATOM ARRAY. The affinity
      // module's distogram indexes `positions[i*3]` for `i < tokens`, and
      // `fold.positions` is indexed by dense ATOM SLOT (`token*dense + atom`,
      // which is what shared/af3/structure/pdb.js writes from). Feeding it
      // read dense slots 0..n-1 - the first ~n/dense tokens' atoms, masked
      // slots included - as if they were n token centres, so every distance in
      // the distogram was wrong. `fold.pseudoBeta` is the disto gather at
      // `tokens * 3` and is what upstream's `token_to_rep_atom` selects;
      // webgpu/af3/fold.js:1321 builds it and pass 2 already captured it.
      //
      // 🔴 THE f64 REFERENCE CAUGHT THIS AND THE GPU PATH DID NOT:
      // cpu/af3/affinity/affinity.js:309 asserts `need(positions, n * 3)` and
      // throws "expected 12 values, got 288"; the GPU loop had no length
      // check, so it silently indexed past the tokens. A guard is added below.
      positions: best.sample.pseudoBeta, feats: pass2.feats,
      signal: options.signal,
    }));
  }

  const mw = molecularWeight(graph);
  const affinity = aggregateAffinity(members, {
    mw, mwCorrection: options.mwCorrection ?? false,
  });

  return {
    id: job.id,
    smiles: job.smiles,
    ...affinity,
    molecularWeight: mw,
    heavyAtoms: limits.heavyAtoms,
    outsideTrainingRange: limits.outsideTrainingRange,
    warning: limits.warning,
    // the structure a reader downloads and the viewer adds, from PASS 1 -
    // which is the pose over the whole target, where pass 2's is a pocket crop
    pdb: structure.pdb,
    // confidence from pass 1 (the whole complex) and the chosen pass-2 sample
    plddt: structure.meanPlddt,
    ptm: structure.ptm,
    iptm: structure.iptm,
    affinitySampleIptm: best.sample.iptm,
    affinitySampleIndex: best.index,
    croppedTokens: crop.kept.length,
    croppedProteinTokens: crop.proteinKept,
    runtimeMs: Date.now() - started,
  };
}

/**
 * The queue: sequential, streaming, stoppable, and resumable.
 *
 * 🔴 SEQUENTIAL IS REQUIRED, NOT A SIMPLIFICATION. Boltz-2 predicts affinity
 * for ONE ligand at a time - `multiplicity=1`, one `affinity_token_mask` - so
 * there is no batch axis to use even if the device had room.
 *
 * 🔴 AND A LIGAND THAT FAILS DOES NOT END THE RUN. Each is caught and recorded
 * with its error, so a 400-ligand screen is not lost to one unparseable SMILES
 * on line 237 - and `results` is always complete-so-far, which is what makes a
 * partial download after a failure possible rather than an afterthought.
 */
export async function runAffinityScreen(jobs, deps, options = {}) {
  const results = [];
  const failures = [];
  const durations = [];
  const startedAt = Date.now();
  // `resumeFrom` is an id, not an index, so adding or removing a line from the
  // .smi file between runs cannot silently restart somewhere else.
  const resumeFrom = options.resumeFrom ?? null;
  let skipping = resumeFrom !== null;

  for (let index = 0; index < jobs.length; index += 1) {
    const job = jobs[index];
    if (skipping) {
      if (job.id === resumeFrom) skipping = false;
      else { results.push({ id: job.id, smiles: job.smiles, skipped: "resumed past" }); continue; }
    }
    if (options.signal?.aborted) {
      return { results, failures, stopped: true, stoppedBefore: job.id,
        elapsedMs: Date.now() - startedAt };
    }

    options.onProgress?.({
      phase: "start", index, total: jobs.length, id: job.id,
      done: results.length, failed: failures.length,
      eta: estimateRemaining(durations, jobs.length - index),
    });

    try {
      const graph = await deps.parseLigand(job.smiles);
      const result = await runOneLigand(job, graph, deps, options);
      results.push(result);
      durations.push(result.runtimeMs);
      options.onProgress?.({ phase: "done", index, total: jobs.length,
        id: job.id, result, done: results.length, failed: failures.length,
        eta: estimateRemaining(durations, jobs.length - index - 1) });
    } catch (error) {
      if (error?.name === "AbortError") {
        return { results, failures, stopped: true, stoppedBefore: job.id,
          elapsedMs: Date.now() - startedAt };
      }
      const record = { id: job.id, smiles: job.smiles,
        error: String(error?.message ?? error), name: error?.name ?? "Error",
        // 🔴 THE STACK IS KEPT. A screen catches every ligand's failure so one
        // bad SMILES cannot end the run - which also means a REAL bug arrives
        // as a one-line string with no way back to it.
        stack: error?.stack };
      failures.push(record);
      options.onProgress?.({ phase: "failed", index, total: jobs.length,
        id: job.id, failure: record, done: results.length,
        failed: failures.length,
        eta: estimateRemaining(durations, jobs.length - index - 1) });
    } finally {
      // 🔴 THE RUN'S BUFFERS, NOT THE WEIGHTS - see this file's header.
      await deps.freeRunBuffers?.();
    }
  }

  return { results, failures, stopped: false, elapsedMs: Date.now() - startedAt,
    throughput: durations.length > 0
      ? { medianMs: estimateRemaining(durations, 1).seconds * 1000,
        samples: durations.length } : null };
}
