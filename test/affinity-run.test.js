/**
 * The screening loop does what it says: aggregates correctly, survives a bad
 * ligand, stops, resumes, and never reports an ETA it has not measured.
 *
 * 🔴 EVERY PATH HERE IS REACHABLE ONLY BECAUSE THE FOLDS ARE INJECTED. A run
 * loop whose failure paths need a real fold to fail is a run loop whose failure
 * paths are untested - and the failure paths are most of what a 400-ligand
 * screen depends on. docs/ records the same shape repeatedly: the arm nothing
 * was running is the arm that was broken.
 *
 * The arithmetic cases below are the ones a plausible implementation gets
 * wrong, and each says which:
 *   * the probability is the mean of the two SIGMOIDS, not the sigmoid of the
 *     mean logit;
 *   * the molecular-weight correction touches the ENSEMBLE value only, so the
 *     aggregate and the per-member values end up on different scales;
 *   * it is OFF by default, because main.py's CLI default is false however the
 *     class signature reads;
 *   * pIC50 is 6 - value, derived rather than stored.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { aggregateAffinity, applyMwCorrection, bestSampleByIptm,
  estimateRemaining, runAffinityScreen, runOneLigand, MW_CORRECTION,
  AFFINITY_PASS } from "../web/affinity-run.js";
import { parseSmiles } from "../shared/chem/smiles.js";

const member = (predValue, probabilityBinary) => ({ predValue, probabilityBinary });

test("upstream's affinity pass settings are what we run", () => {
  assert.equal(AFFINITY_PASS.recyclingSteps, 5);
  assert.equal(AFFINITY_PASS.samplingSteps, 200);
  // 🔴 FIVE, NOT THREE, AND THIS TEST ASSERTED THREE. upstream's `predict`
  // signature says `diffusion_samples_affinity: int = 3`; its click option says
  // `default=5`, and click passes its value, so a plain `boltz predict` runs
  // five. Pinned here against the CLI because that is what a user of upstream
  // gets - the same rule the MW-correction test below applies.
  assert.equal(AFFINITY_PASS.diffusionSamples, 5);
  assert.equal(AFFINITY_PASS.steering, false);
});

test("the ensemble averages the value and the PROBABILITIES", () => {
  const out = aggregateAffinity([member(1.0, 0.8), member(2.0, 0.4)]);
  assert.equal(out.affinity_pred_value, 1.5);
  // 🔴 the mean of the two sigmoids. sigmoid((logit1+logit2)/2) would be a
  // different number, and that is the easy mistake.
  assert.equal(out.affinity_probability_binary, 0.6000000000000001);
});

test("per-member values are kept and labelled 1 and 2", () => {
  const out = aggregateAffinity([member(1.0, 0.8), member(2.0, 0.4)]);
  assert.equal(out.affinity_pred_value1, 1.0);
  assert.equal(out.affinity_pred_value2, 2.0);
  assert.equal(out.affinity_probability_binary1, 0.8);
  assert.equal(out.affinity_probability_binary2, 0.4);
});

test("pIC50 is 6 - value, derived from whatever the value ended up being", () => {
  const plain = aggregateAffinity([member(1.5, 0.5), member(1.5, 0.5)]);
  assert.equal(plain.pIC50, 4.5);
  const corrected = aggregateAffinity([member(1.5, 0.5), member(1.5, 0.5)],
    { mw: 300, mwCorrection: true });
  assert.equal(corrected.pIC50, 6 - corrected.affinity_pred_value);
  assert.notEqual(corrected.pIC50, 4.5);
});

test("🔴 the MW correction is OFF by default", () => {
  // The Boltz2 class default is true, the checkpoint's hparam is false, and
  // main.py's CLI default is false AND passed explicitly - so a plain
  // `boltz predict` does not correct, and neither do we.
  const out = aggregateAffinity([member(1.0, 0.5), member(2.0, 0.5)], { mw: 300 });
  assert.equal(out.mwCorrectionApplied, false);
  assert.equal(out.affinity_pred_value, 1.5);
  assert.equal(out.affinity_pred_value_uncorrected, undefined);
});

test("🔴 the MW correction touches the ENSEMBLE value only", () => {
  const out = aggregateAffinity([member(1.0, 0.8), member(2.0, 0.4)],
    { mw: 300, mwCorrection: true });
  assert.equal(out.mwCorrectionApplied, true);
  assert.equal(out.affinity_pred_value, applyMwCorrection(1.5, 300));
  assert.equal(out.affinity_pred_value_uncorrected, 1.5);
  // the per-member values are RAW - they are built before the correction runs,
  // so with it on the aggregate and the members are on different scales and
  // the CSV header has to say so.
  assert.equal(out.affinity_pred_value1, 1.0);
  assert.equal(out.affinity_pred_value2, 2.0);
  // ...and the probability is untouched
  assert.equal(out.affinity_probability_binary, 0.6000000000000001);
});

test("the correction uses upstream's coefficients and its 0.3 exponent", () => {
  assert.equal(MW_CORRECTION.model, 1.03525938);
  assert.equal(MW_CORRECTION.mw, -0.59992683);
  assert.equal(MW_CORRECTION.bias, 2.83288489);
  assert.equal(MW_CORRECTION.exponent, 0.3);
  const mw = 342.5;
  assert.equal(applyMwCorrection(2.0, mw),
    1.03525938 * 2.0 - 0.59992683 * (mw ** 0.3) + 2.83288489);
});

test("🔴 the correction refuses an unknown molecular weight", () => {
  // Applying it with a missing mw would shift every value by the bias alone
  // and look like a prediction.
  assert.throws(() => aggregateAffinity([member(1, 0.5), member(2, 0.5)],
    { mw: null, mwCorrection: true }), /needs a finite/);
  assert.throws(() => aggregateAffinity([member(1, 0.5), member(2, 0.5)],
    { mw: Number.NaN, mwCorrection: true }), /needs a finite/);
});

test("a single member is not corrected, as upstream's else branch is not", () => {
  const out = aggregateAffinity([member(1.0, 0.7)], { mw: 300, mwCorrection: true });
  assert.equal(out.mwCorrectionApplied, false);
  assert.equal(out.affinity_pred_value, 1.0);
});

test("the affinity sample is the best ipTM, not the first", () => {
  const samples = [{ iptm: 0.4 }, { iptm: 0.9 }, { iptm: 0.6 }];
  assert.equal(bestSampleByIptm(samples).index, 1);
});

test("...and a non-finite ipTM never wins", () => {
  assert.equal(bestSampleByIptm([{ iptm: Number.NaN }, { iptm: 0.1 }]).index, 1);
  assert.equal(bestSampleByIptm([{ iptm: 0.1 }, { iptm: Number.NaN }]).index, 0);
});

test("🔴 the ETA says it does not know rather than guessing zero", () => {
  assert.deepEqual(estimateRemaining([], 10),
    { seconds: null, confident: false, samples: 0 });
  const one = estimateRemaining([1000], 3);
  assert.equal(one.confident, false, "one sample is not a measurement");
  assert.equal(one.samples, 1);
});

test("...and it excludes the FIRST ligand once it has others", () => {
  // The first pays for the weight download and the shader compiles, so using
  // it would overstate the rest by a large factor.
  const eta = estimateRemaining([60000, 1000, 1000, 1000], 2);
  assert.equal(eta.excludedFirst, true);
  assert.equal(eta.seconds, 2, "two ligands at the steady 1 s, not the 60 s first");
  assert.equal(eta.confident, true);
});

test("nothing remaining is zero seconds and confident", () => {
  assert.deepEqual(estimateRemaining([1000], 0),
    { seconds: 0, confident: true, samples: 1 });
});

// --- the loop ---------------------------------------------------------------

function stubDeps(overrides = {}) {
  return {
    parseLigand: async (smiles) => parseSmiles(smiles),
    foldStructure: async () => ({
      pdb: "ATOM", meanPlddt: 90, ptm: 0.8, iptm: 0.7,
      cropTokens: {
        centreCoords: Float64Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        asymId: Int32Array.from([0, 0, 1]),
        resIdx: Int32Array.from([0, 1, 0]),
        molType: Int32Array.from([0, 0, 3]),
        atomNum: Int32Array.from([5, 5, 1]),
        affinityMask: Uint8Array.from([0, 0, 1]),
        resolvedMask: Uint8Array.from([1, 1, 1]),
      },
    }),
    foldAffinityPass: async () => ({
      samples: [{ iptm: 0.5, positions: new Float64Array(9) },
        { iptm: 0.9, positions: new Float64Array(9) }],
      sInputs: new Float64Array(0), z: new Float64Array(0), feats: {},
    }),
    affinityModule: async ({ member: name }) => ({
      predValue: name.endsWith("1") ? 1.0 : 2.0,
      probabilityBinary: name.endsWith("1") ? 0.8 : 0.4,
    }),
    ...overrides,
  };
}

test("one ligand runs both passes and reports both outputs", async () => {
  const jobs = [{ id: "aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" }];
  const run = await runAffinityScreen(jobs, stubDeps(), {});
  assert.equal(run.failures.length, 0);
  assert.equal(run.results.length, 1);
  const r = run.results[0];
  assert.equal(r.affinity_pred_value, 1.5);
  assert.equal(r.affinity_probability_binary, 0.6000000000000001);
  assert.equal(r.affinitySampleIptm, 0.9, "the best-ipTM sample was chosen");
  assert.ok(r.molecularWeight > 180 && r.molecularWeight < 181);
  assert.equal(r.heavyAtoms, 13);
  assert.equal(r.outsideTrainingRange, false);
  assert.ok(typeof r.runtimeMs === "number");
});

test("🔴 a ligand that fails does not end the run", async () => {
  const jobs = [
    { id: "good1", smiles: "CCO" },
    { id: "bad", smiles: "this is not a smiles" },
    { id: "good2", smiles: "CCC" },
  ];
  const run = await runAffinityScreen(jobs, stubDeps(), {});
  assert.equal(run.results.length, 2, "both good ligands finished");
  assert.equal(run.failures.length, 1);
  assert.equal(run.failures[0].id, "bad");
  assert.equal(run.stopped, false);
});

test("a refused ligand is a failure with upstream's sentence, not a crash", async () => {
  const jobs = [{ id: "huge", smiles: "C".repeat(129) }];
  const run = await runAffinityScreen(jobs, stubDeps(), {});
  assert.equal(run.failures.length, 1);
  assert.equal(run.failures[0].name, "LigandRefused");
  assert.match(run.failures[0].error, /more than 128 atoms/);
});

test("stop leaves every finished result in place", async () => {
  const controller = new AbortController();
  const jobs = Array.from({ length: 5 },
    (_, i) => ({ id: `lig${i}`, smiles: "CCO" }));
  const deps = stubDeps({
    foldStructure: async () => {
      if (controller.signal.aborted) {
        const e = new Error("Prediction stopped"); e.name = "AbortError"; throw e;
      }
      return (await stubDeps().foldStructure());
    },
  });
  const run = await runAffinityScreen(jobs, deps, {
    signal: controller.signal,
    onProgress: ({ phase, done }) => { if (phase === "done" && done === 2) controller.abort(); },
  });
  assert.equal(run.stopped, true);
  assert.equal(run.results.length, 2, "partial results survive a stop");
  assert.ok(run.stoppedBefore);
});

test("🔴 resume is keyed on an ID, not an index", async () => {
  // A line added to or removed from the .smi file between runs must not
  // silently restart somewhere else.
  const jobs = ["a", "b", "c", "d"].map((id) => ({ id, smiles: "CCO" }));
  const run = await runAffinityScreen(jobs, stubDeps(), { resumeFrom: "c" });
  const ran = run.results.filter((r) => !r.skipped).map((r) => r.id);
  assert.deepEqual(ran, ["c", "d"]);
  const skipped = run.results.filter((r) => r.skipped).map((r) => r.id);
  assert.deepEqual(skipped, ["a", "b"]);
});

test("the run frees its buffers after every ligand, failures included", async () => {
  let freed = 0;
  const jobs = [{ id: "ok", smiles: "CCO" }, { id: "bad", smiles: "!!!" }];
  await runAffinityScreen(jobs, stubDeps({ freeRunBuffers: async () => { freed += 1; } }), {});
  assert.equal(freed, 2, "the failing ligand's buffers are freed too");
});

test("progress reports an ETA that is honest about its own basis", async () => {
  const seen = [];
  const jobs = Array.from({ length: 3 }, (_, i) => ({ id: `l${i}`, smiles: "CCO" }));
  await runAffinityScreen(jobs, stubDeps(),
    { onProgress: (p) => { if (p.phase === "done") seen.push(p.eta); } });
  assert.equal(seen.length, 3);
  assert.equal(seen[0].confident, false, "after one ligand it is not confident");
  assert.equal(seen[2].seconds, 0, "nothing left to do");
});

test("runOneLigand refuses a multi-fragment ligand before folding anything", async () => {
  let folded = 0;
  const deps = stubDeps({ foldStructure: async () => { folded += 1; throw new Error("x"); } });
  await assert.rejects(
    runOneLigand({ id: "salt", smiles: "[Na+].[Cl-]" },
      parseSmiles("[Na+].[Cl-]"), deps, {}),
    /disconnected fragments/);
  assert.equal(folded, 0, "the refusal came before any GPU work");
});
