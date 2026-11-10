/**
 * The screen page's polarity and sorting decisions.
 *
 * 🔴 THIS MODULE HAD NO TEST AT ALL, AND IT HOLDS EVERY POLARITY DECISION ON
 * THE PAGE. `affinity_pred_value` is log10(IC50) so LOWER is stronger, while
 * `affinity_probability_binary` and the derived pIC50 are higher-is-better -
 * three columns, two directions, and a sort that has to get each right. The
 * logic was correct when written and nothing stopped it regressing; a reversed
 * comparison here silently ranks a screen backwards, which is the worst
 * failure this tool has because the output still looks like a ranking.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { TABLE_COLUMNS, DEFAULT_SORT, sortResults, startsDescending,
  caveatsFor } from "../web/screen-view.js";

const row = (id, over = {}) => ({
  id, smiles: "CCO", affinity_pred_value: 1, affinity_probability_binary: 0.5,
  affinity_pred_value1: 1, affinity_pred_value2: 1, pIC50: 5, iptm: 0.8,
  heavyAtoms: 3, runtimeMs: 1000, ...over,
});

test("🔴 the first click on a LOWER-is-better column sorts ascending", () => {
  // log10(IC50): the strongest binder has the SMALLEST value, so "best first"
  // is ascending. Getting this backwards puts the weakest binder at the top.
  assert.equal(startsDescending("affinity_pred_value"), false);
  // ...and the higher-is-better columns go the other way
  assert.equal(startsDescending("affinity_probability_binary"), true);
  assert.equal(startsDescending("pIC50"), true);
  assert.equal(startsDescending("iptm"), true);
  // runtime: shorter is better
  assert.equal(startsDescending("runtimeMs"), false);
  // text sorts A-Z
  assert.equal(startsDescending("id"), false);
});

test("🔴 sorting by affinity_pred_value ascending puts the STRONGEST first", () => {
  const rows = [row("weak", { affinity_pred_value: 3 }),
    row("strong", { affinity_pred_value: -1 }),
    row("mid", { affinity_pred_value: 1 })];
  const sorted = sortResults(rows, { key: "affinity_pred_value", ascending: true });
  assert.deepEqual(sorted.map((r) => r.id), ["strong", "mid", "weak"]);
});

test("🔴 pIC50 descending agrees with affinity_pred_value ascending", () => {
  // pIC50 = 6 - value, so the two must order a set identically in opposite
  // directions. A disagreement here means one of them has the wrong polarity.
  const rows = [1, -1, 3, 0].map((v, i) =>
    row(`l${i}`, { affinity_pred_value: v, pIC50: 6 - v }));
  const byValue = sortResults(rows, { key: "affinity_pred_value", ascending: true });
  const byPic50 = sortResults(rows, { key: "pIC50", ascending: false });
  assert.deepEqual(byValue.map((r) => r.id), byPic50.map((r) => r.id));
});

test("the default sort is the classifier, best first", () => {
  assert.equal(DEFAULT_SORT.key, "affinity_probability_binary");
  assert.equal(DEFAULT_SORT.ascending, false);
  const rows = [row("lo", { affinity_probability_binary: 0.1 }),
    row("hi", { affinity_probability_binary: 0.9 })];
  assert.deepEqual(sortResults(rows, DEFAULT_SORT).map((r) => r.id), ["hi", "lo"]);
});

test("🔴 a failed ligand sinks whichever way the column points", () => {
  // Its score is MISSING, not zero. Treated as 0 it would be the strongest
  // binder in an ascending sort of log10(IC50) - a ligand that never ran at
  // the top of the table.
  const rows = [row("ok", { affinity_pred_value: 2 }),
    { id: "broke", smiles: "C", error: "device lost", name: "Error" }];
  for (const ascending of [true, false]) {
    const sorted = sortResults(rows, { key: "affinity_pred_value", ascending });
    assert.equal(sorted[0].id, "ok", `ascending=${ascending}`);
    assert.equal(sorted[1].id, "broke");
  }
});

test("neither output is a column named like a merged score", () => {
  const keys = TABLE_COLUMNS.map((column) => column.key);
  assert.ok(keys.includes("affinity_pred_value"));
  assert.ok(keys.includes("affinity_probability_binary"));
  assert.ok(!keys.some((key) => /^score$|combined|composite|total/i.test(key)),
    `a merged column appeared: ${keys.join(",")}`);
});

test("the caveats report the ensemble split, weak poses and extrapolations", () => {
  const text = caveatsFor([
    row("a", { affinity_pred_value1: 2, affinity_pred_value2: 0, iptm: 0.3 }),
    row("b", { outsideTrainingRange: true, iptm: 0.9 }),
  ]);
  assert.match(text, /Ensemble members differ by/);
  assert.match(text, /1 of 2 poses have ipTM below 0\.5/);
  assert.match(text, /1 ligand exceeds 56 heavy atoms/);
  assert.match(text, /two heads trained on different data/);
});

test("caveatsFor says nothing when nothing finished", () => {
  assert.equal(caveatsFor([]), "");
  assert.equal(caveatsFor([{ id: "x", error: "no", name: "Error" }]), "");
});
