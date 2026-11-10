/**
 * Boltz-2's affinity module refuses some ligands, and so must this page.
 *
 * 🔴 BOTH BOUNDS ARE NEW HERE. LocalFold admits 150 heavy atoms (docs/SMILES.md)
 * and upstream's affinity path raises above 128 and warns above 56, "which was
 * the maximum during training". A 140-atom ligand folds perfectly well and has
 * no business being given an affinity, and a 60-atom one gets a number that
 * has to be labelled.
 *
 * 🔴 AND THE WARNING MATTERS MORE THAN THE REFUSAL. A refusal is loud and
 * self-correcting. A silent extrapolation past the training range sorts into a
 * ranked table beside numbers that mean something, which is the failure this
 * whole tool would be judged on - so `outsideTrainingRange` is a per-ligand
 * flag that travels to the CSV, not a toast.
 *
 * The boundaries are asserted at 56/57 and 128/129 because upstream's tests
 * are `>` and an off-by-one here changes which ligands are scored at all.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { parseSmiles } from "../shared/chem/smiles.js";
import { checkAffinityLigand, MAX_HEAVY_ATOMS, TRAINING_MAX_HEAVY_ATOMS }
  from "../shared/chem/affinity-limits.js";

const check = (smiles) => checkAffinityLigand(parseSmiles(smiles), smiles);
const carbons = (n) => "C".repeat(n);

test("the bounds are upstream's numbers", () => {
  assert.equal(MAX_HEAVY_ATOMS, 128);
  assert.equal(TRAINING_MAX_HEAVY_ATOMS, 56);
});

test("an ordinary drug-like ligand is accepted with nothing to say", () => {
  const result = check("CC(=O)Oc1ccccc1C(=O)O");       // aspirin, 13 heavy
  assert.equal(result.ok, true);
  assert.equal(result.heavyAtoms, 13);
  assert.equal(result.reason, null);
  assert.equal(result.warning, null);
  assert.equal(result.outsideTrainingRange, false);
});

test("🔴 the training boundary is > 56, not >= 56", () => {
  const inside = check(carbons(56));
  assert.equal(inside.ok, true);
  assert.equal(inside.outsideTrainingRange, false,
    "56 heavy atoms is INSIDE the training range");
  assert.equal(inside.warning, null);

  const outside = check(carbons(57));
  assert.equal(outside.ok, true, "57 is still scored");
  assert.equal(outside.outsideTrainingRange, true);
  assert.match(outside.warning, /larger than 56 heavy-atoms/);
  assert.match(outside.warning, /might be inaccurate/);
});

test("🔴 the hard boundary is > 128, not >= 128", () => {
  const accepted = check(carbons(128));
  assert.equal(accepted.ok, true, "128 heavy atoms is still accepted");
  assert.equal(accepted.outsideTrainingRange, true, "...but outside training");

  const refused = check(carbons(129));
  assert.equal(refused.ok, false);
  assert.match(refused.reason, /too large/);
  assert.match(refused.reason, /more than 128 atoms/);
  assert.equal(refused.warning, null, "a refusal is not also a warning");
});

test("the refusal uses upstream's own sentence", () => {
  // So a reader who searches for the message finds Boltz-2's documentation.
  assert.equal(check(carbons(200)).reason,
    "The ligand for affinity is too large, ligands with more than 128 atoms"
    + " are not supported in the affinity prediction module");
});

test("a 62-atom paclitaxel core is scored, and flagged", () => {
  // The largest ligand the SMILES gates cover, and a real screening case.
  const result = check("CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)"
    + "C2(O)CC(OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C");
  assert.equal(result.heavyAtoms, 62);
  assert.equal(result.ok, true);
  assert.equal(result.outsideTrainingRange, true);
});

test("🔴 a hydrogen isotope is refused, because the weight cannot be computed", () => {
  // parseSmiles folds a bracket hydrogen into its neighbour's implicit count
  // and keeps no isotope, so affinity_mw would be ~3 daltons wrong and the
  // molecular-weight correction would shift log10(IC50) with nothing said.
  const result = check("[2H]C([2H])([2H])O");
  assert.equal(result.ok, false);
  assert.match(result.reason, /hydrogen isotope/);
});

test("...but an ordinary [nH] or [NH4+] is not a hydrogen isotope", () => {
  for (const smiles of ["c1cc[nH]c1", "[NH4+]", "[CH3:1][OH:2]", "[OH-]"]) {
    assert.equal(check(smiles).ok, true, smiles);
  }
});

test("🔴 a multi-fragment ligand is refused rather than half-scored", () => {
  // A "." is two molecules. The heads pool over the binder chain's tokens, so
  // a salt would be scored as one binder - and dropping the counter-ion
  // silently would change the molecule.
  const result = check("[Na+].[Cl-]");
  assert.equal(result.ok, false);
  assert.match(result.reason, /disconnected fragments/);
});

test("a refusal still reports the heavy-atom count it measured", () => {
  // The number is what a reader needs to fix the input.
  assert.equal(check(carbons(129)).heavyAtoms, 129);
  assert.equal(check("[Na+].[Cl-]").heavyAtoms, 2);
});
