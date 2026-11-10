/**
 * Whether Boltz-2's affinity module will accept this ligand, and what to say.
 *
 * 🔴 THESE BOUNDS ARE UPSTREAM'S AND THEY REFUSE. boltz/data/parse/schema.py
 * raises on a ligand of more than 128 heavy atoms and prints a warning above
 * 56, "which was the maximum during training". LocalFold's page currently
 * admits 150 heavy atoms (docs/SMILES.md), so BOTH bounds are new here: a
 * 140-atom ligand folds perfectly well and has no business being given an
 * affinity.
 *
 * 🔴 AND THE 56 CASE IS THE ONE THAT MATTERS FOR A SCREEN. A refusal is loud
 * and self-correcting; a silent extrapolation past the training range is a
 * number that sorts into a ranked table beside numbers that mean something. So
 * `outsideTrainingRange` is carried per ligand all the way into the CSV, not
 * just raised as a toast the reader has already scrolled past.
 *
 * Upstream's own sentences are reproduced verbatim, so a reader who searches
 * for the message finds Boltz-2's documentation rather than ours.
 */

import { heavyAtomCount, fragmentCount, hydrogenIsotopeInSmiles }
  from "./molecular-weight.js";

/** Upstream's hard ceiling: `mol_no_h.GetNumAtoms() > 128` raises. */
export const MAX_HEAVY_ATOMS = 128;
/** Upstream's soft ceiling: the largest it was trained on. */
export const TRAINING_MAX_HEAVY_ATOMS = 56;

/** Verbatim from boltz/data/parse/schema.py. */
const TOO_LARGE =
  "The ligand for affinity is too large, ligands with more than 128 atoms"
  + " are not supported in the affinity prediction module";
/** Verbatim from boltz/data/parse/schema.py. */
const OUTSIDE_TRAINING =
  "the ligand used for affinity calculation is larger than 56 heavy-atoms,"
  + " which was the maximum during training, therefore the affinity output"
  + " might be inaccurate.";

/**
 * Check one ligand against what the affinity module can be asked.
 *
 * @param {{atoms: unknown[], components?: number[]}} graph from `parseSmiles`
 * @param {string} smiles the string it was parsed from - needed for the
 *   hydrogen-isotope test, which the graph cannot answer
 * @returns {{ok: boolean, heavyAtoms: number, reason: string | null,
 *            warning: string | null, outsideTrainingRange: boolean}}
 */
export function checkAffinityLigand(graph, smiles) {
  const heavyAtoms = heavyAtomCount(graph);

  // Upstream's hard refusal, first and by its own words.
  if (heavyAtoms > MAX_HEAVY_ATOMS) {
    return { ok: false, heavyAtoms, reason: TOO_LARGE, warning: null,
      outsideTrainingRange: true };
  }

  // 🔴 OURS, NOT UPSTREAM'S, AND LABELLED AS SUCH. `parseSmiles` folds a
  // hydrogen written as an atom into its neighbour's implicit count and keeps
  // no isotope for it, so `affinity_mw` cannot be computed for a deuterated
  // ligand: RDKit says 35.0603 for `[2H]C([2H])([2H])O` and any graph-based
  // sum must say 32.042. Upstream has RDKit and so has no such limit. Refusing
  // is the honest move, because the weight feeds the molecular-weight
  // correction and a 3-dalton error there is a real shift in log10(IC50).
  if (hydrogenIsotopeInSmiles(smiles)) {
    return { ok: false, heavyAtoms,
      reason: "this ligand labels a hydrogen isotope (such as [2H]), and this"
        + " port cannot compute a molecular weight for it - the SMILES parser"
        + " keeps hydrogens as counts, not atoms. Remove the label to screen"
        + " it, or fold it without affinity.",
      warning: null, outsideTrainingRange: false };
  }

  // 🔴 ALSO OURS. A `.` in a SMILES is two molecules. The affinity module
  // pools over the binder chain's tokens, so two disconnected fragments are
  // scored as one binder - which is not a thing the model was asked about, and
  // the project's scope is one ligand at a time. A salt or a co-crystallised
  // solvent is the common way this arrives, and dropping the counter-ion
  // silently would change the molecule.
  const fragments = fragmentCount(graph);
  if (fragments > 1) {
    return { ok: false, heavyAtoms,
      reason: `this ligand is ${fragments} disconnected fragments (a "." in the`
        + " SMILES). Affinity is predicted for one molecule; split it, or"
        + " remove the counter-ion.",
      warning: null, outsideTrainingRange: false };
  }

  // Upstream's warning. Note `>` not `>=`: 56 itself is inside the range.
  if (heavyAtoms > TRAINING_MAX_HEAVY_ATOMS) {
    return { ok: true, heavyAtoms, reason: null, warning: OUTSIDE_TRAINING,
      outsideTrainingRange: true };
  }

  return { ok: true, heavyAtoms, reason: null, warning: null,
    outsideTrainingRange: false };
}
