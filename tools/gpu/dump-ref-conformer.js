/**
 * This port's reference conformer for the ligand, in its own dense-24 layout.
 *
 * 🔴 WHAT IS LEFT OF THE G8 GAP IS THIS. After the method-class fix the atom
 * encoder still disagrees with upstream at relRMS 1.16 on LIGAND tokens and
 * 4.4e-2 - the int5 floor - on protein ones, and `perAtomConditioning` reads
 * `refPos`, `refElement`, `refCharge` and `refAtomNameChars` directly. The
 * positions go through a LINEAR layer rather than through a distance, so the
 * same conformer in a different FRAME is a different embedding: centring and
 * orientation matter here in a way they never do for a structure prediction,
 * where docs/ rightly calls ref_pos a reported floor.
 */
import { af3BatchFromA3m } from "../../shared/af3/featurise/batch.js";
import { featuriserDialect } from "../../shared/af3/dialect.js";
import { openAf3Store, trunkWeights, trunkDepths } from "../../shared/af3/weights/weights.js";
import { smilesComponent, ligandName } from "../../shared/chem/component.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

const NAME_CHARS = 4;

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const sequence = option(args, "sequence",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ");
  const smiles = option(args, "ligands", "CCCc1ccccc1O");
  const store = await openAf3Store(manifest);
  const depths = trunkDepths(store);
  const weights = await trunkWeights(store, depths.pairformerBlocks,
                                     depths.msaBlocks, { allowPrefix: true });
  const dialect = featuriserDialect(weights.dialect);
  const component = await smilesComponent(smiles, { code: ligandName(0) });
  const { batch } = af3BatchFromA3m(sequence, null, { ...dialect,
    ligands: [component] });

  // the ligand's live slots, in the dense-24 layout
  const tokens = batch.tokens;
  const dense = batch.dense;
  const atoms = [];
  for (let token = 0; token < tokens; token += 1) {
    if (!batch.isLigand?.[token]) continue;
    for (let slot = 0; slot < dense; slot += 1) {
      const at = token * dense + slot;
      if (!(batch.refMask[at] > 0)) continue;
      let name = "";
      for (let c = 0; c < NAME_CHARS; c += 1) {
        const code = batch.refAtomNameChars[at * NAME_CHARS + c];
        if (code > 0) name += String.fromCharCode(code + 32);
      }
      atoms.push({
        token, slot, name: name.trim(),
        element: batch.refElement[at],
        charge: batch.refCharge[at],
        spaceUid: batch.refSpaceUid[at],
        pos: [batch.refPos[at * 3], batch.refPos[at * 3 + 1],
          batch.refPos[at * 3 + 2]],
      });
    }
  }
  // ...and the protein's first residue, as the control arm: it should AGREE
  const proteinAtoms = [];
  for (let slot = 0; slot < dense; slot += 1) {
    const at = slot;
    if (!(batch.refMask[at] > 0)) continue;
    let name = "";
    for (let c = 0; c < NAME_CHARS; c += 1) {
      const code = batch.refAtomNameChars[at * NAME_CHARS + c];
      if (code > 0) name += String.fromCharCode(code + 32);
    }
    proteinAtoms.push({ slot, name: name.trim(),
      element: batch.refElement[at], charge: batch.refCharge[at],
      pos: [batch.refPos[at * 3], batch.refPos[at * 3 + 1],
        batch.refPos[at * 3 + 2]] });
  }
  return { tokens, dense, smiles, ligandAtoms: atoms.length,
    ligand: atoms, proteinResidue1: proteinAtoms };
}
