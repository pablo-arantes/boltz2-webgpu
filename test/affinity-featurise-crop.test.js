/**
 * The crop's batch is the full batch's subset, with the gaps left in.
 *
 * The elementwise agreement with upstream's own featuriser is
 * `npm run test:boltz2-featurise`, which holds two cropped cases to
 * tools/oracle/dump_boltz2_featurise.py. What is here is the part an oracle
 * case cannot reach: the guards, and the fields whose meaning is internal to
 * this port (`sequence`, `residueOfToken`, the spans).
 */
import test from "node:test";
import assert from "node:assert/strict";
import { featuriseCrop } from "../shared/af3/affinity/featurise-crop.js";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { smilesComponent } from "../shared/chem/component.js";
import { binderChainOf } from "../web/affinity-passes.js";

const dialect = () => featuriserDialect(dialectFor("boltz2"));
async function build(sequence = "AGSKWASG", smiles = "CCCCCC") {
  const component = await smilesComponent(smiles, { code: "LIG" });
  return af3BatchFromA3m(sequence, null,
                         { ...dialect(), ligands: [component] }).batch;
}
const keepExcept = (batch, residues) => {
  const drop = new Set(residues);
  const kept = [];
  for (let t = 0; t < batch.tokens; t += 1) {
    const residue = batch.residueOfToken?.[t] ?? -1;
    if (residue < 0 || !drop.has(residue)) kept.push(t);
  }
  return kept;
};

test("🔴 residueIndex keeps its gaps and tokenIndex is renumbered dense", async () => {
  // The relative-position encoding reads differences of residueIndex, so the
  // gaps are what say those residues are not adjacent. Renumbering it would
  // fuse the crop into a contiguous chain - a different protein.
  const batch = await build();
  const cropped = featuriseCrop(batch, keepExcept(batch, [2, 5]), dialect());
  // dropping 0-based residues {2, 5} removes 1-based {3, 6}: this port's
  // indices are one-based, which is the convention test:boltz2-featurise
  // subtracts one for when comparing against upstream's zero-based ones.
  const protein = [...cropped.residueIndex].slice(0, 6);
  assert.deepEqual(protein, [1, 2, 4, 5, 7, 8], "1-based 3 and 6 are missing");
  assert.deepEqual([...cropped.tokenIndex].slice(0, 6), [1, 2, 3, 4, 5, 6]);
});

test("the atom count drops by exactly the cropped residues' atoms", async () => {
  const batch = await build();
  const cropped = featuriseCrop(batch, keepExcept(batch, [2]), dialect());
  // SER is N CA C O CB OG - six heavy atoms
  assert.equal(cropped.atomCount, batch.atomCount - 6);
});

test("the gathers are re-derived, not copied", async () => {
  const batch = await build();
  const cropped = featuriseCrop(batch, keepExcept(batch, [0, 1]), dialect());
  // a copied gather would still point at the old token count
  assert.equal(cropped.tokenAtomsToPseudoBeta.indices.length, cropped.tokens);
  for (let t = 0; t < cropped.tokens; t += 1) {
    const index = Number(cropped.tokenAtomsToPseudoBeta.indices[t]);
    assert.ok(index < cropped.tokens * cropped.dense,
      `token ${t} gathers atom ${index}, past the cropped layout`);
  }
  assert.ok(cropped.subsets <= batch.subsets);
});

test("sequence and residueOfToken are re-pointed at each other", async () => {
  // `sequence` is what the PDB writer names a residue from, and
  // `residueOfToken` is what indexes into it - carrying pass 1's string across
  // would name every residue after the first gap wrongly.
  const batch = await build("AGSKWASG");
  const cropped = featuriseCrop(batch, keepExcept(batch, [2]), dialect());
  assert.equal(cropped.sequence, "AGKWASG");
  for (let t = 0; t < cropped.tokens; t += 1) {
    const residue = cropped.residueOfToken[t];
    if (residue < 0) continue;
    assert.ok(residue < cropped.sequence.length,
      `token ${t} points at residue ${residue} of "${cropped.sequence}"`);
  }
});

test("a ligand token keeps residueOfToken -1, the sentinel", async () => {
  const batch = await build();
  const cropped = featuriseCrop(batch, keepExcept(batch, [3]), dialect());
  let ligands = 0;
  for (let t = 0; t < cropped.tokens; t += 1) {
    if (!cropped.isLigand[t]) continue;
    ligands += 1;
    assert.equal(cropped.residueOfToken[t], -1, `ligand token ${t}`);
  }
  assert.equal(ligands, 6);
});

test("ligandSpans are renumbered onto the cropped order", async () => {
  const batch = await build();
  const cropped = featuriseCrop(batch, keepExcept(batch, [0, 1, 2]), dialect());
  assert.equal(cropped.ligandSpans.length, 1);
  const span = cropped.ligandSpans[0];
  assert.equal(span.count, 6);
  for (let i = 0; i < span.count; i += 1) {
    assert.equal(cropped.isLigand[span.from + i], 1,
      `span member ${i} is not a ligand token`);
  }
});

test("🔴 a half-cropped component is refused, not silently shortened", async () => {
  // Half a ligand under a name that says otherwise is a molecule the model was
  // never shown. The crop keeps a component whole by construction, so this is
  // a bug in the caller's kept set.
  const batch = await build();
  const kept = [];
  let ligandSeen = 0;
  for (let t = 0; t < batch.tokens; t += 1) {
    if (batch.isLigand?.[t]) { ligandSeen += 1; if (ligandSeen > 3) continue; }
    kept.push(t);
  }
  assert.throws(() => featuriseCrop(batch, kept, dialect()),
    /has to be kept whole/);
});

test("the MSA is subset by column, keeping every row", async () => {
  const deep = [">101\nAGSKWASG", ">h1\nGGSRWGSG", ">h2\nAASKWATG"].join("\n") + "\n";
  const component = await smilesComponent("CCCCCC", { code: "LIG" });
  const batch = af3BatchFromA3m("AGSKWASG", deep,
                                { ...dialect(), ligands: [component] }).batch;
  const rows = batch.msa.length / batch.tokens;
  assert.ok(rows > 1, "the alignment has to be deep for this to mean anything");
  const kept = keepExcept(batch, [2]);
  const cropped = featuriseCrop(batch, kept, dialect());
  assert.equal(cropped.msa.length, rows * cropped.tokens);
  for (let row = 0; row < rows; row += 1) {
    for (let t = 0; t < cropped.tokens; t += 1) {
      assert.equal(cropped.msa[row * cropped.tokens + t],
        batch.msa[row * batch.tokens + kept[t]], `row ${row} token ${t}`);
    }
  }
});

test("the pair matrices subset BOTH axes the same way", async () => {
  const batch = await build();
  const kept = keepExcept(batch, [1, 4]);
  const cropped = featuriseCrop(batch, kept, dialect());
  assert.equal(cropped.bondMatrix.length, cropped.tokens * cropped.tokens);
  for (let i = 0; i < cropped.tokens; i += 1) {
    for (let j = 0; j < cropped.tokens; j += 1) {
      assert.equal(cropped.bondMatrix[i * cropped.tokens + j],
        batch.bondMatrix[kept[i] * batch.tokens + kept[j]]);
    }
  }
});

test("croppedFrom maps a result back to pass 1's numbering", async () => {
  const batch = await build();
  const kept = keepExcept(batch, [3]);
  const cropped = featuriseCrop(batch, kept, dialect());
  assert.deepEqual([...cropped.croppedFrom], kept);
});

test("an empty or malformed kept set is refused", async () => {
  const batch = await build();
  assert.throws(() => featuriseCrop(batch, [], dialect()), /nothing kept/);
  assert.throws(() => featuriseCrop(batch, [0, 0], dialect()), /repeats a token/);
  assert.throws(() => featuriseCrop(batch, [batch.tokens], dialect()),
    /outside 0\.\./);
});

test("🔴 the binder chain is derived, and ambiguity is refused", async () => {
  const batch = await build();
  const kept = keepExcept(batch, []);
  const ligandChain = batch.asymId[kept.find((t) => batch.isLigand?.[t])];
  assert.equal(binderChainOf(batch, kept), ligandChain);
  // no ligand kept at all
  const proteinOnly = kept.filter((t) => !batch.isLigand?.[t]);
  assert.throws(() => binderChainOf(batch, proteinOnly), /no binder/);
});
