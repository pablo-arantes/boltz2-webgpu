/**
 * The crop's centre atom is CA, not the pseudo-beta's CB.
 *
 * 🔴 WHY THIS IS NOT A CORNER CASE. Counted off upstream's own tables,
 * `res_to_center_atom` and `res_to_disto_atom` differ on 28 of 31 residue
 * types and agree only on GLY, N and DN. Ranking the affinity crop on
 * pseudo-beta therefore ranks protein residues by their CB where upstream uses
 * their CA - which reorders the neighbourhood and changes which pocket pass 2
 * sees. The tests below assert the DIFFERENCE as much as the value, because a
 * centre gather that quietly returned the pseudo-beta one would pass any test
 * that only checked shapes.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { atomName } from "../shared/af3/structure/pdb.js";
import { centreAtomGather, centreCoords }
  from "../shared/af3/affinity/centre-atoms.js";
import { smilesComponent } from "../shared/chem/component.js";

const boltz2 = () => featuriserDialect(dialectFor("boltz2"));
const batchOf = (sequence, options = {}) =>
  af3BatchFromA3m(sequence, null, { ...boltz2(), ...options }).batch;
const nameAt = (batch, index) => atomName(batch.refAtomNameChars, index);
const slotOf = (batch, gather, token) =>
  gather.indices[token] - token * batch.dense;

test("every protein token's centre is its CA", () => {
  const batch = batchOf("MKGTAYIAWFRQ");
  const gather = centreAtomGather(batch);
  for (let token = 0; token < batch.tokens; token += 1) {
    assert.equal(nameAt(batch, gather.indices[token]), "CA",
      `token ${token} (aatype ${batch.aatype[token]})`);
    assert.equal(gather.mask[token], 1);
  }
  assert.equal(gather.fromName, batch.tokens, "all found by name");
  assert.equal(gather.fromPseudoBeta, 0);
});

test("🔴 ...and it DIFFERS from the pseudo-beta gather on everything but glycine", () => {
  // Without this, a centre gather that returned the pseudo-beta one would pass
  // every other test here.
  const batch = batchOf("MKGTAYIA");            // one G among seven others
  const gather = centreAtomGather(batch);
  let differing = 0;
  let agreeing = 0;
  for (let token = 0; token < batch.tokens; token += 1) {
    const centre = slotOf(batch, gather, token);
    const pseudo = Number(batch.tokenAtomsToPseudoBeta.indices[token])
      - token * batch.dense;
    if (centre === pseudo) {
      agreeing += 1;
      // the only residue whose two atoms coincide is glycine, which has no CB
      assert.equal(batch.aatype[token], 7, `token ${token} agrees but is not GLY`);
      assert.equal(nameAt(batch, gather.indices[token]), "CA");
    } else {
      differing += 1;
      assert.equal(nameAt(batch, gather.indices[token]), "CA");
      assert.equal(nameAt(batch,
        Number(batch.tokenAtomsToPseudoBeta.indices[token])), "CB");
    }
  }
  assert.equal(agreeing, 1, "exactly the one glycine agrees");
  assert.equal(differing, batch.tokens - 1);
});

test("glycine's centre and pseudo-beta are both CA, as upstream's tables say", () => {
  const batch = batchOf("GGGG");
  const gather = centreAtomGather(batch);
  for (let token = 0; token < batch.tokens; token += 1) {
    assert.equal(nameAt(batch, gather.indices[token]), "CA");
    assert.equal(gather.indices[token],
      Number(batch.tokenAtomsToPseudoBeta.indices[token]));
  }
});

test("a ligand token falls back to the pseudo-beta slot, which IS its centre", async () => {
  // upstream sets atom_center = atom_disto = 0 for a ligand atom
  // (schema.py's own "Placeholder, no center"), so agreement is correct here.
  // 🔴 A LIGAND REACHES THE FEATURISER AS A COMPONENT, NOT A SMILES - the one
  // object `smilesComponent` and `parseCcdComponent` both produce, which is
  // what shared/chem/component.js's header calls the whole integration.
  const component = await smilesComponent("CCO", { code: "LIG" });
  const batch = batchOf("MKGTA", { ligands: [component] });
  const gather = centreAtomGather(batch);
  let ligandTokens = 0;
  for (let token = 0; token < batch.tokens; token += 1) {
    if (!batch.isLigand?.[token]) continue;
    ligandTokens += 1;
    assert.equal(gather.indices[token],
      Number(batch.tokenAtomsToPseudoBeta.indices[token]),
      `ligand token ${token}`);
  }
  assert.ok(ligandTokens >= 3, `expected the ethanol's atoms, got ${ligandTokens}`);
  assert.equal(gather.fromPseudoBeta, ligandTokens);
  assert.equal(gather.fromName, batch.tokens - ligandTokens);
});

test("centreCoords gathers from the positions it is given", () => {
  const batch = batchOf("MKG");
  const positions = new Float64Array(batch.tokens * batch.dense * 3);
  for (let i = 0; i < positions.length; i += 1) positions[i] = i;
  const { coords, fromName } = centreCoords(batch, positions);
  assert.equal(fromName, batch.tokens);
  for (let token = 0; token < batch.tokens; token += 1) {
    const gather = centreAtomGather(batch);
    const from = gather.indices[token] * 3;
    assert.equal(coords[token * 3], positions[from]);
    assert.equal(coords[token * 3 + 2], positions[from + 2]);
  }
});

test("a batch with no pseudo-beta gather is refused, not guessed", () => {
  assert.throws(() => centreAtomGather({ tokens: 2, dense: 24 }),
    /no pseudo-beta gather/);
  assert.throws(() => centreAtomGather({ tokens: "x", dense: 24,
    tokenAtomsToPseudoBeta: {} }), /tokens x/);
});
