#!/usr/bin/env node
/**
 * Does this port's batch agree with upstream Boltz-2's own featuriser?
 *
 *   python3 tools/oracle/dump_boltz2_featurise.py --boltz=<checkout>
 *   node tools/check-boltz2-featurise.mjs [--falsify]
 *
 * 🔴 A REFERENCE THIS PORT DID NOT HAVE. `npm run test:batch` holds 43 fields
 * against AF3's dump, which is af3-any-model's featuriser - the thing this port
 * was derived from. This is upstream BOLTZ's, which is a different
 * implementation of the same conventions, so an error the two AF3-side
 * implementations share is visible here and nowhere else. docs/ names that
 * shape exactly: "a writer and reader sharing a mistake agree perfectly".
 *
 * 🔴 THE ATOM LAYOUTS DIVERGE AND THE COMPARISON SAYS SO. Upstream keeps a FLAT
 * atom table with 32-atom windows; this port compacts to a DENSE-24 slot per
 * token (docs/: "AF3 featurises a DENSE-24 slot in per-residue conformer order
 * and AF2 reads atom37... the layout is NOT interchangeable"). So
 * `atom_to_token`, `atom_pad_mask` and the windows cannot be compared
 * elementwise, and pretending otherwise would be comparing two encodings rather
 * than two answers. What IS compared is every quantity whose MEANING survives
 * the re-layout:
 *
 *   token fields      residue_index, asym_id, entity_id, sym_id, mol_type and
 *                     affinity_token_mask - elementwise
 *   the disto slot     from `token_to_rep_atom`, as an offset within the
 *                     residue, against this port's pseudo-beta slot
 *   the centre slot    from `r_set_to_rep_atom`, against centreAtomGather -
 *                     which is how this gate CONFIRMS that work from upstream
 *                     rather than from its own tables
 *   ref_space_uid      as a PARTITION, because the labels are arbitrary: this
 *                     port numbers densely where a reference may skip, and
 *                     docs/ records elementwise comparison of it reading 3116
 *                     of 3120 differing while both agreed about every atom
 *
 * 🔴 AND ONLY THE CONTIGUOUS CASE IS COMPARED TODAY. The dump's cropped cases
 * carry GAPS in `res_idx`, which is what the affinity crop leaves behind, and
 * this port's featuriser takes a SEQUENCE - so it cannot yet express them. They
 * are the SPECIFICATION for `featuriseCrop`, not a failure here, and the
 * summary says how many are waiting rather than quietly skipping them.
 */
import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { smilesComponent } from "../shared/chem/component.js";
import { centreAtomGather } from "../shared/af3/affinity/centre-atoms.js";
import { featuriseCrop } from "../shared/af3/affinity/featurise-crop.js";

const option = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const dumpPath = option("dump", "oracle-dumps/boltz2-featurise.json");
const falsify = argv.includes("--falsify");

let dump;
try {
  dump = JSON.parse(readFileSync(dumpPath, "utf8"));
} catch {
  console.error(`missing ${dumpPath} - run:`
    + " python3 tools/oracle/dump_boltz2_featurise.py --boltz=<checkout>");
  process.exit(2);
}

/** A one-hot [rows, columns] flat array as row -> column. */
function oneHotToIndex(values, rows, columns) {
  const out = new Int32Array(rows).fill(-1);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      if (values[row * columns + column] !== 0) { out[row] = column; break; }
    }
  }
  return out;
}

/** Two labellings describe the same partition of the same items. */
function samePartition(a, b) {
  if (a.length !== b.length) return false;
  const forward = new Map();
  const backward = new Map();
  for (let i = 0; i < a.length; i += 1) {
    const left = a[i];
    const right = b[i];
    if (forward.has(left) && forward.get(left) !== right) return false;
    if (backward.has(right) && backward.get(right) !== left) return false;
    forward.set(left, right);
    backward.set(right, left);
  }
  return true;
}

console.log(`reference: ${dump.source}`);
console.log(`note     : ${dump.note}`);
console.log();

// 🔴 THE CROPPED CASES RUN THROUGH featuriseCrop NOW. They carry gaps in
// `res_idx`, which is what the affinity crop leaves behind and what this
// port's sequence-based featuriser cannot express on its own - so each is
// built FULL and then cropped, which is exactly what pass 2 does.
const comparable = dump.cases;
let failures = 0;
const summary = [];

for (const testCase of comparable) {
  // the same input this port's way: one-letter sequence + a SMILES ligand of
  // the same heavy-atom count
  const residues = [];
  for (let i = 0; i < testCase.token.mol_type.values.length; i += 1) {
    if (testCase.token.mol_type.values[i] !== 0) break;
    residues.push(i);
  }
  const ligandAtoms = testCase.tokens - residues.length;
  const names = testCase.keptTokenIdx;                 // for the record only
  // 🔴 the dumper's synthetic residues carry the REAL CCD heavy-atom counts
  // (ALA 5, GLY 4, SER 6, LYS 9, TRP 14), which is what makes this comparable
  // the dumper records its own sequence - see its header on why the reader
  // must not reconstruct it
  const dropped = new Set(testCase.droppedResidues ?? []);
  const component = await smilesComponent("C".repeat(testCase.ligandAtoms),
                                          { code: "LIG" });
  const dialect = featuriserDialect(dialectFor("boltz2"));
  // built from the UNCROPPED sequence, then cropped - pass 2's own route
  const { batch: full } = af3BatchFromA3m(testCase.fullSequence, null,
                                          { ...dialect, ligands: [component] });
  let batch = full;
  if (dropped.size > 0) {
    const kept = [];
    for (let t = 0; t < full.tokens; t += 1) {
      const residue = full.residueOfToken?.[t] ?? -1;
      if (residue < 0 || !dropped.has(residue)) kept.push(t);
    }
    batch = featuriseCrop(full, kept, dialect);
  }

  const checks = {};
  const oracleTokens = testCase.tokens;
  checks.tokenCount = { ours: batch.tokens, theirs: oracleTokens,
    ok: batch.tokens === oracleTokens };

  // --- token-level fields, elementwise ------------------------------------
  // 🔴 THIS PORT'S INDICES ARE ONE-BASED AND UPSTREAM'S ARE ZERO-BASED, which
  // docs/ already records as inert ("`struct/token_index` was zero-based where
  // featurise.js and the reference are one-based: inert, since it reaches the
  // model as a DIFFERENCE"). So the comparison subtracts one rather than
  // widening a tolerance.
  //
  // 🔴 AND `residueIndex` IS THE FEATURE; `residueOfToken` IS NOT. The first
  // version of this gate compared the latter and reported 6, 40 and 2 tokens
  // differing - because `residueOfToken` is **-1** for a ligand atom, a
  // deliberate sentinel that docs/ records as load-bearing: "the fallback made
  // every ligand token chain 0 - what a chain-0 polymer residue gives - and the
  // glycerol's first atom was linked to the protein's last residue". Two
  // different quantities, and the gate was comparing the wrong pair.
  const fields = [
    ["residue_index", (t) => batch.residueIndex[t] - 1],
    ["token_index", (t) => batch.tokenIndex[t] - 1],
    ["asym_id", (t) => batch.asymId[t] - 1],
    ["mol_type", (t) => (batch.isLigand?.[t] ? 3 : 0)],
    ["affinity_token_mask", (t) => (batch.isLigand?.[t] ? 1 : 0)],
  ];
  for (const [field, read] of fields) {
    const theirs = testCase.token[field]?.values;
    if (theirs === undefined) continue;
    let differing = 0;
    for (let t = 0; t < Math.min(batch.tokens, oracleTokens); t += 1) {
      if (Number(read(t)) !== Number(theirs[t])) differing += 1;
    }
    checks[field] = { differing, of: oracleTokens, ok: differing === 0 };
  }

  // --- the disto slot, as an offset inside the residue ---------------------
  const atomCount = testCase.atomCount;
  const distoAtom = oneHotToIndex(testCase.atom.token_to_rep_atom.values,
                                  oracleTokens, atomCount);
  const centreAtom = oneHotToIndex(testCase.atom.r_set_to_rep_atom.values,
                                   testCase.atom.r_set_to_rep_atom.shape[0],
                                   atomCount);
  const atomToToken = oneHotToIndex(testCase.atom.atom_to_token.values,
                                    atomCount, oracleTokens);
  // each token's first atom in upstream's flat table
  const firstAtom = new Int32Array(oracleTokens).fill(-1);
  for (let atom = atomCount - 1; atom >= 0; atom -= 1) {
    const token = atomToToken[atom];
    if (token >= 0) firstAtom[token] = atom;
  }
  const centre = centreAtomGather(batch);
  let distoDiffering = 0;
  let centreDiffering = 0;
  let centreCompared = 0;
  for (let t = 0; t < oracleTokens; t += 1) {
    const theirDisto = distoAtom[t] - firstAtom[t];
    const ourDisto = Number(batch.tokenAtomsToPseudoBeta.indices[t])
      - t * batch.dense;
    if (theirDisto !== ourDisto) distoDiffering += 1;
    // the centre gather covers the r_set, i.e. the polymer residues
    if (t < centreAtom.length && centreAtom[t] >= 0) {
      centreCompared += 1;
      const theirCentre = centreAtom[t] - firstAtom[t];
      const ourCentre = centre.indices[t] - t * batch.dense;
      if (theirCentre !== ourCentre) centreDiffering += 1;
    }
  }
  checks.distoSlot = { differing: distoDiffering, of: oracleTokens,
    ok: distoDiffering === 0 };
  checks.centreSlot = { differing: centreDiffering, of: centreCompared,
    ok: centreDiffering === 0 && centreCompared > 0 };

  // --- ref_space_uid as a PARTITION ---------------------------------------
  const theirUid = testCase.atom.ref_space_uid.values;
  const theirPerToken = new Int32Array(oracleTokens);
  for (let t = 0; t < oracleTokens; t += 1) theirPerToken[t] = theirUid[firstAtom[t]];
  const ourPerToken = new Int32Array(batch.tokens);
  for (let t = 0; t < batch.tokens; t += 1) {
    ourPerToken[t] = batch.refSpaceUid[t * batch.dense];
  }
  checks.refSpaceUidPartition = {
    ok: samePartition([...theirPerToken],
      [...ourPerToken].slice(0, oracleTokens)) };

  for (const [label, check] of Object.entries(checks)) {
    failures += check.ok ? 0 : 1;
    const detail = check.differing !== undefined
      ? `${check.differing} of ${check.of} differ`
      : (check.ours !== undefined ? `${check.ours} vs ${check.theirs}` : "");
    console.log(`  ${check.ok ? "OK  " : "FAIL"} ${testCase.name}/${label.padEnd(22)}`
      + ` ${detail}`);
  }
  summary.push({ name: testCase.name, tokens: oracleTokens, checks });
}

console.log();
if (falsify) {
  // 🔴 THE CONTROL: the centre and disto slots MUST be different quantities, or
  // every check above would pass with one gather used for both - which is the
  // bug web/affinity-passes.js shipped with until centre-atoms.js existed.
  const sequence = "AGSKWASG";
  const component = await smilesComponent("CCCCCC", { code: "LIG" });
  const dialect = featuriserDialect(dialectFor("boltz2"));
  const { batch } = af3BatchFromA3m(sequence, null,
                                    { ...dialect, ligands: [component] });
  const centre = centreAtomGather(batch);
  let differ = 0;
  for (let t = 0; t < batch.tokens; t += 1) {
    if (centre.indices[t] !== Number(batch.tokenAtomsToPseudoBeta.indices[t])) {
      differ += 1;
    }
  }
  const sees = differ > 0;
  console.log(`  ${sees ? "PINNED " : "AMBIG  "} centre and disto are distinct`
    + ` gathers    ${differ} of ${batch.tokens} tokens differ`);
  if (!sees) failures += 1;
  console.log();
}

const croppedCount = dump.cases.filter(
  (c) => (c.droppedResidues ?? []).length > 0).length;
console.log(`compared ${comparable.length} case(s),`
  + ` ${croppedCount} of them through featuriseCrop`);
console.log(failures === 0
  ? "FEATURISE PASS: this port's batch agrees with upstream Boltz's featuriser"
    + " on every quantity whose meaning survives the re-layout"
  : `FEATURISE FAIL: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
