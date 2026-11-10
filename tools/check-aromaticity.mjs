// This port's PERCEIVED aromaticity against RDKit's, atom by atom and bond by
// bond, over the whole corpus.
//
// 🔴 WHY IT IS SEPARATE FROM `check-canonical-names.mjs`. That one compares
// bond TYPES, because a type is what reaches the model through the bond-order
// plane - so it sees a misperceived ring only where the ring changes a type.
// It does not compare the per-ATOM aromatic flag at all, and that flag decides
// the conformer's planarity rules and `valenceProblems`. This compares both.
//
// 🔴 AND THE CORPUS HAD TO GROW BEFORE THIS COULD MEAN ANYTHING. Every
// aromatic molecule in the original 74 is written in LOWER CASE with rings
// that each pass Huckel alone, so a parser that trusted the input's case
// scored 73/74 and a model that only ever tried SINGLE rings would score
// 74/74 - both wrong, neither visible. `kekule_benzene`, `azulene` and
// `tropone` are in `dump_rdkit_smiles.py`'s corpus for exactly that reason.
import { readFileSync } from "node:fs";
import { parseSmiles } from "../shared/chem/smiles.js";

const dump = JSON.parse(readFileSync(new URL("../oracle-dumps/rdkit-canon-ranks.json",
  import.meta.url), "utf8"));

const pairKey = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
let atomsExact = 0;
let bondsExact = 0;
let both = 0;
let atomsDiffering = 0;
let bondsDiffering = 0;
const failures = [];

for (const record of dump.records) {
  let graph;
  try {
    // 🔴 THROUGH `parseSmiles`, NOT THE MODULE DIRECTLY. The question is what
    // the PAGE gets, and the parser is what orders kekulise -> hydrogens ->
    // perceive. Calling the model on a hand-built graph would test it against
    // this file's idea of its inputs.
    graph = parseSmiles(record.smiles);
  } catch (error) {
    failures.push(`${record.name}: parse threw - ${error.message}`);
    continue;
  }
  const theirs = record.atoms.slice(0, graph.atoms.length).map((a) => !!a.aromatic);
  const ours = graph.atoms.map((a) => !!a.aromatic);
  const atomsOk = theirs.length === ours.length
    && theirs.every((value, index) => value === ours[index]);
  if (!atomsOk) {
    const at = theirs.findIndex((value, index) => value !== ours[index]);
    atomsDiffering += theirs.filter((v, i) => v !== ours[i]).length;
    failures.push(`${record.name}: atom ${at} aromatic ours ${ours[at]}`
      + ` RDKit ${theirs[at]}`);
  }

  const theirBonds = new Map();
  for (const [a, b, , aromatic] of record.bonds) {
    theirBonds.set(pairKey(a, b), !!aromatic);
  }
  let bondsOk = true;
  for (const bond of graph.bonds) {
    const key = pairKey(bond.from, bond.to);
    if ((theirBonds.get(key) ?? false) !== !!bond.aromatic) {
      bondsOk = false;
      bondsDiffering += 1;
      if (failures.length < 20) {
        failures.push(`${record.name}: bond ${bond.from}-${bond.to} aromatic`
          + ` ours ${!!bond.aromatic} RDKit ${theirBonds.get(key) ?? false}`);
      }
    }
  }
  if (atomsOk) atomsExact += 1;
  if (bondsOk) bondsExact += 1;
  if (atomsOk && bondsOk) both += 1;
}

const total = dump.records.length;
console.log(`RDKit ${dump.rdkit}`);
console.log(`atom aromatic flags exact: ${atomsExact} of ${total}`
  + (atomsDiffering ? `  (${atomsDiffering} atoms differ)` : ""));
console.log(`bond aromatic flags exact: ${bondsExact} of ${total}`
  + (bondsDiffering ? `  (${bondsDiffering} bonds differ)` : ""));
if (failures.length) {
  console.log("\nfirst disagreements:");
  for (const line of failures.slice(0, 20)) console.log(`  ${line}`);
}
console.log(`\nverdict: ${both}/${total}`
  + (both === total ? " - perception agrees with RDKit's model"
    : " - see spec/AROMATICITY-PORT.md"));
process.exit(both === total ? 0 : 1);
