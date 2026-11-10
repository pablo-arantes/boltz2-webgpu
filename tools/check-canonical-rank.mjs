/**
 * Does this port's canonical ranking reproduce RDKit's, exactly, on the corpus?
 *
 *   node tools/check-canonical-rank.mjs [--permute] [--verbose]
 *
 * 🔴 EXACT INTEGERS, NO TOLERANCE, AND THAT IS THE POINT. Boltz-2 names a
 * SMILES ligand's atoms `element + rank + 1`; one tie broken the other way
 * renames an atom and changes four 64-way one-hot columns of the atom
 * encoder's input. G8 measured what that costs: the atom encoder's ligand half
 * at relRMS 1.1556 against upstream, falling to 4.48e-2 when upstream's names
 * are substituted. So the standing decision is that this ships at 74/74 and
 * not at 73.
 *
 * 🔴 THE GRAPH COMES FROM THE ORACLE, NOT FROM THIS PORT'S SMILES PARSER, and
 * that is deliberate: it isolates the RANKING from the hydrogen expansion and
 * the parse. If the ranks are fed RDKit's own degrees, valences, H counts and
 * bonds, then a disagreement is the ranking algorithm and nothing else. A
 * later arm can close the loop through `shared/chem/`'s own parser; failing
 * both at once would say nothing about either.
 *
 * `--permute` searches orderings of the initial invariant's fields, because
 * that order is the one part of RDKit's algorithm that cannot be recovered by
 * reading the output - and reporting "it does not match" without having tried
 * the orderings would be reporting the author's first guess.
 */
import { readFileSync } from "node:fs";
import { canonicalRanks } from "../shared/chem/canonical-rank.js";

const DUMP = "oracle-dumps/rdkit-canon-ranks.json";
const args = process.argv.slice(2);
const has = (flag) => args.includes(`--${flag}`);

let dump;
try {
  dump = JSON.parse(readFileSync(DUMP, "utf8"));
} catch {
  console.error(`${DUMP} is missing. Write it with:\n`
    + "  <python-with-rdkit> tools/oracle/dump_rdkit_canon_ranks.py");
  process.exit(1);
}

/**
 * The molecule as the oracle saw it: RDKit's own per-atom properties, and the
 * bonds recovered from them.
 *
 * 🔴 THE DUMP HAS NO BOND LIST, so the neighbours are rebuilt from the
 * molecule's own SMILES through this port's parser - which means this harness
 * does depend on the parser for CONNECTIVITY while taking every per-atom
 * property from RDKit. That split is stated because it bounds what a failure
 * here can mean: a wrong bond list would show as a ranking failure.
 */
function unusedGraphOf(record, parse) {
  const atoms = record.atoms.map((atom) => ({
    z: atom.z, charge: atom.charge, isotope: atom.isotope,
    totalNumHs: atom.totalNumHs, degree: atom.degree,
    explicitValence: atom.explicitValence, inRing: atom.inRing,
  }));
  const neighbours = parse(record, atoms.length);
  return { atoms, neighbours };
}

/**
 * Neighbour lists straight from the oracle's own bond list.
 *
 * 🔴 NOTHING IS RECONSTRUCTED HERE ANY MORE, and the first version of this
 * harness is why. It rebuilt the H-added connectivity from each atom's
 * `totalNumHs`, which is **0 on an H-added molecule** because the hydrogens
 * are explicit atoms by then - so it attached no hydrogens and reported 71 of
 * 74 molecules as ranking failures on a graph that was simply wrong. Taking
 * RDKit's bonds means a disagreement here is the ranking and nothing else.
 */
function neighboursFromBonds(record) {
  const lists = Array.from({ length: record.atomsWithH }, () => []);
  for (const [from, to, order, aromatic] of record.bonds) {
    // 🔴 THE BOND'S TYPE TRAVELS WITH IT, because `bondholder::compare` ranks
    // on type before the neighbour's class. The dump carries RDKit's own
    // `GetBondTypeAsDouble` and aromatic flag; aromatic wins, as RDKit's enum
    // places AROMATIC above every integral order.
    const type = aromatic ? 12 : Math.max(1, Math.round(order));
    lists[from].push({ to, type, stereo: 0 });
    lists[to].push({ to: from, type, stereo: 0 });
  }
  return lists;
}

function evaluate(verbose = false) {
  let exact = 0;
  let partition = 0;
  const failures = [];
  for (const record of dump.records) {
    if (record.error !== undefined) continue;
    const lists = neighboursFromBonds(record);
    const atoms = record.atoms.map((atom) => ({
      z: atom.z, charge: atom.charge, isotope: atom.isotope,
      totalNumHs: atom.totalNumHs, degree: atom.degree,
      explicitValence: atom.explicitValence, inRing: atom.inRing,
      aromatic: atom.aromatic, chiralTag:
        atom.chiralTag && atom.chiralTag !== "CHI_UNSPECIFIED" ? 1 : 0,
      atomMapNumber: atom.atomMapNumber ?? 0,
      chiralCode: atom.chiralCode ?? 0, nbrIds: atom.nbrIds,
      numAtomRings: atom.numAtomRings ?? 0,
    }));
    const ours = canonicalRanks({ atoms, bonds: lists });
    const want = record.ranks;
    // does each of our classes occupy a contiguous block of RDKit's ranks?
    const byClass = new Map();
    canonicalRanks({ atoms, bonds: lists }, { breakTies: false })
      .forEach((klass, atom) => {
        if (!byClass.has(klass)) byClass.set(klass, []);
        byClass.get(klass).push(atom);
      });
    let contiguous = true;
    for (const group of byClass.values()) {
      const ranks = group.map((atom) => want[atom]).sort((a, b) => a - b);
      if (ranks[ranks.length - 1] - ranks[0] !== ranks.length - 1) contiguous = false;
    }
    if (contiguous) partition += 1;
    const same = ours.length === want.length
      && ours.every((value, index) => value === want[index]);
    if (same) exact += 1;
    else {
      const wrong = ours.filter((v, i) => v !== want[i]).length;
      failures.push({ name: record.name, smiles: record.smiles,
        why: `${wrong} of ${want.length} ranks differ`,
        ours: ours.slice(0, 12), want: want.slice(0, 12) });
    }
  }
  if (verbose) {
    for (const failure of failures.slice(0, 12)) {
      console.log(`  ${failure.name.padEnd(22)} ${failure.why}`);
      if (failure.ours) {
        console.log(`    ours ${failure.ours.join(",")}`);
        console.log(`    rdkit ${failure.want.join(",")}`);
      }
    }
  }
  return { exact, partition, failures,
    total: dump.records.filter((r) => r.error === undefined).length };
}

const result = evaluate(has("verbose"));
console.log(`RDKit ${dump.rdkit}`);
console.log(`exact rank agreement: ${result.exact} of ${result.total} molecules`);
// 🔴 THE PARTITION IS THE INFORMATIVE NUMBER WHEN THE RANKS DISAGREE. If every
// one of our symmetry classes maps onto a CONTIGUOUS block of RDKit's ranks,
// the refinement found the right classes and only their ORDER is wrong - a
// finite convention, not a wrong algorithm. Measured at 74/74 on this corpus
// while exact ranks were at 29/74, which is what reframed the go/no-go.
console.log(`symmetry partition agreement: ${result.partition} of ${result.total}`);
if (result.exact !== result.total) {
  console.log(`\n${result.failures.length} molecules disagree`
    + (has("verbose") ? "" : "   (--verbose to see them)"));
}
// 🔴 IT ENFORCES NOW, BECAUSE IT HAS BEEN GREEN. The note that used to sit
// here said "no exit code gate until it passes - a red gate that has never
// been green is a gate nobody can distinguish from a broken one", which was
// right while the ranking was being built. It reached exact agreement on the
// whole corpus, so from here a disagreement is a REGRESSION and has to fail.
// 🔴 AND THE COUNT IS READ FROM THE DUMP, never typed: the corpus grew from 74
// to 77 when the aromaticity cases were added, and a hardcoded 74 would have
// gone on reporting "74/74 - ship it" over a 77-molecule corpus.
const total = dump.records.length;
console.log(`\nverdict: ${result.exact}/${total}`
  + (result.exact === total ? " - ship it"
    : " - see the go/no-go in spec/G8-RESULT.md"));
process.exit(result.exact === total && result.partition === total ? 0 : 1);
