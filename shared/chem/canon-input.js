/**
 * The molecule RDKit RANKS, built from the molecule this port PARSES.
 *
 * 🔴 UPSTREAM NAMES A SMILES LIGAND'S ATOMS FROM A CANONICAL RANK, AND THE
 * RANK IS TAKEN ON THE HYDROGEN-ADDED GRAPH. `boltz/data/parse/schema.py`:
 *
 *     mol = AllChem.MolFromSmiles(seq)
 *     mol = AllChem.AddHs(mol)
 *     canonical_order = AllChem.CanonicalRankAtoms(mol)
 *     for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
 *         atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
 *     ...
 *     mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
 *
 * so a heavy atom's NAME carries a rank computed with every hydrogen present,
 * and the hydrogens are then dropped while the names stay. Ranking the heavy
 * graph instead gives different numbers for the same molecule - the ranks are
 * a permutation of 0..n-1 and n is the H-ADDED count - so this module exists
 * to build that graph rather than the one the featuriser folds.
 *
 * 🔴 AND IT IS A **MODEL INPUT**, NOT A LABEL. The names reach Boltz-2 as
 * `ref_atom_name_chars`: 256 of the 388 atom-encoder input columns measured in
 * spec/G8-RESULT.md, and substituting upstream's own names took the ligand
 * half of that comparison from 1.1556 to 4.48e-2. A plausible-looking name is
 * a wrong feature, which is why this is gated against RDKit atom by atom
 * rather than inspected.
 */
import { adjacency } from "./smiles.js";
import { neighbourOrder } from "./stereo.js";
import { smallestRings, ringMembership } from "./rings.js";
import { ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";

/** RDKit's `Bond::BondType` ordinal for one of our parsed bonds. */
function bondType(bond) {
  if (bond.aromatic) return 12;              // AROMATIC
  return Math.max(1, Math.round(bond.order));
}

/** The parity of the permutation taking `from` to `to` (same members). */
function oddPermutation(from, to) {
  const working = from.slice();
  let swaps = 0;
  for (let index = 0; index < to.length; index += 1) {
    if (working[index] === to[index]) continue;
    const at = working.indexOf(to[index], index + 1);
    if (at < 0) return null;
    working[at] = working[index];
    working[index] = to[index];
    swaps += 1;
  }
  return swaps % 2 === 1;
}

/**
 * Our parsed molecule as the graph `canonicalRanks` wants, hydrogens explicit.
 *
 * 🔴 THE HYDROGENS GO AT THE END, IN HEAVY-ATOM ORDER, BECAUSE THAT IS WHERE
 * `AddHs` PUTS THEM - and it is not cosmetic. `BreakTies` resolves a surviving
 * tie by taking the FIRST member of the partition in the current order, so the
 * input numbering decides which of two symmetric atoms gets the lower rank.
 * Appending each heavy atom's hydrogens in turn reproduces RDKit's numbering
 * exactly: glucose comes out heavy 0-11 then H 12-23 with H12 on atom 0 and
 * H13/H14 both on atom 1, which is what the oracle dump shows.
 *
 * 🔴 AND `totalNumHs` IS ZERO ON EVERY ATOM OF THIS GRAPH. On the H-added
 * molecule RDKit's `canon_atom.totalNumHs` counts only the hydrogens that are
 * STILL implicit, and there are none. Feeding the original counts instead was
 * the single largest fault in this port's ranking - class order 41 of 74
 * against 62 - because it is a `basecomp` field, so it reorders classes rather
 * than merely failing to split them.
 *
 * @param {{atoms: object[], bonds: object[]}} graph a parsed SMILES molecule
 * @returns {{atoms: object[], bonds: object[][], heavyAtoms: number}}
 */
export function hydrogenatedGraph(graph) {
  const lists = adjacency(graph);
  const heavy = graph.atoms.length;
  const rings = smallestRings(graph);
  const membership = ringMembership(graph, rings);

  // every heavy atom's neighbours in RDKit's BOND order, which is the order
  // the bonds were created in - not the written order, which is a different
  // list for any ring opening (see stereo.js's note).
  // 🔴 A RING-CLOSURE BOND COMES AFTER EVERY CHAIN BOND, WHICH IS NOT THE
  // ORDER EITHER PARSER CREATES IT IN. RDKit's SMILES reader defers every
  // closure to `SmilesParseOps::CloseMolRings`, run once the string is read,
  // so `GetNeighbors()` lists the chain neighbours first whatever order the
  // digits appeared in - naphthalene's atom 8 is [7, 9, 3] where this port
  // built [7, 3, 9], and cholesterol's atom 26 moves TWO closures to the end
  // keeping their order relative to each other. Measured over the corpus this
  // was 40 of the atoms and, through `getChiralRank`'s permutation parity, 7
  // of the stereocentres' tags with it.
  //
  // 🔴 AND IT IS NOT THE SAME LIST `stereo.js` WANTS. That one needs the order
  // the string WROTE, where a closure digit counts at the atom it was written
  // on; this is the order the MOLECULE holds. Two orders, both load-bearing,
  // and the chiral tag below is the permutation between them.
  const bondOrderNbrs = graph.atoms.map((_, atom) => lists[atom].slice()
    .map((entry, position) => ({ entry, position }))
    .sort((a, b) => {
      const ra = graph.bonds[a.entry.bond];
      const rb = graph.bonds[b.entry.bond];
      return (Number(ra.ring === true) - Number(rb.ring === true))
        // ...and among an atom's OWN closures, by the ring digit: see the note
        // on `ringLabel` in smiles.js.
        || ((ra.ringLabel ?? 0) - (rb.ringLabel ?? 0))
        || (a.position - b.position);
    })
    .map(({ entry }) => entry.atom));

  // the hydrogens, appended heavy-atom by heavy-atom
  const hydrogenOf = graph.atoms.map(() => []);
  let next = heavy;
  graph.atoms.forEach((atom, index) => {
    for (let k = 0; k < (atom.hydrogens ?? 0); k += 1) {
      hydrogenOf[index].push(next);
      next += 1;
    }
  });
  const total = next;

  const bonds = Array.from({ length: total }, () => []);
  graph.bonds.forEach((bond) => {
    const type = bondType(bond);
    bonds[bond.from].push({ to: bond.to, type, stereo: 0 });
    bonds[bond.to].push({ to: bond.from, type, stereo: 0 });
  });
  hydrogenOf.forEach((list, parent) => {
    for (const h of list) {
      bonds[parent].push({ to: h, type: 1, stereo: 0 });
      bonds[h].push({ to: parent, type: 1, stereo: 0 });
    }
  });

  const atoms = [];
  graph.atoms.forEach((atom, index) => {
    const nbrIds = [...bondOrderNbrs[index], ...hydrogenOf[index]];
    atoms.push({
      z: atom.element,
      charge: atom.charge ?? 0,
      isotope: atom.isotope ?? 0,
      // 🔴 zero, not `atom.hydrogens` - see the note above
      totalNumHs: 0,
      chiralCode: chiralCodeOf(graph, lists, index, nbrIds, hydrogenOf[index]),
      atomMapNumber: atom.mapClass ?? 0,
      inRing: (membership[index] ?? []).length > 0,
      numAtomRings: (membership[index] ?? []).length,
      nbrIds,
    });
  });
  hydrogenOf.forEach((list, parent) => {
    for (const _h of list) {
      atoms.push({
        z: 1, charge: 0, isotope: 0, totalNumHs: 0, chiralCode: 0,
        atomMapNumber: 0, inRing: false, numAtomRings: 0, nbrIds: [parent],
      });
    }
  });

  return { atoms, bonds, heavyAtoms: heavy };
}

/**
 * A stereocentre's tag as RDKit holds it AFTER `AddHs`: 1 for CW, 2 for CCW.
 *
 * 🔴 THE SMILES SYMBOL IS NOT THE TAG, AND THE DIFFERENCE IS A PERMUTATION.
 * `@` means "anticlockwise reading the neighbours in the order the string
 * WROTE them", with an implicit hydrogen occupying the place right after the
 * preceding atom. RDKit's tag means the same thing read in the atom's own
 * neighbour order, and `AddHs` moves that hydrogen to the END of the list. So
 * the tag is the symbol's hand flipped by the parity of the permutation
 * between the two orders - derived, never special-cased by where the H sat,
 * because a ring opening permutes the heavy atoms too.
 */
function chiralCodeOf(graph, lists, index, nbrIds, hydrogens) {
  const atom = graph.atoms[index];
  if (atom.chirality !== "@" && atom.chirality !== "@@") return 0;
  const written = neighbourOrder(graph, lists, index);
  if (written === null) return 0;
  // the placeholder for the implicit hydrogen becomes its new index
  const placed = written.map((entry) => (entry === "H" ? hydrogens[0] : entry));
  if (placed.length !== nbrIds.length) return 0;
  const odd = oddPermutation(placed, nbrIds);
  if (odd === null) return 0;
  // CHI_TETRAHEDRAL_CCW is 2 in the oracle's coding, CW is 1
  const base = atom.chirality === "@" ? 2 : 1;
  return odd ? (base === 2 ? 1 : 2) : base;
}

/** The element symbol, as `atom.GetSymbol().upper()` gives it. */
function defaultSymbol(atom) {
  return (ELEMENT_SYMBOLS[atom.element - 1] ?? "X").toUpperCase();
}

/**
 * Upstream's atom names for a SMILES ligand's HEAVY atoms, in parse order.
 *
 * @param {{atoms: object[], bonds: object[]}} graph a parsed SMILES molecule
 * @param {(graph: object) => number[]} rank `canonicalRanks`
 * @returns {string[]} one name per heavy atom
 */
export function canonicalAtomNames(graph, rank, symbolOf = defaultSymbol) {
  const built = hydrogenatedGraph(graph);
  const ranks = rank({ atoms: built.atoms, bonds: built.bonds });
  const names = [];
  for (let index = 0; index < built.heavyAtoms; index += 1) {
    const name = `${symbolOf(graph.atoms[index]).toUpperCase()}${ranks[index] + 1}`;
    // 🔴 FOUR CHARACTERS IS UPSTREAM'S OWN LIMIT AND IT RAISES RATHER THAN
    // TRUNCATES: `f"{seq} has an atom with a name longer than 4 characters"`.
    // A truncated name is a silently wrong model input.
    if (name.length > 4) {
      throw new Error(`atom name ${name} is longer than 4 characters`);
    }
    names.push(name);
  }
  return names;
}
