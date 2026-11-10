/**
 * RDKit's canonical atom ranking, ATTEMPTED — and it does NOT reproduce RDKit.
 * Nothing imports this module; it is kept because half of it is right and a
 * future attempt should not start from scratch.
 *
 * 🔴 WHERE IT STANDS, MEASURED (tools/check-canonical-rank.mjs, 74-molecule
 * corpus, RDKit 2025.03.5):
 *
 *     exact ranks                                     66/74
 *     partition vs RDKit's OWN pre-break classes      74/74
 *     tie-free molecules                                6/6
 *
 * 🔴 THE LAST STRUCTURAL FAULT WAS FOUND BY BUILDING RDKit AND READING ITS OWN
 * TRACE, not by reading its source: `rankMolAtoms` does NOT pre-split by
 * basecomp. Its VERBOSE_CANON passes "1" and "1a" print a SINGLE class for the
 * whole molecule, and the one refinement that follows uses the full
 * comparator. Pre-splitting is not an optimisation - from one partition every
 * neighbour class is 0, so the first sort compares bond TYPES, and for
 * indole's atoms 5 and 7 that makes position 2 decide with
 * `single(1) < aromatic(12)`. Fixing it took 64 -> 66/74 and every tie-free
 * molecule to exact. refs/canon-trace/ holds the driver.
 *
 * 🔴 THE REFINEMENT IS RIGHT ON EVERY MOLECULE. Compared against
 * `CanonicalRankAtoms(mol, breakTies=False)` - RDKit's own symmetry classes,
 * not inferred from final ranks - this port finds the IDENTICAL partition on
 * all 74. What differs, on 10 molecules, is the ORDER those classes are
 * numbered in; and because exact ranks are also 64, the tie-break is right
 * wherever the numbering is. The whole remaining fault is one ordering.
 *
 * Ported faithfully from `new_canon.h` / `new_canon.cpp` at Release_2025_03_5
 * (in refs/rdkit-src/): `basecomp`'s field order including the unsigned-charge
 * quirk and the atom-map term, `bondholder::compare`, the descending bond sort
 * and parallel walk, `BreakTies`' last-atom/highest-rank rule, and the
 * `SpecialChirality` pass with `updateAtomNeighborNumSwaps` and
 * `countSwapsToInterconvert`.
 *
 * Both special passes are ported too - `SpecialChirality` and
 * `SpecialSymmetry` with `compareRingAtomsConcerningNumNeighbors` - and
 * TOGETHER THEY GAIN ONE MOLECULE. That is not a defect in the ports; it is
 * what the structure of the algorithm implies, and measuring it is what
 * finally located the fault:
 *
 * 🔴 EVERY REMAINING FAILURE IS A CLASS-**ORDER** ERROR, AND A REFINEMENT PASS
 * CANNOT FIX ONE. `RefinePartitions` sorts WITHIN a partition - it splits
 * classes and never reorders them - so no comparator added after the main
 * refinement can change which class comes first. Both passes fire correctly
 * (naphthalene meets the symmetry gate: 10 ring atoms, branching, ties) and
 * both change nothing.
 *
 * 🔴 AND THE INITIAL PARTITION IS RIGHT ON 74/74, measured: basecomp alone
 * puts every molecule's atoms into classes that are contiguous in RDKit's
 * ranks AND in the same order. So the divergence is introduced DURING
 * refinement.
 *
 * 🔴 WHICH LEAVES ONE THING, AND IT IS THE DRIVER. RDKit carries a persistent
 * `order` array across sweeps and partitions, sorting each partition's slice
 * in place with a STABLE sort (`hanoisort`) and assigning
 * `symclass = offset + i` from positions in that array. This implementation
 * re-ranks globally every sweep and breaks ties by ATOM INDEX. Those agree on
 * the first sweep and can diverge on every one after it, because a stable sort
 * preserves the PREVIOUS order where index tie-breaking does not. Porting
 * `RefinePartitions`' order/count/activeset bookkeeping is the remaining work.
 *
 * 🔴 WHY THIS EXISTS, MEASURED. upstream builds a ligand atom's name as
 * `element + (CanonicalRankAtoms + 1)` over the molecule WITH hydrogens
 * (boltz/data/parse/schema.py). The four name characters are 256 of the 388
 * input columns of `embed_atom_features`. G8 compared this port's names
 * (AF3's per-element counter, `C1..C9, O1`) against upstream's
 * (`C20, C22, ... O13`) on a real job: **not one of ten agreed**, the atom
 * encoder's ligand half read relRMS **1.1556**, and substituting upstream's
 * names took it to **4.48e-2** — the int5 quantisation floor. So this is the
 * whole of the remaining disagreement with upstream, and nothing else is left
 * above quantisation.
 *
 * 🔴 EXACT INTEGERS OR NOTHING. Unlike a conformer, a rank admits no
 * tolerance: one tie broken the other way renames an atom, and a renamed atom
 * is a different 64-way one-hot in four columns. There is no "close". So the
 * gate is exact agreement over the 74-molecule corpus the SMILES gates already
 * use (`tools/check-canonical-rank.mjs`), and the standing decision is that
 * this ships only at 74/74.
 *
 * 🔴 AND IT IS A RANK, NOT CHEMISTRY. `C20` says nothing about that carbon
 * except where it sorted among 22 atoms. The model was trained on these
 * labels, so reproducing them is what agreement requires — but two thirds of
 * the atom encoder's input for a SMILES ligand is an arbitrary index. Recorded
 * because it is surprising and because it bounds how much meaning to read into
 * the feature.
 *
 * THE ALGORITHM, as RDKit's `rankMolAtoms` is structured:
 *   1. an initial invariant per atom, from local properties only;
 *   2. iterative refinement: an atom's new class is its old class plus the
 *      sorted multiset of its neighbours' old classes, until stable;
 *   3. tie-breaking: the lowest-index atom of the lowest tied class is split
 *      off into its own class, then refinement resumes — repeated until every
 *      atom is alone.
 *
 * 🔴 STEP 1'S FIELD ORDER IS THE PART THAT CANNOT BE GUESSED, so it is
 * `INVARIANT_FIELDS` here — one list, in one place, inferred by measurement
 * against the oracle rather than from recollection of RDKit's source.
 */

/**
 * The initial invariant's fields, most significant first.
 *
 * Each entry reads one local property. The ORDER is what the oracle decides;
 * `tools/check-canonical-rank.mjs --permute` is what searched it.
 */
import { createPartitions, seedPartitions, activatePartitions,
  refinePartitions, breakTies } from "./canon-partitions.js";

/**
 * `basecomp`'s field order, read from RDKit's `new_canon.h:299` at
 * Release_2025_03_5 — the version the oracle was dumped with.
 *
 * 🔴 THREE FIELDS THIS PORT HAD ARE NOT IN IT, AND ONE IT LACKED IS. The first
 * attempt used `degree, z, valence, chargeSign, chargeMag, nH, inRing,
 * aromatic` and plateaued at 41/74 on class order under a scored search. The
 * source has no `explicitValence`, no `inRing` and no `aromatic` at all: they
 * were invented, and inventing them reorders classes on every ring molecule,
 * which is why phenol, toluene, naphthalene and indole failed. `totalNumHs`
 * also comes BEFORE charge, not after.
 */
const BASECOMP = [
  // 🔴 THE ATOM MAP NUMBER COMES FIRST, BEFORE DEGREE (`df_useAtomMaps`,
  // default true). A mapped atom is a user-asserted identity and outranks
  // every structural property - which is why it sorts ahead of everything.
  (a) => a.atomMapNumber,
  (a) => a.degree,
  (a) => a.z,
  (a) => a.isotope,
  (a) => a.totalNumHs,
  // 🔴 CHARGE IS SIGNED, READ INTO AN **UNSIGNED** VARIABLE, and that is not a
  // transcription slip - it is what the source does:
  //
  //     unsigned int ivi = dp_atoms[i].atom->getFormalCharge();
  //
  // so -1 becomes 4294967295 and the effective order is
  // `0 < +1 < +2 < ... < -2 < -1`. Negative charges sort ABOVE every neutral
  // and positive atom. This port's first attempt decomposed charge into
  // (sign, magnitude), which sorts negatives BELOW neutrals - the exact
  // opposite - and acetate, methanesulfonate and sulfate were among the
  // molecules it could not place. No amount of fitting against RDKit's output
  // would have produced this; it had to be read.
  (a) => (a.charge < 0 ? a.charge + 4294967296 : a.charge),
  // presence of specified chirality (df_useChiralPresence, default true)
  (a) => (a.chiralTag ? 1 : 0),
];

/** Bond types as RDKit's enum orders them, which is what `compare` ranks on. */
const BOND_TYPE_ORDER = {
  UNSPECIFIED: 0, SINGLE: 1, DOUBLE: 2, TRIPLE: 3, QUADRUPLE: 4,
  QUINTUPLE: 5, HEXTUPLE: 6, ONEANDAHALF: 7, TWOANDAHALF: 8,
  THREEANDAHALF: 9, FOURANDAHALF: 10, FIVEANDAHALF: 11, AROMATIC: 12,
};

/**
 * `bondholder::compare`: bond type, then bond stereo, then the neighbour's
 * current class.
 *
 * 🔴 THE BOND TYPE DOMINATES THE NEIGHBOUR'S CLASS, which is the precedence
 * this port had inverted. Its first attempt compared a sorted multiset of
 * neighbour CLASSES and treated the bond order as an optional extra column
 * appended after it; the source ranks each bond by type first and only then by
 * whom it reaches.
 */
function compareBond(a, b) {
  if (a.type !== b.type) return a.type < b.type ? -1 : 1;
  if (a.stereo !== b.stereo) return a.stereo < b.stereo ? -1 : 1;
  if (a.klass !== b.klass) return a.klass < b.klass ? -1 : 1;
  return 0;
}

function compareKeys(a, b) {
  const n = Math.max(a.length, b.length);
  for (let index = 0; index < n; index += 1) {
    const x = a[index] ?? -1;
    const y = b[index] ?? -1;
    if (x !== y) return x < y ? -1 : 1;
  }
  return 0;
}

/**
 * Swaps to turn `probe` into `ref` — RDKit's `countSwapsToInterconvert`.
 *
 * 🔴 ONLY ITS PARITY IS USED, which is the point: the chirality pass asks
 * whether reading a stereocentre's neighbours in canonical order preserves or
 * inverts the hand it was given, and that is a parity question.
 */
function countSwapsToInterconvert(ref, probe) {
  const working = probe.slice();
  let swaps = 0;
  for (let index = 0; index < ref.length && index < working.length; index += 1) {
    if (working[index] === ref[index]) continue;
    const at = working.indexOf(ref[index], index + 1);
    if (at < 0) continue;
    working[at] = working[index];
    working[index] = ref[index];
    swaps += 1;
  }
  return swaps;
}

/**
 * `updateAtomNeighborNumSwaps`: per bond, a (neighbour class, hand) pair.
 *
 * 🔴 THE HAND IS 1 OR 2 AND ONLY FOR A CHIRAL NEIGHBOUR OF A **RING** ATOM.
 * `isRingAtom && neighbour has a chiral tag` is the whole gate: everything
 * else contributes 0, so this comparator is inert on an acyclic molecule and
 * on any ring atom whose neighbours are not stereocentres. That narrowness is
 * why the pass is cheap and why it only moves the stereocentre-bearing
 * molecules.
 *
 * 🔴 AND `tooManySimilarNbrs` SUPPRESSES IT. If two of the neighbour's own
 * neighbours (excluding this atom) already share a class, the parity is not
 * yet determined and the pair is recorded as 0 rather than as a guess - which
 * is what makes the pass safe to run before the classes have settled.
 */
function neighbourSwaps(atom, atoms, bonds, sortedBondsFor, klass) {
  const isRingAtom = (atoms[atom].numAtomRings ?? 0) > 0
    || atoms[atom].inRing === true;
  const result = [];
  for (const bond of sortedBondsFor(atom)) {
    const other = bond.to;
    const chiral = atoms[other].chiralCode ?? 0;
    if (!isRingAtom || chiral === 0) {
      result.push([bond.klass, 0]);
      continue;
    }
    const ref = atoms[other].nbrIds ?? bonds[other].map((b) => b.to);
    let tooManySimilar = false;
    const seen = [];
    for (const nbrNbr of ref) {
      if (nbrNbr === atom) continue;
      if (seen.includes(klass[nbrNbr])) tooManySimilar = true;
      else seen.push(klass[nbrNbr]);
    }
    if (tooManySimilar) { result.push([bond.klass, 0]); continue; }
    const probe = [atom, ...sortedBondsFor(other).map((b) => b.to)
      .filter((to) => to !== atom)];
    const odd = countSwapsToInterconvert(ref, probe) % 2 === 1;
    result.push([bond.klass, chiral === 1 ? (odd ? 2 : 1) : (odd ? 1 : 2)]);
  }
  result.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return result;
}

/**
 * `compareRingAtomsConcerningNumNeighbors`: two per-ring-atom profiles from a
 * breadth-first walk that only travels through ring atoms.
 *
 * 🔴 BOTH PROFILES ARE LEVEL SEQUENCES SEPARATED BY -1, compared
 * lexicographically as `std::vector<int>` is. `neighborNum` is how many atoms
 * each level newly reached; `revistedNeighbors` is the SORTED multiset, per
 * level, of how many times each atom was reached from the current or previous
 * level. The second is what distinguishes a fused ring system from a chain of
 * the same size - a bridgehead gets revisited where a plain ring atom does
 * not - which is why this pass exists and why it is gated on fused rings.
 *
 * 🔴 NON-RING ATOMS GET EMPTY PROFILES, because the source `continue`s before
 * assigning them. An empty vector sorts below every non-empty one, so every
 * non-ring atom sorts ahead of every ring atom under this comparator - which
 * is harmless only because the pass runs INSIDE existing partitions.
 */
function ringNeighbourProfiles(atoms, bonds) {
  const n = atoms.length;
  const nbrIds = atoms.map((atom, index) =>
    atom.nbrIds ?? bonds[index].map((bond) => bond.to));
  const inRing = atoms.map((atom) => (atom.numAtomRings ?? 0) > 0);
  const profiles = atoms.map(() => ({ neighborNum: [], revisited: [] }));
  for (let start = 0; start < n; start += 1) {
    if (!inRing[start]) continue;
    const visited = new Uint8Array(n);
    let lastLevel = new Uint8Array(n);
    let queue = [start];
    const profile = profiles[start];
    while (queue.length > 0) {
      const currentLevel = new Uint8Array(n);
      const revisited = new Int32Array(n);
      const nextLevel = [];
      let numLevelNbrs = 0;
      for (const atom of queue) {
        // 🔴 THE WALK ONLY EXPANDS THROUGH RING ATOMS, but it still COUNTS
        // non-ring neighbours when it reaches them. Expanding through them
        // would make the profile a whole-molecule walk.
        if (!inRing[atom]) continue;
        lastLevel[atom] = 1;
        visited[atom] = 1;
        for (const other of nbrIds[atom]) {
          if (visited[other]) continue;
          currentLevel[other] = 1;
          numLevelNbrs += 1;
          visited[other] = 1;
          nextLevel.push(other);
        }
      }
      for (let atom = 0; atom < n; atom += 1) {
        if (!currentLevel[atom]) continue;
        for (const other of nbrIds[atom]) {
          if (currentLevel[other] || lastLevel[other]) revisited[other] += 1;
        }
      }
      lastLevel = currentLevel;
      const counts = [];
      for (let atom = 0; atom < n; atom += 1) {
        if (revisited[atom] > 0) counts.push(revisited[atom]);
      }
      counts.sort((a, b) => a - b);
      profile.revisited.push(...counts, -1);
      profile.neighborNum.push(numLevelNbrs, -1);
      queue = nextLevel;
    }
  }
  return profiles;
}

/** Lexicographic comparison of two int sequences, as std::vector<int> is. */
function compareSequence(a, b) {
  const shared = Math.min(a.length, b.length);
  for (let index = 0; index < shared; index += 1) {
    if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1;
  }
  if (a.length !== b.length) return a.length < b.length ? -1 : 1;
  return 0;
}

/**
 * The gate `rankMolAtoms` puts on the symmetry pass:
 * `ties && ringAtoms > 0 && symRingAtoms / ringAtoms > 0.5 && branchingRingAtom`.
 *
 * 🔴 IT IS NARROW ON PURPOSE AND THE FRACTION IS LOAD-BEARING. More than half
 * the ring atoms must sit in symmetry classes of more than two, and at least
 * one atom in more than one ring must still be tied - i.e. a FUSED system with
 * unresolved symmetry. Running the pass unconditionally would change molecules
 * RDKit leaves alone.
 */
function symmetryPassApplies(classes, atoms) {
  const sizes = new Map();
  for (const klass of classes) sizes.set(klass, (sizes.get(klass) ?? 0) + 1);
  if (![...sizes.values()].some((n) => n > 1)) return false;
  let ringAtoms = 0;
  let symRingAtoms = 0;
  let branching = false;
  const counted = new Set();
  classes.forEach((klass, atom) => {
    const rings = atoms[atom].numAtomRings ?? 0;
    if (rings < 1) return;
    ringAtoms += 1;
    const size = sizes.get(klass);
    if (size > 2 && !counted.has(klass)) { symRingAtoms += size; counted.add(klass); }
    if (rings > 1 && size > 1) branching = true;
  });
  return ringAtoms > 0 && symRingAtoms / ringAtoms > 0.5 && branching;
}

/**
 * Canonical ranks, 0-based, one per atom.
 *
 * @param {{atoms: Array<{z: number, charge?: number, isotope?: number,
 *          totalNumHs?: number, degree?: number, explicitValence?: number,
 *          inRing?: boolean}>,
 *          neighbours: number[][]}} graph the molecule WITH hydrogens as
 *   explicit atoms, because that is the graph upstream ranks
 * @param {{breakTies?: boolean, includeIsotopes?: boolean}} [options]
 * @returns {number[]} a permutation of 0..n-1 when ties are broken
 */
export function canonicalRanks(graph, options = {}) {
  const atoms = graph.atoms;
  const bonds = graph.bonds ?? graph.neighbours.map((list) => list.map((to) => ({
    to, type: BOND_TYPE_ORDER.SINGLE, stereo: 0,
  })));
  if (atoms.length !== bonds.length) {
    throw new Error(`${atoms.length} atoms but ${bonds.length} bond lists`);
  }
  const n = atoms.length;
  const degrees = bonds.map((list) => list.length);
  const nbrIds = atoms.map((atom, index) =>
    atom.nbrIds ?? bonds[index].map((bond) => bond.to));

  const base = atoms.map((atom, index) => BASECOMP.map((read) => read({
    degree: degrees[index],
    z: atom.z,
    isotope: options.includeIsotopes === false ? 0 : (atom.isotope ?? 0),
    totalNumHs: atom.totalNumHs ?? 0,
    charge: atom.charge ?? 0,
    chiralTag: atom.chiralTag ?? (atom.chiralCode ? 1 : 0),
    atomMapNumber: atom.atomMapNumber ?? 0,
  })));
  const compareBase = (i, j) => compareKeys(base[i], base[j]);

  // 🔴 ONE PARTITION, AND **NO** BASE-ONLY PRE-SPLIT. This is what RDKit's own
  // trace shows and it was the last fault: `rankMolAtoms` calls
  // `CreateSinglePartition`, sets `df_useNbrs = true`, and refines ONCE with
  // the FULL comparator. Its VERBOSE_CANON passes "1" and "1a" both print a
  // single class of 16 for indole - nothing is split before the neighbours are
  // switched on.
  //
  // 🔴 AND PRE-SPLITTING BY basecomp IS NOT AN OPTIMISATION, IT CHANGES THE
  // ANSWER. From one partition every neighbour's class is 0, so the first
  // sort's neighbour comparison reduces to bond TYPES and list length. For
  // indole's atoms 5 and 7 - both degree-3 aromatic carbons tying through all
  // of basecomp - that makes positions 0 and 1 tie and position 2 decide,
  // where `single(1) < aromatic(12)`, giving atom 5 the lower class. Having
  // pre-split, this port instead compared `(aromatic, class 15)` against
  // `(aromatic, class 7)` at position 0 and got the opposite order. Two
  // sessions of searching could not find that because it is not a key
  // ordering: it is the state the comparator is first applied to.
  const partitions = createPartitions(n);

  // 🔴 EVERY COMPARATOR READS THE **LIVE** CLASSES OUT OF THE PARTITION STATE,
  // which is what `updateAtomNeighborIndex` does on each call: it refreshes
  // each bond's `nbrSymClass` from the neighbour's current class and re-sorts.
  // Precomputing the sorted bond lists once per sweep - as this port did while
  // it re-ranked globally - freezes them at the sweep's start and compares
  // against stale classes partway through.
  const sortedBondsFor = (atom) => {
    const list = bonds[atom].map((bond) => ({
      to: bond.to, type: bond.type, stereo: bond.stereo,
      klass: partitions.klass[bond.to],
    }));
    list.sort((a, b) => -compareBond(a, b));
    return list;
  };
  const walkBonds = (i, j) => {
    const bi = sortedBondsFor(i);
    const bj = sortedBondsFor(j);
    const shared = Math.min(bi.length, bj.length);
    for (let index = 0; index < shared; index += 1) {
      const order = compareBond(bi[index], bj[index]);
      if (order !== 0) return order;
    }
    if (bi.length !== bj.length) return bi.length < bj.length ? -1 : 1;
    return 0;
  };
  /**
   * `getChiralRank`: a stereocentre's hand READ IN THE CURRENT CLASS ORDER.
   *
   * 🔴 THIS IS A FIELD OF `basecomp`, NOT A LATER PASS, AND THAT IS THE WHOLE
   * DIFFERENCE. This port had chirality only in `compareChirality`, which runs
   * after the main refinement - and `refinePartitions` can only SUBDIVIDE an
   * existing partition, never reorder one, so a term that arrives late can
   * split a class but cannot decide which side of its neighbours it lands on.
   * Traced on glucose: RDKit's own `basecomp(2, 4)` is **0** on the first pop
   * (every class is still 0) and **1** on the second, which is only possible
   * for a field that reads the live classes. It sorts carbon 2 above all four
   * of 4/6/8/10 and so to class 23; without it this port sorted it second and
   * got 20, and the six oxygens then inherited the swap through their own
   * neighbour classes.
   *
   * 🔴 AND IT ABSTAINS THE MOMENT TWO NEIGHBOURS SHARE A CLASS, which is what
   * makes it safe to apply before the classes have settled: RDKit `break`s out
   * of the neighbour walk on the first duplicate, leaving `perm` short of the
   * degree, and returns 0. On glucose's second pop that is exactly why only
   * carbon 2 gets a verdict - it alone has four DISTINCT neighbour classes
   * (18, 12, 19, 0); the other four each see two 12s or two 19s and tie.
   *
   * 🔴 AND THE DUMP'S `chiralCode` IS THE OPPOSITE WAY ROUND FROM THIS
   * FUNCTION'S RESULT. `dump_rdkit_canon_ranks.py` writes 1 for CW and 2 for
   * CCW; `getChiralRank` returns 2 for CW and 1 for CCW, then flips on an odd
   * swap parity. Mapping one onto the other by eye inverts every stereocentre.
   */
  const chiralRank = (atom) => {
    const code = atoms[atom].chiralCode ?? 0;
    if (code === 0) return 0;
    const perm = [];
    for (const nbr of nbrIds[atom]) {
      const rank = partitions.klass[nbr];
      if (perm.includes(rank)) return 0;
      perm.push(rank);
    }
    if (perm.length !== degrees[atom]) return 0;
    const sorted = perm.slice().sort((a, b) => a - b);
    const odd = countSwapsToInterconvert(sorted, perm) % 2 === 1;
    const base = code === 1 ? 2 : 1;
    return odd ? (base === 2 ? 1 : 2) : base;
  };
  const compareChiralRank = (i, j) => {
    const a = chiralRank(i);
    const b = chiralRank(j);
    return a === b ? 0 : (a < b ? -1 : 1);
  };
  const compareMain = (i, j) =>
    compareBase(i, j) || compareChiralRank(i, j) || walkBonds(i, j);

  // 🔴 THE HOOK EXISTS TO BE DIFFED AGAINST RDKit'S OWN TRACE. `options.onPass`
  // is handed the same three arrays RDKit's VERBOSE_CANON blocks print -
  // `order`, each atom's class, and `count` - after each pass, named with
  // RDKit's own labels. Comparing them pass by pass is what found the
  // single-partition fault; guessing from final ranks is what failed to.
  const report = (label) => options.onPass?.(label, {
    order: Array.from(partitions.order),
    klass: Array.from(partitions.klass),
    count: Array.from(partitions.count),
  });
  report("1");
  activatePartitions(partitions);
  report("1a");
  refinePartitions(partitions, compareMain, true, nbrIds, options.onPop);
  report("2");

  const tiesRemain = () => {
    for (let i = 0; i < n; i += 1) {
      if (partitions.count[partitions.order[i]] > 1) return true;
    }
    return false;
  };

  if (options.useChirality !== false
      && atoms.some((atom) => (atom.chiralCode ?? 0) !== 0) && tiesRemain()) {
    const compareChirality = (i, j) => {
      const walked = walkBonds(i, j);
      if (walked !== 0) return walked;
      const si = neighbourSwaps(i, atoms, bonds, sortedBondsFor, partitions.klass);
      const sj = neighbourSwaps(j, atoms, bonds, sortedBondsFor, partitions.klass);
      const both = Math.min(si.length, sj.length);
      for (let index = 0; index < both; index += 1) {
        if (si[index][1] !== sj[index][1]) return si[index][1] - sj[index][1];
      }
      return 0;
    };
    activatePartitions(partitions);
    refinePartitions(partitions, compareChirality, true, nbrIds, options.onPop);
    report("2a");
  }

  if (options.useSpecialSymmetry !== false
      && symmetryGate(partitions, atoms) ) {
    const profiles = ringNeighbourProfiles(atoms, bonds);
    const compareSym = (i, j) => {
      const num = compareSequence(profiles[i].neighborNum, profiles[j].neighborNum);
      if (num !== 0) return num;
      const rev = compareSequence(profiles[i].revisited, profiles[j].revisited);
      if (rev !== 0) return rev;
      return walkBonds(i, j);
    };
    activatePartitions(partitions);
    refinePartitions(partitions, compareSym, true, nbrIds, options.onPop);
    report("2b");
  }

  if (options.breakTies === false) return Array.from(partitions.klass);
  activatePartitions(partitions);
  breakTies(partitions, compareMain, true, nbrIds);
  report("3");
  return Array.from(partitions.klass);
}

/** `rankMolAtoms`'s gate on the symmetry pass, over live partition state. */
function symmetryGate(partitions, atoms) {
  const { order, count, klass, n } = partitions;
  let ties = false;
  for (let i = 0; i < n; i += 1) if (count[order[i]] > 1) ties = true;
  if (!ties) return false;
  let ringAtoms = 0;
  let symRingAtoms = 0;
  let branching = false;
  for (let i = 0; i < n; i += 1) {
    const atom = order[i];
    const rings = atoms[atom].numAtomRings ?? 0;
    if (rings < 1) continue;
    ringAtoms += 1;
    if (count[atom] > 2) symRingAtoms += count[atom];
    if (rings > 1 && count[atom] > 1) branching = true;
  }
  return ringAtoms > 0 && symRingAtoms / ringAtoms > 0.5 && branching;
}
