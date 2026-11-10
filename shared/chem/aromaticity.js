/**
 * RDKit's aromaticity model, ported from Code/GraphMol/Aromaticity.cpp.
 *
 * 🔴 WHY PERCEPTION AND NOT THE INPUT'S LOWER CASE. This port used to take a
 * SMILES' case at face value, which is wrong in BOTH directions: porphine
 * came out with all twenty carbons aromatic where RDKit leaves four of them
 * out, and a Kekule-written benzene (`C1=CC=CC=C1`) came out non-aromatic
 * where RDKit perceives a ring. The bond type is a model input - it reaches
 * boltz2 through the bond-order plane and every family through
 * `_ccd_bond_orders` - and `bondholder::compare` ranks the TYPE before the
 * neighbour's class, so one ring perceived differently reorders every atom
 * around it.
 *
 * 🔴 IT RUNS ON KEKULE ORDERS. RDKit's pipeline is parse -> sanitise ->
 * kekulise -> setAromaticity, so perception reads integer bond orders.
 * `kekulize(graph)` must have run first; this reads `bond.order` and writes
 * `atom.aromatic` and `bond.aromatic`.
 *
 * 🔴 AND THE RINGS IT IS HANDED DECIDE EVERYTHING. Measured on the hardest
 * case in the corpus: `smallestRings` returns RDKit's exact five rings for
 * porphine - four 5-rings and the 16-ring - which is what makes this port
 * possible at all. spec/AROMATICITY-PORT.md has the trace.
 */

/**
 * 🔴 THE PARSER CALLS IT `element` AND THE RANKER'S ADAPTER CALLS IT `z`, and
 * reading only one of them is a silent failure rather than an error:
 * `DEFAULT_VALENCE.get(undefined)` is undefined, so every atom looked
 * univalent, every donor type came out None, and NOTHING in the corpus was
 * aromatic - 49 of 74 "matching" because 49 molecules have no aromatic ring to
 * get wrong. Accept both shapes.
 */
function elementOf(atom) {
  return atom.element ?? atom.z;
}

/** Default valence and outer-shell electrons, RDKit's PeriodicTable values. */
const DEFAULT_VALENCE = new Map([
  [1, 1], [5, 3], [6, 4], [7, 3], [8, 2], [9, 1], [14, 4], [15, 3], [16, 2],
  [17, 1], [33, 3], [34, 2], [35, 1], [52, 2], [53, 1],
]);
const OUTER_ELECS = new Map([
  [1, 1], [5, 3], [6, 4], [7, 5], [8, 6], [9, 7], [14, 4], [15, 5], [16, 6],
  [17, 7], [33, 5], [34, 6], [35, 7], [52, 6], [53, 7],
]);
/** Pauling electronegativity, for "does an exocyclic bond steal the electron". */
const ELECTRONEGATIVITY = new Map([
  [1, 2.20], [5, 2.04], [6, 2.55], [7, 3.04], [8, 3.44], [9, 3.98],
  [14, 1.90], [15, 2.19], [16, 2.58], [17, 3.16], [33, 2.18], [34, 2.55],
  [35, 2.96], [52, 2.10], [53, 2.66],
]);

// donor types; the numbers are this port's own, only the BOUNDS below matter
const NO_DONOR = 0;
const VACANT = 1;
const ONE_ELECTRON = 2;
const TWO_ELECTRONS = 3;

/** `getMinMaxAtomElecs`: the electrons a donor type may contribute. */
function electronBounds(type) {
  if (type === ONE_ELECTRON) return [1, 1];
  if (type === TWO_ELECTRONS) return [2, 2];
  return [0, 0];                       // NoElectronDonorType, VacantElectronDonorType
}

/**
 * `countAtomElec`: electrons this atom can put into a pi system, or -1.
 *
 * 🔴 `degree` COUNTS HYDROGENS AND THE LIMIT IS THREE. A four-coordinate atom
 * is never aromatic, which is what keeps an sp3 carbon out without any
 * hybridisation model.
 */
function countAtomElec(atom, context) {
  const dv = DEFAULT_VALENCE.get(elementOf(atom)) ?? 0;
  if (dv <= 1) return -1;                        // univalent: never aromatic
  const degree = context.heavyDegree + (atom.hydrogens ?? 0);
  if (degree > 3) return -1;
  const outer = OUTER_ELECS.get(elementOf(atom)) ?? 0;
  const nlp = Math.max(outer - dv - (atom.charge ?? 0), 0);
  let res = (dv - degree) + nlp;                 // no radicals in this parser
  if (res > 1) {
    // a triple bond or higher donates one electron, detected through the total
    // unsaturation because `isAtomCandForArom` has already excluded more than
    // one multiple bond
    if (context.bondOrderSum - context.heavyDegree > 1) res = 1;
  }
  return res;
}

/** `getAtomDonorTypeArom`. */
function donorType(atom, context, atoms) {
  const nelec = countAtomElec(atom, context);
  if (nelec < 0) return NO_DONOR;
  if (nelec === 0) {
    if (context.exocyclicMultiple >= 0) return VACANT;
    if (context.cyclicMultiple) return ONE_ELECTRON;
    return NO_DONOR;
  }
  if (nelec === 1) {
    if (context.exocyclicMultiple >= 0) {
      // 🔴 THE C=O CASE. The one electron goes to the exocyclic bond and is
      // not available when the partner is more electronegative, which is why
      // a cyclohexadienone ring is not aromatic.
      return moreElectronegative(atoms[context.exocyclicMultiple], atom)
        ? VACANT : ONE_ELECTRON;
    }
    if (context.anyMultiple) return ONE_ELECTRON;
    if ((atom.charge ?? 0) === 1) return VACANT;   // tropylium, cyclopropenyl
    return NO_DONOR;
  }
  let electrons = nelec;
  if (context.exocyclicMultiple >= 0
    && moreElectronegative(atoms[context.exocyclicMultiple], atom)) {
    electrons -= 1;
  }
  return electrons % 2 === 1 ? ONE_ELECTRON : TWO_ELECTRONS;
}

function moreElectronegative(other, atom) {
  return (ELECTRONEGATIVITY.get(elementOf(other)) ?? 0)
    > (ELECTRONEGATIVITY.get(elementOf(atom)) ?? 0);
}

/** `isAtomCandForArom`. */
function isCandidate(atom, type, context) {
  const z = elementOf(atom);
  if (z > 18 && z !== 34 && z !== 52) return false;
  if (type === NO_DONOR) return false;
  const dv = DEFAULT_VALENCE.get(elementOf(atom)) ?? 0;
  // 🔴 THE PERIODIC TABLE IS INDEXED BY Z MINUS THE CHARGE, which is RDKit's
  // own trick rather than a transcription slip: an N+ (Z 7, charge +1) is
  // allowed a valence of 4 because `defaultValence(6)` is 4. Reproduce it.
  const allowed = DEFAULT_VALENCE.get(elementOf(atom) - (atom.charge ?? 0)) ?? 0;
  // 🔴 BOND ORDERS PLUS HYDROGENS, COUNTED ONCE. RDKit's
  // `getValence(EXPLICIT)` does NOT include an implicit hydrogen and
  // `getTotalValence()` does; every hydrogen in this parser is a count, so the
  // total is the bond orders plus that count. Adding the count to a sum that
  // already held it put benzene's carbons at valence 5 against an allowed 4
  // and made every aromatic ring in the corpus non-aromatic - 49 of 74.
  const totalValence = context.bondOrderSum + (atom.hydrogens ?? 0);
  if (dv > 0 && totalValence > allowed) return false;
  // more than one double or triple bond disqualifies (`C1=C=NC=N1`)
  if (context.bondOrderSum - context.heavyDegree > 1 && context.multipleCount > 1) {
    return false;
  }
  return true;
}

/** `applyHuckel`, over an atom SET rather than a ring. */
function huckel(atomIndices, types) {
  let low = 0;
  let high = 0;
  for (const index of atomIndices) {
    const [lw, up] = electronBounds(types[index]);
    low += lw;
    high += up;
  }
  if (high >= 6) {
    for (let count = low; count <= high; count += 1) {
      if ((count - 2) % 4 === 0) return true;
    }
    return false;
  }
  return high === 2;
}

/** every k-subset of 0..n-1, in RDKit's `nextCombination` order */
function* combinations(n, k) {
  const comb = Array.from({ length: k }, (_, i) => i);
  for (;;) {
    yield comb;
    let pos = k - 1;
    while (pos >= 0 && comb[pos] === n - k + pos) pos -= 1;
    if (pos < 0) return;
    comb[pos] += 1;
    for (let i = pos + 1; i < k; i += 1) comb[i] = comb[i - 1] + 1;
  }
}

/**
 * 🔴 SIX RINGS AT MOST IN ONE COMBINATION, which is RDKit's `maxFused`. The
 * search is over SUBSETS of a fused system, so it is exponential without a
 * cap, and RDKit caps it here.
 */
const MAX_FUSED = 6;

/** `maxFusedAromaticRingSize`: a larger ring is not a neighbour of anything. */
const MAX_NEIGHBOUR_RING_BONDS = 24;

/**
 * Perceive aromaticity, writing `atom.aromatic` and `bond.aromatic`.
 *
 * @param {{atoms: object[], bonds: object[]}} graph kekulised
 * @param {number[][]} rings SSSR, each an ORDERED cycle
 * @returns {{atoms: number, bonds: number}} how many of each were marked
 */
export function perceiveAromaticity(graph, rings) {
  const { atoms, bonds } = graph;
  const n = atoms.length;
  const key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  const bondAt = new Map();
  bonds.forEach((bond, index) => bondAt.set(key(bond.from, bond.to), index));

  // each ring as bond indices; an ordered cycle, asserted rather than assumed
  const ringBonds = rings.map((ring) => ring.map((atom, position) => {
    const next = ring[(position + 1) % ring.length];
    const at = bondAt.get(key(atom, next));
    if (at === undefined) {
      throw new Error(`ring ${ring.join("-")} is not an ordered cycle:`
        + ` ${atom} and ${next} are not bonded`);
    }
    return at;
  }));

  const inRingBond = new Set();
  for (const list of ringBonds) for (const index of list) inRingBond.add(index);

  // per-atom context, then the donor type and candidacy
  const incident = Array.from({ length: n }, () => []);
  bonds.forEach((bond, index) => {
    incident[bond.from].push(index);
    incident[bond.to].push(index);
  });
  const types = new Array(n).fill(NO_DONOR);
  const candidate = new Array(n).fill(false);
  for (let atom = 0; atom < n; atom += 1) {
    let bondOrderSum = 0;
    let multipleCount = 0;
    let exocyclicMultiple = -1;
    let cyclicMultiple = false;
    for (const index of incident[atom]) {
      const bond = bonds[index];
      const order = Math.round(bond.order);
      bondOrderSum += order;
      if (order >= 2) {
        multipleCount += 1;
        const other = bond.from === atom ? bond.to : bond.from;
        if (inRingBond.has(index)) cyclicMultiple = true;
        else if (exocyclicMultiple < 0) exocyclicMultiple = other;
      }
    }
    const context = {
      heavyDegree: incident[atom].length,
      bondOrderSum,
      multipleCount,
      exocyclicMultiple,
      cyclicMultiple,
      anyMultiple: multipleCount > 0,
    };
    types[atom] = donorType(atoms[atom], context, atoms);
    candidate[atom] = isCandidate(atoms[atom], types[atom], context);
  }

  // candidate rings: every atom a candidate
  const candidateRings = [];
  rings.forEach((ring, index) => {
    if (ring.every((atom) => candidate[atom])) {
      candidateRings.push({ atoms: ring, bonds: ringBonds[index] });
    }
  });

  // 🔴 TWO RINGS ARE NEIGHBOURS ONLY IF THEY SHARE **EXACTLY ONE** BOND, and
  // getting this wrong is what made this port mark porphine aromatic where
  // RDKit does not. `makeRingNeighborMap(brings, neighMap, 24, 1)` passes
  // `maxOverlapSize = 1`, and its test is `inter.size() > 0 && inter.size() <=
  // maxOverlapSize` - so rings sharing TWO or more bonds are not neighbours at
  // all. Each of porphine's pyrroles shares two bonds with the 16-ring, so
  // every ring there is its own fused system and only single rings are ever
  // tried: the two pyrroles holding a 1-electron nitrogen fail on 5 electrons
  // and their carbons stay non-aromatic. With "shares at least one bond" the
  // five rings became one system, a two-ring combination passed, and atom 0
  // came out aromatic.
  //
  // 🔴 AND A RING OF MORE THAN 24 BONDS IS LEFT OUT OF THE MAP (`maxSize`), so
  // it is a singleton rather than a neighbour of anything.
  const neighbours = (i, k) => {
    if (candidateRings[i].bonds.length > MAX_NEIGHBOUR_RING_BONDS) return false;
    if (candidateRings[k].bonds.length > MAX_NEIGHBOUR_RING_BONDS) return false;
    const shared = candidateRings[i].bonds
      .filter((b) => candidateRings[k].bonds.includes(b)).length;
    return shared === 1;
  };
  const parent = candidateRings.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let i = 0; i < candidateRings.length; i += 1) {
    for (let k = i + 1; k < candidateRings.length; k += 1) {
      if (neighbours(i, k)) parent[find(i)] = find(k);
    }
  }
  const systems = new Map();
  candidateRings.forEach((_, i) => {
    const root = find(i);
    if (!systems.has(root)) systems.set(root, []);
    systems.get(root).push(i);
  });

  const aromaticAtoms = new Set();
  const aromaticBonds = new Set();
  for (const fused of systems.values()) {
    const ringBondCount = new Set();
    for (const r of fused) for (const b of candidateRings[r].bonds) ringBondCount.add(b);
    const done = new Set();
    for (let size = 1; size <= Math.min(fused.length, MAX_FUSED); size += 1) {
      // 🔴 THE SEARCH STOPS ONCE EVERY RING BOND IS ACCOUNTED FOR, which is
      // RDKit's `doneBonds.size() >= nRingBonds`. It is why porphine never
      // tries a two-ring combination.
      if (done.size >= ringBondCount.size) break;
      for (const comb of combinations(fused.length, size)) {
        const chosen = comb.map((i) => fused[i]);
        if (size > 1 && !connected(chosen, neighbours)) continue;
        const appearances = new Map();
        for (const r of chosen) {
          for (const atom of candidateRings[r].atoms) {
            appearances.set(atom, (appearances.get(atom) ?? 0) + 1);
          }
        }
        // 🔴 ATOMS IN EXACTLY ONE OR TWO OF THE RINGS, never more: RDKit issue
        // #2895, acepentalene's central atom being counted as aromatic.
        const set = [...appearances.entries()]
          .filter(([, count]) => count === 1 || count === 2)
          .map(([atom]) => atom);
        if (!huckel(set, types)) continue;
        for (const r of chosen) {
          for (const atom of candidateRings[r].atoms) aromaticAtoms.add(atom);
        }
        // 🔴 ONLY BONDS APPEARING ONCE ACROSS THE COMBINATION. An interior
        // fusing bond, shared by two of the chosen rings, stays non-aromatic.
        const bondCount = new Map();
        for (const r of chosen) {
          for (const b of candidateRings[r].bonds) {
            bondCount.set(b, (bondCount.get(b) ?? 0) + 1);
          }
        }
        for (const [b, count] of bondCount) {
          if (count === 1) { aromaticBonds.add(b); done.add(b); }
        }
      }
    }
  }

  atoms.forEach((atom, index) => { atom.aromatic = aromaticAtoms.has(index); });
  // 🔴 THE FLAG ONLY - THE KEKULE ORDER STAYS. `parseSmiles` leaves a
  // molecule kekulised with `aromatic` beside the integer order, and the
  // conformer's bounds, the valence check and the batch's bond-order plane all
  // read `order`. Putting 1.5 back here would undo the kekulisation the
  // perception itself depends on.
  bonds.forEach((bond, index) => { bond.aromatic = aromaticBonds.has(index); });
  return { atoms: aromaticAtoms.size, bonds: aromaticBonds.size };
}

/** `RingUtils::checkFused`: the chosen rings are connected THROUGH THE
 * NEIGHBOUR MAP, which is the share-exactly-one-bond relation above. */
function connected(chosen, neighbours) {
  const seen = new Set([chosen[0]]);
  const stack = [chosen[0]];
  while (stack.length) {
    const here = stack.pop();
    for (const other of chosen) {
      if (seen.has(other)) continue;
      if (neighbours(here, other)) {
        seen.add(other);
        stack.push(other);
      }
    }
  }
  return seen.size === chosen.length;
}
