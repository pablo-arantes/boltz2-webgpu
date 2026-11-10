/**
 * A ligand's average molecular weight, and its heavy-atom count.
 *
 * WHY THIS EXISTS. Boltz-2's affinity module takes `affinity_mw` as a feature
 * of the correction applied to its predicted value:
 *
 *     value = 1.03525938 * value - 0.59992683 * mw**0.3 + 2.83288489
 *
 * so the weight reaches a number a reader RANKS ON. Upstream computes it as
 * RDKit's `Descriptors.MolWt(RemoveHs(mol))` (boltz/data/parse/schema.py), and
 * nothing in this port had a molecular weight at all.
 *
 * 🔴 `RemoveHs` DOES NOT REMOVE THE HYDROGEN MASS, AND READING THE NAME THE
 * OTHER WAY WOULD BE A SYSTEMATIC ERROR ON EVERY LIGAND. Measured against
 * RDKit 2025.03.5 over thirteen cases, `MolWt(mol)` and `MolWt(RemoveHs(mol))`
 * agree to every digit: `RemoveHs` makes a hydrogen implicit, it does not
 * delete it. So this counts every hydrogen, explicit or implicit.
 *
 * 🔴 AND `MolWt` HONOURS AN ISOTOPE LABEL, WHICH IS NOT WHAT "AVERAGE
 * MOLECULAR WEIGHT" SUGGESTS. Measured: `[13CH3][13CH3]` is **32.0547**, which
 * is 2 x 13.00335 + 6 x 1.008, and not C2H6's standard 30.070. A table of
 * standard weights alone is wrong by a whole mass unit per labelled atom. The
 * isotope masses below are therefore part of the specification, not a nicety.
 *
 * 🔴 AND THE ONE CASE THIS CANNOT DO IS REFUSED RATHER THAN APPROXIMATED.
 * `parseSmiles` folds a hydrogen written as an atom into its neighbour's
 * implicit count and keeps no isotope for it: `[2H]C([2H])([2H])O` parses to a
 * carbon and an oxygen with `hydrogens: 3`, so the three deuteriums are
 * indistinguishable from protons by the time the graph exists. RDKit says
 * 35.0603 and any graph-based sum must say 32.042. The information is gone, so
 * `molecularWeight` cannot detect it - `hydrogenIsotopeInSmiles` reads the
 * STRING for it, and the affinity intake refuses such a ligand by name. A
 * wrong weight here is a wrong IC50, silently.
 *
 * 🔴 AND THE ATOMIC WEIGHTS ARE RDKit'S, NOT IUPAC'S CURRENT ONES. This was
 * written from IUPAC 2021 first and the gate caught it: **51 of 86 elements
 * disagree**, and not by rounding - RDKit ships an older table (Ge 72.610
 * against 72.630, Se 78.96 against 78.971, Mo 95.94 against 95.95, Pt 195.078
 * against 195.084, S 32.067 against 32.06, Zn 65.39 against 65.38). Measured
 * end to end that is `[Na+].[Cl-]` at 58.4398 against RDKit's 58.4430 and
 * phenylboronic acid at 121.9300 against 121.9320.
 *
 * The right table is RDKit's, and the reason is not deference: `affinity_mw`
 * is DEFINED by upstream's `Descriptors.MolWt` call, and the three correction
 * coefficients (1.03525938, -0.59992683, 2.83288489) were FITTED against
 * weights computed from this table. Substituting better physical constants
 * into a fitted correction puts a systematic bias in its output. So the
 * numbers below are the ones upstream used, and being "more correct" than
 * them would be the bug.
 *
 * Indexed by atomic number so the table cannot drift out of step with
 * `ELEMENT_SYMBOLS` in ../af3/featurise/ccd-component.js, which is the list
 * this port supports. Five elements have no stable isotope and so no standard
 * weight at all (Tc, Pm, Po, At, Rn); the mass number of the longest-lived
 * isotope stands in, as RDKit's own table does.
 *
 * The ISOTOPE masses needed no such correction - all 25 agree with RDKit's to
 * better than 1.4e-6.
 */

import { ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";

/**
 * RDKit's standard atomic weights, indexed by atomic number (so `[0]` is
 * unused and `[6]` is carbon). See the header for why these and not IUPAC's.
 * Must stay the same length as ELEMENT_SYMBOLS + 1.
 */
const STANDARD_WEIGHTS = [
  0,
  1.008, 4.003, 6.941, 9.012, 10.812, 12.011, 14.007, 15.999,
  18.998, 20.18, 22.99, 24.305, 26.982, 28.086, 30.974, 32.067,
  35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.944, 51.996,
  54.938, 55.845, 58.933, 58.693, 63.546, 65.39, 69.723, 72.61,
  74.922, 78.96, 79.904, 83.8, 85.468, 87.62, 88.906, 91.224,
  92.906, 95.94, 98, 101.07, 102.906, 106.42, 107.868, 112.412,
  114.818, 118.711, 121.76, 127.6, 126.904, 131.29, 132.905, 137.328,
  138.906, 140.116, 140.908, 144.24, 145, 150.36, 151.964, 157.25,
  158.925, 162.5, 164.93, 167.26, 168.934, 173.04, 174.967, 178.49,
  180.948, 183.84, 186.207, 190.23, 192.217, 195.078, 196.967, 200.59,
  204.383, 207.2, 208.98, 209, 210, 222,
];

if (STANDARD_WEIGHTS.length !== ELEMENT_SYMBOLS.length + 1) {
  // A loud failure at load rather than a silent zero for the last element.
  throw new Error(`molecular-weight.js has ${STANDARD_WEIGHTS.length - 1}`
    + ` weights for ${ELEMENT_SYMBOLS.length} elements`);
}

/** Atomic number from an upper-case symbol, as the parser writes it. */
const ATOMIC_NUMBER = new Map(
  ELEMENT_SYMBOLS.map((symbol, index) => [symbol, index + 1]));

/**
 * Masses of the isotopes a SMILES is likely to label, keyed `SYMBOL-mass`.
 *
 * Deliberately not a general table: an isotope that is not here THROWS rather
 * than falling back to the standard weight, because the fallback is wrong by
 * about a mass unit and nothing downstream would notice.
 */
const ISOTOPE_MASSES = new Map(Object.entries({
  "H-1": 1.00782503207, "H-2": 2.0141017781, "H-3": 3.0160492779,
  "B-10": 10.01293695, "B-11": 11.00930536,
  "C-12": 12, "C-13": 13.00335483507, "C-14": 14.0032419884,
  "N-14": 14.0030740044, "N-15": 15.0001088989,
  "O-16": 15.9949146196, "O-17": 16.9991317565, "O-18": 17.9991596128,
  "F-19": 18.998403163,
  "P-31": 30.973761998,
  "S-32": 31.9720711744, "S-33": 32.971458909, "S-34": 33.967867004,
  "CL-35": 34.968852682, "CL-37": 36.965902602,
  "BR-79": 78.9183376, "BR-81": 80.9162906,
  "I-127": 126.904471839,
  "SE-80": 79.9165218,
  "FE-56": 55.9349375,
}));

/** The standard atomic weight of one element, by the parser's symbol. */
export function atomicWeight(symbol) {
  const z = ATOMIC_NUMBER.get(String(symbol).toUpperCase());
  if (z === undefined) throw new RangeError(`unknown element ${symbol}`);
  return STANDARD_WEIGHTS[z];
}

const HYDROGEN = STANDARD_WEIGHTS[1];

/**
 * Does this SMILES label a HYDROGEN with an isotope?
 *
 * 🔴 READ FROM THE STRING, BECAUSE THE GRAPH NO LONGER KNOWS. See the header:
 * `parseSmiles` folds a bracket hydrogen into its neighbour's implicit count
 * and keeps no isotope for it, so `[2H]`, `[3H]` and `[1H]` are invisible
 * downstream and would make the weight quietly wrong.
 *
 * Matches a bracket atom whose symbol is H and which carries a leading mass
 * number. `[NH4+]`, `[nH]` and `[CH3:1]` are not matches: the H there is a
 * hydrogen COUNT on another element, not the element.
 *
 * @param {string} smiles
 * @returns {boolean}
 */
export function hydrogenIsotopeInSmiles(smiles) {
  return /\[\s*[0-9]+\s*H\s*[+\-\]@:]/.test(String(smiles ?? ""))
    || /\[\s*[0-9]+\s*H\s*\]/.test(String(smiles ?? ""));
}

/**
 * The average molecular weight of a parsed molecule, RDKit `MolWt`'s answer.
 *
 * @param {{atoms: {symbol: string, hydrogens?: number, isotope?: number}[]}} graph
 *   from `parseSmiles`
 * @returns {number} daltons
 */
export function molecularWeight(graph) {
  const atoms = graph?.atoms;
  if (!Array.isArray(atoms)) {
    // 🔴 An unguarded sum over a non-array returns 0, which is a plausible
    // weight for nothing at all - the shape of the relative-rms trap this
    // repository records.
    throw new TypeError("molecularWeight needs a parsed graph with atoms");
  }
  let total = 0;
  for (const atom of atoms) {
    const symbol = String(atom.symbol).toUpperCase();
    const isotope = Number(atom.isotope ?? 0);
    if (isotope > 0) {
      const mass = ISOTOPE_MASSES.get(`${symbol}-${isotope}`);
      if (mass === undefined) {
        throw new RangeError(`no mass for isotope ${isotope}${symbol};`
          + " add it to ISOTOPE_MASSES rather than falling back to the"
          + " standard weight, which is wrong by about a mass unit");
      }
      total += mass;
    } else {
      total += atomicWeight(symbol);
    }
    // Every hydrogen, implicit or written: see the header on RemoveHs.
    total += HYDROGEN * Number(atom.hydrogens ?? 0);
  }
  return total;
}

/**
 * Heavy atoms - what upstream's affinity size limits are counted in.
 *
 * `parseSmiles` keeps only heavy atoms (hydrogens live as counts), so this is
 * the atom count; it is a named function because the limits read better
 * against it and because an explicit hydrogen would change the answer if the
 * parser ever kept one.
 *
 * @param {{atoms: {symbol: string}[]}} graph
 * @returns {number}
 */
export function heavyAtomCount(graph) {
  const atoms = graph?.atoms;
  if (!Array.isArray(atoms)) {
    throw new TypeError("heavyAtomCount needs a parsed graph with atoms");
  }
  let n = 0;
  for (const atom of atoms) {
    if (String(atom.symbol).toUpperCase() !== "H") n += 1;
  }
  return n;
}

/**
 * How many disconnected fragments the molecule has.
 *
 * `parseSmiles` returns `components` as a per-atom label, so the fragment
 * count is the number of distinct labels. A `.` in a SMILES is two molecules,
 * and a ligand being scored for affinity has to be one.
 *
 * @param {{atoms: unknown[], components?: number[]}} graph
 * @returns {number}
 */
export function fragmentCount(graph) {
  const components = graph?.components;
  if (!Array.isArray(components)) {
    throw new TypeError("fragmentCount needs a parsed graph with components");
  }
  return new Set(components).size;
}
