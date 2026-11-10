#!/usr/bin/env node
/**
 * G4 - does this port's molecular weight agree with RDKit's, the one upstream uses?
 *
 *   python3 tools/oracle/dump_rdkit_molwt.py
 *   node tools/check-molecular-weight.mjs
 *   node tools/check-molecular-weight.mjs --falsify
 *
 * WHY IT MATTERS. Boltz-2's affinity correction is
 * `1.03525938 * value - 0.59992683 * mw**0.3 + 2.83288489`, so the weight
 * reaches a log10(IC50) a reader ranks on. A weight that is wrong by a few
 * per cent moves the corrected value by a few tenths, which is the size of the
 * differences a hit-to-lead campaign is looking at.
 *
 * 🔴 IT FOUND A REAL DEFECT THE FIRST TIME IT RAN. The table was written from
 * IUPAC 2021 and **51 of 86 elements disagreed with RDKit's**, which ships an
 * older one. That is not pedantry: the three correction coefficients were
 * FITTED against RDKit-computed weights, so better physical constants put a
 * systematic bias into a fitted model. See the header of
 * shared/chem/molecular-weight.js.
 *
 * 🔴 AND IT ASSERTS `heavyAtoms` TOO, because upstream's affinity size limits
 * (refuse > 128, warn > 56) are counted in heavy atoms, and a count that
 * disagrees with RDKit's would admit a ligand the model refuses or refuse one
 * it accepts.
 *
 * 🔴 THREE MOLECULES ARE EXPECTED REFUSALS, NOT FAILURES. `parseSmiles` folds
 * a hydrogen written as an atom into its neighbour's implicit count and keeps
 * no isotope for it, so a deuterated molecule's weight cannot be recovered
 * from the graph - RDKit says 35.0603 for `[2H]C([2H])([2H])O` and any
 * graph-based sum must say 32.042. The port detects this from the STRING and
 * refuses; the gate asserts that it refuses, and would fail if it silently
 * answered. A deliberate gap held as a failure is a red gate for ever - the
 * treatment this repository already gives `ref_pos`.
 *
 * `--falsify` perturbs one element's weight and one hydrogen count and expects
 * the gate to go red, because a gate nobody has watched fail is a gate nobody
 * has checked.
 */
import { readFileSync } from "node:fs";
import { parseSmiles } from "../shared/chem/smiles.js";
import { molecularWeight, heavyAtomCount, fragmentCount, hydrogenIsotopeInSmiles }
  from "../shared/chem/molecular-weight.js";

const DUMP = "oracle-dumps/rdkit-molwt.json";
// 🔴 SET FROM THE MEASUREMENT, NOT FROM COMFORT. The first version used 5e-4
// and passed with a worst residual of 9.86e-9 - a bound fifty thousand times
// looser than the thing it was bounding, which would have admitted a wrong
// element weight (the smallest IUPAC/RDKit disagreement is 4.4e-5 on Mn, and
// the one that actually bit was 2e-3 on boron). Both sides share the same
// three-decimal table, so a correct port agrees to double-rounding; 1e-6
// leaves two orders of headroom over the measured worst and still catches
// every constant in that table.
const BOUND = 1e-6;

const falsify = process.argv.includes("--falsify");

let dump;
try {
  dump = JSON.parse(readFileSync(DUMP, "utf8"));
} catch {
  console.error(`missing ${DUMP} - run: python3 tools/oracle/dump_rdkit_molwt.py`);
  process.exit(2);
}

const rows = [];
let failures = 0;
let refusals = 0;
let worst = 0;
let worstName = "";

for (const molecule of dump.molecules) {
  const { name, smiles } = molecule;
  if (molecule.error) continue;

  // 🔴 RemoveHs must not change the weight. If this ever trips, the port's
  // "count every hydrogen" rule is the thing that is wrong.
  if (Math.abs(molecule.molWt - molecule.molWtWithH) > 1e-9) {
    console.log(`  REMOVEHS  ${name}: RDKit's MolWt changed under RemoveHs`
      + ` (${molecule.molWt} vs ${molecule.molWtWithH})`);
    failures += 1;
    continue;
  }

  const expectRefusal = hydrogenIsotopeInSmiles(smiles);
  let graph;
  try {
    graph = parseSmiles(smiles);
  } catch (error) {
    console.log(`  PARSE     ${name}: ${error.message}`);
    failures += 1;
    continue;
  }

  if (expectRefusal) {
    // The port must refuse rather than answer. Assert that it WOULD have been
    // wrong, so this arm cannot pass by the weight happening to agree.
    const got = molecularWeight(graph);
    const wrongBy = Math.abs(got - molecule.molWt);
    if (!molecule.hydrogenIsotope) {
      console.log(`  REFUSAL?  ${name}: detected an H isotope RDKit does not see`);
      failures += 1;
    } else if (wrongBy <= BOUND) {
      console.log(`  REFUSAL?  ${name}: expected the graph sum to be WRONG,`
        + ` and it agreed to ${wrongBy.toExponential(2)} - the refusal may be`
        + " unnecessary");
      failures += 1;
    } else {
      refusals += 1;
      rows.push({ name, kind: "refused",
        note: `graph sum would be off by ${wrongBy.toFixed(4)}` });
    }
    continue;
  }

  if (molecule.hydrogenIsotope) {
    console.log(`  MISSED    ${name}: RDKit sees an H isotope the port did not`);
    failures += 1;
    continue;
  }

  let mw = molecularWeight(graph);
  let heavy = heavyAtomCount(graph);
  if (falsify && name === "benzene") mw += 0.01;
  if (falsify && name === "glycerol") heavy += 1;

  const delta = Math.abs(mw - molecule.molWt);
  if (!Number.isFinite(delta)) {
    // 🔴 `delta > BOUND` is FALSE for NaN, which is how a checker passes
    // without computing anything. Tested explicitly.
    console.log(`  NONFINITE ${name}: ${mw}`);
    failures += 1;
    continue;
  }
  if (delta > worst) { worst = delta; worstName = name; }

  const heavyOk = heavy === molecule.heavyAtoms;
  const fragOk = fragmentCount(graph) === molecule.fragments;
  const ok = delta <= BOUND && heavyOk && fragOk;
  if (!ok) {
    failures += 1;
    const why = [
      delta > BOUND ? `mw ${mw.toFixed(4)} vs ${molecule.molWt.toFixed(4)}`
        + ` (${delta.toExponential(2)})` : null,
      heavyOk ? null : `heavy ${heavy} vs ${molecule.heavyAtoms}`,
      fragOk ? null : `fragments ${fragmentCount(graph)} vs ${molecule.fragments}`,
    ].filter(Boolean).join("; ");
    console.log(`  FAIL      ${name}: ${why}`);
  }
  rows.push({ name, kind: "compared", delta, heavy, mw });
}

const compared = rows.filter((r) => r.kind === "compared").length;
console.log();
console.log(`reference        : ${dump.source}`);
console.log(`quantity         : ${dump.quantity}`);
console.log(`compared         : ${compared} molecules`);
console.log(`expected refusals: ${refusals} (hydrogen isotopes)`);
console.log(`worst residual   : ${worst.toExponential(2)} on ${worstName}`
  + ` (bound ${BOUND.toExponential(0)})`);
for (const row of rows.filter((r) => r.kind === "refused")) {
  console.log(`  refused: ${row.name} - ${row.note}`);
}
console.log();

if (falsify) {
  if (failures >= 2) {
    console.log(`FALSIFY OK: ${failures} failures with the weight and the heavy`
      + " count perturbed, as expected");
    process.exit(0);
  }
  console.log(`FALSIFY FAILED: perturbing two molecules produced ${failures}`
    + " failures; this gate cannot see what it is for");
  process.exit(1);
}

console.log(failures === 0
  ? `G4 PASS: ${compared} molecules within ${BOUND.toExponential(0)}, ${refusals} refused`
  : `G4 FAIL: ${failures} molecules`);
process.exit(failures === 0 ? 0 : 1);
