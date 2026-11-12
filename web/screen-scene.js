/**
 * One target, every ligand on top of it: the scene a screen builds up.
 *
 * 🔴 THE FIT IS ON THE TARGET'S ALPHA CARBONS AND NOTHING ELSE. Each ligand is
 * a SEPARATE fold - its own diffusion trajectory, its own random rotation - so
 * two runs of the same protein come back in two frames tens of angstroms and
 * any orientation apart. Drawn as they arrive, the "same" protein appears once
 * per ligand in a different place and no two ligands can be compared. Fitting
 * on the target's CA atoms puts every run in run #1's frame, and because the
 * ligand is moved by the SAME rigid transform, where it sits relative to the
 * pocket is preserved exactly. The brief asks for this by name.
 *
 * 🔴 AND IT IS RUN #1'S FRAME, NOT THE PREVIOUS RUN'S. Chaining each run onto
 * the one before accumulates every fit's residual error, and a screen is
 * hundreds of runs long; it also makes the picture depend on the ORDER ligands
 * happened to finish in, which for a stop/resume run is not even deterministic.
 * One reference, kept.
 *
 * 🔴 AND THE TARGET IS DRAWN ONCE. Every run carries a full copy of the
 * protein, so adding whole structures puts N copies of the same 33 residues in
 * the scene - N times the geometry to draw, and a visibly thickened ribbon
 * where the copies disagree by tenths of an angstrom. `splitTarget` keeps the
 * first run's protein and takes only the LIGAND from every run after it.
 *
 * 🔴 WHAT THIS FILE DOES NOT DO IS TOUCH py2Dmol. It returns PDB text and
 * colours; the page hands those to the viewer it already has. That is what
 * makes the superposition and the colour ramp testable without a browser -
 * and the arithmetic is web/pdb-superpose.js's, injected the way
 * shared/heads/superpose-pdb.js's header asks for.
 */
import { superposePdb } from "../shared/heads/superpose-pdb.js";
import { superposePoints } from "./pdb-superpose.js";

const isAtom = (line) => line.startsWith("ATOM") || line.startsWith("HETATM");
const chainOf = (line) => line[21];

/**
 * Which chain the ligand is, read off the file rather than assumed.
 *
 * 🔴 NOT "CHAIN B" AND NOT "THE LAST CHAIN". web/affinity-passes.js derives the
 * binder from the batch's own ligand flags for exactly this reason; here the
 * only evidence is the PDB, where a ligand is a HETATM record. A protein chain
 * can contain HETATM records too (a modified residue), so the test is that the
 * chain is ENTIRELY het - which is what a SMILES ligand folded as its own chain
 * always is, and what an atomised residue inside a polymer never is.
 */
export function ligandChains(pdb) {
  const het = new Map();
  for (const line of pdb.split("\n")) {
    if (!isAtom(line)) continue;
    const chain = chainOf(line);
    const seen = het.get(chain) ?? { atoms: 0, het: 0 };
    seen.atoms += 1;
    if (line.startsWith("HETATM")) seen.het += 1;
    het.set(chain, seen);
  }
  return [...het.entries()]
    .filter(([, seen]) => seen.het === seen.atoms && seen.atoms > 0)
    .map(([chain]) => chain);
}

/**
 * The chain letters a ligand may take, in preference order.
 *
 * 🔴 THE TARGET'S OWN LETTERS ARE EXCLUDED, AND READING THEM TOOK A FIXED
 * COLUMN. A ligand handed a letter the target already uses lands inside that
 * chain's residue numbering, where `resSeq` 1 is a real residue - so the viewer
 * treats the ligand and that residue as one thing, and `renumberLigand`'s whole
 * point is lost.
 *
 * 🔴 AND THE FIRST VERSION OF THIS READ THE WRONG COLUMN FOR EVERY PROTEIN
 * ATOM. It was `/^(?:ATOM|HETATM).{15}(.)/`, and `ATOM` is FOUR characters
 * where `HETATM` is SIX - so after fifteen more it captured column 19 on an
 * ATOM line and column 21 on a HETATM one. On a real fold it returned "O", the
 * last letter of `PRO`, and excluded nothing: a single-chain target still got
 * its ligands on B, C, D..., so nothing looked wrong. A TWO-chain target put
 * the first ligand on **B, the target's own second chain**. PDB is a
 * fixed-column format and this is why the rest of this file uses `line[21]`.
 *
 * @param {string} pdb the target's records
 * @returns {string[]} letters not used by the target, B-first then A last
 */
export function freeChainLetters(pdb) {
  const taken = new Set();
  for (const line of pdb.split("\n")) {
    if (isAtom(line)) taken.add(chainOf(line));
  }
  // B first because A is conventionally the protein; A is kept as a last
  // resort rather than dropped, so a target using B..Z still leaves something.
  return [..."BCDEFGHIJKLMNOPQRSTUVWXYZA"]
    .filter((letter) => !taken.has(letter));
}

/**
 * A structure split into the part that repeats and the part that does not.
 *
 * @param {string} pdb
 * @returns {{target: string, ligand: string, ligandChains: string[]}}
 */
export function splitTarget(pdb) {
  const ligands = new Set(ligandChains(pdb));
  const target = [];
  const ligand = [];
  for (const line of pdb.split("\n")) {
    if (!isAtom(line)) {
      // 🔴 CONECT RECORDS FOLLOW THEIR ATOMS. A ligand's bonds are written as
      // CONECT and py2Dmol draws a SMILES ligand's bonds from them; left in
      // the target half they reference serial numbers that half does not
      // contain, and dropped entirely the ligand renders as loose spheres.
      if (line.startsWith("CONECT")) ligand.push(line);
      else if (line.startsWith("TER") || line.length === 0) continue;
      else { target.push(line); ligand.push(line); }
      continue;
    }
    (ligands.has(chainOf(line)) ? ligand : target).push(line);
  }
  return {
    target: `${target.join("\n")}\nEND`,
    ligand: `${ligand.join("\n")}\nEND`,
    ligandChains: [...ligands],
  };
}

/** The alpha carbons of every chain but `exclude`, in file order. */
function alphaCarbonPoints(pdb, exclude) {
  return pdb.split("\n")
    .filter((line) => line.startsWith("ATOM")
      && line.slice(12, 16).trim() === "CA" && chainOf(line) !== exclude)
    .map((line) => [30, 38, 46]
      .map((at) => Number.parseFloat(line.slice(at, at + 8))));
}

/**
 * One run, put into the reference run's frame.
 *
 * 🔴 AND IT REPORTS WHAT IS LEFT OVER, because that number changes how the
 * picture must be read. The fit removes the frame; it cannot remove the fact
 * that each ligand was co-folded with its OWN copy of the target, and those
 * copies differ. Measured on this port's real output - two ligands, 33-residue
 * target, 1 recycle and 50 sampler steps - the two runs' alpha carbons are
 * **24.16 A rms apart before the fit and 4.14 A after** (worst 44.55 -> 8.27).
 * So at cheap settings a superposed scene carries several angstroms of the
 * target's own run-to-run variation, and a ligand's position relative to the
 * pocket is only as meaningful as that number is small. A UI that draws the
 * scene without showing it is inviting a reader to compare poses to a
 * precision the data does not have.
 *
 * @param {string} pdb this run's structure
 * @param {string|null} reference run #1's, or null when this IS run #1
 * @returns {{pdb: string, fitted: number, on: string, rms: number|null}}
 *   `rms` is over the fitted alpha carbons, after the fit
 */
export function superposeRun(pdb, reference) {
  if (reference === null || reference === undefined) {
    return { pdb, fitted: 0, on: "reference", rms: 0 };
  }
  // 🔴 `designed` IS THE CHAIN **NOT** FITTED ON - superposePdb's own
  // parameter, which fits on every OTHER chain first and falls back to this
  // one only when there is nothing else. So the ligand's letter goes here and
  // the protein is what the fit uses. Passing the protein's letter instead
  // reaches the same fit by the FALLBACK on a two-chain file, and silently
  // fits on the LIGAND the moment a second protein chain exists.
  const ligands = ligandChains(pdb);
  const fit = superposePdb(superposePoints, pdb, reference,
                           { designed: ligands[0] ?? "B" });
  if (fit.on === "none") return { ...fit, rms: null };
  const exclude = fit.on === "target" ? (ligands[0] ?? "B") : null;
  const moved = alphaCarbonPoints(fit.pdb, exclude);
  const fixed = alphaCarbonPoints(reference, exclude);
  if (moved.length !== fixed.length || moved.length === 0) {
    return { ...fit, rms: null };
  }
  const total = moved.reduce((sum, p, index) => {
    const q = fixed[index];
    return sum + (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  }, 0);
  return { ...fit, rms: Math.sqrt(total / moved.length) };
}

/**
 * One ligand's PDB records renumbered so several can share a scene.
 *
 * 🔴 WITHOUT THIS, EVERY LIGAND'S BONDS LAND ON THE LAST LIGAND'S ATOMS. Each
 * ligand comes from a SEPARATE fold, so serials restart per file: the protein
 * takes 1..P and every ligand takes P+1.. . py2Dmol resolves CONECT by serial,
 * merging partner lists and letting the last atom with a given serial win - so
 * in a two-ligand scene `13 -> [14,14]` and all the bonds attach to ligand 2
 * while ligand 1 renders as loose spheres. That is exactly the failure
 * `splitTarget`'s CONECT handling exists to prevent, reintroduced by
 * concatenation. Chain letter, residue number and residue name are identical
 * in every run too (`LIG`, B, 1), so without renumbering the viewer's
 * residue-level machinery treats N ligands as one residue.
 *
 * @param {string} pdb one ligand's records (ATOM/HETATM/CONECT)
 * @param {number} firstSerial the serial to start at
 * @param {string} chain the chain letter to give it
 * @param {number} resSeq the residue number to give it
 */
export function renumberLigand(pdb, firstSerial, chain, resSeq) {
  const map = new Map();
  let next = firstSerial;
  const lines = pdb.split("\n");
  // first pass: assign new serials in file order
  for (const line of lines) {
    if (!line.startsWith("HETATM") && !line.startsWith("ATOM")) continue;
    map.set(Number.parseInt(line.slice(6, 11), 10), next);
    next += 1;
  }
  const out = [];
  for (const line of lines) {
    if (line.startsWith("HETATM") || line.startsWith("ATOM")) {
      const serial = map.get(Number.parseInt(line.slice(6, 11), 10));
      // 🔴 AND THE CHAIN LETTER IS WHAT CARRIES THE SCORE COLOUR. py2Dmol
      // resolves a per-atom colour through an `{type:"advanced", value:{chain}}`
      // map on the loaded object, so one letter per ligand is what lets the
      // COLOUR BY ramp reach the 3D scene - see `chainColours` in web/screen.js.
      // The B-FACTOR COLUMN IS LEFT ALONE for the same reason: the viewer reads
      // it into `plddts`, and overwriting a real per-atom confidence with a
      // whole-ligand score would be a lie told in the viewer's own units.
      out.push(line.slice(0, 6)
        + String(serial).padStart(5)
        + line.slice(11, 21)
        + chain
        + String(resSeq).padStart(4)
        + line.slice(26));
    } else if (line.startsWith("CONECT")) {
      // 🔴 CONECT IS FIXED-WIDTH, FIVE COLUMNS OF FIVE: the record is rewritten
      // field by field rather than by splitting on whitespace, because a
      // five-digit serial leaves no space between fields.
      let rewritten = "CONECT";
      for (let at = 6; at + 5 <= line.length; at += 5) {
        const field = line.slice(at, at + 5).trim();
        if (field === "") break;
        const mapped = map.get(Number.parseInt(field, 10));
        if (mapped === undefined) continue;
        rewritten += String(mapped).padStart(5);
      }
      out.push(rewritten);
    } else if (line.trim() !== "" && !line.startsWith("END")) {
      out.push(line);
    }
  }
  return { pdb: out.join("\n"), nextSerial: next };
}

/**
 * The scene, built one finished ligand at a time.
 *
 * The first run contributes the target and its ligand; every run after it
 * contributes a ligand only, superposed onto the first.
 */
export class ScreenScene {
  #reference = null;
  #target = null;
  #ligands = new Map();

  /** @returns {{target: string|null, added: object}} */
  add(id, pdb) {
    const first = this.#reference === null;
    const fit = superposeRun(pdb, this.#reference);
    const split = splitTarget(fit.pdb);
    if (first) {
      this.#reference = fit.pdb;
      this.#target = split.target;
    }
    const entry = { id, ligand: split.ligand, fitted: fit.fitted, on: fit.on,
      // how far the target itself moved: see superposeRun's note
      rms: fit.rms, visible: true };
    this.#ligands.set(id, entry);
    return { target: first ? this.#target : null, added: entry };
  }

  get target() { return this.#target; }
  get ligands() { return [...this.#ligands.values()]; }
  has(id) { return this.#ligands.has(id); }
  setVisible(id, visible) {
    const entry = this.#ligands.get(id);
    if (entry !== undefined) entry.visible = visible === true;
    return entry;
  }
  visible() { return this.ligands.filter((entry) => entry.visible); }
  clear() {
    this.#reference = null;
    this.#target = null;
    this.#ligands.clear();
  }
}

/**
 * The colour ramp a ligand is drawn in, from the score it is being ranked on.
 *
 * 🔴 THE DOMAIN IS THE RUN'S OWN RANGE, NOT A FIXED SCALE, and that is a
 * decision with a cost. A fixed 0..1 scale on `affinity_probability_binary`
 * would make two screens comparable by colour, but within one screen of
 * congeneric ligands every probability can sit between 0.3 and 0.4 and the
 * whole scene comes out one colour. Ranking is what a reader does here, so the
 * ramp spans what this run actually produced - which means a colour means
 * nothing across runs, and the legend must print the numbers. Callers pass the
 * domain so the UI can offer both.
 *
 * 🔴 AND A ONE-LIGAND RUN HAS NO RANGE. lo === hi gives 0/0; it takes the
 * middle of the ramp rather than NaN, which renders as black.
 */
export function scoreColour(value, lo, hi) {
  if (!Number.isFinite(value) || !Number.isFinite(lo) || !Number.isFinite(hi)) {
    return "#9aa0a6";
  }
  const span = hi - lo;
  const t = span === 0 ? 0.5 : Math.min(1, Math.max(0, (value - lo) / span));
  // 🔴 NOT A RED-GREEN RAMP. About one man in twelve cannot tell those apart,
  // and this ramp is the whole visual ranking. Blue to orange is the standard
  // colour-blind-safe pair and reads as cool/warm in greyscale too.
  const stops = [[49, 86, 184], [120, 150, 200], [200, 170, 120], [214, 108, 20]];
  const at = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(at));
  const f = at - i;
  const channel = (k) => Math.round(stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f);
  return `#${[0, 1, 2].map((k) => channel(k).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * What the ramp should span for a given column.
 *
 * 🔴 AND THE POLARITY IS PART OF THE ANSWER. `affinity_probability_binary` is
 * higher-is-better and `affinity_pred_value` is lower-is-better, so a single
 * "warm means good" ramp needs the value column REVERSED. Returning the domain
 * rather than sorting the data is what keeps that decision in one place.
 */
export function scoreDomain(results, column) {
  const values = results
    .map((result) => result[column])
    .filter((value) => Number.isFinite(value));
  if (values.length === 0) return { lo: NaN, hi: NaN, reversed: false };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  // lower is better for the regression, so warm must be the LOW end
  const reversed = column === "affinity_pred_value";
  return reversed ? { lo: hi, hi: lo, reversed } : { lo, hi, reversed };
}
