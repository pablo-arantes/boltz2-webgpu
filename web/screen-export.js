/**
 * What a screen leaves behind: one CSV of numbers and one structure per ligand.
 *
 * 🔴 THE TWO AFFINITY OUTPUTS ARE NEVER MERGED, AND THE CSV IS WHERE THAT RULE
 * IS EASIEST TO BREAK. `affinity_probability_binary` is a 0..1 classification
 * - higher is better, trained for hit discovery - and `affinity_pred_value` is
 * a regression on log10(IC50) in micromolar, where LOWER is better and the
 * supervision is a different dataset. A single "score" column would be a
 * number with no units that sorts two meanings together, so both are written
 * under upstream's own names, with pIC50 beside the value as the thing a
 * medicinal chemist actually reads.
 *
 * 🔴 AND THE PER-MEMBER COLUMNS STAY. The ensemble is two members and they
 * DISAGREE: measured on this port, `CCCc1ccccc1O` gives pred_value 1.751 and
 * 0.189, probability 0.583 and 0.194, for a mean of 0.970 / 0.389. A reader
 * shown the mean alone cannot tell a confident prediction from a split one, so
 * `…1`/`…2` are columns and the aggregate is derived in front of them.
 *
 * 🔴 AND A REFUSED OR UNFINISHED LIGAND GETS A ROW TOO. A screen's CSV is also
 * its record of what was attempted: a ligand that was refused for being 129
 * heavy atoms, one that failed mid-fold, and one the reader stopped before are
 * three different outcomes, and a file that silently contains only the winners
 * is how a half-finished screen gets read as a complete one. `status` names
 * which, and every numeric column is EMPTY rather than zero - a zero sorts.
 */
import { writeZip } from "./zip.js";
import { fileSafeId } from "../shared/input/smi.js";
// 🔴 A CYCLE USED ONLY AT CALL TIME: screen-view.js reads `statusOf` from this
// file, so this is export -> view -> export. Every use is inside a function
// body, which is what makes it safe; do not hoist either to the top level.
import { drawScatter } from "./screen-view.js";

/**
 * The columns, in order, with how each is read off a result.
 *
 * 🔴 ONE LIST, BECAUSE A HEADER AND A ROW BUILT SEPARATELY DRIFT. The failure
 * is silent and total: every value lands one column left of its name and the
 * file still parses.
 */
const COLUMNS = [
  ["ligand_id", (r) => r.id],
  ["smiles", (r) => r.smiles],
  ["status", (r) => statusOf(r)],
  // upstream's two names, verbatim, aggregate first
  ["affinity_pred_value", (r) => r.affinity_pred_value],
  ["affinity_probability_binary", (r) => r.affinity_probability_binary],
  // ...and the per-ensemble-member values behind them
  ["affinity_pred_value1", (r) => r.affinity_pred_value1],
  ["affinity_probability_binary1", (r) => r.affinity_probability_binary1],
  ["affinity_pred_value2", (r) => r.affinity_pred_value2],
  ["affinity_probability_binary2", (r) => r.affinity_probability_binary2],
  // derived, and labelled as derived
  ["pIC50", (r) => r.pIC50],
  // the confidence a reader should weigh the affinity by
  ["iptm", (r) => r.iptm],
  ["ptm", (r) => r.ptm],
  ["plddt", (r) => r.plddt],
  ["affinity_sample_iptm", (r) => r.affinitySampleIptm],
  // the chemistry, so a row can be judged without re-parsing the SMILES
  ["molecular_weight", (r) => r.molecularWeight],
  ["heavy_atoms", (r) => r.heavyAtoms],
  ["mw_correction_applied", (r) => boolean(r.mwCorrectionApplied)],
  ["outside_training_range", (r) => boolean(r.outsideTrainingRange)],
  // what the model was actually given, and what it cost
  ["cropped_tokens", (r) => r.croppedTokens],
  ["cropped_protein_tokens", (r) => r.croppedProteinTokens],
  ["runtime_ms", (r) => r.runtimeMs],
  ["warning", (r) => r.warning],
  ["error", (r) => r.error],
];

const boolean = (value) =>
  (value === undefined || value === null ? undefined : (value ? "yes" : "no"));

/** done | refused | failed | skipped - never absent. */
export function statusOf(result) {
  if (result.skipped !== undefined && result.skipped !== null) return "skipped";
  if (result.error !== undefined && result.error !== null) {
    return result.name === "LigandRefused" ? "refused" : "failed";
  }
  if (result.affinity_pred_value === undefined) return "failed";
  return "done";
}

/**
 * 🔴 RFC 4180 QUOTING, BECAUSE A SMILES CONTAINS COMMAS. `CC(=O)O[C@@H]1...`
 * does not, but `C[C@H](N)C(=O)N[C@@H](C)C(=O)O` written with a dot-separated
 * salt does, every ring-closure digit is fine, and upstream's own warning
 * sentence contains both a comma and the word "therefore". An unquoted field
 * shifts every column after it.
 */
function csvField(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "";
    // Enough digits that the CSV round-trips an f32, without 17-digit noise.
    return String(Number(value.toPrecision(9)));
  }
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

/**
 * One CSV for a whole run.
 *
 * @param {object[]} results `runAffinityScreen`'s results, failures included
 * @returns {string}
 */
export function screenCsv(results) {
  const lines = [COLUMNS.map(([name]) => name).join(",")];
  for (const result of results) {
    lines.push(COLUMNS.map(([, read]) => csvField(read(result))).join(","));
  }
  // A trailing newline: a file without one is a file some tools drop the last
  // row of.
  return `${lines.join("\n")}\n`;
}

/**
 * The names inside the archive, deduplicated.
 *
 * 🔴 A ZIP WITH TWO MEMBERS OF ONE NAME IS A ZIP SOME READERS SILENTLY
 * OVERWRITE. `.smi` ids come from a user's file and `fileSafeId` can map two
 * distinct ids onto one name (`lig/1` and `lig_1`), which `uniqueIds` fixes
 * for the queue but cannot for a name derived later.
 */
export function structureNames(results) {
  const used = new Map();
  const names = new Map();
  results.forEach((result, index) => {
    const base = fileSafeId(result.id, index + 1);
    const seen = used.get(base) ?? 0;
    used.set(base, seen + 1);
    names.set(result.id, seen === 0 ? base : `${base}_${seen + 1}`);
  });
  return names;
}

/**
 * The whole download: the CSV, one structure per finished ligand, and a README
 * saying what the two numbers mean.
 *
 * @param {object[]} results
 * @param {{target?: string, sequence?: string, settings?: object,
 *          store?: boolean}} [options]
 * @returns {Promise<Uint8Array>}
 */
export async function screenArchive(results, options = {}) {
  const names = structureNames(results);
  const entries = [["affinity.csv", screenCsv(results)]];
  let written = 0;
  for (const result of results) {
    if (typeof result.pdb !== "string" || result.pdb.length === 0) continue;
    entries.push([`structures/${names.get(result.id)}.pdb`, result.pdb]);
    written += 1;
  }
  // 🔴 THE PLOT GOES IN, BECAUSE THE SCATTER IS HOW A SCREEN IS READ. The
  // archive used to hold the numbers and the structures and leave out the one
  // picture the page exists to draw, so "download the data" lost the thing a
  // reader had been looking at. Reported by a user, which is the tell that no
  // gate asserted the member list against what the page shows.
  //
  // 🔴 IT IS RE-DRAWN FROM THE RESULTS, NOT SCRAPED OFF THE PAGE, so it always
  // agrees with affinity.csv beside it. Serialising the live SVG would export
  // whatever the reader had hidden or highlighted at the moment they clicked,
  // and that picture would disagree with the CSV it ships with.
  const plot = screenPlotSvg(results, options);
  if (plot !== null) entries.push(["affinity.svg", plot]);
  entries.push(["README.txt", archiveReadme(results, written, options)]);
  return writeZip(entries, { store: options.store === true });
}

/**
 * 🔴 THE FILE THAT SAYS WHICH WAY IS BETTER. Two columns of affinity, opposite
 * polarities, from heads trained on different data - and the archive is where
 * they are read weeks later, away from any UI copy.
 */
export function archiveReadme(results, structures, options = {}) {
  const counts = new Map();
  for (const result of results) {
    const status = statusOf(result);
    counts.set(status, (counts.get(status) ?? 0) + 1);
  }
  const tally = [...counts].map(([name, n]) => `${n} ${name}`).join(", ");
  const settings = options.settings ?? {};
  return [
    "Boltz-2 virtual screen",
    "",
    `target:     ${options.target ?? "(unnamed)"}`,
    `ligands:    ${results.length} (${tally})`,
    `structures: ${structures} .pdb under structures/`,
    "plot:       affinity.svg - the scatter, re-drawn from the rows below so it",
    "            cannot disagree with affinity.csv",
    `settings:   ${settings.recyclingSteps ?? "?"} recycles,`
      + ` ${settings.samplingSteps ?? "?"} sampler steps,`
      + ` ${settings.diffusionSamples ?? "?"} diffusion samples`,
    // 🔴 THE SEED AND THE ALIGNMENT, because without them this archive cannot
    // be reproduced and two runs of one target - one single-sequence, one with
    // a 3,000-row MSA - read identically. The first version recorded neither.
    `seed:       ${settings.seed ?? "(not recorded)"}`,
    `msa:        ${settings.msa === "mmseqs2"
      ? `MMseqs2, ${settings.msaDepth ?? "?"} rows`
      : "single sequence (none)"}`,
    ...(options.bundles === null || options.bundles === undefined ? []
      : [`weights:    ${Object.entries(options.bundles)
        .map(([name, revision]) => `${name}@${revision}`).join(", ")}`]),
    // ...and the sequence itself, which was never written: the residue count
    // was printed twice instead.
    ...(options.sequence === undefined ? [] : ["", "target sequence:",
      ...(options.sequence.match(/.{1,60}/g) ?? [])]),
    "",
    "affinity.csv carries BOTH of Boltz-2's affinity outputs and they are not",
    "interchangeable. They come from two heads trained on different data with",
    "different supervision, so there is no single score here on purpose:",
    "",
    "  affinity_probability_binary   0..1, HIGHER is better.",
    "      A classifier: is this a binder at all? Trained on a binary",
    "      binder/decoy label. This is the column for hit discovery - for",
    "      sorting a library and deciding what to look at next.",
    "",
    "  affinity_pred_value           log10(IC50 in micromolar), LOWER is",
    "      better. A regression on measured potency. This is the column for",
    "      hit-to-lead - for comparing analogues of something already known to",
    "      bind. pIC50 = 6 - affinity_pred_value is the same number in the",
    "      units a medicinal chemist reads, so HIGHER pIC50 is better.",
    "",
    "The …1 and …2 columns are the two ensemble members. They can disagree",
    "substantially; the unsuffixed columns are their mean. A wide split is a",
    "reason to distrust the mean rather than a number to average away.",
    "",
    "outside_training_range marks a ligand of more than 56 heavy atoms, which",
    "is the largest Boltz-2's affinity module saw in training. Such a row is",
    "an extrapolation, not a prediction.",
    "",
    "Weigh every affinity by its confidence: iptm is the predicted accuracy of",
    "the protein-ligand interface in the structure the affinity was read from.",
    "A low-iptm pose is a guess, and so is its affinity.",
    "",
    "Structures are from pass 1 (the whole target plus the ligand). The",
    "affinity itself is predicted from pass 2, a re-folded pocket crop.",
  ].join("\n");
}

/**
 * The scatter as a standalone SVG document, or null where it cannot be drawn.
 *
 * 🔴 THE SAME `drawScatter` THE PAGE USES, so the archive cannot drift from
 * the screen. It sets every colour, stroke and font as an ATTRIBUTE rather
 * than through a class, which is what lets the markup stand on its own in a
 * file - the page's stylesheet is not in the zip.
 *
 * 🔴 AND IT NEEDS A DOCUMENT. `screenArchive` is called from the page and from
 * the probes, both in a browser, but a caller without a DOM gets null and the
 * rest of the archive rather than an exception.
 */
export function screenPlotSvg(results, options = {}) {
  if (typeof document === "undefined") return null;
  try {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    drawScatter(svg, results, { colourBy: options.colourBy });
    // a width and height so it opens at a sensible size on its own; drawScatter
    // writes the viewBox
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    if (!svg.getAttribute("width")) svg.setAttribute("width", "440");
    if (!svg.getAttribute("height")) svg.setAttribute("height", "240");
    const markup = new XMLSerializer().serializeToString(svg);
    return `<?xml version="1.0" encoding="UTF-8"?>\n${markup}\n`;
  } catch {
    return null;                 // a lost picture is not a lost archive
  }
}
