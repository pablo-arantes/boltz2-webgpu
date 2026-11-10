/**
 * A `.smi` file: one ligand a line, as a virtual screen is handed in.
 *
 * The format has no specification, only a convention - a SMILES, optional
 * whitespace, and a name that runs to the end of the line:
 *
 *     CCCc1ccccc1O            propofol
 *     CC(=O)Oc1ccccc1C(=O)O   aspirin
 *     c1ccccc1                           # no name; one is generated
 *
 * 🔴 PURELY LEXICAL, AND THAT IS THE LAYERING. Nothing here parses chemistry
 * or imports ../chem/: a file can be read, counted and shown to a reader
 * before any molecule is built, and the chemistry's refusals belong to the
 * intake that applies them (../chem/affinity-limits.js). Mixing the two would
 * mean a typo on line 900 decides whether lines 1-899 are listed.
 *
 * 🔴 AN ID IS AN OUTPUT FILENAME, SO A DUPLICATE IS AN ERROR AND NOT A
 * CURIOSITY. The run writes one structure per ligand into a zip and one row
 * per ligand into a csv; two ligands called `aspirin` means one of them
 * silently overwrites the other in the archive and the csv stops being a key.
 * Duplicates are reported with both line numbers, and `uniqueIds` is how a
 * caller gets a set it can safely write with.
 *
 * 🔴 AND A REPEATED SMILES IS NOT AN ERROR. Screening the same structure twice
 * under two names is a legitimate positive control - it measures the sampler's
 * own spread, which is the only honest baseline for "is this difference real".
 * It is counted and reported, never refused.
 *
 * What is skipped, each with a reason a reader can act on: blank lines, `#`
 * comments, and a first line that looks like a column header.
 */

/** Byte-order mark, which a file saved from a spreadsheet carries. */
const BOM = "﻿";

/**
 * Does this line look like a column header rather than a molecule?
 *
 * Only ever asked of the FIRST content line, and deliberately narrow: the
 * words a header uses are not SMILES. `smiles` parses as far as its aromatic
 * sulfur and then fails on `m`, so a chemistry-aware check would also reject
 * it - but this runs before any chemistry exists, and guessing wrong here
 * would silently drop a real ligand.
 */
function looksLikeHeader(first) {
  const token = first.split(/\s+/, 1)[0].toLowerCase();
  return token === "smiles" || token === "structure" || token === "canonical_smiles"
    || token === "smi";
}

/**
 * Read a `.smi` file.
 *
 * @param {string} text the file's contents
 * @param {object} [options]
 * @param {string} [options.stem] what to call a ligand with no name;
 *   defaults to `ligand`. The generated id is `<stem>_0001`, numbered by
 *   POSITION IN THE FILE rather than by how many were unnamed, so a name can
 *   be added to one line without renumbering the rest.
 * @returns {{entries: {smiles: string, id: string, line: number, named: boolean}[],
 *            problems: {line: number, text: string, why: string}[],
 *            skipped: {line: number, why: string}[],
 *            duplicateIds: {id: string, lines: number[]}[],
 *            repeatedSmiles: {smiles: string, lines: number[]}[]}}
 */
export function parseSmi(text, options = {}) {
  const stem = options.stem ?? "ligand";
  const source = String(text ?? "").replace(/^﻿/, "");
  if (String(text ?? "").startsWith(BOM)) { /* consumed above */ }

  const lines = source.split(/\r\n|\n|\r/);
  const entries = [];
  const problems = [];
  const skipped = [];
  let seenContent = false;

  for (let index = 0; index < lines.length; index += 1) {
    const line = index + 1;
    const raw = lines[index];
    const trimmed = raw.trim();
    if (trimmed === "") { continue; }                       // blank: silent
    if (trimmed.startsWith("#")) {
      skipped.push({ line, why: "comment" });
      continue;
    }
    if (!seenContent && looksLikeHeader(trimmed)) {
      seenContent = true;
      skipped.push({ line, why: "column header" });
      continue;
    }
    seenContent = true;

    // 🔴 A LEADING TAB IS AN EMPTY FIRST COLUMN, NOT INDENTATION. Trimming the
    // line first destroys the difference, and then a tab-separated row whose
    // SMILES cell is blank becomes a ligand named after its SECOND column -
    // `\tnosmiles` read as the molecule `nosmiles`. Checked on the RAW line,
    // and narrow on purpose: a space-indented line in a hand-written file is
    // ordinary and is still trimmed.
    if (/^\t/.test(raw)) {
      problems.push({ line, text: trimmed,
        why: "the first column is empty (tab-separated with no SMILES)" });
      continue;
    }

    // The SMILES is the first whitespace-delimited token; the name is the
    // rest of the line, trimmed. A name may contain spaces, which is why this
    // splits once rather than on every run.
    const at = trimmed.search(/\s/);
    const smiles = at < 0 ? trimmed : trimmed.slice(0, at);
    const name = at < 0 ? "" : trimmed.slice(at).trim();

    if (smiles === "") {
      problems.push({ line, text: trimmed, why: "no SMILES on this line" });
      continue;
    }
    // 🔴 A TAB-SEPARATED FILE WITH AN EMPTY FIRST COLUMN LANDS HERE, and it
    // must not become a ligand called "" - which would collide with every
    // other empty one and overwrite them in the archive.
    entries.push({
      smiles,
      id: name === "" ? `${stem}_${String(line).padStart(4, "0")}` : name,
      line,
      named: name !== "",
    });
  }

  // --- duplicates, reported rather than resolved --------------------------
  const byId = new Map();
  for (const entry of entries) {
    if (!byId.has(entry.id)) byId.set(entry.id, []);
    byId.get(entry.id).push(entry.line);
  }
  const duplicateIds = [...byId.entries()]
    .filter(([, lines]) => lines.length > 1)
    .map(([id, lines]) => ({ id, lines }));

  const bySmiles = new Map();
  for (const entry of entries) {
    if (!bySmiles.has(entry.smiles)) bySmiles.set(entry.smiles, []);
    bySmiles.get(entry.smiles).push(entry.line);
  }
  const repeatedSmiles = [...bySmiles.entries()]
    .filter(([, lines]) => lines.length > 1)
    .map(([smiles, lines]) => ({ smiles, lines }));

  return { entries, problems, skipped, duplicateIds, repeatedSmiles };
}

/**
 * One id per entry: file-safe, and unique.
 *
 * 🔴 SANITISED BEFORE IT IS DE-DUPLICATED, AND THAT ORDER IS THE WHOLE POINT.
 * These ids become entry names in the download zip, so they have to survive
 * being filenames - and a reader's names routinely contain spaces, slashes and
 * commas. De-duplicating first and sanitising afterwards reintroduces exactly
 * the collision this function exists to prevent: `a b` and `a_b` are distinct
 * ids that become one filename, and one ligand's structure then overwrites the
 * other's in the archive with nothing reported.
 *
 * A collision takes a `-2`, `-3` suffix in FILE ORDER, and the first
 * occurrence keeps the bare name, so a clean file is untouched. The suffix is
 * `-n` rather than `#n` because `#` starts a fragment in a URL and the results
 * table links to each structure.
 *
 * @param {{id: string, line: number}[]} entries
 * @returns {string[]} one id per entry, in the same order
 */
export function uniqueIds(entries) {
  const seen = new Map();
  return entries.map((entry) => {
    const safe = fileSafeId(entry.id, entry.line);
    const n = (seen.get(safe) ?? 0) + 1;
    seen.set(safe, n);
    return n === 1 ? safe : `${safe}-${n}`;
  });
}

/**
 * A ligand name reduced to something that can be a filename on any platform,
 * without becoming empty or a reserved name.
 *
 * @param {string} id
 * @param {number} line used only if nothing usable survives
 */
export function fileSafeId(id, line = 0) {
  const cleaned = String(id ?? "")
    // Windows forbids \ / : * ? " < > | ; control characters break archives.
    .replace(/[\u0000-\u001f\u007f\\/:*?"<>|]+/g, "_")
    // 🔴 THE ENDS ARE TRIMMED BEFORE WHITESPACE IS SUBSTITUTED, AND THAT ORDER
    // IS THE BUG THIS COMMENT EXISTS FOR. Windows silently drops a trailing dot
    // or space, so `trailing.` and `trailing` are one file and must collapse to
    // one id HERE, where uniqueIds can see the collision. Substituting first
    // turns the trailing space into `_`, which the trim can no longer match -
    // `trailing. ` came out `trailing._`, a DISTINCT id that Windows would then
    // have merged with `trailing.` downstream, where nothing is watching.
    // A leading dot goes too: it hides the file and is how `..` arrives.
    .replace(/^[.\s]+|[.\s]+$/g, "")
    .replace(/\s+/g, "_")
    .slice(0, 96);
  if (cleaned === "" || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(cleaned)) {
    return `ligand_${String(line).padStart(4, "0")}`;
  }
  return cleaned;
}

/**
 * A one-line summary for the status line, saying what was and was not read.
 *
 * @param {ReturnType<typeof parseSmi>} result
 */
export function describeSmi(result) {
  const parts = [`${result.entries.length} ligand`
    + (result.entries.length === 1 ? "" : "s")];
  const unnamed = result.entries.filter((e) => !e.named).length;
  if (unnamed > 0) parts.push(`${unnamed} unnamed`);
  if (result.problems.length > 0) parts.push(`${result.problems.length} unreadable`);
  if (result.duplicateIds.length > 0) {
    parts.push(`${result.duplicateIds.length} duplicate name`
      + (result.duplicateIds.length === 1 ? "" : "s"));
  }
  if (result.repeatedSmiles.length > 0) {
    parts.push(`${result.repeatedSmiles.length} repeated structure`
      + (result.repeatedSmiles.length === 1 ? "" : "s"));
  }
  return parts.join(" · ");
}
