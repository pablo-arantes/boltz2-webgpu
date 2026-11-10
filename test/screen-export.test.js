/**
 * The CSV and the ZIP a screen leaves behind.
 *
 * 🔴 WHAT THESE ASSERT IS THE BRIEF'S OWN RULE: the two affinity outputs are
 * kept apart, labelled with their polarity, and never merged into one score.
 * That is a rule about a FILE, read weeks later with no UI around it, so it is
 * tested as one - including the sentences in the archive's README, because
 * "label correctly" is a deliverable and not a comment.
 *
 * 🔴 AND THE HEADER/ROW DRIFT TEST IS THE ONE THAT WOULD NOT BE WRITTEN BY
 * HAND. A header and a row built from two lists fail SILENTLY and TOTALLY -
 * every value one column left of its name, the file still valid CSV, every
 * number read as the wrong quantity. One list is the fix and this is what
 * holds it.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { screenCsv, screenArchive, structureNames, statusOf, archiveReadme }
  from "../web/screen-export.js";
import { readZip } from "../web/zip.js";

const done = (over = {}) => ({
  id: "lig1", smiles: "CCO", affinity_pred_value: 2.3866,
  affinity_probability_binary: 0.1975, affinity_pred_value1: 4.5633,
  affinity_probability_binary1: 0.0827, affinity_pred_value2: 0.2099,
  affinity_probability_binary2: 0.3124, pIC50: 3.6134, iptm: 0.509,
  ptm: 0.658, plddt: 83.66, affinitySampleIptm: 0.509,
  molecularWeight: 46.069, heavyAtoms: 3, mwCorrectionApplied: false,
  outsideTrainingRange: false, croppedTokens: 36, croppedProteinTokens: 33,
  runtimeMs: 4012, warning: null, pdb: "ATOM      1  N   MET A   1\nEND",
  ...over,
});

/** A minimal RFC 4180 reader, so the quoting is checked by READING it back. */
function parseCsvRow(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') { field += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { fields.push(field); field = ""; }
    else field += c;
  }
  fields.push(field);
  return fields;
}

test("🔴 every row has exactly as many fields as the header has names", () => {
  const rows = [
    done(),
    done({ id: "refused", error: "too large", name: "LigandRefused",
      affinity_pred_value: undefined, pdb: undefined }),
    { id: "broke", smiles: "CC", error: "device lost", name: "Error" },
    { id: "later", skipped: "resumed past" },
  ];
  const lines = screenCsv(rows).trimEnd().split("\n");
  const header = parseCsvRow(lines[0]);
  assert.equal(lines.length, rows.length + 1);
  for (const line of lines.slice(1)) {
    assert.equal(parseCsvRow(line).length, header.length,
      `"${line}" does not have ${header.length} fields`);
  }
});

test("both affinity outputs are columns, and neither is called a score", () => {
  const header = parseCsvRow(screenCsv([done()]).split("\n")[0]);
  assert.ok(header.includes("affinity_pred_value"));
  assert.ok(header.includes("affinity_probability_binary"));
  // 🔴 the two heads are trained differently; one merged column would be a
  // number with no meaning. The brief says never, so it is asserted.
  assert.ok(!header.some((name) => /^score$|^affinity$|^combined/.test(name)),
    `a merged score column appeared: ${header.join(",")}`);
});

test("per-ensemble-member values are kept beside the aggregate", () => {
  const [header, row] = screenCsv([done()]).split("\n").map(parseCsvRow);
  const at = (name) => row[header.indexOf(name)];
  assert.equal(at("affinity_pred_value1"), "4.5633");
  assert.equal(at("affinity_pred_value2"), "0.2099");
  assert.equal(at("affinity_probability_binary1"), "0.0827");
  assert.equal(at("affinity_probability_binary2"), "0.3124");
  // and the aggregate is still the aggregate, not a member
  assert.equal(at("affinity_pred_value"), "2.3866");
});

test("🔴 a SMILES containing a comma is quoted and reads back whole", () => {
  // A salt arrives like this, and the brief's scope refuses it - but it still
  // gets a ROW, and an unquoted one would shift every column after it.
  const salt = "CC(=O)[O-].[Na+]";
  const withComma = "C[C@H](N)C(=O)O,unexpected";
  for (const smiles of [salt, withComma]) {
    const [header, row] = screenCsv([done({ smiles })]).split("\n")
      .map(parseCsvRow);
    assert.equal(row[header.indexOf("smiles")], smiles);
  }
});

test("🔴 upstream's warning sentence survives the CSV: it has a comma", () => {
  const warning = "the ligand used for affinity calculation is larger than 56"
    + " heavy-atoms, which was the maximum during training, therefore the"
    + " affinity output might be inaccurate.";
  const [header, row] = screenCsv([done({ warning, outsideTrainingRange: true })])
    .split("\n").map(parseCsvRow);
  assert.equal(row[header.indexOf("warning")], warning);
  assert.equal(row[header.indexOf("outside_training_range")], "yes");
});

test("🔴 a refused ligand's numbers are EMPTY, not zero", () => {
  const [header, row] = screenCsv([{ id: "huge", smiles: "C".repeat(129),
    error: "too large", name: "LigandRefused" }]).split("\n").map(parseCsvRow);
  assert.equal(row[header.indexOf("status")], "refused");
  // 🔴 a zero SORTS. A probability of 0 reads as "certainly not a binder",
  // which is a claim; an empty cell is the absence of one.
  for (const name of ["affinity_pred_value", "affinity_probability_binary",
    "pIC50", "iptm"]) {
    assert.equal(row[header.indexOf(name)], "",
      `${name} must be empty for a refused ligand`);
  }
  assert.equal(row[header.indexOf("error")], "too large");
});

test("status tells refused, failed, skipped and done apart", () => {
  assert.equal(statusOf(done()), "done");
  assert.equal(statusOf({ id: "a", error: "x", name: "LigandRefused" }),
    "refused");
  assert.equal(statusOf({ id: "a", error: "x", name: "Error" }), "failed");
  assert.equal(statusOf({ id: "a", skipped: "resumed past" }), "skipped");
  // no affinity and no error is still not a success
  assert.equal(statusOf({ id: "a", smiles: "C" }), "failed");
});

test("a non-finite number is empty rather than NaN or Infinity", () => {
  const [header, row] = screenCsv([done({ iptm: NaN, ptm: Infinity })])
    .split("\n").map(parseCsvRow);
  assert.equal(row[header.indexOf("iptm")], "");
  assert.equal(row[header.indexOf("ptm")], "");
});

test("🔴 two ids that map onto one safe filename get distinct members", () => {
  // fileSafeId maps both of these to `lig_1`, and a ZIP with two members of
  // one name is one some readers silently overwrite.
  const names = structureNames([{ id: "lig/1" }, { id: "lig_1" }]);
  assert.equal(names.get("lig/1"), "lig_1");
  assert.equal(names.get("lig_1"), "lig_1_2");
  assert.equal(new Set(names.values()).size, 2);
});

test("the archive holds the CSV, a README and one structure per finished ligand",
  async () => {
    const results = [
      done(),
      done({ id: "lig2", smiles: "CCCc1ccccc1O" }),
      { id: "lig3", smiles: "C".repeat(129), error: "too large",
        name: "LigandRefused" },
    ];
    const bytes = await screenArchive(results,
      { target: "test", sequence: "MKT",
        settings: { recyclingSteps: 5, samplingSteps: 200, diffusionSamples: 3 } });
    const members = await readZip(bytes);
    const names = [...members.keys()].sort();
    assert.deepEqual(names, ["README.txt", "affinity.csv",
      "structures/lig1.pdb", "structures/lig2.pdb"],
      `got ${names.join(", ")}`);
    // readZip decodes its members as text already
    const csv = members.get("affinity.csv");
    // every ligand is in the CSV, the refused one included
    assert.equal(csv.trimEnd().split("\n").length, 4);
    assert.match(csv, /lig3,/);
  });

test("🔴 the README says which way is better, for BOTH outputs", () => {
  const text = archiveReadme([done()], 1, {});
  assert.match(text, /affinity_probability_binary[\s\S]{0,80}HIGHER is better/);
  assert.match(text, /affinity_pred_value[\s\S]{0,80}LOWER is\s*\n?\s*better/);
  assert.match(text, /pIC50 = 6 - affinity_pred_value/);
  // ...and that they are not one number
  assert.match(text, /trained on different data/);
  assert.match(text, /no single score/);
});

test("the README tallies what happened, so a partial run cannot read as whole",
  () => {
    const text = archiveReadme([done(), { id: "b", skipped: "x" },
      { id: "c", error: "y", name: "LigandRefused" }], 1, {});
    assert.match(text, /ligands:\s+3 \(/);
    assert.match(text, /1 done/);
    assert.match(text, /1 skipped/);
    assert.match(text, /1 refused/);
  });
