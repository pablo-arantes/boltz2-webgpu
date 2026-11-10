/**
 * A `.smi` file reaches the screen as the ligands it names, and no others.
 *
 * 🔴 THE TWO CASES HERE THAT ARE NOT OBVIOUS, BOTH FOUND BY RUNNING IT.
 *
 * A LEADING TAB IS AN EMPTY FIRST COLUMN. `parseSmi` trims each line before
 * splitting, which is right for a hand-indented file and wrong for a
 * tab-separated one whose SMILES cell is blank: `\tnosmiles` became a ligand
 * called `ligand_0011` whose structure was the string `nosmiles`. It would
 * have failed later, in the parser, naming a molecule nobody wrote.
 *
 * AND AN ID IS A FILENAME, SO IT MUST BE SANITISED BEFORE IT IS
 * DE-DUPLICATED. `a b` and `a_b` are two distinct ids that become one
 * filename, so de-duplicating on the raw id and sanitising afterwards puts two
 * ligands' structures at one path in the download zip and loses one of them
 * with nothing reported. Asserted in that order, because the bug is the order.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { parseSmi, uniqueIds, fileSafeId, describeSmi } from "../shared/input/smi.js";

test("reads a SMILES and a name per line", () => {
  const { entries } = parseSmi("CCCc1ccccc1O propofol\nCCO ethanol\n");
  assert.equal(entries.length, 2);
  assert.deepEqual(entries.map((e) => e.smiles), ["CCCc1ccccc1O", "CCO"]);
  assert.deepEqual(entries.map((e) => e.id), ["propofol", "ethanol"]);
  assert.deepEqual(entries.map((e) => e.line), [1, 2]);
  assert.ok(entries.every((e) => e.named));
});

test("a name may contain spaces and runs to the end of the line", () => {
  const { entries } = parseSmi("CCO  ethyl alcohol, 95%\n");
  assert.equal(entries[0].id, "ethyl alcohol, 95%");
});

test("tabs separate as well as spaces", () => {
  const { entries } = parseSmi("CCO\tethanol\n");
  assert.deepEqual([entries[0].smiles, entries[0].id], ["CCO", "ethanol"]);
});

test("a ligand with no name is numbered by its LINE, not by its position", () => {
  // Numbering by position would renumber every later ligand when one line
  // gains a name, which silently renames files between two runs of one file.
  const { entries } = parseSmi("CCO named\n\nCCC\n");
  assert.equal(entries[1].id, "ligand_0003");
  assert.equal(entries[1].named, false);
});

test("blank lines, comments and a column header are skipped with a reason", () => {
  const result = parseSmi("smiles name\nCCO ethanol\n\n# a note\n");
  assert.equal(result.entries.length, 1);
  assert.deepEqual(result.skipped,
    [{ line: 1, why: "column header" }, { line: 4, why: "comment" }]);
});

test("a header is only recognised on the FIRST content line", () => {
  // Otherwise a ligand legitimately named `smiles` further down disappears.
  const result = parseSmi("CCO ethanol\nCCC smiles\n");
  assert.equal(result.entries.length, 2);
  assert.equal(result.entries[1].id, "smiles");
  assert.deepEqual(result.skipped, []);
});

test("a byte-order mark does not become part of the first SMILES", () => {
  const { entries } = parseSmi("﻿CCO ethanol\n");
  assert.equal(entries[0].smiles, "CCO");
});

test("CRLF and CR line endings read the same as LF", () => {
  for (const eol of ["\r\n", "\r", "\n"]) {
    const { entries } = parseSmi(`CCO a${eol}CCC b${eol}`);
    assert.equal(entries.length, 2, `for ${JSON.stringify(eol)}`);
    assert.equal(entries[1].smiles, "CCC");
  }
});

test("🔴 a LEADING TAB is an empty first column, not indentation", () => {
  const result = parseSmi("\tnosmiles\n");
  assert.equal(result.entries.length, 0,
    "a tab-separated row with a blank SMILES cell must not become a ligand");
  assert.equal(result.problems.length, 1);
  assert.match(result.problems[0].why, /first column is empty/);
});

test("...but a SPACE-indented line is ordinary and still read", () => {
  const { entries, problems } = parseSmi("   CCO ethanol\n");
  assert.equal(problems.length, 0);
  assert.equal(entries[0].smiles, "CCO");
});

test("duplicate names are reported with both lines, not resolved silently", () => {
  const result = parseSmi("CCO ethanol\nCCO ethanol\n");
  assert.deepEqual(result.duplicateIds, [{ id: "ethanol", lines: [1, 2] }]);
});

test("a repeated structure is reported and NOT an error", () => {
  // Screening one molecule twice under two names is a positive control: it
  // measures the sampler's own spread, which is the baseline for "is this
  // difference real". Refusing it would remove the only honest baseline.
  const result = parseSmi("CCO first\nCCO second\n");
  assert.equal(result.entries.length, 2);
  assert.equal(result.problems.length, 0);
  assert.deepEqual(result.repeatedSmiles, [{ smiles: "CCO", lines: [1, 2] }]);
});

test("🔴 ids are made file-safe BEFORE they are de-duplicated", () => {
  // `a b` and `a_b` are distinct ids and one filename. Sanitising after
  // de-duplicating puts both structures at the same path in the zip.
  const { entries } = parseSmi("CC a b\nCCC a_b\n");
  const ids = uniqueIds(entries);
  assert.deepEqual(ids, ["a_b", "a_b-2"]);
  assert.equal(new Set(ids).size, 2, "two ligands must not share one filename");
});

test("uniqueIds leaves a clean file untouched and is stable in file order", () => {
  const { entries } = parseSmi("CCO aspirin\nCCC ibuprofen\n");
  assert.deepEqual(uniqueIds(entries), ["aspirin", "ibuprofen"]);
});

test("a path, a reserved name and a trailing dot cannot become a filename", () => {
  assert.equal(fileSafeId("../../etc/passwd", 1), "_.._etc_passwd");
  assert.equal(fileSafeId("con", 7), "ligand_0007");
  assert.equal(fileSafeId("NUL", 8), "ligand_0008");
  assert.equal(fileSafeId("trailing. ", 9), "trailing");
  assert.equal(fileSafeId("   ", 9), "ligand_0009");
  assert.equal(fileSafeId("", 3), "ligand_0003");
  assert.ok(!fileSafeId("a/b\\c:d*e?f\"g<h>i|j").includes("/"));
});

test("an id is capped so it cannot overflow an archive entry name", () => {
  assert.ok(fileSafeId("x".repeat(400)).length <= 96);
});

test("describeSmi says what was read and what was not", () => {
  const result = parseSmi("CCO a\nCCO a\nCCC\n\tbad\n");
  const text = describeSmi(result);
  assert.match(text, /3 ligands/);
  assert.match(text, /1 unnamed/);
  assert.match(text, /1 unreadable/);
  assert.match(text, /1 duplicate name/);
  assert.match(text, /1 repeated structure/);
});

test("an empty file is empty rather than one blank ligand", () => {
  for (const text of ["", "\n", "   \n\n", "# only a comment\n"]) {
    assert.equal(parseSmi(text).entries.length, 0, JSON.stringify(text));
  }
});
