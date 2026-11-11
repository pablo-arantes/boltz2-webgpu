/**
 * The target-length ceiling, and the advice above it.
 *
 * 🔴 THE PAGE HAD NO UPPER LIMIT, AND A READER FOUND IT WITH A 2,244-RESIDUE
 * TARGET: `sequence.length < 4` was the only check, so the run started, the
 * weights downloaded and the screen failed in the middle of the first ligand
 * with nothing said about the length. These tests pin the arithmetic that
 * replaced it, for devices this machine does not have.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  targetSizeAdvice, residueCeiling, pairBytes, CROP_ADVICE, MEASURED_TO,
  AFFINITY_PROTEIN_TOKENS, ASSUMED_BINDING_BYTES, PAIR_CHANNELS,
} from "../web/target-size.js";

const GIB = 1024 ** 3;

test("🔴 the ceiling is the DEVICE's, and it differs by a thousand residues", () => {
  // measured through Chrome 157 on an M3 Pro: the adapter reports 4 GiB - 4
  assert.equal(residueCeiling(4 * GIB - 4), 2896);
  // and a driver that reports 2 GiB puts a 2,000-residue target on the edge
  assert.equal(residueCeiling(2 * GIB), 2048);
  // Chrome's own default, when an adapter offers nothing better
  assert.equal(residueCeiling(128 * 1024 ** 2), 512);
});

test("🔴 a pair tensor is tokens squared, which is the whole wall", () => {
  assert.equal(pairBytes(33), 33 * 33 * PAIR_CHANNELS * 4);
  // the two numbers that decided the reader's case, to two decimals
  assert.equal((pairBytes(2000) / GIB).toFixed(2), "1.91");
  assert.equal((pairBytes(2244) / GIB).toFixed(2), "2.40");
});

test("🔴 a 2,244-residue target is REFUSED on a 2 GiB device", () => {
  const advice = targetSizeAdvice(2244, { bindingBytes: 2 * GIB });
  assert.equal(advice.level, "refuse");
  assert.match(advice.message, /2244 residues/);
  assert.match(advice.message, /2\.40 GiB/, "the pair tensor's real size");
  assert.match(advice.message, /2048 residues/, "this device's own ceiling");
  assert.ok(advice.message.includes(CROP_ADVICE),
    "a refusal without the way forward is just a wall");
});

test("🔴 ...and the SAME target is refused on a 4 GiB device too", () => {
  // 2,244 is under 2,896, so the binding would take ONE tensor - but the note
  // must still warn, because several are live at once.
  const advice = targetSizeAdvice(2244, { bindingBytes: 4 * GIB - 4 });
  assert.notEqual(advice.level, "ok");
  assert.match(advice.note, /2244 residues/);
});

test("🔴 the measured range passes quietly, and 297 is the boundary", () => {
  // docs/MEASURED.md times 99, 198 and 297 residues; nothing beyond is timed
  assert.equal(MEASURED_TO, 297);
  assert.equal(targetSizeAdvice(99).level, "ok");
  assert.equal(targetSizeAdvice(99).note, "99 residues");
  // past the affinity head's own budget, the page says so without warning
  assert.equal(targetSizeAdvice(250).level, "note");
  assert.match(targetSizeAdvice(250).note,
               new RegExp(`${AFFINITY_PROTEIN_TOKENS} of them`));
  assert.equal(targetSizeAdvice(297).level, "note");
  assert.equal(targetSizeAdvice(298).level, "warn");
});

test("🔴 the warning says it is PAST A MEASUREMENT, not past a cliff", () => {
  const advice = targetSizeAdvice(800, { bindingBytes: 4 * GIB - 4 });
  assert.equal(advice.level, "warn");
  assert.match(advice.message, /beyond what has been measured/);
  assert.match(advice.message, /297/, "and names the largest length timed");
  // 🔴 AND IT GIVES NO TIME ESTIMATE FOR 800 RESIDUES. The measured curve runs
  // 99-297; extrapolating it almost three times past its last point would be
  // inventing a number, and an invented number in a warning is worse than the
  // warning being vague.
  assert.doesNotMatch(advice.message, /800 s|minutes each|\d+ s a ligand at 800/);
  assert.ok(advice.message.includes(CROP_ADVICE));
});

test("🔴 an empty box is not a warning", () => {
  const advice = targetSizeAdvice(0);
  assert.equal(advice.level, "empty");
  assert.equal(advice.note, "one protein chain, amino acids");
  assert.equal(advice.message, "");
});

test("🔴 before a device answers, the assumption is CONSERVATIVE", () => {
  // The note is drawn while the weights are still downloading. Assuming this
  // machine's 4 GiB would promise a ceiling a visitor's hardware may not have.
  assert.equal(ASSUMED_BINDING_BYTES, 2 * GIB);
  assert.equal(targetSizeAdvice(2500).level, "refuse");
  assert.equal(targetSizeAdvice(2500, { bindingBytes: 4 * GIB - 4 }).level,
               "warn", "a real 4 GiB device folds what the assumption refuses");
});

test("🔴 the crop advice says cropping costs nothing in the SCORE", () => {
  // which is the fact that makes it the method rather than a workaround:
  // shared/af3/affinity/crop.js is upstream's own AffinityCropper budget
  assert.match(CROP_ADVICE, new RegExp(`${AFFINITY_PROTEIN_TOKENS}`));
  assert.match(CROP_ADVICE, /10 Å|10 Å/);
  assert.match(CROP_ADVICE, /whole domain/);
  assert.match(CROP_ADVICE, /both/, "a site between two domains needs both");
});

test("🔴 the numbers here are the ones shared/af3/affinity/crop.js holds", () => {
  // 🔴 TWO COPIES OF A BUDGET IS ONE COPY THAT GOES STALE, and the stale one
  // would be the advice - which would then tell a reader to crop to a size the
  // model does not use.
  const crop = readFileSync(
    new URL("../shared/af3/affinity/crop.js", import.meta.url), "utf8");
  assert.match(crop,
               new RegExp(`MAX_TOKENS_PROTEIN = ${AFFINITY_PROTEIN_TOKENS}`),
               "the affinity cropper's protein budget moved");
  assert.match(crop, /NEIGHBOURHOOD_SIZE = 10/,
               "the 10 Å in the advice is the cropper's neighbourhood");
});

test("🔴 the README's crop advice carries the SAME numbers as the cropper", () => {
  // 🔴 THE README IS THE THIRD COPY OF THIS BUDGET, after the page's advice and
  // the cropper itself - and prose is always the copy that goes stale. A README
  // telling a reader to crop to a size the model does not use would be worse
  // than saying nothing, because they would act on it.
  const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
  const tips = readme.slice(readme.indexOf("## Tips for a screen"));
  assert.ok(tips.length > 500, "the README has no screening tips section");
  assert.match(tips, new RegExp(`${AFFINITY_PROTEIN_TOKENS} of them protein`),
    "the tips do not give the affinity head's protein budget");
  assert.match(tips, /10 Å/, "the tips do not give the cropper's neighbourhood");
  assert.match(tips, /256 tokens/, "the tips do not give the token budget");
  // the measured table, which is the other thing a reader will act on
  for (const point of ["99 aa", "198 aa", "297 aa", "7.4 s", "20.9 s", "32.2 s"]) {
    assert.ok(tips.includes(point), `the tips lost the measured point ${point}`);
  }
  assert.match(tips, new RegExp(`${MEASURED_TO} residues is the largest`),
    "the tips must say 297 is the largest length timed, not predict beyond it");
});

test("🔴 every level gives the sequence box something to print", () => {
  for (const residues of [0, 1, 99, 250, 297, 298, 1500, 5000]) {
    const advice = targetSizeAdvice(residues);
    assert.equal(typeof advice.note, "string");
    assert.ok(advice.note.length > 0, `no note at ${residues}`);
    assert.ok(advice.note.length < 140,
      `the note at ${residues} is a paragraph, not a line: ${advice.note}`);
    if (advice.level === "warn" || advice.level === "refuse") {
      assert.ok(advice.message.length > 120,
        `${advice.level} at ${residues} has no explanation`);
    }
  }
});
