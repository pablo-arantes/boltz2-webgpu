/**
 * The scene a screen builds: one target, every ligand superposed onto run #1.
 *
 * 🔴 THE TEST THAT MATTERS IS A RECOVERY TEST, NOT A SMOOTHNESS ONE. A fit
 * that is subtly wrong still returns a structure of the right shape in a
 * plausible place, which looks fine in a picture and makes every ligand's
 * position relative to the pocket a lie. So a known rigid transform is applied
 * to a known structure and the fit must take it BACK - exactly, to within PDB
 * rounding - and the control is the distance before fitting, which must be
 * large.
 *
 * 🔴 AND THE LIGAND MUST MOVE WITH THE PROTEIN AND NOT BE FITTED ITSELF. That
 * is the whole point of the exercise: the ligand's place in the pocket is the
 * ANSWER, so the transform is derived from the target alone and applied to
 * everything. A test that only checked the protein landed would pass with the
 * ligand fitted separately, which would put every ligand in the same place
 * regardless of where the model put it.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { ScreenScene, ligandChains, splitTarget, superposeRun, scoreColour,
  scoreDomain, renumberLigand } from "../web/screen-scene.js";

/**
 * 🔴 BUILT BY COLUMN INDEX, NOT BY CONCATENATION. The first version of this
 * helper concatenated the fields and left out the one-character altLoc at
 * column 17, which put the chain id at column 21 instead of 22 - so every
 * chain read as " " and six of these tests failed in a way that looked like
 * the module under test was wrong. A PDB is a fixed-column format; written as
 * one it cannot drift.
 */
const atom = (serial, name, res, chain, resSeq, [x, y, z], het = false) => {
  const line = new Array(80).fill(" ");
  const put = (at, text) => [...String(text)]
    .forEach((c, i) => { line[at + i] = c; });
  put(0, het ? "HETATM" : "ATOM  ");
  // 🔴 THE SERIAL OCCUPIES COLUMNS 7-11, i.e. INDICES 6-10, right justified -
  // so the last digit lands on index 10, not 11. This read `11 - len + 1`,
  // which pushed a two-digit serial into indices 10-11 and left columns 6-10
  // holding a truncated value. Harmless while no test parsed a serial;
  // renumberLigand parses them, and the fixture's error read as the code's.
  put(11 - String(serial).length, serial);         // 7-11, right justified
  put(13, name);                                 // 14-16 for a short name
  put(17, res.padEnd(3));                        // 18-20
  put(21, chain);                                // 22
  put(26 - String(resSeq).length, resSeq);       // 23-26, right justified
  [x, y, z].forEach((value, k) =>
    put(30 + k * 8, value.toFixed(3).padStart(8)));
  put(54, "  1.00 50.00");
  put(77, "C");
  return line.join("").trimEnd();
};

/**
 * A six-residue chain A plus a ligand chain B of `ligandAtoms` atoms.
 *
 * 🔴 THE CHAIN IS A HELIX AND NOT A LINE, AND THAT IS NOT COSMETIC. The first
 * version laid the alpha carbons out along x in the z = 0 plane - six
 * COLLINEAR points - and a Kabsch fit on collinear points is underdetermined:
 * every rotation about that axis fits equally well, so the routine returns an
 * arbitrary one and an identical structure does not come back as the identity.
 * It failed the "fitted on run #1" test at x = 11.56 where 10 was expected,
 * which reads exactly like a wrong reference. A real backbone spans three
 * dimensions; the fixture must too, or the test is about degeneracy rather
 * than about the scene.
 */
function structure(ligandAtoms = 3, shift = [0, 0, 0]) {
  const lines = [];
  let serial = 1;
  for (let residue = 0; residue < 6; residue += 1) {
    const turn = residue * 1.75;
    for (const [name, offset] of [["N", -0.6], ["CA", 0], ["C", 0.6]]) {
      lines.push(atom(serial, name, "ALA", "A", residue + 1,
        [residue * 1.5 + offset + shift[0],
          2.3 * Math.cos(turn) + shift[1],
          2.3 * Math.sin(turn) + shift[2]]));
      serial += 1;
    }
  }
  lines.push("TER");
  for (let i = 0; i < ligandAtoms; i += 1) {
    lines.push(atom(serial, `C${i + 1}`, "LIG", "B", 1,
      [10 + i * 1.5 + shift[0], 6 + shift[1], 2 + shift[2]], true));
    serial += 1;
  }
  for (let i = 0; i < ligandAtoms - 1; i += 1) {
    lines.push(`CONECT${String(18 + i + 1).padStart(5)}${String(18 + i + 2).padStart(5)}`);
  }
  lines.push("END");
  return lines.join("\n");
}

/** Rotate about z by `angle` and translate, as a diffusion run's frame would. */
function moved(pdb, angle, translation) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return pdb.split("\n").map((line) => {
    if (!(line.startsWith("ATOM") || line.startsWith("HETATM"))) return line;
    const [x, y, z] = [30, 38, 46].map((at) => Number.parseFloat(line.slice(at, at + 8)));
    const p = [x * c - y * s + translation[0], x * s + y * c + translation[1],
      z + translation[2]];
    return line.slice(0, 30)
      + p.map((v) => v.toFixed(3).padStart(8)).join("") + line.slice(54);
  }).join("\n");
}

const coords = (pdb, predicate = () => true) => pdb.split("\n")
  .filter((line) => (line.startsWith("ATOM") || line.startsWith("HETATM"))
    && predicate(line))
  .map((line) => [30, 38, 46].map((at) => Number.parseFloat(line.slice(at, at + 8))));

const maxDistance = (a, b) => Math.max(...a.map((p, i) =>
  Math.hypot(p[0] - b[i][0], p[1] - b[i][1], p[2] - b[i][2])));

test("a ligand chain is the one that is entirely HETATM", () => {
  assert.deepEqual(ligandChains(structure()), ["B"]);
});

test("🔴 a protein chain with a HETATM in it is not a ligand chain", () => {
  // An atomised/modified residue inside a polymer is written HETATM. Taking
  // "has a HETATM" as the test would make the protein the ligand and fit the
  // superposition on three ligand atoms.
  const pdb = structure().split("\n").map((line) =>
    (line.slice(12, 16).trim() === "CA" && line.slice(22, 26).trim() === "1"
      ? `HETATM${line.slice(6)}` : line)).join("\n");
  assert.deepEqual(ligandChains(pdb), ["B"]);
});

test("the target and the ligand are split apart, CONECT with the ligand", () => {
  const split = splitTarget(structure(3));
  assert.equal(coords(split.target).length, 18);
  assert.equal(coords(split.ligand).length, 3);
  assert.match(split.ligand, /CONECT/);
  assert.doesNotMatch(split.target, /CONECT/);
  assert.deepEqual(split.ligandChains, ["B"]);
});

test("🔴 a run in another frame is taken back onto run #1, exactly", () => {
  const reference = structure(3);
  const tumbled = moved(structure(3), 1.1, [37, -12, 60]);
  // the control: it really is somewhere else to begin with
  const before = maxDistance(coords(tumbled), coords(reference));
  assert.ok(before > 20, `the arm does not move the structure: ${before}`);

  const fit = superposeRun(tumbled, reference);
  assert.equal(fit.on, "target", "the fit must be on the protein");
  assert.equal(fit.fitted, 6, "six alpha carbons");
  const after = maxDistance(coords(fit.pdb), coords(reference));
  assert.ok(after < 2e-3, `not recovered: ${after} A`);
});

test("🔴 the LIGAND moves with the protein rather than being fitted itself",
  () => {
    const reference = structure(3);
    // A different ligand, in a DIFFERENT PLACE in the pocket, in another frame.
    const other = moved(structure(7).split("\n").map((line) =>
      (line.startsWith("HETATM")
        ? line.slice(0, 30)
          + [Number.parseFloat(line.slice(30, 38)) + 4,
            Number.parseFloat(line.slice(38, 46)),
            Number.parseFloat(line.slice(46, 54))]
            .map((v) => v.toFixed(3).padStart(8)).join("") + line.slice(54)
        : line)).join("\n"), 0.7, [-20, 15, -8]);

    const fit = superposeRun(other, reference);
    // the protein lands...
    const protein = (line) => line[21] === "A";
    assert.ok(maxDistance(coords(fit.pdb, protein), coords(reference, protein))
      < 2e-3);
    // ...and the ligand keeps the 4 A offset it was given, in run #1's frame.
    const ligand = coords(fit.pdb, (line) => line[21] === "B");
    const referenceLigand = coords(reference, (line) => line[21] === "B");
    assert.equal(ligand.length, 7);
    const offset = ligand[0][0] - referenceLigand[0][0];
    assert.ok(Math.abs(offset - 4) < 2e-3,
      `the ligand was fitted rather than carried: offset ${offset}`);
  });

test("🔴 unequal alpha-carbon counts refuse the fit rather than pairing N of each",
  () => {
    const shorter = structure(3).split("\n")
      .filter((line) => !(line.slice(12, 16).trim() === "CA"
        && line.slice(22, 26).trim() === "6")).join("\n");
    const fit = superposeRun(moved(structure(3), 1.1, [37, -12, 60]), shorter);
    assert.equal(fit.on, "none");
    assert.equal(fit.fitted, 0);
  });

test("the scene draws the target once and a ligand per run", () => {
  const scene = new ScreenScene();
  const first = scene.add("lig1", structure(3));
  assert.ok(first.target !== null, "run #1 contributes the target");
  assert.equal(coords(first.target).length, 18);

  const second = scene.add("lig2", moved(structure(7), 1.1, [37, -12, 60]));
  assert.equal(second.target, null, "run #2 must not add a second protein");
  assert.equal(coords(second.added.ligand).length, 7);
  assert.equal(second.added.on, "target");
  assert.equal(scene.ligands.length, 2);
});

test("🔴 every run is fitted on run #1, not on the previous run", () => {
  const scene = new ScreenScene();
  scene.add("lig1", structure(3));
  // run 2 lands far away; run 3 is in run #1's own frame already
  scene.add("lig2", moved(structure(3), 2.5, [120, -80, 40]));
  const third = scene.add("lig3", structure(3));
  assert.equal(third.added.fitted, 6);
  // 🔴 chained onto run #2 this would be moved somewhere else entirely; onto
  // run #1 it is already where it belongs and must not move.
  const ligand = coords(third.added.ligand);
  assert.ok(Math.abs(ligand[0][0] - 10) < 2e-3,
    `fitted onto the wrong reference: x = ${ligand[0][0]}`);
});

test("show and hide is per ligand", () => {
  const scene = new ScreenScene();
  scene.add("a", structure(3));
  scene.add("b", structure(3));
  scene.setVisible("a", false);
  assert.deepEqual(scene.visible().map((entry) => entry.id), ["b"]);
  scene.setVisible("a", true);
  assert.equal(scene.visible().length, 2);
});

test("🔴 the ramp is reversed for the column where LOWER is better", () => {
  const results = [{ affinity_pred_value: 1, affinity_probability_binary: 0.2 },
    { affinity_pred_value: 5, affinity_probability_binary: 0.9 }];
  const probability = scoreDomain(results, "affinity_probability_binary");
  assert.deepEqual([probability.lo, probability.hi, probability.reversed],
    [0.2, 0.9, false]);
  const value = scoreDomain(results, "affinity_pred_value");
  // lower is better, so the ramp's warm end is the LOW value
  assert.deepEqual([value.lo, value.hi, value.reversed], [5, 1, true]);
  // ...and that makes the better ligand the warmer colour in both columns
  const warmer = scoreColour(1, value.lo, value.hi);
  const cooler = scoreColour(5, value.lo, value.hi);
  assert.equal(warmer, scoreColour(0.9, probability.lo, probability.hi));
  assert.equal(cooler, scoreColour(0.2, probability.lo, probability.hi));
});

test("a one-ligand run has no range and still gets a colour", () => {
  const { lo, hi } = scoreDomain([{ affinity_probability_binary: 0.4 }],
    "affinity_probability_binary");
  assert.match(scoreColour(0.4, lo, hi), /^#[0-9a-f]{6}$/);
  // and a missing number is grey rather than black
  assert.equal(scoreColour(undefined, lo, hi), "#9aa0a6");
  assert.equal(scoreColour(NaN, 0, 1), "#9aa0a6");
});

test("🔴 the fit reports the residual, because the target itself differs", () => {
  const reference = structure(3);
  // the same protein, each alpha carbon nudged: a second fold of one sequence
  const nudged = reference.split("\n").map((line) => {
    if (!line.startsWith("ATOM")) return line;
    const [x, y, z] = [30, 38, 46]
      .map((at) => Number.parseFloat(line.slice(at, at + 8)));
    return line.slice(0, 30)
      + [x + 0.3, y - 0.3, z].map((v) => v.toFixed(3).padStart(8)).join("")
      + line.slice(54);
  }).join("\n");
  const fit = superposeRun(nudged, reference);
  // a pure translation is removed entirely by the fit
  assert.ok(fit.rms < 1e-3, `a rigid shift must fit away: ${fit.rms}`);

  // ...where a real conformational difference does not
  const bent = reference.split("\n").map((line) => {
    if (!line.startsWith("ATOM") || line.slice(22, 26).trim() !== "6") return line;
    const [x, y, z] = [30, 38, 46]
      .map((at) => Number.parseFloat(line.slice(at, at + 8)));
    return line.slice(0, 30)
      + [x, y + 3, z].map((v) => v.toFixed(3).padStart(8)).join("")
      + line.slice(54);
  }).join("\n");
  const second = superposeRun(bent, reference);
  assert.ok(second.rms > 0.3,
    `a bent terminus must survive the fit: ${second.rms}`);
  // 🔴 and run #1 itself is 0, not null: it IS the reference
  assert.equal(superposeRun(reference, null).rms, 0);
});

test("🔴 two ligands in one scene get distinct serials, chains and residues", () => {
  // Each ligand comes from its OWN fold, so both restart their serials. py2Dmol
  // resolves CONECT by serial with the last atom winning, so concatenating them
  // raw puts every bond on the last ligand and leaves the first as loose
  // spheres - the exact failure splitTarget's CONECT handling exists to stop.
  const first = splitTarget(structure(3)).ligand;
  const second = splitTarget(structure(5)).ligand;
  const a = renumberLigand(first, 100, "B", 1);
  const b = renumberLigand(second, a.nextSerial, "C", 2);

  const serialsOf = (pdb) => pdb.split("\n")
    .filter((line) => line.startsWith("HETATM") || line.startsWith("ATOM"))
    .map((line) => Number.parseInt(line.slice(6, 11), 10));
  const serials = [...serialsOf(a.pdb), ...serialsOf(b.pdb)];
  assert.equal(new Set(serials).size, serials.length, "serials must be unique");
  assert.equal(a.nextSerial, 103);
  assert.equal(b.nextSerial, 108);

  // chain and residue number distinguish them, so the viewer's residue-level
  // machinery sees two ligands rather than one
  const idOf = (pdb) => new Set(pdb.split("\n")
    .filter((line) => line.startsWith("HETATM"))
    .map((line) => `${line[21]}:${line.slice(22, 26).trim()}`));
  assert.deepEqual([...idOf(a.pdb)], ["B:1"]);
  assert.deepEqual([...idOf(b.pdb)], ["C:2"]);

  // 🔴 AND THE CONECT RECORDS FOLLOW THE NEW SERIALS. Left pointing at the old
  // ones they would reference atoms in the other ligand, or nothing.
  const conect = (pdb) => pdb.split("\n").filter((l) => l.startsWith("CONECT"));
  for (const [part, own] of [[a, serialsOf(a.pdb)], [b, serialsOf(b.pdb)]]) {
    const records = conect(part.pdb);
    assert.ok(records.length > 0, "the ligand must keep its bonds");
    for (const record of records) {
      for (let at = 6; at + 5 <= record.length; at += 5) {
        const field = record.slice(at, at + 5).trim();
        if (field === "") break;
        assert.ok(own.includes(Number.parseInt(field, 10)),
          `CONECT ${field} is not one of this ligand's new serials`);
      }
    }
  }
});
