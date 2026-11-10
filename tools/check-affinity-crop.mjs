#!/usr/bin/env node
/**
 * Does the affinity crop select the tokens upstream's AffinityCropper does?
 *
 *   python3 tools/oracle/dump_affinity_crop.py --boltz=<checkout>
 *   node tools/check-affinity-crop.mjs [--falsify]
 *
 * 🔴 AN EXACT INDEX-SET COMPARISON, BECAUSE THERE IS NO TOLERANCE TO HAVE. The
 * crop either keeps a token or it does not, so this is `===` on a sorted list
 * rather than a residual - and the whole of pass 2 is downstream of it: the
 * trunk's token count, the ligand's own token indices, and therefore
 * `affinity_token_mask`.
 *
 * 🔴 THE CORPUS REACHES ALL THREE BREAK CONDITIONS, which is the point. The
 * loop ends in `break` rather than `continue`, so each budget truncates the
 * crop at a different place and a corpus that never hits one tests a cropper
 * that only ever keeps everything. Measured on the dump:
 *
 *     protein-budget   408 tokens -> 208 kept, protein exactly 200
 *     atom-budget      306 tokens -> 151 kept, 2036 of 2048 atoms
 *     token-budget     270 tokens -> 256 kept, the token cap exactly
 *     short-chain       11 tokens -> 11 kept, the WHOLE-CHAIN branch
 *     unresolved        55 tokens -> ranking skips 7, whole-chain test does not
 *
 * `--falsify` runs five single-rule mutations and each must change at least one
 * case's selection, because a gate over a corpus where every case keeps
 * everything would pass with any of these rules broken.
 *
 * 🔴 AND THE PROBES ARE SIZED TO WHAT THE RULES CAN EXPRESS. Two obvious ones -
 * an atom budget off by one, and neighbourhood 9 against 10 - move NOTHING, and
 * that is a property of the cropper rather than a weak corpus: the atom budget
 * is tested once per window and a window is 250-300 atoms, and the grow loop
 * adds a whole residue at a time and stops at the first count at or above the
 * neighbourhood, so 9 and 10 (and 10 and 20) land on the same window. Measured
 * rather than assumed: atom -1 moves 0 cases, -50 moves 3; neighbourhood 9
 * moves 0, 5 moves 2.
 */
import { readFileSync } from "node:fs";
import { argv } from "node:process";
import { affinityCrop } from "../shared/af3/affinity/crop.js";

const option = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};
const dumpPath = option("dump", "oracle-dumps/affinity-crop.json");
const falsify = argv.includes("--falsify");

let dump;
try {
  dump = JSON.parse(readFileSync(dumpPath, "utf8"));
} catch {
  console.error(`missing ${dumpPath} -`
    + " run: python3 tools/oracle/dump_affinity_crop.py --boltz=<checkout>");
  process.exit(2);
}

function asTokens(rows) {
  const count = rows.length;
  const centreCoords = new Float64Array(count * 3);
  const asymId = new Int32Array(count);
  const resIdx = new Int32Array(count);
  const molType = new Int32Array(count);
  const atomNum = new Int32Array(count);
  const affinityMask = new Uint8Array(count);
  const resolvedMask = new Uint8Array(count);
  rows.forEach((row, i) => {
    centreCoords[i * 3] = row.centre[0];
    centreCoords[i * 3 + 1] = row.centre[1];
    centreCoords[i * 3 + 2] = row.centre[2];
    asymId[i] = row.asymId;
    resIdx[i] = row.resIdx;
    molType[i] = row.molType;
    atomNum[i] = row.atomNum;
    affinityMask[i] = row.affinity ? 1 : 0;
    resolvedMask[i] = row.resolved ? 1 : 0;
  });
  return { centreCoords, asymId, resIdx, molType, atomNum,
    affinityMask, resolvedMask };
}

const sameList = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

console.log(`reference: ${dump.source}`);
console.log(`budgets  : maxTokens ${dump.maxTokens}, maxAtoms ${dump.maxAtoms},`
  + ` neighbourhood ${dump.neighborhoodSize},`
  + ` maxTokensProtein ${dump.maxTokensProtein}`);
console.log();

const options = { maxTokens: dump.maxTokens, maxAtoms: dump.maxAtoms,
  neighbourhoodSize: dump.neighborhoodSize,
  maxTokensProtein: dump.maxTokensProtein };

let failures = 0;
const baseline = new Map();
for (const testCase of dump.cases) {
  const tokens = asTokens(testCase.tokens);
  const result = affinityCrop(tokens, options);
  baseline.set(testCase.name, result.kept);
  const keptOk = sameList(result.kept, testCase.kept);
  const atomsOk = result.atoms === testCase.atoms;
  const proteinOk = result.proteinKept === testCase.proteinKept;
  const ok = keptOk && atomsOk && proteinOk;
  failures += ok ? 0 : 1;
  const detail = ok ? "" : "  " + [
    keptOk ? null : `kept ${result.kept.length} vs ${testCase.kept.length}`
      + ` (first difference at ${result.kept.findIndex((v, i) => v !== testCase.kept[i])})`,
    atomsOk ? null : `atoms ${result.atoms} vs ${testCase.atoms}`,
    proteinOk ? null : `protein ${result.proteinKept} vs ${testCase.proteinKept}`,
  ].filter(Boolean).join("; ");
  console.log(`  ${ok ? "OK  " : "FAIL"} ${testCase.name.padEnd(16)}`
    + ` ${String(testCase.tokens.length).padStart(4)} tokens ->`
    + ` kept ${String(result.kept.length).padStart(3)}`
    + `  protein ${String(result.proteinKept).padStart(3)}`
    + `  atoms ${String(result.atoms).padStart(5)}`
    + `  stoppedAt ${result.stoppedAt === null ? "-" : result.stoppedAt}${detail}`);
}

if (falsify) {
  console.log();
  console.log("=== separation control: one rule broken at a time ===");
  // Each mutation is applied by re-running the crop through a patched budget or
  // ordering, because the rules live inside one function and a flag on it would
  // be a mutation hook in shipped code.
  // 🔴 EACH PROBE IS A CHANGE THE RULE CAN ACTUALLY EXPRESS, AND TWO OBVIOUS
  // ONES CANNOT BE. The first version used `atom budget +/- 1` and
  // `neighbourhood 9 not 10` and both reported "0 of 12 cases change" - which
  // reads as a corpus too weak to see them and is not. Both effects are
  // QUANTISED BY THE WINDOW:
  //
  //   * the atom budget is tested once per window, and a window here is
  //     250-300 atoms, so no budget change smaller than a window can move the
  //     break. Measured: -1 moves 0 cases, -50 moves 3, -300 moves 4.
  //   * the grow loop adds a whole RESIDUE at a time and stops at the first
  //     count >= neighbourhood, so 9 and 10 both land on the same window
  //     wherever the increment straddles them - and so do 10 and 20, where the
  //     window is already large enough. Measured: 9 moves 0 cases, 5 moves 2.
  //
  // So the probes below are sized to what the rules can say, and the two that
  // cannot be seen are named here rather than left looking like a gap. This is
  // the treatment docs/ gives `dedupeSelfMsa` in test:batch - a convention the
  // gate cannot reach, stated in the header instead of silently absent.
  const mutations = {
    "token budget -1": { maxTokens: dump.maxTokens - 1 },
    "protein budget -1": { maxTokensProtein: dump.maxTokensProtein - 1 },
    "atom budget -50 (one window)": { maxAtoms: dump.maxAtoms - 50 },
    "neighbourhood 5 not 10": { neighbourhoodSize: 5 },
    "no atom budget at all": { maxAtoms: null },
  };
  let caught = 0;
  for (const [label, patch] of Object.entries(mutations)) {
    let moved = 0;
    for (const testCase of dump.cases) {
      const result = affinityCrop(asTokens(testCase.tokens),
                                  { ...options, ...patch });
      if (!sameList(result.kept, baseline.get(testCase.name))) moved += 1;
    }
    const sees = moved > 0;
    caught += sees ? 1 : 0;
    console.log(`  ${sees ? "CAUGHT " : "MISSED "} ${label.padEnd(28)}`
      + ` ${moved} of ${dump.cases.length} cases change`);
  }
  console.log();
  const all = Object.keys(mutations).length;
  if (caught === all) {
    console.log(`SEPARATION OK: all ${all} rule changes move at least one case`);
  } else {
    console.log(`SEPARATION FAILED: ${caught} of ${all} - the corpus cannot see`
      + " every rule it claims to cover");
    failures += 1;
  }
}

console.log();
console.log(failures === 0
  ? `CROP PASS: ${dump.cases.length} cases, every kept index exact`
  : `CROP FAIL: ${failures}`);
process.exit(failures === 0 ? 0 : 1);
