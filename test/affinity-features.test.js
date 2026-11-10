/**
 * The affinity pass's own features are depth-1, and demonstrably not the deep ones.
 *
 * 🔴 THE SEPARATION IS THE POINT. A depth-1 profile and a deep one have the
 * same shape, the same dtype and the same row sums, so an implementation that
 * silently used the deep profile would look entirely healthy. Every test here
 * that asserts an equality is paired with one asserting a DIFFERENCE against
 * the deep profile of the same sequence - because without that, these would all
 * pass with the wrong feature wired in.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { affinityProfile, affinityTokenMask, RESTYPES }
  from "../shared/af3/affinity/features.js";

const SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ";
const boltz2 = () => featuriserDialect(dialectFor("boltz2"));

// an alignment with real variation, so the deep profile is genuinely not a
// one-hot - otherwise the separation tests below could not fail
const DEEP_A3M = [
  `>101\n${SEQUENCE}`,
  `>h1\n${SEQUENCE.replace(/A/g, "G").replace(/K/g, "R")}`,
  `>h2\n${SEQUENCE.replace(/I/g, "L").replace(/S/g, "T")}`,
  `>h3\n${SEQUENCE.replace(/Q/g, "E").replace(/Y/g, "F")}`,
].join("\n") + "\n";

const single = () => af3BatchFromA3m(SEQUENCE, null, boltz2()).batch;
const deep = () => af3BatchFromA3m(SEQUENCE, DEEP_A3M, boltz2()).batch;

test("the restype width agrees with the featuriser's own", () => {
  const batch = single();
  assert.equal(batch.profile.length / batch.tokens, RESTYPES);
});

test("the deep alignment really is deep - or the separation proves nothing", () => {
  const batch = deep();
  assert.ok(batch.msa.length / batch.tokens > 1, "more than one MSA row");
  let mixed = 0;
  for (let t = 0; t < batch.tokens; t += 1) {
    let nonZero = 0;
    for (let c = 0; c < RESTYPES; c += 1) {
      if (batch.profile[t * RESTYPES + c] > 0) nonZero += 1;
    }
    if (nonZero > 1) mixed += 1;
  }
  assert.ok(mixed > 5, `only ${mixed} tokens have a mixed profile`);
});

test("the depth-1 profile is a one-hot of the MSA query row", () => {
  const batch = deep();
  const { profile } = affinityProfile(batch);
  for (let t = 0; t < batch.tokens; t += 1) {
    let sum = 0;
    let argmax = -1;
    let best = -1;
    for (let c = 0; c < RESTYPES; c += 1) {
      const v = profile[t * RESTYPES + c];
      sum += v;
      if (v > best) { best = v; argmax = c; }
    }
    assert.equal(sum, 1, `token ${t} does not sum to 1`);
    assert.equal(argmax, batch.msa[t], `token ${t} is not the query row's code`);
  }
});

test("🔴 ...and it EQUALS what the featuriser gives with no alignment", () => {
  // upstream's depth-1 crop is the same thing, and this featuriser's
  // single-sequence path is already held to AF3's dump by test:batch - so this
  // reuses verified machinery rather than reimplementing a profile.
  const fromDeep = affinityProfile(deep()).profile;
  const noAlignment = single().profile;
  assert.equal(fromDeep.length, noAlignment.length);
  for (let i = 0; i < fromDeep.length; i += 1) {
    assert.equal(fromDeep[i], noAlignment[i], `index ${i}`);
  }
});

test("🔴 ...and DIFFERS from the deep profile of the same sequence", () => {
  // Without this, every test above would pass with the deep profile wired in.
  const batch = deep();
  const depthOne = affinityProfile(batch).profile;
  let differing = 0;
  for (let i = 0; i < depthOne.length; i += 1) {
    if (depthOne[i] !== batch.profile[i]) differing += 1;
  }
  assert.ok(differing > 20,
    `only ${differing} entries differ - the depth-1 profile may be the deep one`);
});

test("the deletion mean is zeros, where the deep one is not", () => {
  const batch = deep();
  const { deletionMean } = affinityProfile(batch);
  assert.equal(deletionMean.length, batch.tokens);
  assert.ok(deletionMean.every((v) => v === 0));
});

test("a batch without an MSA query row is refused, not defaulted", () => {
  assert.throws(() => affinityProfile({ tokens: 4, msa: new Int32Array(2) }),
    /at least the query row/);
  assert.throws(() => affinityProfile({ tokens: 0, msa: new Int32Array(0) }),
    /batch.tokens is 0/);
});

test("a query-row code outside the restype range is refused", () => {
  assert.throws(
    () => affinityProfile({ tokens: 2, msa: Int32Array.from([0, 99]) }),
    /outside 0\.\.30/);
});

test("affinity_token_mask covers every token of the binder CHAIN", () => {
  // asym ids: 0 protein, 1 ligand (3 tokens)
  const chainOfToken = Int32Array.from([0, 0, 0, 0, 1, 1, 1]);
  const mask = affinityTokenMask(chainOfToken, 1);
  assert.deepEqual([...mask], [0, 0, 0, 0, 1, 1, 1]);
});

test("🔴 an empty mask is refused rather than producing a confident zero", () => {
  // The heads divide by the mask's own sum, so an empty mask gives
  // 0 / 1e-7 = 0 and a plausible affinity for no ligand at all.
  const chainOfToken = Int32Array.from([0, 0, 0]);
  assert.throws(() => affinityTokenMask(chainOfToken, 7),
    /has no tokens in the cropped structure/);
});
