/**
 * The method conditioning class boltz2 is folded with.
 *
 * 🔴 THIS EXISTS BECAUSE A WRONG CONSTANT HERE MOVED A PREDICTED IC50 BY A
 * FACTOR OF 2.2 AND NOTHING IN THE TREE COULD SEE IT. `targetFeatures` fed
 * method class **1** - X-RAY DIFFRACTION - where genuine Boltz-2 feeds **4**,
 * the OTHER/THEORETICAL MODEL bucket that a de novo prediction belongs to.
 * Measured against upstream on a real job (spec/G8-RESULT.md):
 *
 *   affinity_pred_value, gap from upstream   6.0x its seed spread -> 0.19x
 *   s_inputs, protein tokens                 4.89e-1 -> 3.90e-2 (the int5 floor)
 *   s - a, every non-atom-encoder term        4.82e-1 -> 3.62e-2
 *
 * 🔴 AND IT WAS INVISIBLE TO EVERY EXISTING GATE FOR ONE REASON: the method
 * row is a learned 384-vector added IDENTICALLY to every token, so it shifts
 * `target_feat` wholesale instead of distorting any token's features. A
 * per-token comparison cannot see a constant. docs/ records this port's boltz2
 * `target_feat` at ~1e-8 against af3-any-model's dump - which can only hold
 * because that port feeds class 1 too, so the agreement was two
 * implementations being wrong together. That is CLAUDE.md's own recorded trap:
 * "feeding ONE bundle to both sides makes a tensor at the wrong name
 * invisible, because both forwards read it and both are wrong together."
 *
 * So what is asserted here is the VALUE, against upstream's published table,
 * with the table reproduced so a reader can check it without a checkout.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { targetFeatures } from "../cpu/af3/diffusion/atom-encoder.js";

/** boltz/data/const.py `method_types_ids`, at the pinned revision. */
const UPSTREAM_METHOD_IDS = {
  MD: 0,
  "X-RAY DIFFRACTION": 1,
  "ELECTRON MICROSCOPY": 2,
  "SOLUTION NMR": 3,
  // 4 is shared by every method that is not one of the above, including
  // THEORETICAL MODEL and OTHER - which is what a prediction with no
  // experiment IS, and what `boltz predict` uses by default.
  "THEORETICAL MODEL": 4,
  OTHER: 4,
  AFDB: 5,
  "BOLTZ-1": 6,
};
const DE_NOVO = UPSTREAM_METHOD_IDS.OTHER;

/**
 * A minimal boltz2-dialect input: the sum weights are one-hot selectors so
 * each term's contribution is readable straight out of the output.
 */
function probe({ methodClass } = {}) {
  const tokens = 2;
  const channels = 384;
  const zeros = (n) => new Float32Array(n);
  // the method weight: row k is the constant k, so the output names the row
  const method = new Float32Array(12 * channels);
  for (let k = 0; k < 12; k += 1) {
    for (let c = 0; c < channels; c += 1) method[k * channels + c] = k;
  }
  const input = {
    aatype: new Int32Array(tokens),
    profile: zeros(tokens * 31),
    deletionMean: zeros(tokens),
    atomFeatures: zeros(tokens * channels),
    isLigand: new Uint8Array(tokens),
    sum: {
      resType: zeros(31 * channels),
      msaProfile: zeros(32 * channels),
      molType: zeros(4 * channels),
      cyclic: zeros(channels),
      method,
      modified: zeros(2 * channels),
    },
    ...(methodClass === undefined ? {} : { methodClass }),
  };
  const out = targetFeatures(input, tokens, { targetFeatAtomOnly: true });
  return out[0];
}

test("🔴 boltz2 is folded with upstream's de novo method class, not x-ray", () => {
  // With row k holding the constant k, the output IS the class index.
  assert.equal(probe(), DE_NOVO,
    `targetFeatures used method class ${probe()}; genuine Boltz-2 uses`
    + ` ${DE_NOVO} (OTHER / THEORETICAL MODEL). Class 1 is X-RAY DIFFRACTION,`
    + " which asks the model for a crystal structure of something that was"
    + " never crystallised - measured as a factor of 2.2 in predicted IC50.");
  assert.notEqual(DE_NOVO, UPSTREAM_METHOD_IDS["X-RAY DIFFRACTION"],
    "the table itself must distinguish the two, or this test is vacuous");
});

test("the class is a dialect parameter, so a caller that really has a crystal can say so",
  () => {
    assert.equal(probe({ methodClass: 1 }), 1);
    assert.equal(probe({ methodClass: 5 }), 5);
    // ...and omitting it is upstream's default rather than undefined behaviour
    assert.equal(probe(), DE_NOVO);
  });

test("🔴 the method term is CONSTANT across tokens, which is why nothing saw it",
  () => {
    // Both tokens must get the same contribution. A per-token differential
    // comparison therefore cannot distinguish class 1 from class 4 at all -
    // only a comparison against an independent implementation can, which is
    // what npm run bench:g8 is for.
    const tokens = 2;
    const channels = 384;
    const method = new Float32Array(12 * channels).fill(0);
    for (let c = 0; c < channels; c += 1) method[4 * channels + c] = 7;
    const input = {
      aatype: new Int32Array(tokens), profile: new Float32Array(tokens * 31),
      deletionMean: new Float32Array(tokens),
      atomFeatures: new Float32Array(tokens * channels),
      isLigand: new Uint8Array(tokens),
      sum: { resType: new Float32Array(31 * channels),
        msaProfile: new Float32Array(32 * channels),
        molType: new Float32Array(4 * channels),
        cyclic: new Float32Array(channels), method,
        modified: new Float32Array(2 * channels) },
    };
    const out = targetFeatures(input, tokens, { targetFeatAtomOnly: true });
    assert.equal(out[0], 7);
    assert.equal(out[channels], 7, "both tokens get the identical row");
    for (let c = 0; c < channels; c += 1) {
      assert.equal(out[c], out[channels + c]);
    }
  });
