/**
 * The three features pass 2 adds, and the one it quietly replaces.
 *
 * 🔴 THE AFFINITY MODULE'S `s_inputs` CARRIES A DEPTH-1 MSA PROFILE, HOWEVER
 * DEEP THE ALIGNMENT IS. This is the subtlest thing in the whole port.
 * `InputEmbedder.forward(feats, affinity=True)` differs from the structure
 * call in exactly one respect - it reads `profile_affinity` and
 * `deletion_mean_affinity` instead of `profile` and `deletion_mean` - and those
 * come from
 *
 *     process_msa_features(..., max_seqs_batch=1, max_seqs=1, affinity=True)
 *
 * i.e. the MSA cropped to ONE row. Both embedder calls happen in one forward
 * (boltz2.py:415 with affinity=False for the trunk, :627 with affinity=True for
 * the affinity module), so the pass-2 TRUNK still consumes the full alignment
 * and only the affinity module's own `s_inputs` is depth 1. Feeding the deep
 * profile into the affinity module would be a silent, plausible, wrong input -
 * the same shape of mistake, and about as visible, as reading the wrong atom
 * window.
 *
 * 🔴 AND THE DEPTH-1 PROFILE IS NOT `one_hot(res_type)`. upstream computes
 * `one_hot(msa).float().mean(dim=0)`, so at depth 1 it is the one-hot of the
 * MSA's QUERY ROW - and for boltz2 a ligand or atomised token's query row is
 * the protein UNK rather than its own restype (`ligandMsaUnk`,
 * `atomizedUnknownRestype`). The two therefore differ exactly on the tokens
 * where it is easiest to be wrong, which is why this reads the query row rather
 * than the restype.
 *
 * 🔴 AND IT IS WHAT THIS FEATURISER ALREADY PRODUCES WITH NO ALIGNMENT.
 * Measured: `af3BatchFromA3m(sequence, null, boltz2Dialect)` gives a `profile`
 * whose rows sum to 1.0000, whose argmax is `msa[0][token]`, and a
 * `deletionMean` of zeros - which is upstream's depth-1 crop exactly. So this
 * does not reimplement a profile; it rebuilds the one the featuriser's own
 * already-verified machinery defines (`npm run test:batch` holds that machinery
 * to AF3's dump over 43 fields), and `test/affinity-features.test.js` asserts
 * the equality both ways round.
 */

/** The MSA restype width the featuriser writes, and this must agree with it. */
export const RESTYPES = 31;

/**
 * The affinity module's `s_inputs` profile: depth 1, from the query row.
 *
 * @param {{tokens: number, msa: ArrayLike<number>}} batch from the featuriser;
 *   `msa` is row-major rows x tokens and row 0 is the query
 * @param {number} [restypes]
 * @returns {{profile: Float32Array, deletionMean: Float32Array}}
 */
export function affinityProfile(batch, restypes = RESTYPES) {
  const tokens = batch.tokens;
  if (!Number.isInteger(tokens) || tokens <= 0) {
    throw new RangeError(`affinityProfile: batch.tokens is ${tokens}`);
  }
  if (!batch.msa || batch.msa.length < tokens) {
    throw new RangeError("affinityProfile: batch.msa must hold at least the"
      + " query row; it is what the depth-1 profile IS");
  }
  const profile = new Float32Array(tokens * restypes);
  for (let token = 0; token < tokens; token += 1) {
    const code = batch.msa[token];          // row 0 is the query
    if (!Number.isInteger(code) || code < 0 || code >= restypes) {
      throw new RangeError(`affinityProfile: query row code ${code} at token`
        + ` ${token} is outside 0..${restypes - 1}`);
    }
    profile[token * restypes + code] = 1;
  }
  // 🔴 ZEROS, AND DERIVED RATHER THAN COPIED. upstream's
  // `deletion_mean = (pi/2 * arctan(deletion/3)).mean(axis=0)` over one row is
  // the query's own deletions, and a query row has none by construction - it is
  // the sequence the alignment is of. Copying the structure pass's
  // `deletionMean` would carry the whole alignment's deletions into a depth-1
  // feature.
  return { profile, deletionMean: new Float32Array(tokens) };
}

/**
 * `affinity_token_mask`: every token of the binder chain.
 *
 * 🔴 PER CHAIN, NOT PER LIGAND TOKEN, AND REBUILT AFTER THE CROP. upstream sets
 * it for every token whose `asym_id` equals `affinity.chain_id`
 * (data/tokenize/boltz2.py:172), and the affinity crop renumbers tokens - so
 * carrying pass 1's mask across would mask the wrong ones. `chainOfToken` here
 * is the CROPPED order's.
 *
 * @param {ArrayLike<number>} chainOfToken one asym id per token, cropped order
 * @param {number} binderChain the asym id the job's `properties.affinity.binder`
 *   names
 */
export function affinityTokenMask(chainOfToken, binderChain) {
  const mask = new Uint8Array(chainOfToken.length);
  let count = 0;
  for (let token = 0; token < chainOfToken.length; token += 1) {
    if (chainOfToken[token] === binderChain) { mask[token] = 1; count += 1; }
  }
  if (count === 0) {
    // 🔴 A MASK WITH NOTHING IN IT IS NOT A MASK. The heads divide by the
    // mask's own sum, so an empty one gives 0/1e-7 = 0 and a confident-looking
    // affinity for no ligand at all.
    throw new Error(`affinityTokenMask: chain ${binderChain} has no tokens in`
      + " the cropped structure - the binder was cropped away or named wrong");
  }
  return mask;
}
