/**
 * Boltz-2's affinity crop: the pocket pass 2 actually runs on.
 *
 * A port of upstream `boltz/data/crop/affinity.py` (AffinityCropper), which
 * `data/module/inferencev2.py` instantiates with its defaults and calls with
 * `max_tokens=256, max_atoms=2048`.
 *
 * 🔴 PASS 2 IS NOT PASS 1 ON THE SAME TOKENS. The affinity pass re-featurises
 * pass 1's PREDICTED structure and crops the protein to a neighbourhood of the
 * ligand, so its trunk sees at most 200 protein tokens however long the chain
 * is. That is what makes a 5-recycle, 200-step, 3-sample second pass tractable
 * at all - and it means the ligand's token indices MOVE between the passes,
 * which `affinity_token_mask` has to be rebuilt against rather than carried
 * over.
 *
 * 🔴 IT STOPS AT THE FIRST TOKEN THAT WOULD OVERFLOW; IT DOES NOT SKIP ONE AND
 * CARRY ON. Upstream's loop ends in `break`, not `continue`, so the crop is a
 * prefix of the distance ordering and nothing further out is reached even if it
 * would fit. Reading that as "skip and keep going" produces a crop that is
 * bigger, differently shaped, and plausible.
 *
 * 🔴 THE DISTANCE IS TO THE LIGAND'S CENTRE ATOMS, NOT ITS REPRESENTATIVE ONES.
 * Upstream ranks on `center_coords`, which is the token's `center_idx` atom -
 * CA for a protein residue - and that is a DIFFERENT gather from `disto_idx`,
 * the distogram atom (CB) that the affinity module's own distances use. Both
 * exist per token upstream and the port must not share one for both: ranking on
 * CB would reorder the neighbourhood. The caller passes centres here and
 * pseudo-beta to the module, and `tools/check-affinity-crop.mjs` compares the
 * selected index set against upstream's rather than trusting that.
 *
 * 🔴 AND THE WHOLE-CHAIN TEST READS EVERY TOKEN OF THE CHAIN, RESOLVED OR NOT.
 * The ranking is over resolved tokens only, but `chain_tokens` comes from the
 * full table, so a chain with unresolved tokens can take the whole-chain branch
 * on a count the ranking never saw.
 */

/** Upstream's AffinityCropper defaults. */
export const NEIGHBOURHOOD_SIZE = 10;
export const MAX_TOKENS_PROTEIN = 200;
/** inferencev2.py's call: `self.cropper.crop(tokenized, max_tokens=256, max_atoms=2048)` */
export const CROP_MAX_TOKENS = 256;
export const CROP_MAX_ATOMS = 2048;

const NONPOLYMER = 3;      // const.chain_type_ids["NONPOLYMER"]

/**
 * Which tokens pass 2 keeps.
 *
 * @param {object} tokens one array per field, all the same length, in token order
 * @param {Float64Array|number[]} tokens.centreCoords 3 per token - CENTRE atoms
 * @param {Int32Array|number[]} tokens.asymId
 * @param {Int32Array|number[]} tokens.resIdx
 * @param {Int32Array|number[]} tokens.molType
 * @param {Int32Array|number[]} tokens.atomNum atoms the token contributes
 * @param {ArrayLike<number|boolean>} tokens.affinityMask the binder chain's tokens
 * @param {ArrayLike<number|boolean>} tokens.resolvedMask
 * @param {{neighbourhoodSize?: number, maxTokensProtein?: number,
 *          maxTokens?: number, maxAtoms?: number|null}} [options]
 * @returns {{kept: number[], atoms: number, proteinKept: number,
 *            stoppedAt: number|null, ranked: number[]}}
 *   `kept` is ascending token indices, as upstream's `sorted(cropped)`.
 */
export function affinityCrop(tokens, options = {}) {
  const neighbourhood = options.neighbourhoodSize ?? NEIGHBOURHOOD_SIZE;
  const maxTokensProtein = options.maxTokensProtein ?? MAX_TOKENS_PROTEIN;
  const maxTokens = options.maxTokens ?? CROP_MAX_TOKENS;
  const maxAtoms = options.maxAtoms === undefined ? CROP_MAX_ATOMS : options.maxAtoms;

  const count = tokens.asymId.length;
  // 🔴 A NAMED LIST, NOT EVERYTHING IN THE OBJECT. Validating every key meant a
  // caller could not carry a diagnostic alongside the arrays - `cropTokensFrom`
  // reports how the centres were resolved, and that scalar failed the length
  // check with a message about "undefined entries". A validator that constrains
  // its caller's bookkeeping is validating the wrong thing.
  const REQUIRED = ["asymId", "resIdx", "molType", "atomNum", "affinityMask",
                    "resolvedMask"];
  for (const name of [...REQUIRED, "centreCoords"]) {
    const array = tokens[name];
    const expected = name === "centreCoords" ? count * 3 : count;
    if (array === undefined || array === null) {
      throw new RangeError(`affinityCrop: ${name} is absent`);
    }
    if (array.length !== expected) {
      throw new RangeError(`affinityCrop: ${name} has ${array.length} entries,`
        + ` expected ${expected}`);
    }
  }
  const truthy = (v) => v === true || v === 1;

  const valid = [];
  for (let i = 0; i < count; i += 1) if (truthy(tokens.resolvedMask[i])) valid.push(i);
  if (valid.length === 0) throw new Error("No valid tokens in structure");

  const ligand = valid.filter((i) => truthy(tokens.affinityMask[i]));
  if (ligand.length === 0) {
    throw new Error("affinityCrop: no token carries affinityMask -"
      + " the binder chain has to be named before the pocket can be found");
  }

  // minimum distance from each resolved token's centre to any ligand centre
  const distance = new Float64Array(valid.length);
  for (let at = 0; at < valid.length; at += 1) {
    const i = valid[at];
    let best = Infinity;
    for (const j of ligand) {
      const dx = tokens.centreCoords[i * 3] - tokens.centreCoords[j * 3];
      const dy = tokens.centreCoords[i * 3 + 1] - tokens.centreCoords[j * 3 + 1];
      const dz = tokens.centreCoords[i * 3 + 2] - tokens.centreCoords[j * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < best) best = d;
    }
    distance[at] = best;
  }
  // 🔴 A STABLE ORDER, BECAUSE numpy's argsort IS STABLE AND TIES DECIDE THE
  // CROP. Every ligand token is at distance 0 from itself, so the ligand's own
  // tokens are a block of ties at the front of the ranking; an unstable sort
  // would take them in a different order and reach a different prefix before
  // the break.
  const ranked = Array.from(valid.keys())
    .sort((a, b) => (distance[a] - distance[b]) || (a - b))
    .map((at) => valid[at]);

  // chains, from the FULL table - see the header
  const chainOf = new Map();
  for (let i = 0; i < count; i += 1) {
    const key = tokens.asymId[i];
    if (!chainOf.has(key)) chainOf.set(key, []);
    chainOf.get(key).push(i);
  }
  const ligandIds = new Set();
  for (const i of valid) if (tokens.molType[i] === NONPOLYMER) ligandIds.add(i);

  const cropped = new Set();
  const croppedProtein = new Set();
  let atoms = 0;
  let stoppedAt = null;

  for (const index of ranked) {
    const chain = chainOf.get(tokens.asymId[index]);
    const residue = tokens.resIdx[index];
    let candidate;
    if (chain.length <= neighbourhood) {
      candidate = chain;
    } else {
      const window = chain.filter((i) => tokens.resIdx[i] >= residue - neighbourhood
        && tokens.resIdx[i] <= residue + neighbourhood);
      candidate = window.filter((i) => tokens.resIdx[i] === residue);
      let low = residue;
      let high = residue;
      // Grown one residue at a time, as upstream does, "to handle some edge
      // cases with non-standard chains" - and by RESIDUE rather than token, so
      // a modified residue's or a ligand's tokens always come in together.
      //
      // 🔴 AND IT STOPS WHEN THE WINDOW IS EXHAUSTED, WHERE UPSTREAM LOOPS FOR
      // EVER. `while new_tokens.size < self.neighborhood_size` has no other
      // exit, and `max_token_set` caps what growing can ever reach - so if the
      // +/-neighbourhood RESIDUE window holds fewer tokens than
      // `neighborhood_size`, the condition can never be satisfied. Isolated and
      // confirmed against upstream: residues numbered every third index give a
      // 7-token window against a neighbourhood of 10, and the loop does not
      // return.
      //
      // It is not reachable from a real input - `res_idx` is the tokeniser's
      // own per-chain index and so contiguous, which makes the window at least
      // 21 tokens wherever this branch is taken at all (a chain of 10 or fewer
      // takes the whole-chain branch above). A synthetic case with a stride
      // reached it, which is why the oracle corpus does not carry one and why
      // this guard is here: a port that hangs where upstream hangs is not
      // bug-compatible, it is unusable, and the crop is on the path of every
      // ligand in a screen.
      while (candidate.length < neighbourhood) {
        low -= 1;
        high += 1;
        const grown = window.filter((i) => tokens.resIdx[i] >= low
          && tokens.resIdx[i] <= high);
        // Nothing new AND the window is already fully covered: no further
        // expansion can change the answer.
        if (grown.length === candidate.length && grown.length === window.length) break;
        candidate = grown;
      }
    }

    const fresh = candidate.filter((i) => !cropped.has(i));
    let freshAtoms = 0;
    for (const i of fresh) freshAtoms += tokens.atomNum[i];
    const freshProtein = fresh.filter((i) => !ligandIds.has(i));
    const proteinAfter = new Set(croppedProtein);
    for (const i of freshProtein) proteinAfter.add(i);

    // 🔴 BREAK, NOT CONTINUE. See the header: the crop is a prefix of the
    // distance ordering.
    if (fresh.length > maxTokens - cropped.size
      || (maxAtoms !== null && atoms + freshAtoms > maxAtoms)
      || proteinAfter.size > maxTokensProtein) {
      stoppedAt = index;
      break;
    }

    for (const i of fresh) cropped.add(i);
    atoms += freshAtoms;
    for (const i of freshProtein) croppedProtein.add(i);
  }

  return {
    kept: [...cropped].sort((a, b) => a - b),
    atoms,
    proteinKept: croppedProtein.size,
    stoppedAt,
    ranked,
  };
}

/**
 * Bonds that survive the crop: both endpoints kept.
 * Upstream filters `token_1` and `token_2` with `np.isin` against the kept set.
 */
export function cropBonds(bonds, kept) {
  const keep = new Set(kept);
  return bonds.filter((bond) => keep.has(bond.from) && keep.has(bond.to));
}
