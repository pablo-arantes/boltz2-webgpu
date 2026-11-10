/**
 * The token CENTRE atom, which is not the token's pseudo-beta.
 *
 * 🔴 TWO GATHERS, AND THIS PORT ONLY HAD ONE. upstream keeps both per token -
 * `center_idx` from `const.res_to_center_atom_id` and `disto_idx` from
 * `res_to_disto_atom_id` - and uses them for different things:
 *
 *     the affinity CROP ranks on `center_coords`   (data/crop/affinity.py)
 *     the affinity MODULE's distances use `disto`  (token_to_rep_atom)
 *
 * Counted off upstream's own tables, the two **differ on 28 of 31 residue
 * types** and agree only on GLY, N and DN:
 *
 *     centre   CA for every amino acid including UNK and GLY;  C1' for a base
 *     disto    CB for every amino acid EXCEPT GLY (CA);  C4 / C2 for a base
 *
 * So using the pseudo-beta gather to rank the crop is wrong nearly everywhere,
 * not in a corner - it ranks protein residues by their CB where upstream ranks
 * them by their CA, which reorders the neighbourhood and so changes which
 * pocket pass 2 sees. `web/affinity-passes.js` said so in a comment; this is
 * the fix.
 *
 * 🔴 AND IT IS DERIVED FROM THE BATCH, NOT ADDED TO THE FEATURISER. Everything
 * needed is already there - `refAtomNameChars`, `refMask`, `dense` and the
 * chain kinds - so a centre gather is a read over the batch rather than a new
 * field in featurise.js. That keeps it out of the path of `npm run test:batch`,
 * which holds 43 fields of that featuriser exact against AF3's own dump: a new
 * field there would be a new thing for the dump to disagree about, for a
 * quantity AF3 does not carry.
 *
 * 🔴 THE FALLBACK IS THE PSEUDO-BETA SLOT, WHICH IS CORRECT RATHER THAN LAZY.
 * For a ligand atom or a modified residue upstream sets
 * `atom_center = atom_disto = 0` (parse_ccd_residue, and schema.py's own
 * "Placeholder, no center"), so centre and disto genuinely coincide there - and
 * this port's pseudo-beta slot for those tokens is already the residue's first
 * held atom. Falling back is therefore agreement, not approximation.
 */
import { atomName } from "../structure/pdb.js";

/** upstream `res_to_center_atom`, by chain kind: the only two values it takes. */
const CENTRE_NAME = { protein: "CA", dna: "C1'", rna: "C1'" };

/**
 * One centre atom per token, in the same `{indices, mask}` shape as the
 * batch's own gathers.
 *
 * @param {object} batch from `featuriseProtein` / `af3BatchFromA3m`
 * @returns {{indices: Int32Array, mask: Uint8Array, fromName: number,
 *            fromPseudoBeta: number}}
 *   `fromName` counts tokens whose centre was found by name and
 *   `fromPseudoBeta` those that fell back - a caller that expects a protein
 *   target can assert the split rather than trust it.
 */
export function centreAtomGather(batch) {
  const tokens = batch.tokens;
  const dense = batch.dense;
  if (!Number.isInteger(tokens) || !Number.isInteger(dense)) {
    throw new RangeError(`centreAtomGather: tokens ${tokens}, dense ${dense}`);
  }
  const pseudo = batch.tokenAtomsToPseudoBeta;
  if (pseudo === undefined) {
    throw new Error("centreAtomGather: the batch has no pseudo-beta gather to"
      + " fall back to; it is what a ligand or modified token's centre IS");
  }
  const indices = new Int32Array(tokens);
  const mask = new Uint8Array(tokens);
  let fromName = 0;
  let fromPseudoBeta = 0;

  for (let token = 0; token < tokens; token += 1) {
    // the fallback first, so an early `continue` is always safe
    indices[token] = Number(pseudo.indices[token]);
    mask[token] = pseudo.mask[token] ? 1 : 0;

    // a ligand or atomised token has no named centre: upstream's is atom 0,
    // which is what the pseudo-beta slot already holds here
    if (batch.isLigand?.[token]) { fromPseudoBeta += 1; continue; }
    const residue = batch.residueOfToken?.[token] ?? token;
    if (residue < 0) { fromPseudoBeta += 1; continue; }
    const kind = batch.chainKinds?.[batch.chainOfResidue?.[residue]] ?? "protein";
    const wanted = CENTRE_NAME[kind];
    if (wanted === undefined) { fromPseudoBeta += 1; continue; }

    let found = -1;
    for (let slot = 0; slot < dense; slot += 1) {
      if (!batch.refMask[token * dense + slot]) continue;
      if (atomName(batch.refAtomNameChars, token * dense + slot) === wanted) {
        found = slot;
        break;
      }
    }
    if (found >= 0) {
      indices[token] = token * dense + found;
      mask[token] = 1;
      fromName += 1;
    } else {
      // 🔴 A MODIFIED RESIDUE INSIDE A POLYMER LANDS HERE and must not be
      // silently given its CB. Its centre is its first held atom, which is the
      // pseudo-beta slot already written above - the same thing upstream's
      // `atom_center = 0` means.
      fromPseudoBeta += 1;
    }
  }
  return { indices, mask, fromName, fromPseudoBeta };
}

/**
 * Centre coordinates per token, which is what the crop ranks on.
 *
 * @param {object} batch
 * @param {ArrayLike<number>} positions flat xyz per dense atom slot
 */
export function centreCoords(batch, positions) {
  const gather = centreAtomGather(batch);
  const out = new Float64Array(batch.tokens * 3);
  for (let token = 0; token < batch.tokens; token += 1) {
    if (!gather.mask[token]) continue;
    const from = gather.indices[token] * 3;
    for (let axis = 0; axis < 3; axis += 1) out[token * 3 + axis] = positions[from + axis];
  }
  return { coords: out, ...gather };
}
