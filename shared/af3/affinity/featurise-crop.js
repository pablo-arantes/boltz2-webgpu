/**
 * Pass 1's batch, cropped to the pocket pass 2 runs on.
 *
 * 🔴 upstream CROPS THE TOKEN TABLE; IT DOES NOT REBUILD A SEQUENCE. That is
 * why this exists rather than a second call to `featuriseProtein`: a crop keeps
 * a residue NEIGHBOURHOOD per chain, so the protein it leaves is generally not
 * contiguous and has no faithful sequence form. Joining the kept residues into
 * a string would imply a peptide bond across every gap.
 *
 * 🔴 `residueIndex` IS PRESERVED AND `tokenIndex` IS RENUMBERED. Measured off
 * upstream's own featuriser (tools/oracle/dump_boltz2_featurise.py), on a crop
 * that drops residues 1, 4 and 5:
 *
 *     keptTokenIdx    [0, 2, 3, 6, 7, 8, 9, 10]
 *     token_index     [0, 1, 2, 3, 4, 5, 6, 7]     <- dense, renumbered
 *     residue_index   [0, 2, 3, 6, 0, 0, 0, 0]     <- ORIGINAL, gaps kept
 *
 * The gaps are the point. The relative-position encoding reads differences of
 * `residue_index`, so keeping them is what tells the model those residues are
 * not adjacent; renumbering would fuse the crop into a contiguous chain, which
 * is a different protein that folds perfectly well.
 *
 * 🔴 AND THE GATHERS ARE RE-DERIVED BY `atomGathers`, NOT REWRITTEN HERE.
 * That function was extracted "so the structural tokeniser can reuse it rather
 * than copy it", and its own header says why a second copy is dangerous: the
 * windowing rule is delicate enough that docs/ records FOUR OF SEVEN MODELS
 * having had the wrong one, with three distinct conventions living in the gap.
 * A crop changes the token count and the atom ordering, which is exactly the
 * input that function takes - so the right move is to call it again, not to
 * reimplement a window.
 *
 * 🔴 `refSpaceUid` KEEPS ITS ORIGINAL LABELS. It answers "are these two atoms
 * in the same reference space?", so the labels are arbitrary and only the
 * PARTITION matters - docs/ records an elementwise comparison of it reading
 * 3116 of 3120 differing while both sides agreed about every atom. Subsetting
 * leaves a valid partition with gaps in its numbering, which is all it has to be.
 */
import { atomGathers } from "../featurise/featurise.js";

/**
 * @param {object} batch from `featuriseProtein` / `af3BatchFromA3m`
 * @param {number[]|Set<number>} kept token indices to keep, any order
 * @param {{paddedAtomKeys?: boolean, qblockAtomKeys?: boolean}} [dialect]
 *   the same two flags `featuriseProtein` passes on; a crop must use the SAME
 *   window convention as the fold it came from or the two disagree about which
 *   atoms attend
 * @returns {object} a batch of the same shape over the kept tokens
 */
export function featuriseCrop(batch, kept, dialect = {}) {
  const order = [...(kept instanceof Set ? [...kept] : kept)]
    .map((v) => Number(v))
    .sort((a, b) => a - b);
  if (order.length === 0) throw new RangeError("featuriseCrop: nothing kept");
  const seen = new Set(order);
  if (seen.size !== order.length) {
    throw new RangeError("featuriseCrop: the kept list repeats a token");
  }
  for (const token of order) {
    if (!Number.isInteger(token) || token < 0 || token >= batch.tokens) {
      throw new RangeError(`featuriseCrop: token ${token} is outside`
        + ` 0..${batch.tokens - 1}`);
    }
  }

  const tokens = order.length;
  const dense = batch.dense;
  const out = {};

  // --- per-token vectors ---------------------------------------------------
  const perToken = ["aatype", "deletionMean", "residueIndex", "asymId",
    "entityId", "symId", "seqMask", "residueOfToken", "isDna", "isRna",
    "isLigand", "isModified", "cyclicPeriod", "method"];
  for (const field of perToken) {
    const source = batch[field];
    if (source === undefined || source === null) continue;
    if (source.length !== batch.tokens) continue;
    const target = new source.constructor(tokens);
    for (let t = 0; t < tokens; t += 1) target[t] = source[order[t]];
    out[field] = target;
  }
  // 🔴 RENUMBERED, where residueIndex above is not - see the header.
  // One-based, as this port's indices are.
  out.tokenIndex = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) out.tokenIndex[t] = t + 1;

  // --- per-token x width ---------------------------------------------------
  const strided = [["profile", batch.profile ? batch.profile.length / batch.tokens : 0],
    ["refMask", dense], ["refElement", dense], ["refCharge", dense],
    ["refSpaceUid", dense], ["predDenseAtomMask", dense],
    ["refPos", dense * 3], ["refAtomNameChars", dense * 4],
    ["displayAtomNameChars", dense * 4]];
  for (const [field, width] of strided) {
    const source = batch[field];
    if (source === undefined || source === null || width === 0) continue;
    if (source.length !== batch.tokens * width) continue;
    const target = new source.constructor(tokens * width);
    for (let t = 0; t < tokens; t += 1) {
      target.set(source.subarray(order[t] * width, (order[t] + 1) * width),
                 t * width);
    }
    out[field] = target;
  }

  // --- the MSA: rows x tokens, so a column subset --------------------------
  const rows = batch.msa === undefined ? 0 : batch.msa.length / batch.tokens;
  for (const field of ["msa", "msaMask", "deletionMatrix"]) {
    const source = batch[field];
    if (source === undefined || source === null) continue;
    const target = new source.constructor(rows * tokens);
    for (let row = 0; row < rows; row += 1) {
      for (let t = 0; t < tokens; t += 1) {
        target[row * tokens + t] = source[row * batch.tokens + order[t]];
      }
    }
    out[field] = target;
  }

  // --- pair-shaped: both axes subset the same way --------------------------
  for (const field of ["bondMatrix", "bondOrderMatrix", "declaredBondMatrix"]) {
    const source = batch[field];
    if (source === undefined || source === null) continue;
    if (source.length !== batch.tokens * batch.tokens) continue;
    const target = new source.constructor(tokens * tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        target[i * tokens + j] = source[order[i] * batch.tokens + order[j]];
      }
    }
    out[field] = target;
  }

  // --- the gathers, RE-DERIVED by the one function that knows the rule -----
  const realAtoms = [];
  const pseudoBetaSlot = new Int32Array(tokens).fill(-1);
  const refMask = out.refMask ?? batch.refMask;
  for (let t = 0; t < tokens; t += 1) {
    for (let slot = 0; slot < dense; slot += 1) {
      if (refMask[t * dense + slot]) realAtoms.push(t * dense + slot);
    }
    // the kept token's own pseudo-beta slot, as an offset inside its row
    const original = Number(batch.tokenAtomsToPseudoBeta.indices[order[t]])
      - order[t] * dense;
    pseudoBetaSlot[t] = batch.tokenAtomsToPseudoBeta.mask[order[t]]
      ? original : -1;
  }
  const gathers = atomGathers({
    tokens, dense, realAtoms, pseudoBetaSlot,
    paddedKeys: dialect.paddedAtomKeys === true,
    qblockKeys: dialect.qblockAtomKeys === true,
  });
  for (const [field, value] of Object.entries(gathers)) out[field] = value;

  // --- spans, renumbered onto the kept order ------------------------------
  const position = new Map();
  for (let t = 0; t < tokens; t += 1) position.set(order[t], t);
  for (const field of ["ligandSpans", "modifiedSpans"]) {
    const source = batch[field];
    if (!Array.isArray(source)) continue;
    const target = [];
    for (const span of source) {
      // 🔴 A SPAN SURVIVES ONLY IF IT IS WHOLE. A ligand half-cropped would
      // give the model part of a molecule under a name that says otherwise,
      // and the crop keeps a ligand entire by construction - so a partial span
      // is a bug in the caller's kept set, not something to paper over.
      const members = [];
      for (let offset = 0; offset < span.count; offset += 1) {
        const at = position.get(span.from + offset);
        if (at !== undefined) members.push(at);
      }
      if (members.length === 0) continue;
      if (members.length !== span.count) {
        throw new Error(`featuriseCrop: ${field} ${span.code} is ${span.count}`
          + ` tokens and only ${members.length} survive the crop; a component`
          + " has to be kept whole");
      }
      for (let i = 1; i < members.length; i += 1) {
        if (members[i] !== members[i - 1] + 1) {
          throw new Error(`featuriseCrop: ${field} ${span.code} is not`
            + " contiguous after the crop");
        }
      }
      target.push({ ...span, from: members[0] });
    }
    out[field] = target;
  }

  // --- residue-level and scalar -------------------------------------------
  // 🔴 `sequence` IS REBUILT FROM THE KEPT RESIDUES, and it is a LABEL here
  // rather than an input: the PDB writer reads it to name a residue, and
  // `residueOfToken` is what indexes into it. Carrying pass 1's string across
  // would name every residue after the first gap wrongly - the same fault
  // docs/ records for a modified residue ("a chain that reads as a real
  // protein and is not the one that was folded").
  const keptResidues = [];
  const residueSeen = new Set();
  for (const token of order) {
    const residue = batch.residueOfToken?.[token] ?? -1;
    if (residue >= 0 && !residueSeen.has(residue)) {
      residueSeen.add(residue);
      keptResidues.push(residue);
    }
  }
  const residueAt = new Map(keptResidues.map((r, i) => [r, i]));
  out.sequence = keptResidues
    .map((residue) => batch.sequence?.[residue] ?? "X").join("");
  if (batch.chainOfResidue !== undefined) {
    out.chainOfResidue = new batch.chainOfResidue.constructor(keptResidues.length);
    for (let i = 0; i < keptResidues.length; i += 1) {
      out.chainOfResidue[i] = batch.chainOfResidue[keptResidues[i]];
    }
  }
  // ...and `residueOfToken` is re-pointed at the new residue numbering, since
  // it is an INDEX INTO `sequence` and that string just changed.
  if (out.residueOfToken !== undefined) {
    for (let t = 0; t < tokens; t += 1) {
      const original = batch.residueOfToken[order[t]];
      out.residueOfToken[t] = original >= 0
        ? (residueAt.get(original) ?? -1) : -1;
    }
  }

  // 🔴 `features` IS THE SAME FIVE ROWS AGAIN, AND THE EMBEDDER READS THAT
  // COPY. `Af3EmbedderGpu.run` packs its integer feature rows out of
  // `input.features`, not out of the batch's top level - so subsetting the
  // top-level vectors and leaving `features` pointing at pass 1's arrays gives
  // the trunk the UNCROPPED indices at the cropped token count, which is a
  // read past the end rather than a wrong answer only because the lengths
  // disagree. Rebuilt from the subset values, so the two cannot drift.
  if (batch.features !== undefined) {
    out.features = {};
    for (const name of Object.keys(batch.features)) {
      out.features[name] = out[name] ?? batch.features[name];
    }
  }

  out.tokens = tokens;
  out.dense = dense;
  out.chainKinds = batch.chainKinds;
  out.chains = batch.chains;
  out.chainLengths = batch.chainLengths;
  out.sequences = batch.sequences;
  out.pairedRows = batch.pairedRows;
  out.numMsa = batch.numMsa;
  // `queries` and `keys` come from the re-derived gathers, not pass 1's shape
  out.shape = { ...(batch.shape ?? {}), tokens, dense,
    subsets: gathers.subsets, atomCount: gathers.atomCount,
    keys: gathers.keys ?? batch.shape?.keys };
  out.subsets = gathers.subsets;
  out.atomCount = gathers.atomCount;
  // what was kept, so a caller can map a result back to pass 1's numbering
  out.croppedFrom = Int32Array.from(order);
  return out;
}
