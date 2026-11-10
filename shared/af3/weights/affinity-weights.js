/**
 * Boltz-2's affinity weights, from the bundle's PyTorch names into the shapes
 * the affinity modules read.
 *
 * 🔴 THE NAMES ARE UPSTREAM'S, UNCHANGED, and that is the convention rather
 * than a shortcut. tools/export_af3_model.py keeps Haiku paths because "the
 * names ARE the description... a tensor that is wrong can be found by grepping
 * for it in both places". af3-any-model has no affinity module, so for this
 * one bundle the oracle IS upstream PyTorch and its dotted paths are what
 * preserve that property.
 *
 * 🔴 AND A MISSING TENSOR THROWS, NAMING ITSELF. `weight-sources.js` records
 * the twin trap: `=== undefined` is the wrong presence test because a loader
 * writes `null`, and that null reaching a consumer killed boltz2 once already.
 * Everything here goes through `take`, which refuses both.
 */

/** The ensemble members, with the pairformer depth each was trained at. */
export const AFFINITY_MEMBERS = Object.freeze([
  Object.freeze({ scope: "affinity_module1", layers: 8 }),
  Object.freeze({ scope: "affinity_module2", layers: 4 }),
]);

/** TriangleAttention(c_in=128, c_hidden=32, no_heads=4) - from the shapes. */
export const AFFINITY_HEADS = 4;
export const AFFINITY_HEAD_DIM = 32;
export const AFFINITY_TOKEN_S = 384;
export const AFFINITY_TOKEN_Z = 128;
export const AFFINITY_DIST_BINS = 64;
export const AFFINITY_MAX_DIST = 22;

/**
 * The 63 bin edges, DERIVED rather than read.
 *
 * The checkpoint carries `boundaries` as a buffer, and it is exactly
 * `linspace(2, 22, 63)` - checked against the export at 1.9e-6, which is f32
 * rounding. Deriving it means a bundle cannot silently ship a different ramp,
 * and it is the one tensor here whose correctness can be asserted rather than
 * trusted.
 */
export function affinityBoundaries(bins = AFFINITY_DIST_BINS,
                                   maxDist = AFFINITY_MAX_DIST) {
  const count = bins - 1;
  const out = new Float64Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = 2 + (maxDist - 2) * (i / (count - 1));
  }
  return out;
}

/**
 * @param {(name: string) => Float64Array | null | undefined} read
 *   one tensor by its full bundle name
 * @param {string} scope `affinity_module1` or `affinity_module2`
 * @param {number} layers that member's pairformer depth
 */
export function affinityWeights(read, scope, layers) {
  const take = (suffix) => {
    const name = `${scope}.${suffix}`;
    const value = read(name);
    // 🔴 null AND undefined, deliberately: see this file's header.
    if (value === null || value === undefined) {
      throw new Error(`affinity weights: ${name} is absent from this bundle`);
    }
    return value;
  };
  const transitionAt = (prefix) => ({
    normWeight: take(`${prefix}.norm.weight`),
    normBias: take(`${prefix}.norm.bias`),
    fc1: take(`${prefix}.fc1.weight`),
    fc2: take(`${prefix}.fc2.weight`),
    fc3: take(`${prefix}.fc3.weight`),
  });
  const triMul = (prefix) => ({
    normInWeight: take(`${prefix}.norm_in.weight`),
    normInBias: take(`${prefix}.norm_in.bias`),
    pIn: take(`${prefix}.p_in.weight`),
    gIn: take(`${prefix}.g_in.weight`),
    pOut: take(`${prefix}.p_out.weight`),
    gOut: take(`${prefix}.g_out.weight`),
    normOutWeight: take(`${prefix}.norm_out.weight`),
    normOutBias: take(`${prefix}.norm_out.bias`),
  });
  const triAtt = (prefix) => ({
    layerNormWeight: take(`${prefix}.layer_norm.weight`),
    layerNormBias: take(`${prefix}.layer_norm.bias`),
    linear: take(`${prefix}.linear.weight`),
    q: take(`${prefix}.mha.linear_q.weight`),
    k: take(`${prefix}.mha.linear_k.weight`),
    v: take(`${prefix}.mha.linear_v.weight`),
    g: take(`${prefix}.mha.linear_g.weight`),
    o: take(`${prefix}.mha.linear_o.weight`),
  });

  const pairformer = [];
  for (let layer = 0; layer < layers; layer += 1) {
    const at = `pairformer_stack.layers.${layer}`;
    pairformer.push({
      triMulOut: triMul(`${at}.tri_mul_out`),
      triMulIn: triMul(`${at}.tri_mul_in`),
      triAttStart: triAtt(`${at}.tri_att_start`),
      triAttEnd: triAtt(`${at}.tri_att_end`),
      transitionZ: transitionAt(`${at}.transition_z`),
    });
  }

  return {
    zNormWeight: take("z_norm.weight"),
    zNormBias: take("z_norm.bias"),
    zLinear: take("z_linear.weight"),
    sToZ1: take("s_to_z_prod_in1.weight"),
    sToZ2: take("s_to_z_prod_in2.weight"),
    binEmbed: take("dist_bin_pairwise_embed.weight"),
    conditioner: {
      initNormWeight: take("pairwise_conditioner.dim_pairwise_init_proj.0.weight"),
      initNormBias: take("pairwise_conditioner.dim_pairwise_init_proj.0.bias"),
      initProject: take("pairwise_conditioner.dim_pairwise_init_proj.1.weight"),
      transitions: [
        transitionAt("pairwise_conditioner.transitions.0"),
        transitionAt("pairwise_conditioner.transitions.1"),
      ],
    },
    pairformer,
    heads: AFFINITY_HEADS,
    headDim: AFFINITY_HEAD_DIM,
    heads2: {
      outMlp0Weight: take("affinity_heads.affinity_out_mlp.0.weight"),
      outMlp0Bias: take("affinity_heads.affinity_out_mlp.0.bias"),
      outMlp2Weight: take("affinity_heads.affinity_out_mlp.2.weight"),
      outMlp2Bias: take("affinity_heads.affinity_out_mlp.2.bias"),
      predValue0Weight: take("affinity_heads.to_affinity_pred_value.0.weight"),
      predValue0Bias: take("affinity_heads.to_affinity_pred_value.0.bias"),
      predValue2Weight: take("affinity_heads.to_affinity_pred_value.2.weight"),
      predValue2Bias: take("affinity_heads.to_affinity_pred_value.2.bias"),
      predValue4Weight: take("affinity_heads.to_affinity_pred_value.4.weight"),
      predValue4Bias: take("affinity_heads.to_affinity_pred_value.4.bias"),
      predScore0Weight: take("affinity_heads.to_affinity_pred_score.0.weight"),
      predScore0Bias: take("affinity_heads.to_affinity_pred_score.0.bias"),
      predScore2Weight: take("affinity_heads.to_affinity_pred_score.2.weight"),
      predScore2Bias: take("affinity_heads.to_affinity_pred_score.2.bias"),
      predScore4Weight: take("affinity_heads.to_affinity_pred_score.4.weight"),
      predScore4Bias: take("affinity_heads.to_affinity_pred_score.4.bias"),
      logitsWeight: take("affinity_heads.to_affinity_logits_binary.weight"),
      logitsBias: take("affinity_heads.to_affinity_logits_binary.bias"),
    },
  };
}


// --- the AF3 kernel contracts ----------------------------------------------
//
// 🔴 ONE MAPPING, TWO CONSUMERS. tools/check-affinity-kernel-contract.mjs
// verifies these against the float64 reference and webgpu/af3/affinity/ drives
// the kernels with them. docs/ records what a second copy costs: a reserved
// offset list written over one loop and read over another put rf3's bias in
// the source, in the offsets and not in the buffer, and every oracle seam
// matched the previous run to the last digit. One list for both loops.

const rowsOf = (matrix, from, count, inDim) =>
  Float32Array.from(matrix.subarray(from * inDim, (from + count) * inDim));
/** Transpose an [out, in] matrix into AF3's (in, out) layout. */
export function transposeOutIn(matrix, outDim, inDim) {
  const out = new Float32Array(matrix.length);
  for (let o = 0; o < outDim; o += 1) {
    for (let c = 0; c < inDim; c += 1) out[c * outDim + o] = matrix[o * inDim + c];
  }
  return out;
}

/**
 * Boltz's triangle weights in the AF3 triangle kernel's contract.
 *
 * 🔴 AF3 SPLITS THE TWO PROJECTIONS INTO FOUR TENSORS WHERE BOLTZ CARRIES TWO.
 * `p_in` and `g_in` are [2*cz, cz] and upstream splits them with
 * `torch.chunk(x, 2, dim=-1)` AFTER projecting, so rows 0..cz-1 are `a` and
 * cz..2cz-1 are `b`. AF3's kernel wants linearAP/AG/BP/BG separately.
 *
 * 🔴 AND THE SIX BIAS SLOTS ARE ZEROS BECAUSE BOLTZ HAS NO BIASES HERE. Every
 * projection in the affinity pairformer is a LinearNoBias, and an absent weight
 * reaching a consumer is what `weight-sources.js` records killing boltz2 once.
 *
 * 🔴 AND THE INCOMING DIRECTION SWAPS a AND b, WHICH COST A GATE TO FIND. The
 * two projects contract DIFFERENT index pairs:
 *
 *     AF3    incoming   kjc,kic->ijc     out[i][j] = sum_k a[k][j] * b[k][i]
 *     Boltz  incoming   bkid,bkjd->bijd  out[i][j] = sum_k a[k][i] * b[k][j]
 *
 * so a and b's roles are exchanged between them, and feeding Boltz's halves
 * straight through computes the transpose of the right answer. Outgoing is
 * `ikc,jkc->ijc` against `bikd,bjkd->bijd`, which DO agree - so the mistake is
 * invisible in one direction and wrong in the other.
 *
 * It was found by tools/gpu/check-affinity-pair-stack.js and NOT by
 * check-affinity-kernel-contract.mjs, whose host reference
 * (`webgpu/kernels/triangle/cpu-reference.js`) implements outgoing ONLY - a gap
 * that gate's own header declared rather than papered over. Measured:
 * `tri_mul_out` exact at 1.0e-7 while `tri_mul_in` read 0.304, and every step
 * after it inherited that.
 */
export function af3TriangleContract(tri, channels, direction = "outgoing") {
  const zeros = () => new Float32Array(channels);
  // a is rows 0..channels-1 of p_in/g_in and b is the rest - exchanged for
  // incoming, per the note above.
  const aFrom = direction === "incoming" ? channels : 0;
  const bFrom = direction === "incoming" ? 0 : channels;
  return {
    layerNormInWeight: Float32Array.from(tri.normInWeight),
    layerNormInBias: Float32Array.from(tri.normInBias),
    linearAPWeight: rowsOf(tri.pIn, aFrom, channels, channels),
    linearAPBias: zeros(),
    linearBPWeight: rowsOf(tri.pIn, bFrom, channels, channels),
    linearBPBias: zeros(),
    linearAGWeight: rowsOf(tri.gIn, aFrom, channels, channels),
    linearAGBias: zeros(),
    linearBGWeight: rowsOf(tri.gIn, bFrom, channels, channels),
    linearBGBias: zeros(),
    layerNormOutWeight: Float32Array.from(tri.normOutWeight),
    layerNormOutBias: Float32Array.from(tri.normOutBias),
    linearZWeight: Float32Array.from(tri.pOut),
    linearZBias: zeros(),
    linearGWeight: Float32Array.from(tri.gOut),
    linearGBias: zeros(),
  };
}

/**
 * Boltz's triangle-attention weights in AF3's grid-attention contract.
 *
 * 🔴 THREE OF THE SIX TENSORS NEED TRANSPOSING AND THREE DO NOT, and the split
 * is not a guess - `gridSelfAttention` reads q, k and the gate with
 * `transposed = true` and v and the output projection without, and its own
 * comment says why: "q and k carry `transpose_weights`, v does not. The
 * exported shapes say so: (heads, dimension, channels) against (channels,
 * heads, dimension)." PyTorch stores ALL of them (out, in), so v and the
 * output projection have to be turned around and q, k and the gate must not
 * be. `linear_o` is square, so getting it wrong is not a shape error.
 */
export function af3GridContract(att, channels, heads, dimension) {
  const width = heads * dimension;
  return {
    heads,
    dimension,
    actNormScale: Float32Array.from(att.layerNormWeight),
    actNormOffset: Float32Array.from(att.layerNormBias),
    // bias projection: (channels, heads) read untransposed
    pairBiasProjection: transposeOutIn(Float32Array.from(att.linear), heads,
                                       channels),
    qProjection: Float32Array.from(att.q),
    kProjection: Float32Array.from(att.k),
    gatingQuery: Float32Array.from(att.g),
    vProjection: transposeOutIn(Float32Array.from(att.v), width, channels),
    outputProjection: transposeOutIn(Float32Array.from(att.o), channels, width),
  };
}

/**
 * Boltz's Transition in AF3's contract.
 *
 * 🔴 AF3 PACKS fc1 AND fc2 INTO ONE TENSOR, GATE HALF FIRST. Its reference
 * computes `swish(wide[0..h]) * wide[h..2h]`, and Boltz computes
 * `silu(fc1(x)) * fc2(x)` - so fc1 is the GATE half and fc2 the value half,
 * in that order. Swapping them is a different function of the same shape.
 */
export function af3TransitionContract(t, channels, hidden) {
  const fc1 = Float32Array.from(t.fc1);
  const fc2 = Float32Array.from(t.fc2);
  const packed = new Float32Array(fc1.length + fc2.length);
  packed.set(fc1, 0);
  packed.set(fc2, fc1.length);
  return {
    inputLayerNormScale: Float32Array.from(t.normWeight),
    inputLayerNormOffset: Float32Array.from(t.normBias),
    transition1: transposeOutIn(packed, hidden * 2, channels),
    transition2: transposeOutIn(Float32Array.from(t.fc3), channels, hidden),
  };
}

/**
 * Boltz's triangle weights in AF3's INTERLEAVED trunk contract.
 *
 * 🔴 A THIRD LAYOUT FOR THE SAME ARITHMETIC, AND IT IS WHY THE INCOMING BUG
 * SURVIVED A GATE. There are three triangle contracts in this repository:
 *
 *   1. `webgpu/kernels/triangle/` - sixteen separate names
 *      (linearAP/AG/BP/BG...), OUTGOING only on the host.
 *   2. `cpu/af3/trunk/pairformer.js` - AF3's own, with `projection` and `gate`
 *      INTERLEAVED (even channel = a, odd = b) and BOTH directions.
 *   3. upstream Boltz - `p_in`/`g_in` CONCATENATED, both directions.
 *
 * The first gate used (1), so incoming was unreachable and `af3TriangleContract`
 * shipped with a/b the wrong way round for it until the GPU differential found
 * it. This maps to (2) so the host can check both directions, which is what
 * closes that gap.
 *
 * @param {"outgoing"|"incoming"} direction
 */
export function af3InterleavedTriangleContract(tri, channels, direction) {
  const wide = channels * 2;
  const aFrom = direction === "incoming" ? channels : 0;
  const bFrom = direction === "incoming" ? 0 : channels;
  // AF3 reads `projection` and `gate` untransposed, i.e. (in, out).
  const weave = (matrix) => {
    const out = new Float32Array(channels * wide);
    for (let c = 0; c < channels; c += 1) {
      for (let input = 0; input < channels; input += 1) {
        out[input * wide + c * 2] = matrix[(aFrom + c) * channels + input];
        out[input * wide + c * 2 + 1] = matrix[(bFrom + c) * channels + input];
      }
    }
    return out;
  };
  return {
    leftNormInputScale: Float32Array.from(tri.normInWeight),
    leftNormInputOffset: Float32Array.from(tri.normInBias),
    projection: weave(tri.pIn),
    gate: weave(tri.gIn),
    centerNormScale: Float32Array.from(tri.normOutWeight),
    centerNormOffset: Float32Array.from(tri.normOutBias),
    outputProjection: transposeOutIn(Float32Array.from(tri.pOut), channels,
                                     channels),
    gatingLinear: transposeOutIn(Float32Array.from(tri.gOut), channels, channels),
  };
}
