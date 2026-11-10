/**
 * Boltz-2's affinity module, in float64, as the reference its kernels are held to.
 *
 * A port of upstream `boltz/model/modules/affinity.py` (AffinityModule and
 * AffinityHeadsTransformer) with `layers/pairformer.py`'s PairformerNoSeq,
 * `layers/triangular_mult.py`, `layers/triangular_attention/`,
 * `layers/transition.py` and `modules/encodersv2.py`'s PairwiseConditioning.
 * `AGENTS.md` requires every GPU kernel to have a deterministic differential
 * test against an independent reference; this is that reference.
 *
 * WHAT IS DELIBERATELY NOT HERE, because upstream carries it and never runs it:
 *   * `GaussianSmearing` - defined in affinity.py, never instantiated.
 *   * `use_cross_transformer` - a constructor argument nothing reads.
 *   * `groups` - threaded through and unused.
 *   * `transformer_args.num_blocks` (12) and `num_heads` (8) - the checkpoint
 *     has NO tensors for them. Only `token_s` is read. "AffinityHeadsTransformer"
 *     contains no transformer; it is four MLPs over a pooled vector, and a port
 *     that believed the name would build twelve blocks that do not exist.
 *   * dropout - `get_dropout_mask` computes `dropout * training`, so at
 *     inference the mask is exactly 1.0. There is nothing to port.
 *
 * 🔴 CONVENTIONS THAT A PLAUSIBLE PORT GETS WRONG, each measured against the
 * oracle rather than assumed:
 *
 *   * `s_to_z_prod_in1/in2` IS A SUM, NOT A PRODUCT, whatever `prod` says in
 *     the name: two independent projections of `s_inputs` broadcast along
 *     opposite axes and ADDED.
 *   * THE DISTOGRAM BUCKETS WITH A STRICT `>`, over 63 boundaries
 *     (`linspace(2, 22, 63)`), giving 64 bins. `>=` moves every distance that
 *     lands on an edge.
 *   * `pairwise_conditioner` IS ADDED TO ITS OWN INPUT: the call site is
 *     `z = z + conditioner(z, distogram)` and the conditioner internally
 *     residuals its two transitions as well. Two residuals, not one.
 *   * THE POOLING MASK EXCLUDES THE DIAGONAL and the pairformer's does not.
 *     Same expression, one `* (1 - eye)` apart, and using one for the other
 *     changes `g` on every ligand of more than one token.
 *   * ATTENTION IS OVER THE SECOND PAIR AXIS, PER ROW OF THE FIRST, and the
 *     ending-node variant transposes the pair in and out rather than changing
 *     the kernel.
 *   * THE TRIANGLE BIAS INDEXES (head, query, key) WHERE QUERY IS THE ROW's
 *     OWN J. `tri_bias` is `[H, I, J]` and broadcasts into `[*, I, H, Q, K]`
 *     with its `I` against `Q` - so the bias is shared across the attended
 *     row index, not indexed by it.
 *
 * Every tensor is a flat Float64Array in row-major order; a weight `W` of
 * shape [out, in] is indexed `W[row * inDim + column]`, which is torch's own
 * layout, so `y = x @ W.T + b`.
 */

const EPS_LAYER_NORM = 1e-5;        // torch's nn.LayerNorm default, as constructed
const EPS_POOL = 1e-7;              // the heads' masked-mean denominator
const ATTENTION_INF = 1e9;          // TriangleAttention(inf=1e9)

/** Shape check at a public boundary, as AGENTS.md asks for. */
function need(array, length, what) {
  if (!(array instanceof Float64Array) && !Array.isArray(array)) {
    throw new TypeError(`${what}: expected an array, got ${typeof array}`);
  }
  if (array.length !== length) {
    throw new RangeError(`${what}: expected ${length} values, got ${array.length}`);
  }
  return array;
}

/** y[r, o] = sum_i x[r, i] * w[o, i] (+ b[o]) */
export function linear(x, rows, inDim, w, outDim, b = null) {
  need(x, rows * inDim, "linear input");
  need(w, outDim * inDim, "linear weight");
  if (b !== null) need(b, outDim, "linear bias");
  const out = new Float64Array(rows * outDim);
  for (let r = 0; r < rows; r += 1) {
    const xr = r * inDim;
    const or = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      let sum = b === null ? 0 : b[o];
      const wr = o * inDim;
      for (let i = 0; i < inDim; i += 1) sum += x[xr + i] * w[wr + i];
      out[or + o] = sum;
    }
  }
  return out;
}

/** torch's nn.LayerNorm over the last axis: BIASED variance, then scale+shift. */
export function layerNorm(x, rows, dim, w, b, eps = EPS_LAYER_NORM) {
  need(x, rows * dim, "layerNorm input");
  need(w, dim, "layerNorm weight");
  const out = new Float64Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const at = r * dim;
    let mean = 0;
    for (let c = 0; c < dim; c += 1) mean += x[at + c];
    mean /= dim;
    let variance = 0;
    for (let c = 0; c < dim; c += 1) {
      const d = x[at + c] - mean;
      variance += d * d;
    }
    variance /= dim;                        // biased, as torch computes it
    const scale = 1 / Math.sqrt(variance + eps);
    for (let c = 0; c < dim; c += 1) {
      out[at + c] = (x[at + c] - mean) * scale * w[c] + (b === null ? 0 : b[c]);
    }
  }
  return out;
}

export const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const silu = (v) => v * sigmoid(v);
const relu = (v) => (v > 0 ? v : 0);

/**
 * upstream's Transition: fc3( silu(fc1(norm(x))) * fc2(norm(x)) ), no biases.
 * @param {{normWeight, normBias, fc1, fc2, fc3}} w
 */
export function transition(x, rows, dim, hidden, w, outDim = dim) {
  const normed = layerNorm(x, rows, dim, w.normWeight, w.normBias);
  const a = linear(normed, rows, dim, w.fc1, hidden);
  const b = linear(normed, rows, dim, w.fc2, hidden);
  const gated = new Float64Array(rows * hidden);
  for (let i = 0; i < gated.length; i += 1) gated[i] = silu(a[i]) * b[i];
  return linear(gated, rows, hidden, w.fc3, outDim);
}

/**
 * TriangleMultiplication{Outgoing,Incoming}.
 *
 * 🔴 THE TWO DIFFER ONLY IN WHICH PAIR INDEX IS SUMMED OVER, and nothing else:
 * outgoing is `bikd,bjkd->bijd` and incoming is `bkid,bkjd->bijd`. Getting it
 * backwards returns a tensor of the right shape and magnitude.
 */
export function triangleMultiplication(z, n, cz, w, mask, outgoing) {
  need(z, n * n * cz, "triangle input");
  need(mask, n * n, "triangle mask");
  const xIn = layerNorm(z, n * n, cz, w.normInWeight, w.normInBias);
  const projected = linear(xIn, n * n, cz, w.pIn, 2 * cz);
  const gate = linear(xIn, n * n, cz, w.gIn, 2 * cz);
  // p_in(x) * g_in(x).sigmoid(), then the pair mask
  const x = new Float64Array(n * n * 2 * cz);
  for (let pair = 0; pair < n * n; pair += 1) {
    const m = mask[pair];
    const at = pair * 2 * cz;
    for (let c = 0; c < 2 * cz; c += 1) {
      x[at + c] = projected[at + c] * sigmoid(gate[at + c]) * m;
    }
  }
  // a = x[..., :cz], b = x[..., cz:]
  const product = new Float64Array(n * n * cz);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const out = (i * n + j) * cz;
      for (let k = 0; k < n; k += 1) {
        const aAt = (outgoing ? (i * n + k) : (k * n + i)) * 2 * cz;
        const bAt = (outgoing ? (j * n + k) : (k * n + j)) * 2 * cz;
        for (let d = 0; d < cz; d += 1) {
          product[out + d] += x[aAt + d] * x[bAt + cz + d];
        }
      }
    }
  }
  const normedOut = layerNorm(product, n * n, cz, w.normOutWeight, w.normOutBias);
  const pOut = linear(normedOut, n * n, cz, w.pOut, cz);
  const gOut = linear(xIn, n * n, cz, w.gOut, cz);
  const result = new Float64Array(n * n * cz);
  for (let i = 0; i < result.length; i += 1) result[i] = pOut[i] * sigmoid(gOut[i]);
  return result;
}

/** Transpose a [n, n, c] pair tensor on its two leading axes. */
function transposePair(z, n, c) {
  const out = new Float64Array(n * n * c);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const from = (j * n + i) * c;
      const to = (i * n + j) * c;
      for (let d = 0; d < c; d += 1) out[to + d] = z[from + d];
    }
  }
  return out;
}

function transposeMask(mask, n) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) out[i * n + j] = mask[j * n + i];
  }
  return out;
}

/**
 * TriangleAttention{StartingNode,EndingNode}.
 *
 * 🔴 THE ENDING NODE IS THE STARTING NODE ON A TRANSPOSED PAIR. Upstream
 * transposes in, runs one kernel, and transposes out; there is no second
 * arithmetic. A port that instead swapped indices inside the kernel has to get
 * the bias and the mask right too, and they do not transpose together.
 */
export function triangleAttention(z, n, cz, w, mask, heads, headDim, starting) {
  need(z, n * n * cz, "triangle attention input");
  const pair = starting ? z : transposePair(z, n, cz);
  const pairMask = starting ? mask : transposeMask(mask, n);
  const x = layerNorm(pair, n * n, cz, w.layerNormWeight, w.layerNormBias);
  // triangle bias: linear(x) is [i, j, heads] -> read as bias[h][q][k]
  const biasRaw = linear(x, n * n, cz, w.linear, heads);
  const inner = heads * headDim;
  const q = linear(x, n * n, cz, w.q, inner);
  const k = linear(x, n * n, cz, w.k, inner);
  const v = linear(x, n * n, cz, w.v, inner);
  const g = linear(x, n * n, cz, w.g, inner);
  const scale = 1 / Math.sqrt(headDim);
  const gathered = new Float64Array(n * n * inner);
  const logits = new Float64Array(n);
  for (let row = 0; row < n; row += 1) {
    for (let head = 0; head < heads; head += 1) {
      const headOff = head * headDim;
      for (let qj = 0; qj < n; qj += 1) {
        const qAt = (row * n + qj) * inner + headOff;
        let max = -Infinity;
        for (let kj = 0; kj < n; kj += 1) {
          const kAt = (row * n + kj) * inner + headOff;
          let dot = 0;
          for (let d = 0; d < headDim; d += 1) dot += q[qAt + d] * k[kAt + d];
          // mask_bias = inf * (mask - 1): 0 where live, -1e9 where not
          const maskBias = ATTENTION_INF * (pairMask[row * n + kj] - 1);
          // 🔴 the triangle bias's own I axis lands on the QUERY, and it is
          // shared across `row` - see this file's header.
          const tri = biasRaw[(qj * n + kj) * heads + head];
          const value = dot * scale + maskBias + tri;
          logits[kj] = value;
          if (value > max) max = value;
        }
        let total = 0;
        for (let kj = 0; kj < n; kj += 1) {
          logits[kj] = Math.exp(logits[kj] - max);
          total += logits[kj];
        }
        const outAt = (row * n + qj) * inner + headOff;
        for (let kj = 0; kj < n; kj += 1) {
          const weight = logits[kj] / total;
          if (weight === 0) continue;
          const vAt = (row * n + kj) * inner + headOff;
          for (let d = 0; d < headDim; d += 1) {
            gathered[outAt + d] += weight * v[vAt + d];
          }
        }
      }
    }
  }
  // _wrap_up: gate with sigmoid(linear_g(q_x)), then linear_o
  for (let i = 0; i < gathered.length; i += 1) gathered[i] *= sigmoid(g[i]);
  const out = linear(gathered, n * n, inner, w.o, cz);
  return starting ? out : transposePair(out, n, cz);
}

/** One PairformerNoSeqLayer: four pair ops and a transition, all residual. */
export function pairformerNoSeqLayer(z, n, cz, w, mask, heads, headDim) {
  let pair = z;
  const add = (update) => {
    const next = new Float64Array(n * n * cz);
    for (let i = 0; i < next.length; i += 1) next[i] = pair[i] + update[i];
    pair = next;
  };
  add(triangleMultiplication(pair, n, cz, w.triMulOut, mask, true));
  add(triangleMultiplication(pair, n, cz, w.triMulIn, mask, false));
  add(triangleAttention(pair, n, cz, w.triAttStart, mask, heads, headDim, true));
  add(triangleAttention(pair, n, cz, w.triAttEnd, mask, heads, headDim, false));
  add(transition(pair, n * n, cz, cz * 4, w.transitionZ));
  return pair;
}

/**
 * PairwiseConditioning (Algorithm 21), num_transitions = 2.
 *
 * LayerNorm over the CONCATENATION of z and the distogram embedding, then one
 * LinearNoBias back down, then two residual transitions.
 */
export function pairwiseConditioning(z, distogram, n, cz, w) {
  const joined = new Float64Array(n * n * 2 * cz);
  for (let pair = 0; pair < n * n; pair += 1) {
    const to = pair * 2 * cz;
    for (let c = 0; c < cz; c += 1) {
      joined[to + c] = z[pair * cz + c];
      joined[to + cz + c] = distogram[pair * cz + c];
    }
  }
  const normed = layerNorm(joined, n * n, 2 * cz, w.initNormWeight,
                           w.initNormBias);
  let pair = linear(normed, n * n, 2 * cz, w.initProject, cz);
  for (const t of w.transitions) {
    const update = transition(pair, n * n, cz, 2 * cz, t);
    const next = new Float64Array(n * n * cz);
    for (let i = 0; i < next.length; i += 1) next[i] = pair[i] + update[i];
    pair = next;
  }
  return pair;
}

/**
 * The distogram: representative-atom distances, bucketed, embedded.
 *
 * 🔴 THE DISTANCE IS COMPUTED DIRECTLY, NOT BY A MATMUL EXPANSION. upstream
 * reaches it through `torch.cdist`, whose |a-b|^2 expansion cancels
 * catastrophically near zero - measured at 4.77e-07 on the DIAGONAL, which
 * should be exactly 0. Both bucket the same provided no distance sits on an
 * edge, which is what the oracle's input is built to guarantee and what the
 * gate asserts rather than assumes.
 */
export function distogramBins(positions, n, boundaries) {
  need(positions, n * 3, "positions");
  const bins = new Int32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      let bin = 0;
      // (d > boundaries).sum(): a STRICT comparison, 63 edges, 64 bins
      for (let e = 0; e < boundaries.length; e += 1) if (d > boundaries[e]) bin += 1;
      bins[i * n + j] = bin;
    }
  }
  return bins;
}

/**
 * The cross-pair mask: ligand-receptor, receptor-ligand and ligand-ligand.
 * @param {boolean} excludeDiagonal the HEADS' mask does; the pairformer's does not
 */
export function crossPairMask(molType, affinityMask, padMask, n,
                              excludeDiagonal = false) {
  const receptor = new Float64Array(n);
  const ligand = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    receptor[i] = (molType[i] === 0 ? 1 : 0) * padMask[i];
    ligand[i] = (affinityMask[i] > 0 ? 1 : 0) * padMask[i];
  }
  const mask = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      let value = ligand[i] * receptor[j] + receptor[i] * ligand[j]
        + ligand[i] * ligand[j];
      if (excludeDiagonal && i === j) value *= 0;
      mask[i * n + j] = value;
    }
  }
  return mask;
}

/** AffinityHeadsTransformer: a masked mean and four MLPs. No transformer. */
export function affinityHeads(z, n, cz, tokenS, w, poolMask) {
  let denominator = 0;
  const pooled = new Float64Array(cz);
  for (let pair = 0; pair < n * n; pair += 1) {
    const m = poolMask[pair];
    if (m === 0) continue;
    denominator += m;
    const at = pair * cz;
    for (let c = 0; c < cz; c += 1) pooled[c] += z[at + c] * m;
  }
  for (let c = 0; c < cz; c += 1) pooled[c] /= denominator + EPS_POOL;

  // affinity_out_mlp: Linear(cz, cz), ReLU, Linear(cz, tokenS), ReLU
  const h1 = linear(pooled, 1, cz, w.outMlp0Weight, cz, w.outMlp0Bias);
  for (let i = 0; i < h1.length; i += 1) h1[i] = relu(h1[i]);
  const g = linear(h1, 1, cz, w.outMlp2Weight, tokenS, w.outMlp2Bias);
  for (let i = 0; i < g.length; i += 1) g[i] = relu(g[i]);

  const head = (prefix) => {
    const a = linear(g, 1, tokenS, w[`${prefix}0Weight`], tokenS,
                     w[`${prefix}0Bias`]);
    for (let i = 0; i < a.length; i += 1) a[i] = relu(a[i]);
    const b = linear(a, 1, tokenS, w[`${prefix}2Weight`], tokenS,
                     w[`${prefix}2Bias`]);
    for (let i = 0; i < b.length; i += 1) b[i] = relu(b[i]);
    return linear(b, 1, tokenS, w[`${prefix}4Weight`], 1, w[`${prefix}4Bias`])[0];
  };

  const predValue = head("predValue");
  const predScore = head("predScore");
  const logitsBinary = predScore * w.logitsWeight[0] + w.logitsBias[0];
  return {
    pooled, g, predValue, predScore, logitsBinary,
    probabilityBinary: sigmoid(logitsBinary),
  };
}

/**
 * The whole module, with every seam returned so a gate can localise a fault.
 *
 * @param {object} input `{ sInputs, zAffinity, positions, molType,
 *   affinityMask, padMask, tokens, tokenS, tokenZ, boundaries }`
 * @param {object} weights as `affinityWeights()` lays them out
 * @returns {{seams: Record<string, Float64Array>, out: object}}
 */
export function affinityModule(input, weights) {
  const n = input.tokens;
  const cz = input.tokenZ;
  const cs = input.tokenS;
  const seams = {};

  const zNorm = layerNorm(input.zAffinity, n * n, cz,
                          weights.zNormWeight, weights.zNormBias);
  seams.z_norm = zNorm;
  const zLinear = linear(zNorm, n * n, cz, weights.zLinear, cz);
  seams.z_linear = zLinear;

  const s1 = linear(input.sInputs, n, cs, weights.sToZ1, cz);
  const s2 = linear(input.sInputs, n, cs, weights.sToZ2, cz);
  seams.s_to_z_1 = s1;
  seams.s_to_z_2 = s2;

  // 🔴 A SUM over the two axes, despite `prod` in the upstream name.
  const zSummed = new Float64Array(n * n * cz);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const at = (i * n + j) * cz;
      for (let c = 0; c < cz; c += 1) {
        zSummed[at + c] = zLinear[at + c] + s1[i * cz + c] + s2[j * cz + c];
      }
    }
  }
  seams.z_plus_s = zSummed;

  const bins = distogramBins(input.positions, n, input.boundaries);
  seams.bins = Float64Array.from(bins);
  const embedded = new Float64Array(n * n * cz);
  for (let pair = 0; pair < n * n; pair += 1) {
    const row = bins[pair] * cz;
    const at = pair * cz;
    for (let c = 0; c < cz; c += 1) embedded[at + c] = weights.binEmbed[row + c];
  }
  seams.distogram_embed = embedded;

  const conditioned = pairwiseConditioning(zSummed, embedded, n, cz,
                                           weights.conditioner);
  seams.conditioner_out = conditioned;
  let pair = new Float64Array(n * n * cz);
  for (let i = 0; i < pair.length; i += 1) pair[i] = zSummed[i] + conditioned[i];
  seams.z_after_conditioner = pair;

  const mask = crossPairMask(input.molType, input.affinityMask, input.padMask,
                             n, false);
  seams.cross_pair_mask = mask;
  for (let layer = 0; layer < weights.pairformer.length; layer += 1) {
    pair = pairformerNoSeqLayer(pair, n, cz, weights.pairformer[layer], mask,
                                weights.heads, weights.headDim);
    seams[`pairformer${String(layer).padStart(2, "0")}`] = pair;
  }

  // 🔴 THE HEADS' MASK EXCLUDES THE DIAGONAL; the pairformer's does not.
  const poolMask = crossPairMask(input.molType, input.affinityMask,
                                 input.padMask, n, true);
  const out = affinityHeads(pair, n, cz, cs, weights.heads2, poolMask);
  seams.pooled = out.pooled;
  seams.out_mlp = out.g;
  return { seams, out };
}
