# Phase 2 — Boltz-2 affinity: reverse-engineering result and port plan

**Pins.**
- LocalFold: `357518613af612b4e1e887deecd93bd612fdac76` (as Phase 1).
- Upstream Boltz: `jwohlwend/boltz` @ **`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`** =
  **`v2.2.1-6-gb1ebfc4`**. (Phase 1 noted CLAUDE.md benchmarks against "genuine
  Boltz-2 2.2.1" — same minor version, so the comparison is like-for-like.)
- Weights: `boltz-community/boltz-2` @ repo sha
  **`6fdef46d763fee7fbb83ca5501ccceff43b85607`**, licence **MIT**. Files:
  `boltz2_conf.ckpt` (2 286 561 469 B), `boltz2_aff.ckpt` (2 062 139 170 B),
  `mols.tar`.

**Phase 1's unverified item is now closed.** `python3 tools/check_remote_bundle.py boltz2`
→ **all 24 shards present, 365 MiB**, every length agreeing with the committed manifest.

**Method note.** The two checkpoints were *not* downloaded. A `torch.save` file is a ZIP
whose `data.pkl` carries every tensor's dtype/shape/storage-key, so
`tools/ckpt_index.py` reads the end-of-central-directory, the central directory and that
one member by HTTP range — ~0.9 MB instead of 1.92 GiB — and `tools/ckpt_compare.py`
range-fetches individual tensor members to compare them byte-for-byte. Both use a
**restricted unpickler that resolves nothing**: every global becomes an inert local type
and every persistent id a placeholder, so no code from the downloaded pickle can run. We
only ever read the metadata the opcodes carry.

---

## 1. The two findings that reshape the plan

### 1.1 🔴 `boltz2_aff.ckpt`'s trunk is byte-identical to `boltz2_conf.ckpt`'s

Phase 1 flagged the naming problem as "the largest structural decision Phase 2 has to
make". It is smaller than feared, because almost none of the affinity checkpoint needs
exporting at all.

| | tensors | params |
|---|---|---|
| `boltz2_aff.ckpt` `state_dict` | 5 539 | 529 378 245 |
| `boltz2_conf.ckpt` `state_dict` | 5 102 | — |
| shared names | **5 019** | — |
| shared names with a different **shape** | **0** | — |
| only in `aff` | 520 (`affinity_module1` 334, `affinity_module2` 186) | 8 742 022 |
| only in `conf` | 83 (`template_module` 81, `bfactor_module` 2) | — |

And **33 of 33 sampled shared tensors are byte-identical** (sha256 of the raw storage),
spread deliberately across every shared scope:

| scope | sampled | identical |
|---|---|---|
| `pairformer_module` | 8 | 8 |
| `msa_module` | 5 | 5 |
| `input_embedder` | 5 | 5 |
| `structure_module` | 5 | 5 |
| `confidence_module` | 5 | 5 |
| `diffusion_conditioning` | 5 | 5 |
| **total** | **33** | **33 / 0 differing** |

So the affinity checkpoint is the structure checkpoint **plus** two affinity modules
**minus** the template and b-factor modules. Consequences:

* **No second trunk export.** LocalFold's shipped `boltz2-int5` bundle already holds the
  weights the affinity pass's trunk needs. The 1.92 GiB file is **98.3% redundant** with
  what the page already downloads.
* **We export 520 tensors / 8.74 M params — 1.70% of the checkpoint.**
* The same resident GPU weight buffers serve both passes. No extra residency, no second
  `releaseAllWeights` cycle.
* The absent `template_module` is consistent with upstream skipping templates on the
  affinity pass (§4.3), and with our scope.

> **Caveat, stated plainly.** 33 of 5 019 is a *sample*, chosen to cover every scope
> rather than to be exhaustive. It is strong evidence, not proof. The Phase 1 pipeline
> already ships the exhaustive check — `tools/check-bundle-vs-params.py --bundle= --digest=`,
> the tool whose header says it is "the one question every other checker assumes" and
> which found four negated tensors in Boltz-2's own export. **Gate G0 below runs it over
> all 5 019 before any of this is relied on.**

### 1.2 Tensor naming: keep upstream's PyTorch names verbatim

Phase 1's open question was whether to invent Haiku-style paths for the affinity export,
since af3-any-model has no affinity module to borrow names from. **Keep the PyTorch
names.** `tools/export_af3_model.py`'s own stated reason for keeping Haiku paths is:

> *"the names ARE the description, they match the oracle's module paths character for
> character, and a tensor that is wrong can be found by grepping for it in both places."*

For the affinity module the oracle **is** upstream PyTorch. Matching
`affinity_module1.pairformer_stack.layers.0.tri_mul_out.norm_in.weight` preserves exactly
the property that rule exists to protect; inventing a Haiku path would destroy it against
the only reference that exists. So this is not a departure from the convention — it is the
convention applied to a different oracle.

🔴 **And that has one concrete consequence that would otherwise fold to noise.**
`quantize_af3.py`'s `KEEP_FLOAT32` is
`re.compile(r"/(scale|offset|bias)$|_bias$|_weight$|/output_b$")` — **slash**-separated and
Haiku-shaped. Our names are **dot**-separated (`….z_norm.weight`, `….norm_in.bias`), so
**that regex matches none of them** and every LayerNorm scale and bias would be
group-quantised — which the tool's own header calls "the worst thing to group-quantise".

It is already survivable, because line 245 is a union:

```python
if KEEP_FLOAT32.search(name) or name in named_float32 or here == 32:
```

and `named_float32` comes from the manifest's own `float32Tensors`, validated to name only
tensors that exist (it raises otherwise). That override was added for ESM-C's embedding
table, "a `weights` by name". **So the new exporter must write a complete and correct
`float32Tensors` list, and nothing else needs to change in the quantiser.** Gate G1 below
is the falsification: delete the list and the export must get measurably worse.

---

## 2. The affinity module, as reverse-engineered

`src/boltz/model/modules/affinity.py` is **223 lines** and the whole of the new maths.

### 2.1 `AffinityModule.forward(s_inputs, z, x_pred, feats)`

```
z  = z_linear(z_norm(z))                                  # LN(128) then LinearNoBias 128→128
z += s_to_z_prod_in1(s_inputs)[:, :, None, :]             # LinearNoBias 384→128, broadcast over j
   + s_to_z_prod_in2(s_inputs)[:, None, :, :]             #   ...and over i   (an outer SUM)
x_rep     = token_to_rep_atom @ x_pred                    # one representative atom per token
d         = cdist(x_rep, x_rep)                           # token × token distance
distogram = dist_bin_pairwise_embed[ sum(d[...,None] > boundaries, -1) ]
            # boundaries = linspace(2, 22, 63) → 64 bins; Embedding(64,128); note STRICT >
z += pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)
z  = pairformer_stack(z, pair_mask=cross_pair_mask)       # PairformerNoSeq, 8 or 4 layers
return affinity_heads(z, feats)
```

🔴 **Despite the name, `s_to_z_prod_in*` is a SUM, not a product** — two independent
projections of `s_inputs` broadcast along opposite axes and added. Porting it as an outer
product would be a plausible-looking, wrong tensor.

**`PairwiseConditioning`** (`encodersv2.py`, "Algorithm 21"), `num_transitions=2`:
```
z = LinearNoBias_256→128( LayerNorm(256)( cat(z_trunk, distogram) ) )
for t in transitions:  z = t(z) + z          # 2 × Transition(128, hidden 256), residual
```
Note it is `z + conditioner(z, …)` at the call site, so the conditioner's own output is
added to `z` **again** on top of its internal residuals.

**`PairformerNoSeqModule`** — `num_blocks` layers of, all residual, in this order:
```
z += tri_mul_out (z, mask)          TriangleMultiplicationOutgoing(128)
z += tri_mul_in  (z, mask)          TriangleMultiplicationIncoming(128)
z += tri_att_start(z, mask)         TriangleAttentionStartingNode(128, width 32, heads 4, inf 1e9)
z += tri_att_end  (z, mask)         TriangleAttentionEndingNode  (128, width 32, heads 4, inf 1e9)
z  = transition_z(z) + z            Transition(128, hidden 512)
```
🔴 **Dropout vanishes at inference, exactly.** `get_dropout_mask` computes
`dropout = dropout * training`, so with `training=False` it returns a tensor of exactly
`1.0`. There is no dropout to port and no `p=0.25` to honour.

**`Transition`** (`layers/transition.py`): `LayerNorm(dim, eps=1e-5)`, then
`fc3( silu(fc1(x)) * fc2(x) )`, all three `bias=False` — plain SwiGLU, the same shape as
LocalFold's AF3 transition.

### 2.2 `AffinityHeadsTransformer` — there is no transformer in it

Confirmed from the checkpoint: **18 tensors, pure MLP.**

```
g      = Σ(z · pool_mask) / (Σ pool_mask + 1e-7)      # masked mean over BOTH pair axes → [128]
g      = affinity_out_mlp(g)                           # Lin 128→128, ReLU, Lin 128→384, ReLU
value  = to_affinity_pred_value(g)                     # Lin 384→384,ReLU, 384→384,ReLU, 384→1
score  = to_affinity_pred_score(g)                     # same shape, separate weights
logits = to_affinity_logits_binary(score)              # Lin 1→1
→ {affinity_pred_value: value, affinity_logits_binary: logits}
```

🔴 `transformer_args` is `{num_blocks: 12, num_heads: 8, token_s: 384,
activation_checkpointing: true}` and **only `token_s` is used**. `num_blocks: 12` and
`num_heads: 8` are dead config, the checkpoint has no tensors for them, and a port that
believed the name would build twelve transformer blocks that do not exist.

**Also dead:** `GaussianSmearing` (defined, never instantiated), `use_cross_transformer`
(accepted, never read), `groups` (`{0:1,1:4,2:6,3:4,4:1}`, threaded through and unused).
**None of it is ported.**

### 2.3 Three different masks, and they are not interchangeable

```
rec = (mol_type == 0) & token_pad_mask                  # PROTEIN tokens
lig = affinity_token_mask & token_pad_mask              # every token of the binder chain
M   = lig⊗rec + rec⊗lig + lig⊗lig                       # includes the diagonal
```
1. **`boltz2.py` pre-masks the trunk pair**: `z_affinity = z * M[None,:,:,None]` — so the
   module receives a `z` that is already zero outside the cross region.
2. **`AffinityModule`** recomputes the same `M` and uses it as the pairformer `pair_mask`.
3. **`AffinityHeadsTransformer`** recomputes it and then removes the diagonal,
   `M * (1 - eye)`, for the pooling denominator.

Using (2)'s mask for (3) would change `g` on every ligand that is more than one token.

### 2.4 Exact tensor inventory

Per `PairformerNoSeq` layer, 37 tensors:
`tri_mul_{out,in}`: `norm_in.{weight,bias}`, `p_in[256,128]`, `g_in[256,128]`,
`p_out[128,128]`, `g_out[128,128]`, `norm_out.{weight,bias}` — 8 each.
`tri_att_{start,end}`: `layer_norm.{weight,bias}`, `linear[4,128]` (the pair→4-head bias),
`mha.linear_{q,k,v,g,o}[128,128]` — 8 each.
`transition_z`: `norm.{weight,bias}`, `fc1[512,128]`, `fc2[512,128]`, `fc3[128,512]` — 5.

| | pairformer layers | tensors | params |
|---|---|---|---|
| `affinity_module1` | **8** | 334 | 5 490 755 |
| `affinity_module2` | **4** | 186 | 3 251 267 |
| **ensemble total** | | **520** | **8 742 022** |

`token_s = 384`, `token_z = 128`, `num_dist_bins = 64`, `max_dist = 22`.
`tri_att_*.linear[4,128]` and `mha.linear_q[128,128]` together confirm **4 heads × 32**.

---

## 3. Scope: confirmed and corrected

The brief expected **(a)** the affinity module, **(b)** ligand-from-SMILES featurisation,
**(c)** a batch loop + results UI, and asked me to confirm from the Phase 1 spec.

* **(a) Confirmed and now fully specified** — and *larger than "a module"*: it is a
  second pass that re-runs the trunk and the sampler (§4).
* **(b) Already done.** Phase 1 established `shared/chem/` is complete and
  RDKit-validated. Phase 2 adds only two things upstream enforces: the **≤128 heavy-atom
  hard refusal** and the **>56 heavy-atom accuracy warning** (§5.3), plus **molecular
  weight** (§5.2).
* **(c) Confirmed, unchanged.**
* **New, not in the brief:** the affinity pass's own **crop**, **single-sequence MSA
  profile**, **best-sample-by-ipTM selection**, and the optional **molecular-weight
  correction** — §4 and §5.

No part of upstream's Python package is vendored. What is extracted is listed in §6, and
nothing else is read at runtime.

---

## 4. 🔴 The affinity prediction is a second fold, not a head

This is the single most important correction to the brief's model of the work. The brief
said the affinity module is "a separate checkpoint and a separate forward pass, not part
of the structure head" — true, and it understates it. `main.py` loads `boltz2_aff.ckpt`
as a **complete `Boltz2` model** and runs a **second `trainer.predict`**:

```python
predict_affinity_args = {
    "recycling_steps": 5,            # the structure pass's CLI default is 3
    "sampling_steps":   sampling_steps_affinity,     # default 200
    "diffusion_samples": diffusion_samples_affinity, # default 3
    "max_parallel_samples": 1,
    "write_confidence_summary": False, ...
}
steering_args.fk_steering = False
steering_args.physical_guidance_update = False
steering_args.contact_guidance_update = False
model_module = Boltz2.load_from_checkpoint(affinity_checkpoint, strict=True, ...)
```

### 4.1 The full flow, per ligand

**Pass 1 — structure** (`boltz2_conf.ckpt`; LocalFold's existing bundle)
1. Ordinary fold of protein + ligand.
2. The rank-0 structure is written as `pre_affinity_<id>.npz`
   (`write/writer.py:178`, gated on `idx_to_rank[model_idx] == 0`).

**Pass 2 — affinity** (`boltz2_aff.ckpt`; same trunk weights + 8.74 M new)
3. **Input structure is pass 1's prediction** — `inferencev2.py:62` loads
   `pre_affinity_<id>.npz` instead of the original input.
4. **Crop** — `AffinityCropper(neighborhood_size=10, max_tokens_protein=200)` called with
   `max_tokens=256, max_atoms=2048`. It ranks protein tokens by **minimum distance to the
   ligand in the predicted pose** and keeps a residue neighbourhood around the nearest, so
   the affinity trunk sees a **pocket**, not the whole protein.
5. **Featurise** with `compute_affinity=True`, which adds
   `affinity_token_mask`, `affinity_mw`, and `profile_affinity` /
   `deletion_mean_affinity`; templates are skipped
   (`if data.templates and not compute_affinity`).
6. **Trunk** at `recycling_steps=5`, **diffusion** 200 steps × 3 samples, **confidence**.
7. `best_idx = argsort(iptm, descending)[0]`; `coords_affinity = sample_atom_coords[best_idx]`.
8. `s_inputs = input_embedder(feats, affinity=True)`; `z_affinity = z * M`.
9. `affinity_module1` and `affinity_module2` on `(s_inputs, z_affinity, coords_affinity)`,
   `multiplicity=1`.

### 4.2 🔴 The affinity `s_inputs` uses a SINGLE-SEQUENCE MSA profile

`input_embedder(feats, affinity=True)` differs from the structure call in exactly one
respect: it reads `profile_affinity` / `deletion_mean_affinity` instead of `profile` /
`deletion_mean`. And those are produced by

```python
process_msa_features(..., max_seqs_batch=1, max_seqs=1, affinity=True)
```

— **depth 1**. So the affinity module's `s_inputs` carries a query-only profile *however
deep the alignment is*, while the pass-2 **trunk** still consumes the full
`feats["msa"]`. Both `input_embedder` calls happen (`boltz2.py:415` with `affinity=False`
for the trunk, `:627` with `affinity=True` for the affinity module).

Feeding the deep-MSA profile into the affinity module would be a silent, plausible,
wrong input. This is the subtlest thing in the port.

### 4.3 What this costs, and the recommendation

The sampler budget is the whole cost. Pass 2 is **200 steps × 3 samples = 600 denoiser
calls** against pass 1's default 25, on ≤256 tokens instead of the full protein, plus a
5-recycle trunk. The crop is what makes it tractable; the step count is what makes it
expensive. On the mockup's own target (164 residues + ligand, 10 s for pass 1), pass 2
should be expected to **dominate per-ligand wall time**, and the brief's "ETA from
measured throughput" is therefore essential rather than a nicety.

**Recommendation: implement the faithful two-pass flow at upstream's defaults, and expose
`sampling_steps_affinity` and `diffusion_samples_affinity` as controls.** Lowering them is
an *upstream-supported configuration*, not a port deviation, so a user trading accuracy
for throughput stays inside the model's own envelope. Correctness is gated at the
defaults (§7).

**Explicitly NOT recommended: the one-pass shortcut** — feeding pass 1's `z` and
coordinates straight into the affinity modules and skipping pass 2. It is tempting and it
would roughly halve the cost, but it changes four inputs at once: the recycle count
(3 vs 5), the token set (full protein vs 200-token pocket crop), the coordinate source
(pass-1 rank-0 vs pass-2 best-of-3-by-ipTM), and it still needs the single-sequence
profile of §4.2. The affinity heads were trained on pass-2's distribution. The error is
**unmeasured**, and this codebase's own standard (`AGENTS.md`: "Establish numerical
correctness before optimizing performance") says not to ship it on an argument. If it is
wanted later, §7's G5 is the measurement that would justify it: run both arms over a
labelled set and report the rank correlation and the shift in both outputs. Until that
number exists, the shortcut stays out.

---

## 5. Outputs, and three ways to get them wrong

### 5.1 What upstream emits

`affinity_<id>.json` (`write/writer.py:306-334`):

```
affinity_pred_value            affinity_probability_binary
affinity_pred_value1           affinity_probability_binary1
affinity_pred_value2           affinity_probability_binary2
```

Per member: `probability = sigmoid(logits_binary)`.
Ensemble: `value = (v1+v2)/2`, `probability = (p1+p2)/2`.

🔴 **The probability is the mean of the two sigmoids, not the sigmoid of the mean logit.**
The two are not equal and the second is the easy mistake.

### 5.2 🔴 The molecular-weight correction, and its exact scope

```python
model_coef, mw_coef, bias = 1.03525938, -0.59992683, 2.83288489
mw = feats["affinity_mw"][0] ** 0.3
ensemble["affinity_pred_value"] = model_coef * ensemble["affinity_pred_value"] \
                                 + mw_coef * mw + bias
```

Four things about it, all load-bearing and none in the brief:

1. **It applies to the ensemble `affinity_pred_value` only.** Not to the probability, and
   not to `affinity_pred_value1`/`2` — those are built *before* it runs. So with the
   correction on, **the aggregate and the per-member values are on different scales**, and
   a CSV that puts them in adjacent columns without saying so invites exactly the kind of
   silent misreading the brief is trying to avoid. Our column headers must say which are
   corrected.
2. **It exists only on the ensemble path.** The `else` branch has no correction at all.
3. **The effective default is OFF.** The `Boltz2` class default is `True`, the
   checkpoint's stored hparam is `False`, and `main.py`'s CLI default is
   `affinity_mw_correction: Optional[bool] = False`, passed explicitly to
   `load_from_checkpoint` — so a plain `boltz predict` does **not** correct. We default to
   off and expose it, matching upstream behaviour rather than upstream's class signature.
4. **`affinity_mw` is RDKit `Descriptors.MolWt(mol_no_h)`** — ordinary average molecular
   weight (implicit hydrogens included in the figure; `RemoveHs` makes them implicit
   rather than deleting their mass). We compute it from the parsed SMILES graph plus
   standard atomic weights; `shared/chem/smiles.js` already has `molecularFormula(graph)`
   to build on. **A unit gate against RDKit's value over the existing 74-molecule corpus
   is cheap and removes all doubt** (G4).

### 5.3 Ligand size limits are upstream's, and they refuse

`parse/schema.py:1265-1275`, for a SMILES ligand with `affinity`:
* **> 128 heavy atoms → `ValueError`**: "ligands with more than 128 atoms are not supported
  in the affinity prediction module".
* **> 56 heavy atoms → warning**: "larger than 56 heavy-atoms, which was the maximum
  during training, therefore the affinity output might be inaccurate."

LocalFold's page currently admits 150 heavy atoms, so **both bounds are new** and both
belong in the `.smi` intake: refuse past 128 with upstream's sentence, and carry a
per-ligand "outside training range" flag from 57 up — into the UI *and* the CSV, because
a screening table that silently mixes in-range and out-of-range predictions is the
failure mode worth engineering against.

### 5.4 UI semantics

Per the brief, kept separate and labelled, never merged:
* **`affinity_probability_binary`** — 0..1, predicted probability the ligand is a binder.
  Binder-vs-decoy / hit discovery. **Higher is better.**
* **`affinity_pred_value`** — **log10(IC50), IC50 in μM**. Hit-to-lead, ranking small
  modifications of real binders. **Lower is better.** Also show
  **pIC50 = 6 − affinity_pred_value** so the axis reads conventionally.
* Per-member `…1`/`…2` in the CSV, aggregate in the UI.

On the brief's "the two heads are trained on different data with different supervision":
the **architecture** is confirmed — two sibling MLP stacks off one pooled vector `g`, with
the binary logit a 1→1 linear on a *separate* `to_affinity_pred_score` scalar rather than
on `affinity_pred_value`. The *training-data* claim is not visible in this repository and
I have not verified it; it is consistent with the design and I will attribute it to the
Boltz-2 paper rather than present it as something the code shows. The UI copy should say
what is certain — different heads, different quantities, different directions, different
intended use — and not over-claim provenance.

---

## 6. Upstream module → ported kernel

Nothing is vendored. These are the only upstream modules read, and each maps onto a kernel
LocalFold already ships.

| upstream | what it is | ported to |
|---|---|---|
| `modules/affinity.py` `AffinityModule.forward` | the wiring above | **new** `webgpu/af3/affinity/affinity.js` + `cpu/af3/affinity/affinity.js` |
| `modules/affinity.py` `AffinityHeadsTransformer` | 18-tensor MLP + masked mean | **new**, in the same pair |
| `layers/pairformer.py` `PairformerNoSeqModule/Layer` | 4 pair ops + transition, residual | **new pair-only driver** over existing kernels (below) |
| `layers/pairformer.py` `TriangleMultiplication{Outgoing,Incoming}` | tri-mul | `webgpu/af3/trunk/triangle.js` — `af3TriangleMultiplication`, `af3TriangleWeights` |
| `layers/pairformer.py` `TriangleAttention{StartingNode,EndingNode}` | 4 heads × 32, pair-bias attention | `webgpu/af3/trunk/grid-attention.js` — `createGridAttentionShaders`, `packGridAttentionWeights` |
| `layers/transition.py` `Transition` | SwiGLU, LN eps 1e-5 | `webgpu/af3/trunk/transition.js` — `Af3TransitionGpu`, `packTransitionWeights` |
| `modules/encodersv2.py` `PairwiseConditioning` | LN(256)→LinearNoBias→2× Transition residual | **new** (12 lines) over the transition kernel |
| `GaussianSmearing`, `use_cross_transformer`, `groups`, `transformer_args.{num_blocks,num_heads}` | dead | **not ported** |
| `data/crop/affinity.py` `AffinityCropper` | pocket crop, 10-neighbourhood, ≤200 protein tokens | **new** `shared/af3/affinity/crop.js` (CPU, 164 lines upstream) |
| `data/feature/featurizerv2.py` `process_msa_features(affinity=True)` | depth-1 profile | **new** flag on `shared/af3/featurise/msa-features.js` |
| `data/tokenize/boltz2.py` `affinity_mask` | binder chain's tokens | **new** field in `featuriseProtein` options |
| `data/parse/schema.py` MW + size limits | `Descriptors.MolWt`, 128/56 | **new** `shared/chem/molecular-weight.js` + intake checks |
| `models/boltz2.py:609-721` | pre-mask, best-by-ipTM, ensemble, MW correction | **new** `web/affinity-run.js` orchestration |

**Driver, not a modification of the trunk.** `Af3PairformerStackGpu` reads
`blocks[0].singleAttention.heads` unconditionally — the single track is intrinsic to it. So
the affinity stack gets its **own pair-only driver** calling the three kernels directly,
rather than a `noSingle` flag threaded through the shipped trunk. That keeps the stack the
page folds with untouched.

🔴 **And it must carry a distinct pipeline-cache key part.** `ComputePipelineCache`
indexes by `entryPoint + source` as well as by key, so a collision is *refused and named*
rather than silently served — but CLAUDE.md records two real collisions from keys that did
not name constants their shaders embed, and our affinity stack is 128 channels with 4
heads of 32, which is in the same family as the trunk's shapes. The key gets an explicit
`boltz2-affinity-pair:` prefix plus channels/heads/dimension/block-count, and
`npm run test:cache` (`check-pipeline-key-collisions.js`) is extended with a
boltz2-structure-then-affinity arm **in both orders**, since a collision is only reported
by the second compile.

---

## 7. Gates — how each claim gets proved

Written before the code, because `AGENTS.md` requires a deterministic differential test
against an independent reference for every GPU kernel, and because every number in §1 is
currently a sample rather than a proof. Each gate names what it falsifies.

| # | gate | asserts | falsified by |
|---|---|---|---|
| **G0** | `tools/g0_verify.py` over **all 5 019** shared tensors, `aff` vs `conf` (ZIP central-directory CRC-32 + size, escalating any disagreement to sha256) | §1.1's byte-identity, exhaustively — the claim the whole plan rests on | a single differing tensor ⇒ a second trunk export is needed and §4 changes |
| **G1** | re-export with `float32Tensors` **deleted** | that the list is what keeps norms/biases f32 under dot-separated names | if the export is unchanged, the list is not being honoured and §1.2 is wrong |
| **G2** | `cpu/af3/affinity/` f64 reference vs upstream PyTorch on one real complex, stage by stage (`z` after `z_linear`, after the `s_inputs` sum, after the distogram, after the conditioner, after each pairformer layer, `g`, both heads) | the port computes upstream's function | any stage out of band; a one-stage dump cannot localise, so every seam is tapped |
| **G3** | GPU vs the G2 CPU reference, same seams | the kernels compute the reference's answer | — |
| **G4** | `molecularWeight(smiles)` vs RDKit `Descriptors.MolWt(RemoveHs(mol))` over the existing **74-molecule** corpus | §5.2's MW, which only matters when the correction is on | any molecule off by more than f32 rounding |
| **G5** | two-pass vs one-pass over a labelled ligand set: rank correlation + shift in both outputs | **the measurement that would justify the shortcut of §4.3**. Reported, never asserted | — (it exists to produce a number, not to pass) |
| **G6** | `test:cache` with a boltz2-structure + boltz2-affinity arm, **both orders** | no pipeline-cache collision | swapping the order must still pass |
| **G7** | `test:switch`-style: N ligands in one page, device must hold one model's worth | the screening loop does not leak GPU memory across ligands — the failure `releaseAllWeights` was written for | removing the per-ligand teardown must make it red |
| **G8** | end-to-end vs upstream `boltz predict --use_msa_server` on the `examples/affinity.yaml` complex, all six output fields | the whole pipeline, against the real thing | — |
| **G9** | a 129-heavy-atom ligand is **refused** with upstream's sentence; a 57-atom one is **flagged** | §5.3's bounds reach the user | removing either check must make it red |

🔴 **Two traps from Phase 1 that apply directly here.** `relative-rms.js` returns **exactly
0** for an unguarded comparison over a non-array — "a perfect score from comparing
nothing" — so every seam in G2/G3 asserts its operand is an array first. And
`if (x > bound) throw` never fires on `NaN`, which is how a checker passed for as long as
OpenDDE's bundle existed without computing anything; the comparisons here test
`Number.isFinite` explicitly.

---

## 8. Work breakdown

**New files**
```
tools/export_boltz2_affinity.py        read boltz2_aff.ckpt → f32 model dir, PyTorch
                                       names verbatim, correct float32Tensors (§1.2)
shared/bundles/manifests/boltz2-affinity.js    generated by write_manifest_module.py
shared/af3/affinity/crop.js            AffinityCropper, pocket crop ≤200 protein tokens
shared/af3/affinity/features.js        affinity_token_mask, depth-1 profile, mw
shared/chem/molecular-weight.js        Descriptors.MolWt equivalent
cpu/af3/affinity/affinity.js           f64 reference (AGENTS.md requires it)
webgpu/af3/affinity/affinity.js        the module: conditioner + pair-only stack + heads
webgpu/af3/affinity/pair-stack.js      pair-only driver over triangle/grid/transition
web/affinity-run.js                    two-pass orchestration, best-by-ipTM, ensemble, MW
web/screen.js + web/screen.css         queue, table, scatter, per-ligand show/hide
shared/input/smi.js                    .smi parsing
tools/check-affinity-path.mjs          G2/G3 driver
tools/gpu/check-affinity-oracle.js     G3
tools/oracle/dump_boltz2_affinity.py   the upstream dump G2 compares against
```

**Edited**
```
shared/bundles/manifests/index.js      register boltz2-affinity (lazy load, pinned remote)
tools/write_manifest_module.py         its BUNDLES twin
shared/af3/featurise/{featurise,batch,msa-features}.js   affinity options, forwarded
                                       FIELD BY FIELD (the documented throw-away trap)
web/job-json.js                        properties: affinity: {binder: <chain>} — OPEN
                                       dialect, since a SMILES ligand already forces it
web/fold-archive.js                    the screening ZIP: per-ligand structure + CSV
package.json                           test:affinity, test:smi; extend test:cache
.github/workflows/                     one more bundle download step
```

**Registry entry** (shards fetched only when affinity is enabled):
```js
"boltz2-affinity": {
  model: "boltz2-affinity",
  directory: "./model-boltz2-affinity-int5/",
  remote: "https://huggingface.co/<ours>/resolve/<pinned sha>/boltz2-affinity-int5/",
  release: "boltz2-affinity-int5",
  variable: "LOCALFOLD_INCLUDE_BOLTZ2_AFFINITY",
  load: () => import("./boltz2-affinity.js"),   // literal import(), per the registry's rule
}
```

**Export size** — 520 tensors, 8 742 022 params:

| | tensors | params | bytes |
|---|---|---|---|
| kept float32 (110 bias, 90 norm weights, 6 output projections, 4 1-D, 2 embedding tables) | 212 | 46 214 | 180.5 KiB |
| int5 g32 asym (f16 scale + f16 zero) | 308 | 8 695 808 | 6 369 KiB |
| **float32 export** | | | **33.35 MiB** |
| **int5 bundle** | | | **6.40 MiB** (5.21×) |

**6.40 MiB on top of the existing 365 MiB — a 1.8% add-on, fetched lazily.**

**Order of work.** G0 first — it is cheap and the entire plan is void if it fails. Then the
exporter + G1, then the CPU reference + G2, then the GPU kernels + G3/G6, then the
orchestration + G7/G8, then the UI, plots and ZIP, and G9. G5 is run once the two-pass
path works, to produce the number that decides whether a shortcut is ever worth offering.

---

## 9. Risks and open questions

1. **G0 is a precondition, not a formality.** §1.1 rests on 33 of 5 019 tensors. Run it
   before anything else.
   🔴 **Correction to this plan's first draft:** the G0 row named
   `tools/check-bundle-vs-params.py`. That is the wrong instrument — it compares a **local
   LocalFold bundle directory** against an **af3-any-model params digest** (`--bundle=`,
   `--digest=`), which is a different and also-useful question (and it is still what G8's
   neighbourhood wants). It cannot compare two PyTorch checkpoints to each other, which is
   what G0 claims. The instrument is `tools/g0_verify.py`.
2. **`token_to_rep_atom` ≡ LocalFold's pseudo-beta gather — likely, not proven.** Upstream
   builds it as `atom_idx + token["disto_idx"] - start`, the **distogram** atom, held
   separately from `center_idx` (CA). LocalFold's `batch.tokenAtomsToPseudoBeta` is the CB
   pseudo-beta gather and `foldBatch` already returns `pseudoBeta` from it; the BOLTZ2
   dialect's modified-residue note ("the pseudo-beta is **CB** and not slot zero") points
   the same way. But this decides every distance in §2.1, and this codebase has already
   been bitten four times out of seven models by an atom-window convention nobody
   compared. **Compare the integers against a dump, `check-atom-windows.js`-style, before
   trusting it.**
3. **`token_pad_mask` has no obvious browser analogue.** Upstream pads to a bucket; the
   affinity masks are built from `token_pad_mask`. If LocalFold's featuriser does not pad,
   the mask is all-ones and this is free — but it must be checked rather than assumed,
   since a wrong pad mask changes the pooling denominator in §2.3 and so changes every
   score.
4. **The pass-2 crop changes the token set, so the ligand's own token indices move.**
   The crop keeps the ligand whole and crops the protein, but the binder chain's token
   offsets after cropping are what `affinity_token_mask` must address. Off-by-one here
   produces a plausible score from the wrong pooling region.
5. **Our port's trunk is af3-any-model's Haiku Boltz-2, not upstream's PyTorch.** Phase 1
   recorded its parity (`trunk_out_pair` 4.56e-2 on int5; the f32 five-model sweep has
   nothing above 1.4e-3). The affinity module consumes that `z`, so the port's trunk error
   propagates into the score. G8 is what bounds the end-to-end consequence; G2/G3 bound
   only the new code.
6. **Cost.** 600 denoiser calls per ligand in pass 2 (§4.3). The screening ETA must come
   from measurement, and the two sampler knobs must be reachable from the UI.
7. **int5 on an 8.74 M-param module is unmeasured.** Phase 1's int5 evidence is for
   300–500 MiB trunks, where quantisation error averages over many blocks. A 4-layer
   pairformer feeding a scalar regression is a different risk profile, and the output is a
   *number a user ranks on*, not a coordinate. **Export f32 as well (33 MiB) and measure
   both before choosing** — and if int5 moves the ranking, ship f32: 33 MiB is an
   acceptable price for the thing the whole tool is for. This is the one place I would not
   take Phase 1's int5 result on trust.
8. **Licensing stays clean.** Boltz-2 weights are MIT and our bundle would be a derived
   quantisation of them — redistributable with attribution, unlike the AF3 parameters
   Phase 1 flagged. The README must state the MIT origin, the pinned HF revision, and that
   no AF3 parameters are redistributed.
