# Phase 1 — LocalFold architecture spec (reverse-engineered)

**Pinned revision.** `sokrypton/localfold` @ `357518613af612b4e1e887deecd93bd612fdac76`
(Mon 9 Nov 2026 09:15:11 -0500, *"WebGL2 AF3: boltz2's and rf3's outer product mean bias,
Protenix/Mol-1 with an alignment - the input-kind matrix"*). Cloned to `refs/localfold/`
on 2026-11-09. Everything below was read at that SHA.

**Live-site diff: no drift.** `https://localfold.org/index.html` is **byte-identical** to
the pinned tree (sha256 `d58d5d6b…5ac2`, 88 536 bytes both sides), and all 31
decision-relevant modules fetched from the live origin are byte-identical too
(`web/app.js`, `web/model.js`, `shared/bundles/manifests/index.js`,
`shared/bundles/manifests/boltz2.js`, `shared/af3/dialect.js`, `shared/chem/*`,
`shared/input/mmseqs2-api.js`, `shared/heads/superpose-pdb.js`, `webgpu/af3/fold.js`,
`webgpu/chem/conformer.js`, `webgl/index.js`, …; report in `refs/live/REPORT.txt`).
There is **no bundler** — the site serves the module tree as written (`AGENTS.md`:
"Keep `src/` loadable by the browser as written… no build step and no bundler"), so a
file-by-file comparison *is* the bundle comparison. **No capability exists in one and
not the other.** The expectation that the deploy lags HEAD does not hold at this SHA.

> Caveat, stated rather than papered over: `huggingface.co` is **unreachable from this
> machine** (TCP reset on the root domain, not just on a path; `curl` and the fetch tool
> both fail). So the weight *shards* were not verified against their pinned revisions.
> Everything about the shard format below is derived from the committed manifest modules
> and the exporter/reader source, which are self-consistent and digest-checked. The repo
> ships the exact tool for the remote check — `python3 tools/check_remote_bundle.py boltz2`
> (HEAD-only, compares every shard's content-length against what the manifest's tensors
> need) — run it from an unblocked network.

---

## 0. Corrections to the Phase-1 brief

Four of the brief's leads are **false at this SHA**, and two capability assumptions are
wrong in our favour. Recorded first because the rest of the plan depends on them.

| Brief's claim | Reality |
|---|---|
| `localfold.json`'s `entries` is the extension point | **No such file.** The registry is `MODEL_BUNDLES` in `shared/bundles/manifests/index.js`. |
| `tools/sync-localfold.mjs` regenerates `web/localfold.js` | **Neither file exists.** Nothing named `sync-localfold`; `web/localfold.css` is the only `web/localfold.*`. |
| a stamping step `npm run stamp` exists | **No `stamp` script** in `package.json` (32 real scripts; none stamp). |
| weights are "int8 with float16 block scales, int5 asymmetric per-block" | Partly. **Three** codecs ship. The AF3 lineage (incl. Boltz-2) is **int5 asymmetric, group 32, float16 scale *and* float16 zero point**. int8 symmetric group 64 is the *legacy AF2* path (`tools/quantize_model.py`). int3 asymmetric group 128 is the AF2-delta / ESM-C codec. |
| *(assumed missing)* ligand-from-SMILES featurisation | **Already implemented and validated.** See §6. |
| *(assumed present)* CCD assets may be needed | **Not needed for SMILES-only input.** See §6.3. |

**The one thing that really is absent: affinity.** `grep -rni 'affinit'` over the entire
repository — all source, all 27 files in `docs/`, `CLAUDE.md`, `README.md` — returns
**0 matches**. There is no affinity module, no affinity head, no affinity weights, and no
prior investigation of one. That is the whole of Phase 2's model work.

---

## 1. Weight pipeline — checkpoint to browser

### 1.1 Where Boltz-2's weights actually come from

The Boltz-2 bundle is **not** exported from upstream PyTorch `boltz2_conf.ckpt`. Its
manifest records `source: "AF3-lineage parameters from boltz2.bin.zst"`, and every tensor
name is a **JAX/Haiku module path**:

```
diffuser/evoformer/…
diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/act_norm/scale
diffuser/confidence_head/~_boltz2_reembed/contact_encoder/weights
diffuser/boltz2_msa_profile_encoding/weights
```

`boltz2.bin.zst` is a **ColabDesign2 blob**: one zstd stream of `(scope, name, array)`
records, which `sokrypton/alphafold3` ("af3-any-model") emits for all AF3-lineage
checkpoints. `tools/export_af3_model.py`'s header states the design rule plainly:

> *"Reading the blob rather than any one checkpoint format is the whole point — a second
> model becomes a different `--blob`, not a second exporter."*
> *"THE TENSOR NAMES ARE THE HAIKU PATHS, UNCHANGED … the names ARE the description, they
> match the oracle's module paths character for character."*

**Consequence for Phase 2 (flagged, not acted on):** the affinity checkpoint
`boltz2_aff.ckpt` is PyTorch with PyTorch parameter names. It has **no** Haiku path and
af3-any-model has no affinity module to borrow names from, so an exporter for it must
either invent a naming convention or port the module into Haiku first. This is the single
largest structural decision Phase 2 has to make, and it was invisible from the brief.

### 1.2 The quantisation scheme (int5, exactly)

From `tools/quantize_af3.py` and `shared/bundles/manifests/boltz2.js`:

```json
"quantisation": {
  "scheme": "asymmetric-per-group", "bits": 5, "group": 32,
  "scaleDtype": "float16", "zeroDtype": "float16"
}
```

* **Asymmetric, with a zero point.** A group of weights is not centred on zero; a
  symmetric fit wastes half its codes and leaves a *systematic bias*, and bias is what
  matters because zero-mean rounding noise averages out over blocks while a mean shift
  compounds. Worth 1.28× at five bits.
* **Group 32 is chosen for packing, not tuned.** 32 codes × 5 bits = **exactly 160 bits =
  20 bytes**, so no group ever straddles another. At group 24 or 48 the packer needs a
  split-value case, "and that case is where a packer goes wrong silently."
* **Why 5 bits and not 4.** A scheme costs codes *plus* metadata; int4 with scale+zero
  every 16 weights is also 6 bits/weight and 1.6× less accurate. Measured on 6MRR against
  the crystal (not against pLDDT):

  | scheme | bits/w | MiB | × | RMSD | TM | pLDDT |
  |---|---|---|---|---|---|---|
  | float32 | 32.00 | 1405.3 | 1.0 | 0.69 | 0.950 | 81.6 |
  | int8 g64 sym | 8.25 | 362.3 | 3.9 | 0.70 | 0.949 | 81.7 |
  | int5 g32 asym | **6.00** | **263.5** | **5.3** | **0.66** | **0.953** | **81.2** |
  | int4 g32 asym | 5.00 | 219.6 | 6.4 | 1.21 | 0.896 | 76.2 |

  0.66 vs 0.69 is inside the diffusion-seed spread, so int5 is free; int4 falls off a
  cliff.
* **Norms, offsets and biases stay float32** — 0.09% of the parameters and the worst
  thing to group-quantise (a 128-wide LayerNorm scale is four groups, so four scales
  carry a tensor whose job is to set the scale of everything after it). They are listed
  by name in `float32Tensors` (**173** of Boltz-2's 445).
* **Known cost at the fold level:** int5 reads 3.71e-1 on the AF3 real-denoise oracle vs
  9.92e-4 on the f32 bundle — that gap *is* the quantisation, not a port defect
  (3.4–4.6% per tensor, side-chain bond rms 0.051 → 0.059 Å). Any oracle residual taken
  on an int5 bundle is not comparable with one taken on f32: `boltz2 int5 0.167` vs
  `boltz2 f32 8.25e-7` on the same template check.

### 1.3 Manifest and shard format

A bundle directory is `manifest.json` + `weights-NN.int5.bin` shards. The manifest module
compiled into the page (`shared/bundles/manifests/<family>.js`, `export const MANIFEST`)
is the same JSON plus `shardDigests`. Top-level keys:

```
formatVersion   1
source          "AF3-lineage parameters from boltz2.bin.zst"
model           { name: "boltz2", recycles: 0 }
bundle          { purpose, model, encoding, encodingNote, tensors, shards, bytes }
coverage        { included: ["diffuser"], excludedScopes: [] }
float32Tensors  [ …173 names… ]          # which tensors escaped quantisation
tensors         { <haiku/path>: <record> }
quantisation    { scheme, bits, group, scaleDtype, zeroDtype }
shardDigests    { "weights-00.int5.bin": "<sha256>", … }
```

Per-tensor record — **this is the authoritative description**, one entry per tensor:

```js
// quantised
"diffuser/boltz2_method_conditioning/weights": {
  file: "weights-12.int5.bin", shape: [12, 384], byteOffset: 0,
  dtype: "int5", block: 32, scaleOffset: 2884, zeroOffset: 3172 }
// kept float32
"diffuser/boltz2_dn_res_type": {
  file: "weights-10.int5.bin", shape: [384], byteOffset: 0, dtype: "float32" }
```

Offsets are **per-file**. A tensor's span from `byteOffset` covers its codes, its padding
*and* its scales — which is why `repack_shards.py` can shift `byteOffset`, `scaleOffset`
and `zeroOffset` by one common delta. Every tensor is aligned to 4 bytes and the readers
rely on it (`ALIGNMENT = 4`).

🔴 **`bundle.encoding` is stale on five shipped bundles and must not be trusted.** `af3`,
`protenix2`, `opendde`, `openbind0` and `intellifold2` all declare
`"encoding": "float32-le"` with no `tensors`/`shards`/`bytes`, while their tensors are
demonstrably int5 in `.int5.bin` shards (af3: 255 int5 + 151 f32 over 8 shards). This is
harmless *only* because nothing reads it: `grep` for `bundle.encoding` across
`shared/ webgpu/ webgl/ web/` returns nothing, and `shared/weights/dtype.js` dispatches on
the **per-tensor `record.dtype`**. A new exporter must therefore write correct per-tensor
records and should not mimic `bundle.encoding`. (Boltz-2's own block is correct and
carries the counts — it was re-exported later.)

**Shard layout** (`tools/repack_shards.py`): bytes are moved, never made — each tensor is
copied whole, read back and compared byte-for-byte before the manifest is written. Fewer,
even shards are preferred because what the split buys is *parallel fetch of individually
cacheable pieces*: a whole shard returns `200` and the Cache API can store it, where byte
ranges of one big file return `206` and it cannot. Measured: 4 connections ≈ 27.5 MB/s vs
14 serial, 8 ≈ 26.3 — so four already saturate and 26 shards buy nothing but mess. **A
shard is at least one whole tensor** (letting a tensor span files is "a reader change
where an off-by-one is a different protein", for ~3% of a cold load).

Per-bundle survey at this SHA (derived from the committed modules):

| bundle | encoding | tensors | f32 kept | shards | size |
|---|---|---|---|---|---|
| **boltz2** | **int5-le g32** | **445** (269 int5 / 176 f32) | 173 | **24** | **382 207 764 B ≈ 365 MiB** |
| af3 | int5 g32 *(block mislabelled)* | 406 (255/151) | 150 | 8 | — |
| rosettafold3 | int5-le g32 | 448 | 193 | 16 | 265 MiB |
| mol1 | int5-le g32 | 441 | 173 | 32 | 510 MiB |
| opendde | int5 g32 *(mislabelled)* | 481 (301/180) | 177 | 12 | — |
| protenix2 | int5 g32 *(mislabelled)* | 404 (250/154) | 153 | 8 | — |
| esmfold2 / ef2-fast-300m | int5-le g32 | 953 | 426 | 8 | 128 MiB |
| bioemu2 | int5-le g32 | 1908 | 942 | 24 | 371 MiB |
| monomer / multimer | int5-le g32 | 341 / 354 | 58 / 52 | 9 / 8 | 73 MiB |
| monomer-2…5, multimer-2…5 | **int3-le g128** (delta) | 263–347 | 64–84 | 8–9 | 43 MiB |
| esmc, esmc-300m | int3 g128 | — | 2 | — | — |

### 1.4 "Dense atom layout"

The brief's phrase maps to the **featuriser's** layout, not the weight layout: AF3
featurises a **dense-24 slot** in per-residue conformer order, while AF2 reads **atom37**
(0 N, 1 CA, 2 C, 3 CB, 4 O). `templateSlotAtom37` converts by atom *name* and
`test/template-atom37-layout.test.js` pins it. The two are explicitly **not**
interchangeable. Separately, a *reference* featurised batch uses AF3's dense grid of
**51 atom subsets** where this port compacts to **18** — reading the subset count is how
you tell a dump-fed fold from a sequence-featurised one.

### 1.5 CacheStorage caching

`shared/bundles/http-tensor-store.js`. The Cache API rather than IndexedDB "because the
thing being stored is a `Response` and this is what it is for."

* **Cache name:** `localfold-shards-<bundleStem>@<fnv1a32(manifestUrl), base36>-<cacheToken(manifest)>`
  * `bundleStem` is the **directory** name from the manifest URL, not `bundle.model` —
    both ESM-C towers are "esmc", both EF2-fast bundles are "esmfold2-trunk", and AF2
    deltas carry no model name, so keying on the model made each look like a stale export
    of the other (measured: choosing ESM-C 600M deleted 300M's 130 MiB, and every switch
    re-downloaded).
  * `@` terminates the stem because directory names may contain `-`, so `model-int5-`
    would otherwise claim `model-int5-2`'s caches.
  * The URL hash carries the **pinned revision**, so a re-pin is a new cache.
* **Stale sweep** (`staleShardCaches`): other exports of the same bundle, plus the legacy
  `localfold-model-` prefix. It is **deliberately not awaited** — deleting a stale 224 MiB
  entry can outlast the wait, and when it did the store ran with no cache at all.
* **Guarded everywhere.** Unavailable on an insecure origin (`file://` included) and can
  refuse on quota; every call is wrapped so a failure costs only the speed-up.
  `CACHE_TIMEOUT_MS = 2000`.
* **Fetch policy:** `MAX_CONCURRENT_DOWNLOADS = 8`; 5 attempts with 250 ms × 2^n backoff;
  retries on `429, 500, 502, 503, 504` and on a thrown (dropped-connection) fetch, which
  "is the likelier failure across a few hundred MiB on a laptop's wifi."
* **Returning-visitor cost is real:** AF3 page 4901 → 4215 ms, OpenDDE 7123 → 5896 ms
  with a kept profile — much of it Chrome serving *compiled pipelines*, not bytes.

### 1.6 Pinned-revision URL resolution

`MODEL_BUNDLES[family].remote` is an absolute, revision-pinned HF URL ending in `/`;
`directory` is the relative fallback. The resolved base is `remote ?? directory`.

```js
boltz2: {
  model: "boltz2",
  directory: "./model-boltz2-int5/",
  remote: "https://huggingface.co/sokrypton/localfold/resolve/651a25ba8a57c6d18caacf50ab7180b255eb5a82/boltz2-int5/",
  release: "boltz2-int5",
  variable: "LOCALFOLD_INCLUDE_BOLTZ2_MODEL",
  load: () => import("./boltz2.js"),
}
```

🔴 **The revision is pinned in the URL, never left at `main`** — a shard fetched from a
moving branch can change under a manifest that did not, and "three separate hours have
gone into `<file> has an invalid byte length` already." A commit SHA makes the URL
immutable, which is also what lets the browser cache it forever.

🔴 **A `remote: null` family silently falls back to a local export the build never
copies.** GitHub Pages caps a site at 1 GiB; IntelliFold-2 alone is 612 MiB. So
`build_site.py` now **drops an unservable `<option>` from `dist/index.html`** and says so
every build — never from the checkout, because a developer with the local export is who
those options are for. (Since live `index.html` is byte-identical to the checkout's at
this SHA, every offered option is servable.)

### 1.7 Producing a new export — the full chain

```
# 1. float32 model directory from an AF3-lineage blob
python3 tools/export_af3_model.py --model <name> [--blob <x.bin.zst>] [--include diffuser]
#    -> manifest.json + weights-NN.f32.bin, every tensor float32, Haiku names verbatim

# 2. quantise to the shipping codec
python3 tools/quantize_af3.py --source model-<name>-full-f32 --out model-<name>-int5

# 3. lay it into a few even shards (bytes moved, verified, not transformed)
python3 tools/repack_shards.py model-<name>-int5 --files 8 --in-place

# 4. compile the manifest into the module the page loads, WITH shard digests
python3 tools/write_manifest_module.py <name>        # or --all

# 5. register in BOTH twins
#    - MODEL_BUNDLES  in shared/bundles/manifests/index.js  (JS)
#    - BUNDLES        in tools/write_manifest_module.py     (Python)
#    build_site.py checks the two agree.

# 6. verify, then publish
python3 tools/check_remote_bundle.py <name> [--revision <sha>]
python3 tools/check-bundle-vs-params.py --bundle=… --digest=…   # vs the reference's own params
npm run test:site        # python3 tools/build_site.py — digest + unresolved-import check
```

🔴 **Why the digests are the safety interlock.** Compiling the manifest in *inverts the
dependency*: the manifest no longer comes from the weights, so re-exporting the shards
leaves the committed module describing the previous ones. Nothing crashes — every offset
still lands inside a file of roughly the right size — and "the page loads tensors sliced
at the wrong byte and folds to noise." The sha256 of each shard is written into the
module and `build_site.py` refuses to package an export whose bytes disagree. That is the
only reason compiling the manifest in is safe. A module without digests is rejected
rather than trusted.

🔴 **The one check every other checker assumes.** `tools/check-bundle-vs-params.py`
compares the bundle against the digest of the parameters the *reference* loads. All the
other checkers feed **one** bundle to both sides, so a tensor of the wrong **sign** or
**width** is invisible to all of them. It found four negated tensors in Boltz-2's own
export and ten stale ones in OpenDDE's.

---

## 2. Model registration — how a model is actually added

Five edits, no codegen, no stamping step.

1. **`shared/bundles/manifests/<family>.js`** — generated by
   `write_manifest_module.py`; `export const MANIFEST`. Committed.
2. **`MODEL_BUNDLES`** in `shared/bundles/manifests/index.js` — keys: `model`,
   `directory`, `remote`, `release`, `variable`, `load`, plus optional
   `delta: {base}` and `noTemplateEmbedder`. `load` must be a **literal `import()`**, not
   built from `directory`: "a bundler cannot follow a computed specifier." Manifests are
   loaded **lazily** — each is ~100 KiB of JSON-in-JS, so static imports would put
   200 KiB in front of every visitor.
3. **`AF3_FAMILIES`** (same file) — `["af3","openbind0","opendde","boltz2","protenix2",
   "intellifold2","rosettafold3","mol1"]`. Membership is *derived* into other tables
   rather than retyped; `CHAIN_MERGES` in `mmseqs2-api.js` is built from it precisely so
   a fifth AF3-lineage bundle needs no edit there. (That derivation exists because the
   hand-typed version threw `unknown model opendde: expected monomer, multimer, af3,
   openbind0` — "the allow-list trap", which this codebase records hitting at least four
   times.)
4. **A dialect** in `shared/af3/dialect.js` — a frozen object of ~120 convention flags,
   selected by `dialectFor(model)`. §5.2 has Boltz-2's.
5. **`<option value="…">`** in `index.html`, and `BUNDLES` in
   `tools/write_manifest_module.py`.

**Current menu** (`index.html`): af3 *(selected)*, openbind0, opendde, **boltz2**,
protenix2, mol1, intellifold2, rosettafold3, monomer, multimer, ef2-fast-600m, bioemu2.

Relevant gates: `test/model-bundles.test.js`, `test/model-family.test.js`,
`npm run test:site` (digests + constructed-path imports), `npm run test:switch`
(ten models in one page, device must hold one model's worth, < 2 GiB),
`npm run test:cache` (two models in **one** pipeline cache, both orders).

---

## 3. Compute layer

### 3.1 The split

Layering is **`shared/` ← `cpu/` ← `webgpu/`**. A directory names the backend, so files
carry no suffix: a stage's reference and its kernel **share a name** —
`cpu/af3/trunk/embedder.js` beside `webgpu/af3/trunk/embedder.js`.

| tree | contents |
|---|---|
| `shared/` | input, chem, bundles, featurisers, weight loaders, dialects — what every backend reads |
| `cpu/` | the f64 reference implementations and the design/MPNN path |
| `webgpu/` | kernels, runtime, models — the page's primary backend |
| `webgl/` | WebGL2 fallback for a browser without WebGPU; every AF3-lineage family **except OpenDDE** (which re-tokenises after the trunk) |
| `web/` | the page: `app.js`, `model.js`, `viewer.js`, `entities.js`, `job-json.js`, `fold-archive.js`, `zip.js`, `vendor/py2Dmol.full.min.js` |

🔴 **`cuda/`, `metal/` and `python/` are documented at length in `CLAUDE.md` but are
absent from the public repository.** So `npm run test:native`, `test:bridge`,
`test:standalone` and `test:featurise` are unrunnable from this checkout. Irrelevant to
us — we are browser-only — but it means `CLAUDE.md` describes a larger tree than ships,
and a reader should not go looking for those gates.

### 3.2 Kernel set (`webgpu/`)

* **Generic kernels** (`webgpu/kernels/`): `attention.js`, `attention-matrix.js`,
  `attention-project-matrix.js`, `attention-geometry.js`, `matrix-linear.js`,
  `outer-product-mean.js`, `transition.js`, `vector-gemm-block.js`, `triangle/`
  (`webgpu.js`, `shaders.js`, `project-matrix.js`, `weights.js`, `cpu-reference.js`).
  **The four model families share these**, which is why a cross-family pipeline-cache
  collision is a real case and not a hypothetical.
* **AF3 trunk** (`webgpu/af3/trunk/`): `trunk.js`, `pairformer-block.js`, `triangle.js`,
  `grid-attention.js` + `grid-attention-matrix.js` + `grid-project-matrix.js`,
  `single-attention.js`, `msa-attention.js`, `msa-stack.js`, `outer-product-mean.js`,
  `transition.js`, `template.js`, `embedder.js`, `pair-track.js`, `pair-width.js`,
  `sequential-groups.js`.
* **AF3 diffusion** (`webgpu/af3/diffusion/`): `diffusion-head.js`,
  `diffusion-transformer.js`, `diffusion-sampler.js`, `diffusion-conditioning.js`,
  `atom-encoder.js`, `atom-decoder.js`.
* **Heads**: `webgpu/af3/confidence/confidence.js` (+ OpenDDE's own),
  `webgpu/heads/{confidence,distogram}.js`.
* **Chemistry on the device**: `webgpu/chem/conformer.js`.
* **Entry**: `webgpu/af3/fold.js` → `foldBatch(device, batch, weights, options)`.

Every GPU kernel has a deterministic differential test against an independent reference
(`AGENTS.md`), and `O(L³)` storage is forbidden.

### 3.3 Chunking of the pair / MSA tracks

Chunking is expressed as **tuning knobs** resolved per device in
`webgpu/runtime/device-profile.js` (~50 knobs). The ones that chunk:

* **Pair track:** `pairTransitionChunkBytes`, `pairTransitionSplit`,
  `pairTransitionSplitMinChannels`, `transitionChunkBytes`, `opmPairBlockBytes`,
  `opmBlockI`, `opmBlockITokens`, `opmCellChunk`, `pairLogitsCacheBytes`.
* **MSA track:** `msaAverageRows`, `msaProjectRows`.
* **Attention:** `attentionKeyChunk`, `attentionBlockBytes`, `attentionTiled`,
  `attentionMatrixTile`, `diffusionAttendKeyChunk`.
* **Diffusion:** `diffusionSplitK`, `diffusionTokenTile`, `diffusionNormSplit`,
  `diffusionFlashSplits`, `atomRowTile`.

Two load-bearing facts: blocking **reorders no sum**, so a blocked arm is held to
`=== 0` against the unblocked one (`check-opm-paths.js`); and the real ceiling is a
**binding** limit, not memory — `sqrt(maxStorageBufferBindingSize / (cZ·4))`, which is
2 047 residues on a 2 GiB-binding card with **28 labels tied at that exact residue**, so
windowing any one of them moves the fold's ceiling by **zero**. The next wall is
`maxBufferSize` at 4 GiB → 2 896 residues. Irrelevant at our target sizes (a few hundred
residues plus a ligand) but it bounds any claim about scaling.

### 3.4 GPU memory budgeting and recovery

`webgpu/runtime/device-memory.js` is the one accounting point.

* `setMemoryBudget(device, bytes)`, `budgetForDevice(fallbackGiB = 4)`,
  `memoryBudgetBytes`, `memoryTotals`, `memorySnapshot`,
  `noteAllocation`/`noteDestroy`, `residencyAllowed`, `keepResidentAffordable`.
* **`GpuMemoryBudgetError` is raised *before* `createBuffer`**, naming the tensor, the
  resident total and the ceiling. The reason is sharp: *"an over-allocation on this
  hardware is not an error, it is a frozen machine"* — Metal accepts allocations well past
  where macOS starts paging, a phone's driver takes them and is then killed by the system,
  and WebGPU reports nothing until far too late.
* The budget is **opt-in**; without one set it only counts. Every allocator on a device
  shares one total, because the device does.
* **Residency:** `webgpu/runtime/resident.js` — `residentWeightBuffer`,
  `residentWeightBufferFilled`, `releaseResidentWeights(device, prefix)`,
  **`releaseAllWeights(device)`**, `residentPackStats`. Keyed by the weight *object*,
  device first, so the cache lives exactly as long as the model and two devices cannot
  hand each other a buffer. Not routed through the pooling allocator, because these must
  outlive the run that made them.
* **Cross-model recovery is already solved and is exactly our case.**
  `npm run test:switch` found that residency kept between folds was also kept across a
  *change of model*: 677 → 1281 → 2171 → 2874 MiB, the fourth fold dying in validation
  and the device **lost** on a Colab L4. `web/app.js:5035` now calls
  `releaseAllWeights(await getDevice())` on a family change; the gate reads 6151 MiB with
  it removed.
* **Pooling:** `webgpu/runtime/allocator.js` — `GpuBufferAllocator` pools by
  `byteLength:usage` and **retires rather than destroys**, so it leaks by design: a
  released buffer is still named by an in-flight command buffer, which is *why* the pool
  exists. `settleReleasedMemory(device)` drains. `setBufferPooling(false)` is the control
  arm for "does recycling a buffer change the answer".
* **Spill:** `webgpu/runtime/host-spill.js` — `HostSpill`, `SPILL_PIECE_BYTES = 64 MiB`,
  held in pieces because a browser refuses one `ArrayBuffer` past ~1 GiB.
* **Device loss / driver reset:** `web/model.js:173–193`. Neither a rejection nor a lost
  device may stay memoised — a failed request used to leave the page saying "No compatible
  WebGPU adapter was found" until a reload, and a lost device (crashed GPU process, driver
  reset) the same. `void device.lost.then(() => { if (devicePromise === request) devicePromise = undefined; })`.
* **Abort:** `webgpu/runtime/abort.js` — `predictionAbortError()` (`name: "AbortError"`),
  stopping at a boundary where no JS-owned GPU resources are in flight, since submitted
  WebGPU commands cannot be preempted.

**All of this is reusable as-is for the screening loop.** Nothing new is needed for
per-ligand teardown, budgeting, or driver-reset recovery.

---

## 4. Featurisation and output

### 4.1 Featurisation

* **`shared/af3/featurise/featurise.js`** → `featuriseProtein(sequence, options)`.
  Chains are joined by `:`. Options carry `msa`, `deletionMatrix`, `unpairedFrom`,
  `chainKinds`, `modifications`, `ligands`, `bonds`, and the dialect flags.
  🔴 **A ligand is one token per heavy atom**, not one token — 68 residues + glycerol is
  74 tokens, + heme is 111, checked against AF3's own batch. Every array is sized from
  that total.
* **`shared/af3/featurise/batch.js`** → `af3BatchFromA3m(sequence, alignment, options)` —
  the page's entry. `batch.js` forwards the dialect **field by field**;
  `featuriserDialect()` exists because three call sites listed those fields by hand.
  🔴 "A key the batch carries and this literal does not name is a key thrown away here."
* Supporting modules: `ccd-component.js` (`parseCcdComponent`, `ELEMENT_SYMBOLS`,
  `ccdUrl`), `standard-components.js` (33 compiled-in components),
  `reference-conformers*.js`, `modified-parents.js`, `msa-features.js`,
  `template-features.js` / `template-fused-features.js`, `declared-leaving.js`,
  `constraints.js`, `boltz2-conformers.js`, `structural-tokens.js`.
* **One idealised reference conformer set is shared by every family**, while the reference
  featurises CCD geometry — so a *sequence*-featurised oracle run has a floor
  (openbind0's `target_feat` is 2.89e-2 from a sequence and 4.93e-8 from the reference's
  batch, same weights, same code). `ref_pos` is a **reported floor, never a failure**:
  intra-token pairwise distances agree to rms 0.65 Å, worst 3.77 Å on a lysine — the same
  molecule in a different rotamer and frame.
* **`ref_pos` is a feature, not a template**: replacing a conformer with noise fails
  loudly (bond rms 0.22 vs 0.03), shrinking one 15% changes the fold by **nothing**.

### 4.2 Structure output

**`shared/af3/structure/pdb.js`** — `toPdb(batch, positions, plddt, options)`, plus
`THREE_LETTER` / `ONE_LETTER`, `atomName(nameChars, slot)`, `linkLine`,
`pdbLinkRecords(records, bonds)`, `pdbSlots(batch)`. Re-exported from
`webgpu/af3/fold.js`.

* **Chain letters — 62, not 26.** `CHAIN_CHARACTERS = "A–Z a–z 0–9"`, indexed
  `(asymId − 1) % 62`. A PDB chain is one character and the 27th chain of a large complex
  wrapped onto `A` and was **merged with it in every reader**. (mmCIF names them all;
  only the absent `cuda/af3` writes `.cif`.) **Our scope is 2 chains (A protein,
  B ligand), so this never binds** — but the writer is PDB-only on the web path, which
  matters for the ZIP export decision.
* **Numbering** walks **tokens**, mapping back through `residueOfToken`; `sequence` is
  indexed by *residue*, and the two stopped being the same number when a residue could be
  several tokens. Getting this wrong named every residue after the first modification by
  whatever letter sat at its token index.
* **A ligand is `HETATM` and has a name** — from `ligandSpans`/`modifiedSpans`, the
  component's own code; indexed into `sequence` it used to come out `UNK`.
* **`ligandName(index)` in `shared/chem/component.js` gives distinct 3-char codes**
  (`LIG`, `LG2`, …) — because `fold.js` writes the name with `.padEnd(3)` and `LIG2`
  truncates back to `LIG`. 🔴 Two different SMILES ligands sharing the code `LIG` was a
  real bug with an *invisible* half: `featuriseProtein` keys a ligand's **entity** on its
  code, so the two came out sharing an `entity_id` — telling the model that six carbons
  and a glycerol are two copies of one thing. The featuriser now keys on what the
  component **is** (element, charge, bonds). **Directly relevant: our loop names one
  ligand per run, so this stays single-ligand and safe — but any future multi-ligand
  variant must respect it.**
* **Element symbols matter for bonds.** The element table was once four entries (C, N, O,
  S) with everything falling through to carbon; across 51 hetero components **28 carry an
  element it dropped**. A viewer with no `CONECT` derives a ligand's bonds from
  inter-atomic distance between atoms of *known* element, so a disulfide at 2.05 Å read as
  C–C (ceiling 1.8) vanishes.
* **B-factor carries pLDDT.**

### 4.3 Superposition — already built, and fits on the target

**`shared/heads/superpose-pdb.js`** is exactly the Phase-3 requirement:

```js
coordinateAtoms(pdb) · alphaCarbons(atoms, wanted) · rewriteCoordinates(atoms, points)
superposePdb(superpose, pdb, reference, options = {})
superposeCycle(superpose, frames, settled, reference, options = {})
```

Its own header states both the need and the right fit:

> 🔴 *"AF3's sampler calls `randomAugmentation` at the top of every step, so consecutive
> **folds** sit in unrelated reference frames."*
> 🔴 *"AND THE FIT IS ON THE TARGET, NOT ON EVERYTHING. The target is the one thing that
> does not change across a run, so fitting on it holds it still and lets the binder move
> against it… Fitting on every atom instead splits the difference, so BOTH drift, and a
> binder that found its site looks like a binder that wandered."*

It also records why the viewer cannot do it: py2Dmol's `addFrame` superposes onto the
previous frame **only when the two have the same number of positions**, and a different
ligand changes the atom count every run.

The **Kabsch arithmetic is injected, not imported** (`superpose` is a parameter) — it is
py2Dmol's, which ships at `web/vendor/py2Dmol.full.min.js` (`sym3Eigen`, `mat3Mul`,
`mat3Det`, `calculateMean`, `matNx3Mul3`). `web/pdb-superpose.js` is the page's wiring.
So Phase 3 needs **no new geometry code** — only to pass run #1's PDB as `reference`.

---

## 5. The existing Boltz-2 path

### 5.1 What it is

* **Registry key** `boltz2`, menu label **Boltz-2**, bundle
  `boltz2-int5` @ HF rev `651a25ba8a57c6d18caacf50ab7180b255eb5a82`, local fallback
  `./model-boltz2-int5/`, gated by `LOCALFOLD_INCLUDE_BOLTZ2_MODEL`.
* **Checkpoint:** the **structure/confidence model only** — `coverage.included:
  ["diffuser"]`, from `boltz2.bin.zst` (af3-any-model's Haiku conversion of Boltz-2),
  **445 tensors, 24 shards, ~365 MiB, int5 g32**. `model.recycles: 0`.
* **Graph:** runs AF3's trunk + diffusion + confidence through `webgpu/af3/fold.js`
  with the `BOLTZ2` dialect.
* **Also runs on the WebGL2 fallback** — `WEBGL2_FAMILIES = ["af3","openbind0","boltz2",
  "protenix2","intellifold2","rosettafold3","mol1"]` (every AF3-lineage family but
  OpenDDE, which re-tokenises after the trunk and is refused by name).
  🔴 **This is brand new and landed in the commit we pinned.** `webgl/af3/trunk/template.js`
  took the fused template embedder on **2026-11-09** together with "boltz2's bond-order
  z-init terms and confidence re-embedding", each held to the WebGPU fold on the same
  bundle (`tools/gpu/fold-af3-webgl.js --seams`). `test/webgl2-families.test.js` holds the
  list to the lineage in both directions. Treat the WebGL2 Boltz-2 path as working but
  days old — WebGPU is the path to build and test against.
* Tensor scopes present: `diffuser/evoformer*` (169 + 27), `diffuser/~*` (141),
  `diffuser/confidence_head/*` (79), `diffuser/distogram_head` (2), and the
  boltz2-specific conditioning embeddings (`boltz2_res_type_encoding`,
  `boltz2_msa_profile_encoding`, `boltz2_mol_type_conditioning`,
  `boltz2_method_conditioning`, `boltz2_cyclic_conditioning`,
  `boltz2_modified_conditioning`, `boltz2_dn_res_type`, `boltz2_dn_msa_profile`).

### 5.2 Its dialect (the Boltz-2-specific conventions)

Selected by `dialectFor("boltz2")` → `BOLTZ2` in `shared/af3/dialect.js`. The flags that
distinguish it:

```
fusedTemplateEmbedder: true       realignSampler: true        msaDoubleAddPair: true
modifiedAsOneToken: true          keepLigandLeavingAtoms: true  dedupeSelfMsa: true
proteinOnlyMsa: true              ligandMsaUnk: true          zeroDeletions: true
profileOverAllRows: true          dropTerminalAtoms: true     symmetriseBonds: true
paddedAtomKeys: true              keyMaskedAtomAttention: true  distogramBias: true
msaUpdateBeforeOuterProduct: true perBlockPairLayerNorm: true   projectedRelpos: true
targetFeatAtomOnly: true          centreRefConformers: true     rawLastConformer: true
preSymmetrisedPde: true           msaPairedQueryRow: true     declaredPolymerBonds: "all"
atomizedUnknownRestype: true      templateModified: "unk_n"
sampler: { gamma0: 0.8, gammaMin: 1.0, noiseScale: 1.003, stepScale: 1.5,
           rho: 7.0, sigmaMin: 0.0001, sigmaMax: 160.0, scheduleTail: 1 }
```

Two with teeth for us. **`modifiedAsOneToken`** — Boltz-2 alone keeps a modified residue
in one token (see §5.5). And, not a dialect flag but a **weight tensor only Boltz-2
carries**: `tokenBondsTypeEmbed`, read by `cpu/af3/trunk/embedder.js:203` and
`cpu/af3/confidence/confidence.js:323` — Boltz-2's z-init takes a second **bond-order**
plane, and it is the only family that does. 🔴 That was two halves and neither showed
alone: the featuriser never built `bondOrderMatrix` (five consumers, no producer) *and*
`fold.js`'s embedder-input literal did not name it, so fixing the featuriser alone left
the fold **byte-identical**. Any new caller must name it in that literal.

### 5.3 Entities it accepts — ligands and SMILES **are supported today**

`web/entities.js`:

```js
export const CHAIN_TYPES = ["protein", "dna", "rna", "ligand", "smiles"];
```

plus a **`contact`** row type for declared covalent bonds (`A12:SG - B1:C25`), which is
not a chain. So the page already offers, under Boltz-2: protein, DNA, RNA, ligand by
**CCD code**, ligand by **SMILES**, modified residues, templates, and declared bonds.

The attached mockup is a live screenshot of exactly this: `Protein` + `Ligand (SMILES)
CCCc1ccccc1O`, model **Boltz-2**, MSA MMseqs2 128, *"Boltz-2 · 164 residues + CCCc1ccccc1O ·
in 10 s · 3148 MSA rows · 4 passes · pLDDT 98.3"*.

**Measured state of the ligand path** (`npm run test:ligand`, 68-residue protein +
glycerol, asserting five bond lengths): **boltz2 0.062 Å**, af3 0.061, protenix2 0.044,
if2 0.055, openbind0 0.058, rf3 0.069. The gate asserts **bond lengths** because nothing
else can see them — the fold's RMSD is dominated by 68 residues of protein, **pLDDT read
92.38 on a glycerol torn apart at 3.602 Å**, and `chain-geometry.js` steps over a ligand
by design.

**SMILES gates** (all green at this SHA): `test:smiles` (4 gates vs real RDKit over 74
molecules), `test:smiles-batch` (**43 fields × 5 ligands × 4 models = 0 differing** — the
model sees the same thing from a SMILES as from a CCD code), `test:smiles-path`
(same molecule both routes agree to 0.035 Å; a 62-atom paclitaxel core folds at bond rms
0.052 Å). The page admits **150 heavy atoms**.

### 5.4 Affinity — absent, with certainty

**Zero matches** for `affinit` (case-insensitive) across the whole repository: source,
`docs/` (all 27 files, 2.9 MB of findings), `CLAUDE.md`, `README.md`. There is:

* no affinity head, module, or kernel;
* no affinity tensors in the Boltz-2 bundle — `coverage.included: ["diffuser"]` is the
  structure/confidence model only, and the 445 tensor names contain no affinity scope;
* no `properties:`/`binder` handling in `web/job-json.js`;
* no record of anyone having looked at it.

So Phase 2's model work is: export `boltz2_aff.ckpt` through the §1.7 pipeline (under the
naming problem raised in §1.1) and port its forward pass as a **second** pass over a
completed structure prediction.

### 5.5 Known Boltz-2 defects and their state

* **Phosphoserine inflation — FIXED.** Boltz-2 inflated a modified residue 1.8× (N–CA
  2.007 Å vs 1.469) because this port atomised modified residues for everyone, where
  Boltz-2 keeps one token. `modifiedAsOneToken` → **1.813 → 0.938** against genuine
  Boltz-2 2.2.1's 0.986–1.011. Three things had to be right: the flag must be in
  `batch.js`'s forwarding list, the pseudo-beta is **CB** not slot zero, and an atom keeps
  its **component slot** so a removed leaving atom leaves a *hole*. (`docs/BOLTZ2_PTM.md`.)
* **Confidence head** was 8.26e-2 against the oracle while a self-comparison passed at
  9.56e-5 — found only by `check-af3-confidence-oracle.js`.
* **`target_feat`** read relRMS 1.00e+0 against the oracle while `check-af3-trunk` read
  2.74e-5 — "the difference between an oracle and a self-comparison."
* **Four negated tensors** in Boltz-2's export, found only by
  `check-bundle-vs-params.py`.
* **Current trunk parity** (own reference batch, f32): finished pair ≈ **4.56e-2** at
  `trunk_out_pair` is rf3's comparison point; the five-model sweep puts nothing above
  1.4e-3 at matched f32, ordered by pair width.
* **Open, documented, and ours to respect:** `realignSampler` and the sampler constants
  above are Boltz-2's own; the diffusion sampler is the **only** sampler (Flow and ODE
  were removed 2026-10-16) and `noFlowSampler` in the dialect makes `foldBatch` **throw**
  rather than switch silently.

### 5.6 What `foldBatch` returns (the integration surface)

```js
{ positions, trunk, targetFeat, scores /* {plddt, pae} */,
  ptm, iptm, chainPairIptm, chainPtm, chainIptm,
  meanPlddt, perResiduePlddt, atoms, pseudoBeta, recycleDeltas,
  structuralTokens, geometry /* backboneGeometry */,
  pdb /* toPdb(batch, positions, scores.plddt) */,
  reusable: { trunk, targetFeat, recycles } | undefined }
```

Everything the plots and CSV need — pLDDT, pTM, ipTM, PAE, per-residue pLDDT — is here.
An affinity pass would hang off `trunk`/`positions`/`scores`.

🔴 **On `reusable`, and the brief's warning:** it exists, but it is "what a caller hands
back to skip the trunk **next time**" for the **same batch** — re-sampling one input with
another seed. A different ligand is a different batch: ligand tokens enter the pair and
MSA representation (one token per heavy atom, §4.1), the token count changes, and
`boltz2`'s z-init reads a bond-order plane derived from that ligand. **Trunk reuse across
ligands is therefore invalid**, and nothing here suggests otherwise. The brief's
instruction was right and I am not proposing to relax it: the trunk runs per ligand.

---

## 6. Ligand handling — the decision, with the evidence

The brief asks to evaluate RDKit WASM and to report before implementing. **Both questions
are already answered in the tree, with measurements.**

### 6.1 Recommendation: keep LocalFold's own chemistry. Do not add `@rdkit/rdkit`.

`shared/chem/` is a complete, self-contained SMILES→3D stack:

| module | lines of responsibility |
|---|---|
| `smiles.js` | `parseSmiles`, `adjacency`, `molecularFormula` |
| `stereo.js` | `chiralCentres`, `doubleBondStereo` |
| `rings.js`, `kekulize.js` | ring perception; Kekulé assignment |
| `conformer.js` | `distanceBounds`, `smoothBounds`, `embedBounds`, `refineCoordinates`, `projectOntoBounds`, `planarQuadruples`, `linearTriples`, `signedVolume` — distance geometry + SHAKE-style constraint projection |
| `geometry-tables.js` | ideal bond lengths / VSEPR angles |
| `component.js` | **`smilesComponent(smiles, options)`** — the bridge; `ligandName`, `nameSmilesLigands`, `hashOf` |
| `webgpu/chem/conformer.js` | the same refinement on the GPU |

**Why not RDKit** — `CLAUDE.md`'s row, measured rather than argued:

> 🔴 *"Because MinimalLib is **2D ONLY**, which decides it before size does. Run here on
> biotin: every z is `0.0000`, the molblock header reads `RDKit 2D`, and the only
> coordinate entry points are `set_new_coords`/`generate_aligned_coords` — CoordGen, a
> **depiction** library. No `EmbedMolecule`, no ETKDG, no distance geometry, so it cannot
> produce `ref_pos`, which is the one thing the models need and the hard half of the
> work."*

And the size: **2.39 MB gzipped against this port's 50 KB** (18 KB comments stripped) —
48× the bytes — to replace a parser already at **74/74** against real RDKit. A custom WASM
build goes the wrong way (adding DistGeom makes it bigger). OpenChemLib at ~400 KB gz is
the only real alternative and would still need every gate repointed at it.

**What the existing path is validated to** (`npm run test:smiles`, against real RDKit as
an *oracle*, not a dependency — nothing under `shared/` imports anything derived from it):

* graph **74/74**; stereo **43/43** tetrahedral, **2/2** double-bond;
* conformer **0.022 Å** mean bond and **2.61°** mean angle from RDKit's MMFF, every
  aromatic ring within **0.003 Å** of flat;
* **49/49** chiral centres built with the right hand;
* a case-free fourth gate: RDKit rewrites each molecule as arbitrarily many different
  SMILES (different start atom, branch nesting, ring-closure digits) and every one must
  parse to the same graph with the same hand — **1148 rewritings, 1622 centre checks**, and
  *it found nothing*, "which is the evidence the parser is done."

Two honesty notes carried forward: the corpus was **doubled after the first 51 all
passed, and that found three more bugs**; and the conformer reached that quality only
after four fixes (a collapsing embedding axis, a flattened sulfoxide, gradient descent
replaced by constraint projection, adaptive attempt counts, a VSEPR lean, and rings
solved as cyclic polygons).

**Bonus for a screening loop:** `webgpu/chem/conformer.js` is **16.3× faster at 256
conformers** (host 670 ms → device 41 ms), and the device cost is **flat** (35 ms from 1 to
256) where the host's is linear. It *loses 5× on a single ligand* — WGSL forbids a barrier
inside control flow depending on a workgroup reduction, so the kernel cannot stop early and
runs a fixed budget. **So: build all N ligand conformers in one batched dispatch up front,
not one per run.** `bench-smiles-conformer.js` is the bench;
`check-smiles-conformer-gpu.js` the differential.

### 6.2 The integration is one function, by design

`shared/chem/component.js`'s header is explicit that SMILES support is *not* a second
featuriser:

> 🔴 *"A ligand reaches AF3's featuriser as `{code, atoms: [{name, element, charge, x, y,
> z, leaving}], bonds: [{from, to, order}]}` and as nothing else … So SMILES support is
> not a second featuriser or a second code path through the model: it is a second way of
> producing this object, and everything downstream cannot tell which one it got."*

So the screening loop feeds `smilesComponent(smiles, { code })` into the same
`featuriseProtein` → `foldBatch` path the page already runs.

### 6.3 CCD assets: **not needed**, and droppable for our scope

* Upstream's `ccd.pkl` / `mols.tar` have **no counterpart here** and are referenced
  nowhere.
* **33 standard components** (amino acids, nucleotides) are compiled into
  `shared/af3/featurise/standard-components.js`, generated offline by
  `tools/gen-standard-components.mjs` from the wwPDB `components.cif`. `modified-parents.js`
  is generated the same way.
* A ligand named by **CCD code** fetches **one** `.cif` on demand from
  `https://files.rcsb.org/ligands/download/<CODE>.cif` (`ccdUrl` in `ccd-component.js`) —
  no bundled dictionary, no `ccd.pkl`.
* A ligand named by **SMILES** needs **none of that**: `smilesComponent` produces the
  component object directly.

**Decision: our scope is SMILES-only, so the CCD path is never entered and no CCD asset
ships.** One consequence to keep: a SMILES ligand has no dictionary entry, so AF3's own
`_ccd_bond_orders` gives such bonds `OTHER` (genuine Boltz-2 uses RDKit's) — a reported,
understood deviation in `test:batch`, not a defect. Atom names are AF3's
"molecule with no dictionary names" convention (every atom numbered: biotin's lone sulfur
is `S1`, not `S`).

---

## 7. MSA client

**`shared/input/mmseqs2-api.js`**, `DEFAULT_API_URL = "https://api.colabfold.com"`,
`QUERY_ID = 101`. The **only** request the page makes off the machine.

* **Search:** `generateMmseqs2Msa(sequence, options)` (one chain),
  `generateMmseqs2ComplexMsa(sequences, options)` (complex, adds the paired request),
  `generateMmseqs2PairedMsa`. Progress is reported as `{phase, status, elapsedMilliseconds}`;
  transient states `UNKNOWN/PENDING/RUNNING/RATELIMIT` are retried.
* **Results** arrive as a tar: `readTarFiles`, `extractMmseqs2A3m(tar, useEnvironmental)`,
  `extractMmseqs2PairedA3ms`, `extractMmseqs2TemplateHits`, `fetchMmseqs2Templates`.
  `gunzip` is local (`shared/input/inflate.js`).
* **Per-model merge:** `CHAIN_MERGES` — `monomer` block-diagonal, `multimer` dense within
  an entity, and **every AF3-lineage family (Boltz-2 included) row-aligned**, *derived*
  from `AF3_FAMILIES` rather than listed.
* **Deduplication:** `dedupeSelfMsa` — the query used to appear **twice** (both the uniref
  and environmental blocks start with their own `>101`). On a deep alignment worth
  *nothing* (Δ pLDDT −0.005 ± 0.288), but on a **query-only search it changes the answer**:
  pTM 0.4148 → 0.3965, a 4.6% shift on a number the page shows.
* **Single-sequence mode:** `singleSequenceA3m(sequences, model)`, plus
  `singleSequenceIfOnlyQuery` / `foldsAsSingleSequence` — an alignment carrying only the
  query becomes a single-sequence fold from **all four inputs** (searched, pasted,
  uploaded A3M, uploaded archive), keeping the search's template hits. 🔴 There is **no
  speed-up in it** (855/840/843 ms across arms) — it is correctness and honesty.
  Reachable from the UI as *Single sequence*; nothing is sent.

### 7.1 MSA reuse across ligands — **already works, and here is why**

`planSearchReuse({ cache, chains, family })` keys the cache on
`JSON.stringify(chains)`, and the call site is:

```js
// web/app.js:5111, 5167
const proteinChains = chains.filter((_, index) => chainKinds[index] === "protein");
alignmentResult = proteinChains.length === 0
  ? null : await alignmentText(proteinChains, signal, family, proteinWantsMsa);
```

**The cache key contains protein chains only.** A SMILES ligand is `chainKinds ===
"smiles"` and is filtered out before the key is built. So for one fixed target and N
ligands, ligands 2…N hit `plan.reuse === "single"` and the status line reads
**"MSA reused · <depth> sequences"** — **no code change required.** `searchCacheEntry`
also keeps the **template hits** with the alignment (they come from the same tar and cost
nothing), after a bug where a reused search reported no hits.

**What that saves, precisely** (stated as the brief demands): it removes the **repeated
MMseqs2 search** — tens of seconds each, network-bound. It does **not** avoid the trunk:
ligand tokens enter the pair/MSA representation, so **the trunk runs per ligand**. §5.6
explains why trunk reuse is invalid here.

**The one genuine gap:** the cache is `let searchCache;` (`web/app.js:2869`) — **in-memory
only**, so a reload re-searches. The brief's CacheStorage/IndexedDB persistence keyed by
sequence hash is a real, small addition. `hashOf(text)` already exists in
`shared/chem/component.js`.

---

## 8. What Phase 1 establishes for Phase 2

**Already present and reusable — no porting needed:**

1. Boltz-2 structure + confidence, int5, on WebGPU **and** WebGL2.
2. SMILES → component → featuriser → fold, validated against RDKit on 74 molecules; GPU
   batch conformer generation at 16.3×.
3. MSA search, per-model merge, single-sequence mode, and **ligand-independent search
   reuse**.
4. Kabsch superposition that **fits on the target** (`superposePdb`).
5. GPU memory budgeting, residency, cross-model teardown (`releaseAllWeights`), host
   spill, pooling, abort, and device-loss recovery.
6. PDB writer with HETATM ligands, 62 chain letters, pLDDT in B-factor;
   `fold-archive.js` + `zip.js` for ZIP export; `prediction-results.js`, `scores-card.js`,
   py2Dmol viewer.
7. The complete quantise → shard → manifest → register → digest-verify export pipeline.

**Genuinely missing — the Phase 2 scope:**

1. **The affinity module.** Zero prior art. Needs: `boltz2_aff.ckpt` exported through
   §1.7, and its forward pass ported as a second pass. **Open problem raised in §1.1:**
   the existing bundle's names are Haiku paths from af3-any-model's conversion, and the
   affinity checkpoint is PyTorch with no Haiku counterpart — the naming convention for
   this export is an unforced decision that must be made deliberately.
2. **The two affinity outputs** (`affinity_probability_binary`,
   `affinity_pred_value` = log10 IC50 in μM, + per-ensemble `…1`/`…2`), kept separate and
   labelled, with pIC50 = 6 − value.
3. **`properties: affinity: binder: <chain>`** in `web/job-json.js`.
4. **The batch loop + results UI**: queue, sequential run, streaming results, per-ligand
   status, ETA from measured throughput, stop/resume, partial download on failure.
5. **Scatter + ranked table**, live-updating.
6. **Multi-run scene assembly**: fold the target once, superpose each run onto run #1 via
   the existing `superposePdb`, per-ligand show/hide and colour-by-score.
7. **MSA persistence** to CacheStorage/IndexedDB keyed by sequence hash (small).
8. **`.smi` parsing** and lazy affinity-weight fetch.

The brief's expectation was **(a) affinity module, (b) ligand-from-SMILES featurisation,
(c) batch loop + results UI**. Corrected: **(a) and (c) are right; (b) is already done**
and is one of the better-tested parts of the codebase. The effort (b) would have taken
should be redirected to (a), where the checkpoint-naming problem is the real work.

---

## 9. Licensing (for the README, per the working agreement)

* **LocalFold code:** Beerware (Revision 42) — `LICENSE`.
* **Boltz-2 weights:** MIT (`boltz-community/boltz-2`). Not redistributed by LocalFold;
  the bundle names the HF revision it fetches from.
* 🔴 **AF3 parameters carry DeepMind's own terms** (CC BY 4.0 parameters licence +
  Prohibited Use Policy) and are **not redistributed**. `export_af3_model.py` records
  *which* checkpoint filled a bundle in `manifest.model.name`, and `build_site.py` reads
  it and **requires `LOCALFOLD_ACCEPT_MODEL_TERMS` before publishing a restricted one**;
  the page gates the download behind licence acceptance. **Our build must not
  redistribute AF3 parameters.** Since our scope is Boltz-2 only, the clean move is to
  ship/point at the `boltz2-int5` bundle alone and keep the AF3 option out.
* **py2Dmol** — vendored in `web/vendor/`, its own licence.
* **Upstream credit:** forked from Martin Steinegger's AlphaFold2-WebGPU (first eight
  commits are his); models beyond AF2 reach the browser via `sokrypton/alphafold3`.
* **ColabFold / MMseqs2** must be cited if the search is used (Mirdita et al. 2022).
