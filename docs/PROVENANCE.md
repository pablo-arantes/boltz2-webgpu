# Provenance: what is pinned, what is ported, what is new

Split out of the README so that file can stay short. Nothing here is
abridged.

## Pinned revisions

**LocalFold** — this repository is a clone of this commit, and every "ported from
LocalFold" claim below is against this tree:

```
https://github.com/sokrypton/localfold
357518613af612b4e1e887deecd93bd612fdac76   2026-11-09
```

**Upstream Boltz** — read as a reference, never vendored:

```
https://github.com/jwohlwend/boltz
b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc   v2.2.1-6-gb1ebfc4
```

**Weights** — `boltz-community/boltz-2` at repo revision
`6fdef46d763fee7fbb83ca5501ccceff43b85607`:

| file | bytes | what |
|---|---|---|
| `boltz2_conf.ckpt` | 2,286,561,469 | structure; LocalFold's `boltz2` bundle derives from this |
| `boltz2_aff.ckpt` | 2,062,139,170 | affinity; **98.3 % redundant with the above** |
| `mols.tar` | — | CCD assets, **not needed** — SMILES-only scope |

🔴 **The two checkpoints share 5,019 tensors byte for byte.** Verified
exhaustively, not sampled, by `tools/g0_verify.py` (CRC-32 and size per
tensor from the ZIP central directory, escalating to sha256). That is why the
affinity bundle is **520 tensors, 6.4 MiB** rather than a second 500 M-parameter
export, and why no second trunk is downloaded.

## What is ported from LocalFold

Used as it stands, or adapted with its own gates still passing. None of this
was written here:

- **The whole weight pipeline** — `export_af3_model.py` → `quantize_af3.py` →
  `repack_shards.py` → `write_manifest_module.py`, the `MODEL_BUNDLES` registry
  with revision-pinned remotes, `HttpTensorStore` and its CacheStorage shard
  cache, and `build_site.py`'s digest verification. The affinity bundle goes
  through it unchanged: int5 asymmetric, group 32, float16 scale **and** zero
  point.
- **The AF3-lineage model** in its boltz2 dialect — trunk, MSA stack, diffusion
  sampler, confidence head, atom encoder, `target_feat`.
- **Every pair kernel the affinity stack runs on** — triangle multiplication in
  both directions (`webgpu/af3/trunk/triangle.js`), grid attention
  (`grid-attention.js`), the SwiGLU transition (`transition.js`). The affinity
  pairformer is a new *driver*, not new arithmetic.
- **The featuriser** — `shared/af3/featurise/`, including the `atomGathers`
  windowing rule and the dense-24 per-token atom layout.
- **SMILES** — the parser and the distance-geometry conformer builder in
  `shared/chem/`, gated against real RDKit over 74 molecules and 1,148
  re-writings.
- **The runtime** — `GpuBufferAllocator`, the memory budget and
  `GpuMemoryBudgetError`, weight residency and release, the pipeline cache that
  indexes by source, driver-reset recovery.
- **The MSA client** (`shared/input/mmseqs2-api.js`), the PDB writer, the ZIP
  writer, py2Dmol and its viewer DOM, and the Horn/Kabsch superposition in
  `web/pdb-superpose.js`.

## What is reverse-engineered from upstream Boltz

Written here, from reading upstream's source and holding the result to
upstream's own PyTorch output. **Nothing is vendored** — no upstream Python is
copied into this tree.

| upstream | ported to | lines |
|---|---|---|
| `modules/affinity.py` `AffinityModule.forward` | `cpu/af3/affinity/affinity.js` (f64 reference) + `webgpu/af3/affinity/affinity.js` | 458 + 253 |
| `modules/affinity.py` `AffinityHeadsTransformer` — **no transformer: 18 tensors of MLP** | the same pair | — |
| `layers/pairformer.py` `PairformerNoSeqModule` | `webgpu/af3/affinity/pair-stack.js`, a **pair-only driver** over LocalFold's kernels | 194 |
| `modules/encodersv2.py` `PairwiseConditioning` (Algorithm 21) | the same, over the transition kernel | — |
| `data/crop/affinity.py` `AffinityCropper` | `shared/af3/affinity/crop.js` | 220 |
| `data/feature/featurizerv2.py` `process_msa_features(affinity=True)` | `shared/af3/affinity/features.js` — the depth-1 profile | 104 |
| `center_idx` / `disto_idx` | `shared/af3/affinity/centre-atoms.js` | 122 |
| the crop's re-featurisation | `shared/af3/affinity/featurise-crop.js` | 253 |
| `data/parse/schema.py` MW and size limits | `shared/chem/molecular-weight.js` + `shared/chem/affinity-limits.js` | 220 + 96 |
| `models/boltz2.py` pre-mask, best-by-ipTM, ensemble, MW correction | `web/affinity-run.js` | 341 |
| the weight layouts | `shared/af3/weights/affinity-weights.js` | 323 |

**Deliberately not ported**, because upstream never runs them: `GaussianSmearing`,
`use_cross_transformer`, `groups`, and `transformer_args.{num_blocks,num_heads}`.
They are dead in the checkpoint this reads.

**One new kernel.** `webgpu/af3/affinity/pair-project.js` — a fused pair
LayerNorm + bias-free projection. It exists because measurement refuted the
argument that the host side could not matter: the host was ~10 s a ligand, and
this took a ligand from **16,673 ms to 2,332 ms**, with 82 % of that spent on
the GPU.

**New here, not from Boltz:** `shared/input/smi.js` (`.smi` intake),
`shared/input/msa-cache.js` (persisted alignment), `web/screen*.js` and
`screen.html` (the screening page, its scatter, table, scene and ZIP).
