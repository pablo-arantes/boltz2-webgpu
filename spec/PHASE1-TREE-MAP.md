# Annotated map of the LocalFold tree

`sokrypton/localfold` @ **`357518613af612b4e1e887deecd93bd612fdac76`** (2026-11-09).
Annotations mark what the Boltz-2 virtual-screening project **reuses (R)**, **extends (E)**,
**must write new (N)**, or can **ignore (—)** for a browser-only, protein+SMILES scope.

```
localfold/
├── AGENTS.md                      R  engineering invariants. Binding on our changes:
│                                     no CPU fallback for NN ops in production paths;
│                                     every GPU kernel needs a deterministic differential
│                                     test vs an independent reference; no O(L^3) storage;
│                                     all GPU allocations through the shared allocator;
│                                     plain ES modules, relative specifiers, .js extension,
│                                     NO BUILD STEP AND NO BUNDLER; shapes in JSDoc.
├── CLAUDE.md               258 KB  R  operational manual + ~500 measured findings as a
│                                     question->tool table. The single most useful file.
│                                     ⚠ describes cuda/ metal/ python/ which are ABSENT here.
├── HANDOFF.md / docs/HANDOFF.md   —  cross-machine handoff state
├── README.md                      R  licences + acknowledgments; model of what ours must say
├── LICENSE                        R  Beerware r42 (code only; weights excluded)
├── index.html               88 KB  E  THE PAGE. Model <option> list (af3 … boltz2 …),
│                                     entity rows, MSA controls, viewer panes, download row.
│                                     Byte-identical to the live site at this SHA.
├── dev.html                       —  dev harness
├── package.json              8 KB  R  32 npm scripts; every gate is here, with a
│                                     "//name" comment line above most explaining the trap
│                                     it exists to catch. No `stamp` script (brief was wrong).
├── playwright.config.js           —  test/browser runner (needs a browser installed)
├── tsconfig.json                  —  ADVISORY typecheck only; 1075 errors on this tree
│
├── shared/                           ===== BACKEND-INDEPENDENT: what every backend reads =====
│   ├── af3/
│   │   ├── dialect.js       1920 L  R  ~120 per-family convention flags as frozen objects
│   │   │                               (ALPHAFOLD3, BOLTZ2, PROTENIX2, OPENDDE, OPENBIND0, …).
│   │   │                               `dialectFor(model)`; `featuriserDialect(dialect)`.
│   │   │                               ** BOLTZ2 is the one we run. **
│   │   ├── featurise/
│   │   │   ├── featurise.js         R  `featuriseProtein(sequence, options)` — THE featuriser.
│   │   │   │                           A ligand = ONE TOKEN PER HEAVY ATOM.
│   │   │   ├── batch.js             E  `af3BatchFromA3m(sequence, alignment, options)` —
│   │   │   │                           the page's entry. Forwards the dialect FIELD BY FIELD;
│   │   │   │                           a key not named here is a key thrown away.
│   │   │   ├── ccd-component.js     R  `parseCcdComponent`, `ELEMENT_SYMBOLS`, `ccdUrl()`
│   │   │   │                           (files.rcsb.org, on demand, per code)
│   │   │   ├── standard-components.js R 33 compiled-in CCD components (generated)
│   │   │   ├── reference-conformers*.js R one idealised set shared by every family
│   │   │   ├── boltz2-conformers.js  R  Boltz-2's own conformer conventions
│   │   │   ├── modified-parents.js   —  generated; modified-residue parent table
│   │   │   ├── msa-features.js       R  MSA -> features, per dialect
│   │   │   ├── template-features.js  —  out of scope (no templates in our flow)
│   │   │   ├── template-fused-features.js — boltz2's fused embedder (inert without a template)
│   │   │   ├── declared-leaving.js   —  leaving atoms at declared bonds
│   │   │   ├── constraints.js        —  pocket/contact constraints
│   │   │   ├── structural-tokens.js  —  OpenDDE's second token space
│   │   │   └── batch-from-dump.js    —  oracle-dump path (testing)
│   │   ├── structure/
│   │   │   ├── pdb.js               R  `toPdb(batch, positions, plddt)`, THREE_LETTER,
│   │   │   │                           `atomName`, `pdbLinkRecords`, `pdbSlots`.
│   │   │   │                           62 chain letters (A-Z a-z 0-9); HETATM ligands;
│   │   │   │                           pLDDT in B-factor; numbering walks TOKENS.
│   │   │   └── warm-start.js        —  warm-start coordinates
│   │   ├── chain-geometry.js        R  backbone geometry check. ⚠ STEPS OVER A LIGAND
│   │   │                               by design — cannot validate ligand placement.
│   │   ├── weights/weights.js       R  `af3Dialect(store)`; the AF3 weight loader
│   │   ├── weights/diffusion-weights.js R diffusion weight packing
│   │   ├── feature-convergence.js   —  recycle convergence signal
│   │   ├── random.js                R  `uniformFrom`, `normalFrom` — seeded RNG
│   │   ├── msa-draw.js              —  MSA panel rendering data
│   │   └── diffusion/chiral-gradient.js — chirality steering
│   │
│   ├── chem/                         ===== SMILES -> 3D. ALREADY COMPLETE. =====
│   │   ├── component.js             R  ** `smilesComponent(smiles, {code})` — THE BRIDGE. **
│   │   │                               Returns exactly `parseCcdComponent`'s shape, so
│   │   │                               nothing downstream can tell which produced it.
│   │   │                               Also `ligandName(i)` (3-char, LIG/LG2/…), `hashOf`.
│   │   ├── smiles.js                R  `parseSmiles` — 74/74 vs real RDKit; 1148 rewritings
│   │   ├── stereo.js                R  `chiralCentres`, `doubleBondStereo` — 49/49 right hand
│   │   ├── conformer.js             R  distance geometry + SHAKE projection: `distanceBounds`,
│   │   │                               `smoothBounds`, `embedBounds`, `refineCoordinates`,
│   │   │                               `projectOntoBounds`, `signedVolume`.
│   │   │                               0.022 A bond / 2.61 deg vs RDKit MMFF.
│   │   ├── rings.js, kekulize.js    R  ring perception; Kekule assignment
│   │   ├── geometry-tables.js       R  ideal lengths / VSEPR angles
│   │   └── crosslinkers.js          —  af3x linkers (not done upstream either)
│   │
│   ├── bundles/                      ===== WEIGHT LOADING =====
│   │   ├── manifests/
│   │   │   ├── index.js       30 KB  E  ** `MODEL_BUNDLES` — THE REGISTRY (the real
│   │   │   │                            extension point; `localfold.json` does not exist). **
│   │   │   │                            Keys: model, directory, remote (revision-PINNED HF
│   │   │   │                            URL), release, variable, load (literal import()),
│   │   │   │                            +delta, +noTemplateEmbedder. Also `AF3_FAMILIES`,
│   │   │   │                            `graphFamily()`, the WebGL2-capable list.
│   │   │   ├── boltz2.js     125 KB  R  ** our bundle: 445 tensors (269 int5/176 f32),
│   │   │   │                            24 shards, 382,207,764 B, int5 g32 asym,
│   │   │   │                            f16 scale+zero, Haiku tensor names, shardDigests. **
│   │   │   ├── af3.js, protenix2.js, opendde.js, openbind0.js, intellifold2.js,
│   │   │   │   rosettafold3.js, mol1.js  —  other AF3-lineage bundles.
│   │   │   │                            ⚠ five declare bundle.encoding "float32-le" while
│   │   │   │                            holding int5 tensors — stale, and harmless only
│   │   │   │                            because nothing reads it (dtype.js dispatches
│   │   │   │                            per-tensor). Do not copy that field's value.
│   │   │   ├── monomer*.js, multimer*.js  —  AF2; the *-2..5 are int3 g128 DELTAS
│   │   │   ├── esmfold2.js, ef2-fast-300m.js, esmc*.js, bioemu2.js  —  other families
│   │   ├── http-tensor-store.js 30 KB R ** `HttpTensorStore.open(url, …)`; the CacheStorage
│   │   │                               layer: `shardCacheName` =
│   │   │                               localfold-shards-<stem>@<hash(url)>-<token>,
│   │   │                               `staleShardCaches` (sweep NOT awaited),
│   │   │                               8 concurrent, 5 retries, 429/5xx backoff,
│   │   │                               2 s cache timeout, `tensorDecodeStats`. **
│   │   ├── delta-tensor-store.js    —  AF2 delta reader
│   │   ├── tensor-store.js / node.js / script-tensor-store.js  R  store interface + Node lane
│   │   ├── manifest.js              R  manifest helpers
│   │   └── alphafold-fixture.js     —  AF2 fixture loader
│   ├── weights/
│   │   ├── dtype.js                 R  ** THE AUTHORITATIVE READER. `readTensor`,
│   │   │                               `readTensorRange`, `readTensorAsFloat16`,
│   │   │                               `tensorByteLength`, `packedBits`. Dispatches on
│   │   │                               record.dtype: int8 | int5 | float16 | float32. **
│   │   ├── float16.js               R  f16 conversion (tested against the platform's own)
│   │   ├── weight-pack.js           R  packing for upload
│   │   └── weight-sources.js        R  ⚠ `=== undefined` is the WRONG presence test here:
│   │                                   the loader writes `null`, and that null reaching the
│   │                                   SOURCES map once killed boltz2.
│   ├── input/
│   │   ├── mmseqs2-api.js           E  ** api.colabfold.com client: generateMmseqs2Msa /
│   │   │                               ComplexMsa / PairedMsa, extractMmseqs2A3m,
│   │   │                               extractMmseqs2TemplateHits, CHAIN_MERGES,
│   │   │                               `planSearchReuse` + `searchCacheEntry` (the reuse
│   │   │                               cache — key is PROTEIN CHAINS ONLY, so a ligand
│   │   │                               change does NOT invalidate it), `singleSequenceA3m`. **
│   │   ├── a3m.js, a3m-features.js  R  A3M parsing and feature construction
│   │   ├── chains.js                R  merges; `deduplicateUnpairedAgainstPaired`;
│   │   │                               `mergeJobAlignments`
│   │   ├── inflate.js               R  local gunzip (no dependency)
│   │   └── query-only-features.js   R  single-sequence features
│   ├── heads/
│   │   ├── superpose-pdb.js         R  ** `superposePdb(superpose, pdb, reference, opts)`;
│   │   │                               `coordinateAtoms`, `alphaCarbons(atoms, wanted)`,
│   │   │                               `rewriteCoordinates`, `superposeCycle`.
│   │   │                               FITS ON THE TARGET, NOT ON EVERYTHING.
│   │   │                               Kabsch is INJECTED (py2Dmol's `superpose`).
│   │   │                               => Phase 3 needs no new geometry code. **
│   │   ├── tm-score.js              R  TM-score
│   │   ├── distogram.js             R  distogram -> contacts
│   │   └── contact-threshold.js     R  contact probability threshold
│   ├── esmfold2/*, bioemu2/*, af2/* —  other families
│
├── cpu/                              ===== f64 REFERENCES (differential-test targets) =====
│   ├── af3/trunk/{embedder,pairformer,msa,template,trunk}.js   R  reference for each stage
│   ├── af3/diffusion/{diffusion,diffusion-sampler,atom-encoder, R
│   │                  atom-conditioning,rigid-align,steering}.js
│   ├── af3/confidence/confidence.js R  `plddtByRank`
│   ├── af3/structure/*              R
│   ├── design/mpnn/*, esmc/, esmfold2/, bioemu2/  —  other paths
│
├── webgpu/                           ===== THE PAGE'S BACKEND =====
│   ├── af3/fold.js                  E  ** `foldBatch(device, batch, weights, options)` —
│   │                                   THE ENTRY. Returns positions, scores{plddt,pae},
│   │                                   ptm, iptm, chainPairIptm, chainPtm, chainIptm,
│   │                                   meanPlddt, perResiduePlddt, geometry, pdb,
│   │                                   reusable{trunk,targetFeat,recycles}.
│   │                                   Re-exports toPdb/THREE_LETTER/dialects.
│   │                                   `warmTrunkPipelines`, DIFFUSION_DEFAULTS.
│   │                                   ** Affinity hangs off here. ** **
│   ├── af3/trunk/                   R  trunk.js, pairformer-block.js, triangle.js,
│   │                                   grid-attention{,-matrix}.js, grid-project-matrix.js,
│   │                                   single-attention.js, msa-attention.js, msa-stack.js,
│   │                                   outer-product-mean.js, transition.js, template.js,
│   │                                   embedder.js, pair-track.js, pair-width.js,
│   │                                   sequential-groups.js, grid-attend-race.js
│   ├── af3/diffusion/               R  diffusion-head.js, diffusion-transformer.js,
│   │                                   diffusion-sampler.js, diffusion-conditioning.js,
│   │                                   atom-encoder.js, atom-decoder.js
│   ├── af3/confidence/confidence.js R  pLDDT / PAE / pTM / ipTM head
│   ├── af3/weights/{device-weights,pair-track-device-weights}.js R on-device weight decode
│   ├── af3/structure/structural-expander.js  —  OpenDDE only
│   ├── chem/conformer.js            R  ** device conformer: 16.3x at 256 conformers,
│   │                                   FLAT in batch size (35 ms for 1..256), loses 5x on
│   │                                   ONE ligand (WGSL forbids the early exit).
│   │                                   => batch all N ligands' conformers up front. **
│   ├── kernels/                     R  attention{,-matrix,-project-matrix,-geometry}.js,
│   │                                   matrix-linear.js, outer-product-mean.js,
│   │                                   transition.js, vector-gemm-block.js, triangle/*.
│   │                                   SHARED ACROSS FAMILIES (hence cache collisions).
│   ├── runtime/                     R  ** the memory + execution layer we reuse wholesale:
│   │   ├── device-memory.js            `setMemoryBudget`, `budgetForDevice(4)`,
│   │   │                               `GpuMemoryBudgetError` (raised BEFORE createBuffer —
│   │   │                               over-allocation here is a FROZEN MACHINE, not an
│   │   │                               error), `memorySnapshot`, `residencyAllowed`
│   │   ├── resident.js                 `residentWeightBuffer`, `releaseResidentWeights`,
│   │   │                               ** `releaseAllWeights(device)` ** (the fix for
│   │   │                               folding several models/runs in one page),
│   │   │                               `residentPackStats`
│   │   ├── allocator.js                `GpuBufferAllocator` (pools by byteLength:usage,
│   │   │                               RETIRES rather than destroys — leaks by design
│   │   │                               because a released buffer is still named by an
│   │   │                               in-flight command buffer), `settleReleasedMemory`
│   │   ├── host-spill.js               `HostSpill`, 64 MiB pieces (browser refuses one
│   │   │                               ArrayBuffer past ~1 GiB)
│   │   ├── abort.js                    `predictionAbortError()` — stop/resume
│   │   ├── device.js                   adapter/device acquisition
│   │   ├── device-profile.js           ~50 tuning knobs + per-device priors; all the
│   │   │                               chunking lives here (pairTransitionChunkBytes,
│   │   │                               transitionChunkBytes, opmPairBlockBytes,
│   │   │                               attentionKeyChunk, msaAverageRows, diffusionSplitK…)
│   │   ├── pipeline-cache.js           `ComputePipelineCache` — indexes by entryPoint+SOURCE
│   │   │                               as well as by key, so a collision is refused and
│   │   │                               NAMED rather than handing a caller someone else's
│   │   │                               kernel
│   │   ├── execution.js                dispatch; checks binding offset alignment itself
│   │   ├── validation.js, storage.js, elementwise.js, segmented.js,
│   │   ├── queue-pacing.js, yield.js, occupancy.js, cost-model.js,
│   │   └── shader-source-cache.js, device-pack.js **
│   ├── weights/quantised-upload.js  R  `planBlockUpload(…, {accumulate})`, `blockUploadStats`
│   ├── input/{a3m-features,nearest-centres}.js  R  device featurisation
│   ├── heads/{confidence,distogram}.js  R
│   ├── esmfold2/*, esmc/*, bioemu2/*, af2/*  —  other families
│   └── index.js / node.js           R  backend entry points
│
├── webgl/                            R  WebGL2 FALLBACK — `WEBGL2_FAMILIES` = every
│                                        AF3-lineage family EXCEPT OpenDDE. Boltz-2 runs
│                                        here too, so a no-WebGPU browser still works.
│                                        ⚠ boltz2's WebGL2 path (fused template embedder,
│                                        bond-order z-init, confidence re-embedding) landed
│                                        in the commit we pinned — 2026-11-09. Days old.
│                                        `runtime/int5.js` decodes our codec in GLSL.
│
├── web/                              ===== THE PAGE =====
│   ├── app.js             ~6 KL   E  ** the orchestrator. Fold entry, `alignmentText()`
│   │                                  (MSA resolution + the reuse cache at :1554),
│   │                                  `searchCache` (:2869, IN-MEMORY ONLY — our
│   │                                  persistence gap), `releaseAllWeights` on family
│   │                                  change (:5035), `proteinChains` filter (:5111),
│   │                                  `setMemoryBudget` (:6039), `foldStem` (one resolver
│   │                                  for all fold names), `syncDownloads`,
│   │                                  `recordPrediction`, status line. **
│   ├── model.js                   R  bundle/device acquisition; `bundleBaseUrl =
│   │                                  remote ?? directory`; caches a store BY FAMILY;
│   │                                  ⚠ neither a rejection nor a LOST DEVICE stays
│   │                                  memoised (:173-193) — driver-reset recovery.
│   ├── af3-model.js               E  AF3-lineage page glue; `samplerModeFor`
│   ├── entities.js                E  ** `CHAIN_TYPES = [protein,dna,rna,ligand,smiles]`,
│   │                                  `isChainEntity`, `parseBond` (contact rows). **
│   ├── entity-ui.js               E  the entity rows. ⚠ the BLUR handler once ran a
│   │                                  SMILES through `cleanSequence` and turned benzene
│   │                                  into hexane; `.entity-value-ligand` is
│   │                                  text-transform:uppercase (would display c1ccccc1
│   │                                  as C1CCCCC1). Both fixed; both are traps for us.
│   ├── job-json.js                E  ** job read/write. The OPEN vs SERVER dialect rule:
│   │                                  a SMILES ligand forces the OPEN dialect (the server
│   │                                  dialect has no SMILES field), and `dialect` must be
│   │                                  written EXPLICITLY. ** `properties: affinity:` goes here.
│   ├── fold-archive.js            E  ** `buildFoldArchive` — the AF3-server-shaped ZIP
│   │                                  (structure, confidences, PAE, contacts, the request,
│   │                                  one alignment per chain). Our CSV+structures ZIP
│   │                                  extends this. **
│   ├── zip.js                     R  ZIP writer
│   ├── prediction-results.js      E  result objects the panels read
│   ├── scores-card.js             E  the pLDDT/pTM/ipTM card — where affinity labels go
│   ├── viewer.js                  R  py2Dmol wiring
│   ├── pdb-superpose.js           R  page-side superposition wiring
│   ├── morph.js                   R  frame playback; `superposeOnto()` (same atom count only)
│   ├── align.js, sequence.js, plddt.js, hydrophobicity.js, mutate.js  R  panels
│   ├── fold-session.js            R  session save/restore
│   ├── link-builder.js            R  shareable URL
│   ├── template-source.js         —  template fetch (out of scope)
│   ├── native-bridge.js           —  Colab/CUDA bridge (no backend in this repo)
│   ├── impossible-fold.js         R  refusals
│   ├── bioemu2-*.js               —  other family
│   ├── dev*.js                    —  dev panel
│   ├── vendor/py2Dmol.full.min.js R  ** THE VIEWER + the Kabsch `superpose` that
│   │                                 superpose-pdb.js injects (sym3Eigen, mat3Mul, …) **
│   ├── localfold.css, style.css   E  styles
│   ├── examples/, public/         —
│
├── tools/                            ===== EXPORT + GATES (110 files + 291 in gpu/) =====
│   ├── export_af3_model.py          R  ** STEP 1: AF3-lineage blob -> f32 model dir.
│   │                                   Reads a ColabDesign2 zstd blob of (scope, name,
│   │                                   array). TENSOR NAMES ARE THE HAIKU PATHS, UNCHANGED.
│   │                                   Records which checkpoint filled the bundle in
│   │                                   manifest.model.name (licence gating). **
│   ├── quantize_af3.py              R  ** STEP 2: -> int5 asym g32, f16 scale+zero.
│   │                                   32x5 bits = exactly 20 bytes/group. Norms/offsets/
│   │                                   biases stay f32 (0.09% of params). **
│   ├── repack_shards.py             R  ** STEP 3: even shards. Moves bytes only; shifts
│   │                                   byteOffset/scaleOffset/zeroOffset by one delta;
│   │                                   reads every tensor back and compares byte-for-byte. **
│   ├── write_manifest_module.py     R  ** STEP 4: manifest.json -> the compiled JS module,
│   │                                   WITH sha256 shardDigests (the interlock that makes
│   │                                   a committed derived artefact safe). Holds `BUNDLES`,
│   │                                   the Python twin of MODEL_BUNDLES. **
│   ├── check_remote_bundle.py       R  ** is every shard at the remote, at the right
│   │                                   length? HEAD-only. Run this after publishing. **
│   ├── check-bundle-vs-params.py    R  ** is this bundle the weights the REFERENCE loads?
│   │                                   The one question every other checker assumes —
│   │                                   found 4 negated tensors in boltz2's own export. **
│   ├── quantize_model.py            —  the LEGACY AF2 path (int8 sym g64 default)
│   ├── pack_delta_model.py          —  AF2 int3 g128 deltas
│   ├── analyse_quantisation.py      R  the codec comparison behind "int5 not int4"
│   ├── build_site.py                E  ** assembles dist/, VERIFIES shard digests, drops
│   │                                   unservable <option>s, runs its own unresolved-import
│   │                                   check over the tree it is about to publish.
│   │                                   `npm run test:site`. **
│   ├── deploy.py, bundle.py         E  deploy / offline bundle
│   ├── check-smiles-*.mjs           R  the SMILES gate set (vs real RDKit as an ORACLE):
│   │   check-smiles-vs-rdkit, check-stereo-vs-rdkit, check-smiles-rewrites,
│   │   check-smiles-conformer, check-smiles-batch, check-smiles-path
│   ├── check-ligand-path.mjs        R  ** ligand bonds through every family;
│   │                                   boltz2 0.062 A. Asserts BOND LENGTHS because
│   │                                   pLDDT read 92.38 on a ligand 6 A out. **
│   ├── check-modified-path.mjs      R  SEP@3 through 8 bundles (boltz2 0.938)
│   ├── check-template-path.mjs      —  templates (out of scope)
│   ├── check-atom-windows.js        R  atom key windows vs the reference, 7 models
│   ├── check-batch-fields.js        R  ** 43 of 60 batch fields vs AF3's own dump,
│   │                                   7 models x 2 targets. The sharpest featuriser gate. **
│   ├── check-model-switch.py        R  ** 10 models in ONE page, device < 2 GiB —
│   │                                   the gate that found residency leaking across a
│   │                                   model change. Closest existing analogue to our
│   │                                   N-ligand loop. **
│   ├── check-model-pending.py       R  a result belongs to the model that made it
│   │                                   (veils stale panels) — measures PIXELS
│   ├── fold-in-page.py              E  ** drives the REAL page over CDP: --smiles-ui
│   │                                   (dropdown+box+BLUR, which the API path cannot see),
│   │                                   --contact-ui, --job, --job-round-trip, --drop-job,
│   │                                   --throttle=8 (a first visit on a real link),
│   │                                   --keep-profile (a RETURNING visitor). **
│   ├── check-job-archive.py         R  does the archive describe the job handed in?
│   │                                   A SEPARATE PROCESS that never imports the page's reader.
│   ├── gpu-chrome.mjs               R  ** serves the repo, drives headless Chrome, calls a
│   │                                   module's `main(device, args)`. THE working GPU lane
│   │                                   (npm run test:gpu needs a Dawn this glibc lacks). **
│   ├── gpu/  (291 files)            R  every differential checker, bench and probe.
│   │   Most relevant to us: check-af3-denoise.js, check-af3-confidence-oracle.js,
│   │   check-af3-trunk / --trunk-oracle=, fold.js, check-pipeline-key-collisions.js,
│   │   bench-smiles-conformer.js, check-smiles-conformer-gpu.js,
│   │   check-bundle-device-decode.js (does a REAL bundle's int5 decode the same on the
│   │   device as on the host?), probe-compiles.js, probe-submits.js, probe-memory.js,
│   │   probe-binding-ceiling.js, relative-rms.js (⚠ unguarded relRMS over a non-array
│   │   returns exactly 0 — a perfect score from comparing nothing).
│   ├── oracle/  (62 files)          —  PyTorch/JAX dumpers; need a reference checkout
│   ├── index-tools.py               R  derives docs/TOOLS.md from each tool's own header;
│   │                                   `npm run test:tools` fails when stale
│   ├── gen-standard-components.mjs  R  generates standard-components.js from components.cif
│   ├── fixtures/                    R  ccd/ (29 files), msa/, af3-jobs/, rcsb/, crystals
│   └── ppi-bench/ (245), esmc/      —
│
├── test/  (162 + fixtures)           R  `npm test` = the CPU suite, MUST PASS.
│   ⚠ needs --js-float16array on a node lacking Float16Array (12 tests otherwise never RUN).
│   ⚠ 24 of test/*.gpu.test.js name fixtures NOT in the repository (ENOENT on a fresh clone).
│   Ours to keep green: af3-dialect, af3-ligand-bonds, weight-pack, model-bundles,
│   model-family, job-json, prediction-results, fold-session, mmseqs2-api, imports-resolve,
│   af3-example-jobs, template-atom37-layout, msa-attention-pipeline-key.
│
├── docs/  (2.9 MB, 27 files)         R  the measured findings. Read the one for a stage
│   │                                   BEFORE changing that stage.
│   ├── ARCHITECTURE.md       68 KB  R  the architecture narrative
│   ├── AF3.md               396 KB  R  ** the AF3 port's state, parity sweep, dead ends.
│   │                                   AGENTS.md says read it before touching any af3 tree. **
│   ├── SMILES.md             42 KB  R  ** the SMILES path's whole history — the RDKit
│   │                                   rejection, the conformer fixes, the gates. **
│   ├── BOLTZ2_PTM.md         14 KB  R  ** boltz2's modified-residue defect and its fix
│   │                                   (modifiedAsOneToken; 1.813 -> 0.938). **
│   ├── WEB.md               271 KB  R  the page's findings (UI traps, archives, jobs)
│   ├── PERF.md              167 KB  R  where time goes
│   ├── HOSTING.md            29 KB  R  ** first-visit cost, HF bandwidth (27.9-30.1 MiB/s
│   │                                   measured remote, vs 371 MB/s on the dev server). **
│   ├── A100.md              391 KB  —  NVIDIA/flag findings
│   ├── AF2.md, EF2FAST.md, OPENDDE.md, BIOEMU2.md, DAWN.md, PARITY.md,
│   ├── TOOLS.md              61 KB  R  the EXHAUSTIVE tool index (derived)
│   ├── RUNNING.md             4 KB  R  ** how to run from a checkout + the two Chrome
│   │                                   flags worth ~2x on Linux/NVIDIA + the library API. **
│   ├── DEVELOPING.md         38 KB  R  reference numbers, benchmarks, deployment
│   └── M2/M5/PALLAS handoffs        —
│
├── notebooks/localfold.ipynb        —  Colab (needs the absent python/ + cuda/)
├── oracle-dumps/                    —  reference batches (af3-6mrr.json etc.)
├── .github/workflows/               E  Pages deploy (downloads bundles for monomer,
│                                        multimer, af3 — a new family needs a step)
│
└── ABSENT but documented in CLAUDE.md:
    cuda/   metal/   python/          —  the native + Colab backends. NOT in the public
                                         repo, so test:native / test:bridge /
                                         test:standalone / test:featurise cannot run here.
                                         Irrelevant to a browser-only project, but do not
                                         go looking for them.
```

## Dependency direction (enforced)

```
shared/  <-  cpu/  <-  webgpu/        (and  shared/ <- webgl/)
            ^
            +-- web/  reads shared/ and one backend
```

`shared/` imports no backend. A stage's reference (`cpu/`) and its kernel (`webgpu/`)
share a filename. Plain ES modules with `.js` specifiers throughout — **no bundler**, which
is why the live site can be diffed file-by-file against the checkout.

## The five files a Boltz-2 affinity screening build actually touches

| file | change |
|---|---|
| `shared/bundles/manifests/boltz2-affinity.js` | **new** — generated manifest module for the affinity bundle |
| `shared/bundles/manifests/index.js` | **edit** — register it (lazy `load`, pinned `remote`); fetch only when affinity is on |
| `webgpu/af3/affinity/*.js` + `cpu/af3/affinity/*.js` | **new** — the ported module and its differential reference (AGENTS.md requires both) |
| `web/job-json.js` | **edit** — `properties: affinity: { binder: <chain id> }`, OPEN dialect |
| `web/` screening page + `web/fold-archive.js` | **new/edit** — queue, results table, scatter, CSV+structures ZIP |

Everything else in the list above is reuse.
