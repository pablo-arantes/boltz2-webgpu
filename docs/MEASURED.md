# How it is checked, what was measured, and what is still wrong

Split out of the README so that file can stay short. Every number here is
measured; `spec/G8-RESULT.md` carries the full comparisons against upstream
PyTorch Boltz-2, including the retractions.

## How it is checked

Every gate is a `npm run` script at the repository root, with its own one-paragraph
description in `package.json` saying what it caught.

| gate | what it establishes |
|---|---|
| `test:affinity-mw` | the molecular weight agrees with **RDKit's** table, which is what the MW correction's coefficients were fitted against. Found this port's IUPAC-2021 table disagreeing with RDKit on **51 of 86 elements** — being more correct than RDKit is the bug here. |
| `test:affinity-quant` | the manifest's `float32Tensors` list actually protects the norms. `quantize_af3.py`'s `KEEP_FLOAT32` regex is slash-separated and upstream's names are dot-separated, so it matches **none** of them; without the list every LayerNorm scale would be int5. `--falsify` empties it and requires degradation. |
| `test:affinity-oracle` | the f64 reference computes upstream's function, seam by seam, against a dump of the **real module** (forward hooks on upstream's own `AffinityModule`). 9 real weight-swap mutations as a separation control. |
| `test:affinity-contract` | LocalFold's own pair kernels accept Boltz-2's weights. Five layouts pinned by measurement; each re-run with the wrong orientation and required to disagree. |
| `test:affinity-gpu` | the GPU pairformer computes the f64 reference's answer. **Found the incoming triangle's a/b order reversed** — `tri_mul_out` was exact at 1.0e-7 while `tri_mul_in` read 0.304, which the host gate could not see because its reference is outgoing-only. |
| `test:affinity-forward` | the whole function — LayerNorms, projections, broadcast sum, distogram, conditioner, stack, pooling, all three heads — against upstream's dump. |
| `test:affinity-crop` | an **exact index-set** comparison against upstream's `AffinityCropper` over 12 cases reaching all three break conditions. |
| `test:affinity-sinputs` | upstream's `affinity=True` substitution reaches `s_inputs`, and only through the profile. |
| `test:affinity-project` | the new fused kernel against a reference written in the gate. Its falsify input needs rows whose **variance is the size of eps**, because the convention is invisible at variance ≈ 1 *and* at variance exactly 0 — the control missed it twice before the input could express it. |
| `test:affinity-keys` | the affinity stack does not collide with the trunk in one pipeline cache, in both orders. The shapes are identical (128 channels, 4 heads of 32). |
| `test:affinity-memory` | the screening loop does not leak GPU memory across ligands. Two phases, because growth across ligands of *different sizes* is correct behaviour. |
| `test:boltz2-featurise` | the batch agrees with **upstream Boltz's own featuriser** — a reference this port did not have, since `test:batch` compares against the AF3 featuriser this port derives from. Needs no CCD and no 1.86 GB `mols.tar`. |
| `test:screen-page` | the **page**, driven through its own controls. Caught `uniqueIds` returning ids rather than entries (every job `id: undefined`) and py2Dmol silently overwriting the page's status line. |
| `npm test` / `test:boltz2-method-class` | pins boltz2's method conditioning class against upstream's published table. It exists because a wrong constant here moved a predicted IC50 by a factor of 2.2 and no per-token differential could see it — the row is added identically to every token. Fails 3/3 with the fix reverted. |
| `npm run bench:g8` | the whole pipeline against **upstream Boltz running the same job** — the one thing the function-level gates cannot see, since they compare a function to a function on identical input. Not a CI gate: upstream runs on CPU here, ~2 minutes a seed, and needs 6.1 GB of its own assets. **It found a systematic 0.343 log-unit gap that every other gate missed.** |
| `python3 tools/check_readme_claims.py` | this README's own checkable claims — the pinned SHAs against `spec/PINNED_COMMIT.txt`, every file path it names, every gate it lists, the licence text, and the affinity bundle's tensor and shard counts. It caught this file claiming 6.5 MiB where the manifest says 6.4. |
| `npm test` | 1,704 unit tests (1,703 pass, 1 skipped), including the `.smi` parser, the run loop's stop/resume and failure paths, the CSV/ZIP, the superposition and the MSA cache. |

## Measured

On an M2 (`apple / metal-3`), a 33-residue target, cropping to 36–46 tokens:

| setting | per ligand |
|---|---|
| 1 recycle / 12 steps / 1 sample | **1.7 s** |
| 3 / 100 / 1 | **6.0 s** |
| 5 / 200 / 3 | **22.8 s** |
| 3/200/1 then 5/200/5 — **upstream's own settings, both passes** | **34.6 s** |

Bundles: 365 MiB structure (24 shards) + 6.4 MiB affinity (8 shards, 33 MiB
decoded). GPU memory across a run: resident **19 MiB flat**, peak **727 MiB**;
with buffer pooling disabled the same loop grows **492 MiB a ligand**, which is
the failure the gate exists for.

🔴 **Upstream's own settings are 3/200/1 for the structure pass and 5/200/5
for the affinity pass**, and the second half of that was recorded here as
5/200/**3** until the G8 work read it off the CLI rather than the function
signature: `predict`'s signature says `diffusion_samples_affinity: int = 3`,
its click option says `default=5`, and click passes its value — so a plain
`boltz predict` runs five. The same question had already been answered the
other way for the molecular-weight correction, where this port deliberately
follows the CLI.

🔴 **Step count is not a free knob.** At 4 sampler steps the ligand comes out as
a collapsed blob (ethanol's three atoms within 0.4 Å); at 50 it is a molecule
(C–C 1.46 Å, C–O 1.34 Å against ideals of 1.52 and 1.43) on a backbone with
CA–CA median 3.818 Å. The page defaults below upstream's settings and says so on
its own status line.

🔴 **The superposed scene carries the target's own variation.** Each ligand is
co-folded with its own copy of the target, so Kabsch alignment on the target's
CA atoms removes the frame but not the conformational difference. Two ligands at
50 steps: **24.16 Å rms apart before the fit, 4.14 Å after** (worst 44.55 →
8.27). Three ligands at 3 recycles / 100 steps: residuals of **0.745 Å and
8.814 Å** against run #1, with the ligand centroids 9.2–14.1 Å apart. The page
prints the worst residual rather than letting a reader compare poses to a
precision the data does not have.

## Known limits

- **The affinity bundle has no published remote.** `MODEL_BUNDLES` names a local
  directory, so a deployed site would 404; `build_site.py` treats the page as
  optional for that reason. Publishing the export and pinning its revision is a
  one-line change in the registry.
- 🔴 **Neither output is upstream's yet, and which one is closer depends on the
  question.** G8 — `npm run bench:g8`, upstream's own checkpoints on CPU
  against this port — found a systematic gap and localised it to two inputs.
  One is **fixed**: this port fed method conditioning class 1 (X-ray
  diffraction) where genuine Boltz-2 feeds 4 (de novo), a constant 384-vector
  on every token; that took `s_inputs`'s protein tokens to the int5
  quantisation floor. The other is **not**: a SMILES ligand's **atom names**,
  256 of the atom encoder's 388 input columns, where upstream uses RDKit's
  canonical rank over the hydrogen-bearing molecule (`C20 C22 … O13`) and this
  port uses AF3's per-element counter (`C1 … O1`) — **not one of ten names
  agrees**. Substituting upstream's names takes the atom encoder's ligand half
  from relRMS **1.1556 to 4.48e-2**, the quantisation floor, so the names are
  the whole of what is left.
- 🔴 **Measured across six chemically varied ligands, not one, and re-measured
  after two defects were fixed:**

  | Spearman ρ vs upstream | first measured | + review fixes | **+ atom names and the bond code** | discordant |
  |---|---|---|---|---|
  | `affinity_pred_value` / pIC50 | −0.03 | +0.31 | **+0.886** | 8 → 5 → **2 of 15** |
  | `affinity_probability_binary` | +0.89 | +0.89 | **+1.000** | 2 → 2 → **0 of 15** |

  🔴 **BOTH HEADS RANK NOW, AND THE VALUE HEAD'S "DOES NOT RANK" VERDICT IS
  RETRACTED.** It was true of every state measured before the SMILES atom names
  and the bond-type code were fixed; with both, aspirin goes 1.0201 → 1.5561
  against upstream's 1.5887, ethanol 2.3878 → 1.4400 against 1.4100, and
  nicotine 1.5061 → 0.9977 against 0.8530.

  🔴 **AND THE TWO REMAINING INVERSIONS ARE WEIGHED AGAINST UPSTREAM'S OWN SEED
  SPREAD, NOT COUNTED.** Its half-range on this target is 0.0570 log units.
  `propylphenol/naphthol` are **0.0165 apart in upstream's own numbers** — well
  inside that, so upstream does not resolve them at one seed and neither side
  can be said to have them the wrong way round. `ibuprofen/nicotine` are 0.0771
  apart, outside it, and that one is a genuine inversion.

  🔴 **AND n = 6 IS STILL SMALL, WHICH THE ρ DOES NOT HIDE.** At six ligands
  +0.886 sits essentially ON the 5% two-tailed critical value, so it is a real
  but thinly supported correlation; +1.000 is p ≈ 0.003. The ranking control is
  also **one upstream seed per ligand**, so it carries no spread of its own —
  the per-ligand numbers are compared against a single draw.
- 🔴 **Two science defects were found by a cold code review and fixed, and they
  invalidated the measurement above until it was re-run.** Both were in the
  composition, which every function-level gate is blind to by construction:
  * the affinity module's distogram was fed `fold.positions`, which is indexed
    by dense **atom slot** (`token*dense + atom`), where the module indexes one
    coordinate **per token** — so it read the first ~11 tokens' atoms, masked
    slots included, as if they were 256 token centres. It now receives
    `fold.pseudoBeta` (the disto gather, `tokens*3`). The float64 reference
    asserted this contract and threw; the GPU path had no length check, and now
    has one.
  * the trunk pair was handed over **unmasked**, directly below a comment
    claiming it was masked. Upstream applies `z * cross_pair_mask` before the
    module's LayerNorm and projections; the dump tool did and the page did not,
    so the two disagreed about the module's own input.

  Fixing both moved one ligand's value 1.3596 → 1.1708 and its probability
  0.3321 → 0.2964 (upstream 1.3541 ± 0.057, 0.2243 ± 0.009). **The classifier's
  ρ did not move at all** — the same two pairs invert — which says its ordering
  is robust to both defects and is driven by what reaches it through
  `s_inputs`. That is where the atom-name difference lives, so the attribution
  below survives the re-measurement; it was **not** safe to assert before it.
- 🔴 **The blocker that made this "not shippable as a screening tool" is now
  closed, and it was closed the way it was stated.** Reproducing RDKit's
  `CanonicalRankAtoms` was called required rather than optional, gated on exact
  integer agreement over the 74-molecule corpus: `npm run test:canonical-rank`
  is **74/74** and `npm run test:canonical-names` carries the names through the
  page's own parser. Fixing it, together with an off-by-one in boltz2's
  bond-type embedding index, took the module's two conditioning inputs to the
  int5 quantisation floor (`s_inputs` 4.38e-2 against a floor of 4.14e-2; `z`
  3.10e-2 against 3.04e-2, from 2.00e-1) and both affinity heads to the
  rankings above. What is still NOT established is agreement with EXPERIMENT:
  every number here is agreement with upstream Boltz-2 on one 33-residue
  target, single-sequence, over six ligands.
- 🔴 **The method-class fix changes every boltz2 structure fold too**, since
  `target_feat` is shared: pLDDT 87.9 → 82.9 and pTM 0.666 → 0.525 on one
  target, while the ligand gate's bond rms improves 0.062 → 0.049 Å. Less
  confident, more correct — but **no RMSD against a known structure has been
  measured either way**, and that is the outstanding check on the change.
- **`s_inputs` is not compared to upstream's embedder elementwise.** The
  substitution is shown to reach the output and to move only where it should;
  comparing the embedder itself needs upstream's whole featuriser stood up.
- The scene is handed to py2Dmol as one concatenated structure per update, so
  show/hide is a redraw. Cheap for hundreds of small ligands, wrong for
  thousands.
- A 33-residue test target has no real pocket. The ligand placements measured
  above are a plumbing result, not a docking result.

## Publishing

The site builds and deploys from `main` through
`.github/workflows/pages.yml`.

- 🔴 **THE DEPLOY REFUSES A SITE WITH NO SCREENING PAGE, WHICH IS THE ONE
  FAILURE THAT WOULD LOOK LIKE SUCCESS.** `screen.html` opens the
  `boltz2-affinity` bundle through `openStore`; while that bundle's `remote`
  was null, `bundleBaseUrl` fell back to `./model-boltz2-affinity-int5/` - a
  local export the build deliberately never copies - and every visitor would
  have 404'd on shard zero. `tools/build_site.py` drops an unservable page and
  says so, the same rule it already applied to an unservable model `<option>`
  one control further out, and the workflow turns that into a failed build.
  Watched failing both ways before being trusted: with the remote null the page
  is dropped and the deploy refuses, with it pinned the page ships and nothing
  is dropped.

- 🔴 **AND IT REFUSES ANY ROOT-ABSOLUTE REFERENCE**, because a project page is
  served from `/<repo>/` and a single `/`-rooted asset, worker, WASM or
  manifest path is the usual reason a fork 404s on Pages. `dist/` is entirely
  relative; the workflow fails if that stops being true, so no base-path
  rewriting step is needed and none is guessed at.

- 🔴 **THE AFFINITY BUNDLE IS PINNED TO A REVISION, NOT A BRANCH.** 6.51 MiB in
  eight shards plus the manifest at
  `pabloarantes/boltz2-webgpu-weights@adf393c7`, fetched only when a reader
  asks for an affinity. `resolve/<sha>/` cannot change under a visitor where
  `resolve/main/` would re-point on any re-upload. Verified as a BROWSER sees
  it rather than as an upload tool reports it: fetched with a
  `https://pablo-arantes.github.io` Origin, `access-control-allow-origin`
  echoed back on both the redirect and the file, and the bytes compared against
  the local export - manifest and shard zero identical. `python3
  tools/check_remote_bundle.py boltz2-affinity` finds all 8 shards.

  The export itself is checked against the float32 one by
  `tools/check-affinity-quantisation.py --f32=model-boltz2-affinity-f32
  --int5=model-boltz2-affinity-int5`: norms and output projections byte-exact,
  the large projections at 3.88e-2, which is int5's own level.

- 🔴 **NO WEIGHTS ARE TRACKED, AND ONE OVERSIZED FILE IS - DECLARED, NOT
  HIDDEN.** `npm run test:sizes` (`tools/check_repo_sizes.py`) fails on any
  tracked file over 10 MiB that is not declared with a reason, and fails again
  if a declared one leaves the tree, so the brief's "report anything over
  ~10 MB before committing" is a gate rather than a promise. Today: **1536
  tracked files, 70.1 MiB**, one file over the limit -
  `oracle-dumps/af3-6mrr.json` at 18.5 MiB, AlphaFold 3's own featurised batch
  for 6MRR, inherited from LocalFold and read by four tools. It is **not
  published**: the same gate checks `dist/` and finds nothing over the limit
  there, so it costs a visitor nothing.

- 🔴 **NOTHING NEEDS COOP/COEP.** No `SharedArrayBuffer` and no threaded WASM,
  so the headers GitHub Pages cannot set are not required. Checked with a
  pattern over the tree and over `dist/`, not assumed.

### The live site, driven rather than eyeballed

`tools/screen-in-page.py --url=<the published page>` wipes the Chrome profile
and drives the deployed site through CDP - no HTTP cache, no CacheStorage, no
compiled shaders - so what it measures is a first visit. One target (33
residues) and three ligands, at 1 recycle / 20 steps / 1 sample:

| | single sequence | with the MMseqs2 search |
|---|---|---|
| cold start to a runnable page | **79.2 s / 103.6 s** (two runs) | **50.2 s** |
| three ligands scored | 12 s / 6 s | 7 s |
| `api.colabfold.com` requests | **0** | **2, for 3 ligands** |
| ZIP | README.txt, affinity.csv, 3 PDBs | same, 19,171 B |
| CSV | 5 lines | 5 lines |
| ligands superposed onto run #1 | 3 of 3 | 3 of 3 |

🔴 **COLD START IS ALMOST ENTIRELY THE STRUCTURE BUNDLE** - 365 MiB of it
against the affinity bundle's 6.51 - and it varies with the connection more
than with anything this code does, which is why three numbers are given rather
than one.

🔴 **AND THE SINGLE-SEQUENCE OPT-OUT IS A MEASURED ZERO, NOT A PROMISE.**
Resource timing records every request the page made; in that mode the count
aimed at api.colabfold.com is exactly 0. With the search on it is 2 for THREE
ligands rather than six, which is the page's "searched once per target and
cached by sequence hash" as a count. The alignment demonstrably reached the
model: the same three ligands come back with different ipTM (0.63/0.58/0.43
against 0.38/0.58/0.28), which only an alignment would do.

🔴 **ONE THING WORKED LOCALLY AND NOT ON PAGES, AND IT WAS THE PROBE.**
`import('/web/screen-export.js')` resolves at `/` under a local server and
fetches `<host>/web/screen-export.js` - base dropped - under a project page.
The page's own imports are all relative; the harness's was not. Fixing it also
exposed that the workflow's root-absolute guard matched only DOUBLE quotes, so
a single-quoted `import('/...')` would have passed the check written to stop
exactly that. Both fixed, and `dist/` is clean under the wider pattern.


## The screening page's 3D scene was blank — FIXED (2026-11-10)

Reported by a user. Two faults, and the second one is a lesson about where to
look.

### Fault 1: the viewer was never revealed

py2Dmol's DOM is copied into `screen.html` from `index.html` with
`style="display: none"` on `#viewer-container`, because `index.html` reveals it
when a fold lands - and this page never did. `drawScene` now reveals it BEFORE
handing the structure over (py2Dmol fits its camera at ingest, reading the
canvas size, and a canvas laid out inside `display: none` is 0x0), re-asserts
after the loader settles, and pairs `_updateCanvasDimensions()` with `render()`
the way the vendor's own ResizeObserver does.

### Fault 2: an unguarded `.checked` on a checkbox this page did not have

`buildPendingObject` reads two sibling checkboxes and guards only one:

```js
const p = document.getElementById("loadAsFramesCheckbox"),
      m = document.getElementById("alignFramesCheckbox"),
      g = !!m && m.checked;    // guarded
const w = p.checked;           // NOT guarded  -> TypeError
```

The copied subtree omitted `#loadAsFramesCheckbox`, so `p` was null and the
throw landed in `processFiles`' PER-FILE `try/catch`:

```
Error processing file screen.pdb:
TypeError: Cannot read properties of null (reading 'checked')
  at buildPendingObject -> processFiles
```

🔴 **AND EVERY OUTWARD SIGNAL SAID SUCCESS.** The promise resolved, no error
reached the page, and the loader returned `objectsLoaded: 1` - which counts the
files it MEANT to load, not the objects it added. The honest numbers were
`framesAdded: 0`, `buildViewerState().objects: 0` and
`renderer.coords: array(0)`.

| | before | after |
|---|---|---|
| `framesAdded` per hand-over | 0 | **1** |
| `buildViewerState().objects` | 0 | **1** (`{name: 'screen', frames: 1}`) |
| `renderer.coords` | 0 | **59** (33 cartoon CAs + 26 ligand atoms) |
| console errors | 1, unread | **0** |

The element is added **unchecked** where `index.html` has it checked, which is a
different question answered differently: that page loads a sampler TRAJECTORY
and wants each file to become a frame of one object, while this page hands over
the whole scene on every redraw - frames would accumulate one per ligand and
the viewer would sit on the first.

### 🔴 THE ANSWER WAS ALREADY WRITTEN DOWN IN THIS TREE

`index.html` carries a comment about this exact element, measured there by
deleting it: *"`processFiles` ... reads `p.checked` with NO guard ... the fold
still computes and the status line still reads Done in 1.2 s ... and the
structure never arrives ... A right answer nobody can see."* Several runs went
into reverse-engineering a minified bundle while the sibling page said it
plainly. Two input-shape guesses were spent on the way and both were refuted by
reading rather than by running: a `_scores.json` beside the PDB, and wrapping
the scene in `MODEL`/`ENDMDL` - the parser pushes a model at the end regardless
(`return n.length>0&&t.push(n), 0===t.length&&n.length>0&&t.push(n)`).

### What now holds it closed

`tools/screen-in-page.py` asserts, at runtime on the real page:

- the console is **empty** - the fault was a CAUGHT exception, and the only
  record of it was a console line nothing read;
- **`framesAdded == 1`** on every hand-over - the number that actually moves,
  against the one that merely looked right;
- `buildViewerState().objects >= 1` and `renderer.coords.length > 0`;
- `#viewer-container`'s computed display, and the canvas's drawing buffer.

What it asserted before was the scene NOTE's text, which the page writes from
its own bookkeeping whether or not an atom reached the viewer - so a blank box
reported "3 of 3 ligands superposed onto run #1".


## HIV-1 protease: 9 approved inhibitors among 100 drugs (2026-11-11)

The first measurement in this project against EXPERIMENT rather than against
upstream Boltz-2. Everything else here asks whether the port computes what
Boltz-2 computes; this asks whether the ranking puts known binders on top, which
a numerically perfect port can still fail - because then the answer is about
Boltz-2, not about this code.

Target: HIV-1 protease, 99 residues, HXB2 wild type, taken from UniProt P04585
chain "Protease" (489-587) rather than from memory and cross-checked against the
3OXC crystal construct: the only differences are the V3I/Q7K/L33I/L63I
polymorphisms and C67A/C95A, which is the standard cysteine-free construct.

Ligands: `tools/fixtures/hiv1-protease-100.smi`. Canonical isomeric SMILES from
PubChem, every one re-parsed by `shared/chem/smiles.js` so each is a molecule
the page actually accepts. 9 approved protease inhibitors, 91 approved drugs
with unrelated targets - including 17 other antivirals, so the decoys are not
separable as "not an antiviral".

```
screened 100 of 100 ligands in 39.0 min (22.8 s a ligand)
actives present: 9 of 9

P(binder)  - affinity_probability_binary, higher is better
  ROC AUC                 0.762   95% CI 0.540-0.935   (0.5 = indistinguishable)
  EF at    1% (top   1)    11.11x   1 of 9 actives
  EF at    5% (top   5)     6.67x   3 of 9 actives
  EF at   10% (top  10)     4.44x   4 of 9 actives
  EF at   20% (top  20)     3.33x   6 of 9 actives
  ranks of the actives    [1, 2, 3, 10, 11, 16, 49, 63, 85]
  median active rank      11 of 100
  top 5: saquinavir*, tipranavir*, atazanavir*, favipiravir, levothyroxine

pIC50      - affinity_pred_value, LOWER log10 IC50 is better
  ROC AUC                 0.873   95% CI 0.733-0.967   (0.5 = indistinguishable)
  EF at    1% (top   1)     0.00x   0 of 9 actives
  EF at    5% (top   5)     2.22x   1 of 9 actives
  EF at   10% (top  10)     4.44x   4 of 9 actives
  EF at   20% (top  20)     4.44x   8 of 9 actives
  ranks of the actives    [5, 6, 7, 10, 11, 12, 16, 18, 64]
  median active rank      11 of 100
  top 5: digoxin, levothyroxine, itraconazole, azithromycin, lopinavir*

heavy atom count - THE BASELINE, not a prediction
  ROC AUC                 0.958   95% CI 0.916-0.990   (0.5 = indistinguishable)
  EF at    1% (top   1)     0.00x   0 of 9 actives
  EF at    5% (top   5)     6.67x   3 of 9 actives
  EF at   10% (top  10)     6.67x   6 of 9 actives
  EF at   20% (top  20)     5.00x   9 of 9 actives
  ranks of the actives    [3, 4, 5, 7, 8, 9, 12, 13, 17]
  median active rank      8 of 100
  top 5: digoxin, azithromycin, atazanavir*, ritonavir*, saquinavir*

written spec/HIV1-BENCHMARK.json
```

### 🔴 THE BASELINE BEATS THE MODEL, AND THAT IS THE FINDING

Heavy-atom count alone ranks at **0.958 (0.916-0.990)** against `P(binder)`'s
0.762 (0.540-0.935) and `pIC50`'s 0.873 (0.733-0.967). The actives are
peptidomimetics of 40-55 heavy atoms and the decoy median is 24, so size IS a
ranking here - and a better one over the whole list, with an interval clear of
both heads. `P(binder)`'s interval reaches 0.540, next to chance.

So this does not demonstrate recognition of protease binders. It is consistent
with the heads ordering ligands largely by size. The control was written into
the benchmark from the start for exactly this reason and the ligand file's
header warned of it before the screen ran; it would have been easy, and wrong,
to report 0.873 on its own as a success.

### Where the model does beat size

The TOP of the list. `P(binder)` ranks saquinavir, tipranavir and atazanavir
first, second and third - EF at 1% of 11.1x - where size puts digoxin and
azithromycin first and scores 0x. Whole-list AUC and early enrichment are
different questions and only the second is the one a screener acts on. `pIC50`
shows the opposite shape: better AUC, no early enrichment, and a top five of
digoxin, levothyroxine, itraconazole, azithromycin and lopinavir - four large
non-binders, which is what tracking size looks like.

### What this is not

Nine actives and one target. The intervals are wide because of that nine.
The decoys are NOT property-matched, which is the whole reason size works -
DUD-E-style matching on weight, logP and charge is what would make the AUC mean
something. Settings were 3 recycles / 100 steps for the pose and 5 / 200 / 1
for the affinity, below upstream's 5 / 200 / 5, and single-sequence with no
alignment. A real evaluation needs matched decoys, more actives, several
targets, and ideally a held-out set the model's authors did not train on.

### Two things the run itself found

🔴 **`gpu-chrome.mjs` CAPS A RUN AT 600 s** (`LOCALFOLD_GPU_TIMEOUT_MS`), so the
first attempt at 100 ligands was killed at ten minutes and wrote nothing. A
3-ligand timing run finished in 89 s, comfortably inside the cap, so the limit
was invisible at the scale the job was sized on.

🔴 **AND THE RUNNER THROWS ON ANY FAILURE, DISCARDING THE WHOLE SCREEN.** One
ligand - lithium carbonate, `[Li+].[Li+].C(=O)([O-])[O-]` - was refused
correctly ("3 disconnected fragments... affinity is predicted for one
molecule"), and that threw `1 ligand(s) failed` AFTER 39 minutes of successful
work, returning no JSON at all. The per-ligand lines had been printed as they
finished, so `tools/parse_screen_log.py` recovers them. Refusing to report a
partial screen as complete is right; also discarding the part that succeeded is
not. The salt was my curation error and the file now carries a note saying why
every line must be one connected molecule.
