# G8 — this port's affinity numbers against upstream Boltz, end to end

Run 2026-11-09 on an M2 (upstream on CPU, this port on WebGPU/metal-3), with
upstream's own assets: `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `ccd.pkl` and
`mols.tar`, all four byte-exact against the sizes recorded in
`PINNED_COMMIT.txt`. Harness: `tools/g8_compare.py` (`npm run bench:g8`).

**Verdict: the pipeline composes and runs, and its numbers do NOT yet match
upstream's. The gap is systematic, it is not noise, and it is not the pose.**

## What was run

One target, one ligand, both implementations at upstream's own settings —
structure pass 3 recycles / 200 steps / 1 sample, affinity pass 5 recycles /
200 steps / **5** samples (upstream's click default; see §5). Single sequence
on both sides. Five seeds each.

```
target  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ   (33 residues)
ligand  CCCc1ccccc1O                        (10 heavy atoms)
```

Upstream 107–119 s a seed; this port 46–48 s a seed.

## 1. The aggregate gap, against upstream's own spread

🔴 **The unit is upstream's seed-to-seed spread, not a bound chosen by hand.**
The affinity is read off a sampled pose, so upstream disagrees with itself and a
hand-picked tolerance would measure the author's optimism. Half-range over five
seeds:

| | upstream | ours | gap | gap / upstream's spread |
|---|---|---|---|---|
| `affinity_pred_value` | 1.3541 ± 0.0570 | 1.0109 ± 0.0486 | 0.3432 | **6.0 ×** |
| `affinity_probability_binary` | 0.2243 ± 0.0089 | 0.3786 ± 0.0020 | 0.1543 | **17.4 ×** |

Six and seventeen times the noise floor. This is a real difference, not
sampling. In units a reader would use: 0.343 log units is a factor of **2.2 in
IC50**, and the binding probability differs by 0.15 on a 0..1 scale.

## 2. It is not the pose — which is the opposite of what one seed suggested

After the first (single-seed) run the two structures were 9.43 Å CA rms apart
with the ligand centroid 16.69 Å away, and the obvious reading was "different
poses, so the affinity difference means nothing". **Five seeds refute that
reading**, and the refutation is the useful part:

| | CA rms | ligand centroid |
|---|---|---|
| upstream against **itself**, seed to seed | 2.48 – 4.16 Å | **2.20 – 21.93 Å** |
| ours against **itself** | 4.44 – 10.16 Å | 2.44 – 13.13 Å |
| ours against upstream, same seed | 2.36 – 9.43 Å | 3.45 – 16.69 Å |

The cross-implementation structural difference lies **inside** each
implementation's own seed-to-seed spread: upstream's own seeds put the ligand
up to 21.9 Å apart, further than any gap to ours.

And over those same seeds upstream's affinity moves by only ±0.057. Our own
seeds 0–3 return `affinity_pred_value` **1.0304 to four decimals, identically**,
while their structures differ by 5–10 Å CA rms. So on this target **the
affinity is insensitive to the pose on both sides**, and the 6 × gap cannot be
attributed to it.

🔴 That also reverses the first conclusion drawn from one seed — that this
target "cannot settle G8". It settles it better than a well-folded target
would: because the pose contributes almost nothing here, what remains is a
difference in the inputs or the weights, with the pose's confounding effect
measured rather than assumed.

## 3. Both ensemble members are off, in opposite directions

This is the sharpest measurement in the run. Means over five seeds:

| | upstream | ours | gap |
|---|---|---|---|
| `affinity_pred_value1` | 1.6360 | 1.8277 | **+0.19** |
| `affinity_pred_value2` | 1.0723 | 0.1942 | **−0.88** |
| `affinity_pred_value` (mean) | 1.3541 | 1.0109 | −0.34 |
| `affinity_probability_binary1` | 0.1209 | 0.5628 | **+0.44** |
| `affinity_probability_binary2` | 0.3277 | 0.1944 | −0.13 |
| `affinity_probability_binary` (mean) | 0.2243 | 0.3786 | +0.15 |

🔴 **THE MEAN IS HIDING MOST OF IT.** Member 2's value is off by 0.88 log units
and member 1's probability by a factor of 4.6 — and the two errors have
opposite signs, so averaging them cancels most of the damage and leaves an
aggregate gap of 0.34 that understates both. Any gate that compared only the
ensemble output would have reported a third of the real discrepancy.

It is **not a swapped pair**: ours-1 does not equal upstream-2 (1.83 against
1.07) and ours-2 does not equal upstream-1 (0.19 against 1.64).

Ensemble disagreement within each implementation, for scale — ours is three
times upstream's:

| | member1 − member2, value | probability |
|---|---|---|
| upstream | +0.564 | −0.207 |
| ours | **+1.634** | +0.368 |

## 4. Where the fault is, and where it is not

**Not the affinity module's arithmetic.** `test:affinity-oracle` holds the f64
reference to upstream's own `AffinityModule` seam by seam, and
`test:affinity-forward` holds the GPU forward to upstream's dump — both
members, both at ~1e-5 to 1e-7, with a nine-mutation separation control. A
0.88-log-unit error is eight orders of magnitude outside those bounds, so it
cannot be inside the module given correct inputs.

**Not the pose** (§2), and **not the crop**: `test:affinity-crop` is an exact
index-set comparison against upstream's own `AffinityCropper`.

🔴 **So it is the inputs the module is conditioned on — and the two that were
never compared elementwise to upstream are exactly the remaining suspects**,
both named as open in this project's notes *before* this run:

1. **`s_inputs`.** `test:affinity-sinputs` shows upstream's `affinity=True`
   substitution reaching the output and moving only where it should, but the
   port's `s_inputs` has never been compared to upstream's own
   `input_embedder(feats, affinity=True)` elementwise — that needs upstream's
   whole featuriser stood up.
2. **The trunk pair `z`.** The affinity module is conditioned on the trunk's
   pair representation. This port's trunk is af3-any-model's boltz2 port, which
   `docs/AF3.md` records at **4.56e-2 relRMS on `trunk_out_pair`** against its
   own reference — the largest residual anywhere in the stack, and it feeds the
   affinity module directly.

A shared input error explains member-specific, opposite-signed output errors
without difficulty: the two members are differently-shaped networks (8
pairformer layers against 4), so one perturbation at their common input
propagates differently through each.

## 5. What this run also corrected

**`diffusion_samples_affinity` is 5, not 3.** upstream's `predict` signature
says `3`; its click option says `default=5` and click passes its value, so a
plain `boltz predict` runs five. This port had 3 recorded in `AFFINITY_PASS`,
its test, the page copy, the runner's banner and the README — all now 5. The
same question had already been resolved the other way for the molecular-weight
correction three constants lower, where the note explicitly says to follow the
CLI rather than the class signature.

**The runner could not express upstream's settings.** It had no `--seed` and
drove both passes from one `--steps`, where upstream sets them separately. A
comparison run before that fix could not have claimed to match upstream.

## 6. Next, in order

1. **Compare the trunk pair `z` and `s_inputs` against upstream's own forward
   on this exact input.** Both are reachable with a forward hook on upstream's
   affinity module, the way `dump_boltz2_affinity.py` already hooks its seams —
   the difference being real featurised input rather than synthetic. This is
   what localises §4 from two suspects to one.
2. **Feed upstream's own pass-1 structure into this port's pass 2.** That makes
   the pose identical by construction instead of by argument, and separates an
   input error from a structure error with no statistics at all.
3. Re-run G8 on a target with a determined fold and a shared alignment
   (`--a3m`, now plumbed through both arms) to confirm the gap is not a property
   of this one 33-mer.

🔴 **Until (1) and (2) are done, this port's affinity numbers should be read as
a ranking and not as calibrated log10(IC50).** The ordering of two ligands is
likely preserved — the gap is a shift of similar size on both — but a value
quoted as an IC50 is, on this evidence, wrong by about a factor of two, and one
ensemble member by nearly an order of magnitude.

---

# Localisation (same day, after the run above)

`tools/oracle/dump_boltz2_affinity_inputs.py` wraps upstream's
`AffinityModule.forward` and writes the four inputs its call site actually
passes; `tools/gpu/dump-affinity-inputs.js` writes this port's;
`tools/g8_localise.py` compares them. Both dumps are at seed 0, upstream's own
settings, the same job.

🔴 **The comparison is legitimate because the crop is the identity.** Both
sides report **43 of 43 tokens kept**, and the trunk reads features rather than
coordinates, so `s_inputs` and `z` are pose-independent given the same token
set. On a target the crop cuts, they would not be comparable and
`g8_localise.py` refuses rather than printing a number.

## What agrees

| | result |
|---|---|
| `mol_type` | **0 of 43 differ** |
| `affinity_token_mask` | **0 of 43 differ** |
| the pre-mask | masked half is **exactly 0.0** on both sides |
| token count / crop | 43 of 43 on both sides |

So the featurisation of what each token *is*, and the `z * cross_pair_mask`
convention, are exact. Those are excluded.

## What does not

| tensor | relRMS against upstream |
|---|---|
| `s_inputs`, whole | **9.02e-1** |
| `s_inputs`, 33 protein tokens | 4.89e-1 |
| `s_inputs`, **10 ligand tokens** | **1.18e+0** |
| `z`, over the 760 live pairs | **3.03e-1** |

🔴 **A relRMS of 0.90 is not a precision problem, it is a different tensor** —
and it is worst on the ligand tokens, which are the only tokens the affinity
module exists to score. Every other gate in this project reads 1e-5 to 1e-8.

🔴 **And it explains why the OUTPUT gap looked modest.** The end-to-end gap is
0.343 log units from inputs that disagree at relRMS 0.90. The module is far
less sensitive to its inputs than one would assume — which is the same
insensitivity §2 measured against the pose — so the output gap understates the
input error by a wide margin. **A gate on the output alone could never have
found this.**

## The vocabulary question, asked and answered

LocalFold's boltz2 bundle carries `boltz2_res_type_encoding/weights` at
**[31, 384]** and `boltz2_msa_profile_encoding/weights` at **[32, 384]**, where
upstream's checkpoint has `input_embedder.res_type_encoding.weight` at
**[384, 33]** and `input_embedder.msa_profile_encoding.weight` at **[384, 34]**
(33 profile channels plus one deletion mean). Two columns narrower, which
looked like the obvious cause.

**It is not, and the measurement says so.** No contiguous slice of upstream's
matches (best 6.8e-1 and 5.2e-1), but row by row the two are a **clean
one-to-one permutation**:

```
ours row i -> upstream row   relRMS 3.5e-2 .. 4.1e-2   (next best 1.13 .. 1.24)
mapping  [2,3,...,22, 1, 23,24,25,26, 28,29,30,31, 27]      one-to-one: True
```

3.5–4.1e-2 is exactly the int5 quantisation error this project records
(3.4–4.6 % per tensor), against a 30× separation to the next candidate. And the
INDICES agree with that permutation: for `MKTAY…` this port's featuriser emits
AF3's order — M 12, K 11, T 16, A 0, Y 18 — and upstream's `res_type` argmax
reads 14, 13, 18, 2, 20, each exactly our index + 2, which is what the mapping
says. The ligand is our 20 (`X`) against upstream's 22, likewise.

🔴 So the restype and profile paths are **correct, merely reordered**, and the
two missing columns are inert on any job whose tokens stay inside the shared 31
(this one's highest index is 22). They would matter for a token at upstream's 0
or 32 — a case worth a gate, but not this fault.

## What is left, and it is one term

`s_inputs` is `buildTargetFeat` at 384 channels, which sums an **atom-encoder
output**, the restype one-hot, the MSA profile, the deletion mean, and boltz2's
four extra conditionings (method, modified, cyclic, mol type). Of those:

- the restype and profile terms are shown above to be consistent;
- `mol_type` is exact, so the mol-type conditioning's input is exact;
- which leaves the **atom-encoder contribution**, the dominant term, never
  compared to upstream for boltz2. `docs/` records `target_feat` at 1e-8
  against **AF3's** reference batch; the boltz2 dialect's equivalent has only
  ever been compared against af3-any-model's port, never against upstream.

🔴 **And `z` inherits it.** The trunk is conditioned on `s_inputs`, so a 0.90
input error propagates into the pair representation, which is consistent with
`z` reading 3.03e-1 — a smaller number because the trunk's 64 blocks of
LayerNorms compress it, not because the pair track is independently fine.

## Next measurement, in one step

Dump upstream's atom-encoder output and `target_feat` summands on this same job
(the same forward-hook technique, one module earlier) and compare term by term.
That turns "the atom-encoder contribution" from the remaining suspect into
either the cause or the next exclusion. Everything needed is already in place:
the hook technique, the identity-crop condition, and `g8_localise.py`'s
live-only comparison.

**Until then the standing advice is unchanged and now better evidenced: this
port's affinity output is a ranking, not a calibrated IC50.**

---

# Fix 1 and what it left (same day)

## The method conditioning class: 1 -> 4

`cpu/af3/diffusion/atom-encoder.js` fed method class **1** (X-RAY DIFFRACTION)
where genuine Boltz-2 feeds **4** (`OTHER` / `THEORETICAL MODEL`), which is what
a de novo prediction is. One line, now a dialect parameter defaulting to
upstream's value.

| against upstream's five-seed control | class 1 | class 4 |
|---|---|---|
| `affinity_pred_value` gap | 0.3432 = **6.0 ×** spread | 0.0110 = **0.19 ×** |
| our five-seed range inside upstream's? | no | **yes** (1.3596–1.3734 in 1.2920–1.4061) |
| `affinity_probability_binary` gap | 0.1543 = 17.4 × | 0.1148 = 13.0 × |
| `s_inputs` protein tokens | 4.89e-1 | **3.90e-2** (the int5 floor) |
| `s_inputs` ligand tokens | 1.1834 | 1.1419 |
| `a` (atom encoder) | 1.0981 | 1.0981 — unchanged |
| `s - a` (every non-encoder term) | 4.82e-1 | **3.62e-2** |
| `z` live pairs | 3.03e-1 | 2.10e-1 |

🔴 **The protein half of the conditioning is exact. THE "REGRESSION HEAD IS
FIXED" CLAIM IS RETRACTED - see the six-ligand sweep below.** On this one
ligand `affinity_pred_value` came inside upstream's own run-to-run variation
(0.19 x its spread), and that is what this section originally concluded from.
It does not generalise: across six ligands the value head's ordering is
uncorrelated with upstream's. One ligand cannot establish the agreement of a
function, which is the same mistake this file insists on about seeds, made one
axis along.

🔴 **A CONSTANT 384-VECTOR MOVED A PREDICTED IC50 BY 2.2x AND NOTHING COULD SEE
IT.** The method row is added identically to every token, so it shifts
`target_feat` wholesale rather than distorting any token - invisible to every
per-token differential in this project. It survived a `target_feat` agreement
at ~1e-8 against af3-any-model, which can only hold because that port feeds
class 1 too: two implementations wrong together, which is the exact trap
CLAUDE.md records for feeding one bundle to both sides of a comparison. Only a
comparison against the checkpoint's own authors could find it.
`test/boltz2-method-class.test.js` now pins the value against upstream's
published table and fails 3/3 with the fix reverted.

🔴 **AND IT REACHES EVERY BOLTZ2 STRUCTURE FOLD, not just affinity.** Measured
on the same input: pLDDT **87.9 -> 82.9** and pTM 0.666 -> 0.525, while the
ligand gate's bond rms goes **0.062 -> 0.049 A**. Less confident, more correct -
which is what asking the model for a crystal structure of something never
crystallised would be expected to cost. No RMSD against a known structure has
been measured either way; that is the outstanding check on this change.

## RETRACTED: "it is the frame". It is the ATOM NAMES.

🔴 **THIS SECTION PREVIOUSLY SAID THE REMAINING FAULT WAS THE LIGAND
CONFORMER'S FRAME, AND THE PROTEIN CONTROL DISPROVES IT.** The claim rested on
reading upstream's `embed_atom_features.weight` as `[128, 388]` =
3 position + 1 charge + 128 element + 256 name characters and concluding that
absolute positions enter, so a rotation is a different input. The arithmetic is
right and the conclusion was wrong: `embed_atompair_ref_pos` works on pairwise
DIFFERENCES, which absorb the frame. What settled it was measuring the protein
residues, which both sides read from the CCD:

```
                     frame difference    shape difference      a relRMS
protein residue 1    3.157 A mean        pairwise rms 0.2426   4.4e-2
ligand               2.004 A mean        pairwise rms 0.2027   1.1556
```

The protein's conformer sits **further** away in raw coordinates than the
ligand's and its shape differs **more**, yet its atom-encoder output agrees at
the int5 quantisation floor. So the encoder is insensitive to the frame, and
neither frame nor shape can explain a 26x worse result on the ligand.

🔴 **THE LIGAND'S ATOM NAMES ARE TWO THIRDS OF THE ATOM ENCODER'S INPUT AND NOT
ONE OF THEM AGREES.**

```
PROTEIN  upstream ['N','CA','C','O','CB','CG','SD','CE']
         ours     ['N','CA','C','O','CB','CG','SD','CE']      identical

LIGAND   upstream ['C20','C22','C21','C19','C17','C15','C14','C16','C18','O13']
         ours     ['C1','C2','C3','C4','C5','C6','C7','C8','C9','O1']
         names in common: NONE
```

Of the 388 input columns, **256 are the atom name** - four characters, each a
64-way one-hot. Protein names come from the CCD on both sides and match, which
is why protein `a` agrees; a SMILES ligand has no dictionary names, so each
side invents them and they share nothing. That is the whole of the remaining
gap, and it explains the ordering no conformer metric could.

## Upstream's naming rule, read and reproduced

`boltz/data/parse/schema.py`, the SMILES-ligand branch:

```python
mol = AllChem.MolFromSmiles(seq)
mol = AllChem.AddHs(mol)                      # hydrogens added FIRST
canonical_order = AllChem.CanonicalRankAtoms(mol)
Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
    atom.SetProp("name", atom.GetSymbol().upper() + str(can_idx + 1))
...
mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
```

**Element symbol, upper case, plus RDKit's canonical rank + 1, over the
molecule WITH hydrogens**, which are then removed. Run here on `CCCc1ccccc1O`
it returns `['C20','C22','C21','C19','C17','C15','C14','C16','C18','O13']` -
the dumped names exactly, so the rule is confirmed rather than inferred. The
22-atom H-bearing molecule is what the ranks count, which is why the ten heavy
atoms are non-contiguous and `O13` sorts below every carbon.

This port uses AF3's convention instead - element plus a per-element counter,
`C1..C9, O1`, which docs/ describes as "every atom numbered: biotin's lone
sulfur S1, not S".

🔴 **AND THIS IS NOT A ONE-LINE FIX, UNLIKE THE METHOD CLASS.**
`CanonicalRankAtoms` is RDKit's own canonicalisation - a Morgan-style
refinement with RDKit-specific invariants and tie-breaking. It is not a
published convention that can be reimplemented from a description, and
docs/SMILES.md has already ruled out shipping RDKit (MinimalLib is 2D-only,
2.39 MB gzipped against this port's 50 KB parser). Matching upstream's ligand
names means reproducing that ranker - substantial work, though verifiable
against RDKit over the 74-molecule corpus `npm run test:smiles` already uses.

🔴 **AND THE NAME CARRIES NO CHEMISTRY**, which is worth recording as a fact
about Boltz-2's featurisation rather than as a defect in either side: the name
is a canonical RANK, so `C20` says nothing about that carbon except where it
sorted. The model was trained on those labels, so reproducing them matters for
agreement; but two thirds of the atom encoder's input for a SMILES ligand is
chemically an arbitrary index.

## A second convention difference, found in the same read

```python
if affinity:
    seq = standardize(seq)
```

The affinity path **standardises the SMILES** before parsing. On
`CCCc1ccccc1O` it is a no-op - verified, so it affected no measurement here -
but it will differ on a charged, salted or tautomer-ambiguous ligand, which is
most of a real screening library. This port does not standardise at all, and
no gate would catch it.

## The six-ligand sweep: which output ranks, and which does not

🔴 **THE SINGLE-LIGAND AGREEMENT ABOVE WAS A COINCIDENCE, AND THIS IS THE
MEASUREMENT THAT SHOWS IT.** Six chemically varied ligands - not analogues,
because rank correlation on a congeneric series is an easier test than the
tool's real job - one seed each, both implementations, same target and
settings.

```
affinity_pred_value  (lower is better)          upstream    ours    rank
  ibuprofen      CC(C)Cc1ccc(cc1)C(C)C(=O)O       0.9301   0.8564   1 -> 0
  nicotine-ish   CN1CCC[C@H]1c1cccnc1             0.8530   1.5061   0 -> 4
  propofol-ish   CCCc1ccccc1O                     1.3702   1.3596   2 -> 3
  1-naphthol     c1ccc2c(c1)cccc2O                1.3867   1.2670   3 -> 2
  ethanol        CCO                              1.4100   2.3878   4 -> 5
  aspirin        CC(=O)Oc1ccccc1C(=O)O            1.5887   1.0201   5 -> 1
  Spearman rho = -0.0286     discordant pairs: 8 of 15

affinity_probability_binary  (higher is better)
  Spearman rho =  0.8857     discordant pairs: 2 of 15   (p ~ 0.02)
```

🔴 **IT IS THE OPPOSITE SPLIT FROM WHAT THIS FILE ORIGINALLY CLAIMED.** The
potency value does **not** rank - aspirin goes from upstream's weakest binder
to this port's strongest - while the binder probability largely **does**,
despite sitting systematically high (0.339 against 0.224 on the propofol-ish
case). The advice "read pIC50 as a value, the probability as a ranking" was
exactly backwards.

🔴 **AND THE MECHANISM EXPLAINS BOTH.** The method-class error was a CONSTANT,
so it shifted every ligand together and left ordering intact; the atom-name
error is **per molecule** - each ligand draws a different set of wrong names -
which is precisely the perturbation that leaves one molecule's value looking
right while destroying comparisons between molecules. The two faults had to be
diagnosed differently for that reason, and the ranking test is what separated
them.

**Statistical honesty.** n = 6 gives rho wide error bars: it distinguishes
"preserved" from "scrambled" and not fine degrees, and rho = -0.03 is *no
evidence of correlation* rather than proven independence. All six are scored
against a 33-residue target with no pocket, so this measures whether the two
implementations agree with each other, not whether either is right about
binding.

## Standing advice, revised twice

- 🔴 **Not shippable as a screening tool as it stands.** The value head is
  unusable for potency (offset) and for ranking (uncorrelated). The binary
  head ranks at rho 0.89 with 2 of 15 pairs inverted, which is thin for a tool
  whose entire output is an ordering.
- **Reproducing RDKit's `CanonicalRankAtoms` is required, not optional.** The
  payoff is measured and total: substituting upstream's ten names takes the
  atom encoder's ligand half from 1.1556 to **4.48e-2**, and the implied
  `s_inputs` to **4.38e-2** - the int5 floor, nothing left above quantisation.
- **Go/no-go, fixed before starting:** implement Morgan refinement with
  RDKit's documented initial invariants and measure EXACT rank agreement over
  the 74-molecule corpus `npm run test:smiles` already holds against real
  RDKit. 74/74 - finish it. Materially short - the divergence is in canonical
  tie-breaking, which is the genuinely implementation-specific part; stop,
  record the number, and the tool ships with the binary head labelled as a
  weak ranking signal.
- **Ruled out: RDKit WASM for names only.** MinimalLib exposes no rank
  accessor. `_smilesAtomOutputOrder` is reachable
  (`[21,9,8,7,20,6,19,5,18,4,17,3,2,15,16,1,13,14,0,10,11,12]`) but is the
  SMILES traversal order, from which the ranks are not recoverable - atom 0
  ranks 19 and sits at position 18, atom 1 ranks 21 at position 15. A custom
  RDKit build with a new binding would be needed, at >= 2.4 MB.
- 🔴 **Unmeasured and unmatched: the affinity path's SMILES `standardize`.**
  A no-op on every ligand tested here, so it has affected no measurement - and
  it will not be a no-op on a charged, salted or tautomer-ambiguous ligand,
  which is most of a real library. Nothing checks it.

---

# Gate, re-registered 2026-11-09 (second attempt at the canonical ranking)

🔴 **THE FIRST GATE TESTED THE WRONG QUANTITY AND IS REPLACED, WITH THE REASON
RECORDED RATHER THAN THE CRITERION QUIETLY MOVED.** It read: "implement Morgan
refinement with RDKit's documented initial invariants and measure exact rank
agreement over the 74-molecule corpus; 74/74 - finish it; materially short -
the divergence is in canonical tie-breaking, stop." The attempt came out at
**33/74 exact**, which by that letter is a stop. But the criterion was written
on the assumption that falling short meant the ALGORITHM was wrong, and the
measurement says it is not:

```
partition right (our classes = contiguous blocks of RDKit's ranks)   74/74
class ORDER right (those blocks in our class order)                  37/74
exact ranks (class order AND tie-break right)                        33/74
```

Refinement to the correct symmetry classes - the part that is an algorithm -
is right on every molecule. Tie-breaking is right on 33 of the 37 molecules
whose class order is right. **The dominant fault is the order the classes sort
in**, which is a comparator, not a canonicalisation: a finite convention over
data already in hand (1,406 atoms with their ranks and their invariants).

## The new criterion, fixed before the work

Two stages, each with its own number, so "progress" cannot be claimed from the
one that is easy:

1. **Class order: 74/74.** Every molecule's classes must sort into RDKit's rank
   blocks in the same sequence. This is the comparator - the initial
   invariant's field order, and how the refinement compares a neighbour
   multiset.
2. **Exact ranks: 74/74.** Then tie-breaking, which currently costs 4
   molecules of the 37 it can be measured on.

**Stopping rule, so this cannot run forever.** The search space is the
comparator's configuration: field order, neighbour-multiset direction, and
where degree enters. It is searched by SCORE against the oracle, not by
intuition. If stage 1 cannot reach 74/74 by that search, stop and record the
best configuration and its number - that is the honest end, and the tool ships
with the binary head labelled a weak ranking signal (rho 0.89, 2 of 15 pairs
inverted) and the value head labelled unusable (rho -0.03).

🔴 **AND THE SEARCH IS NOT ALLOWED TO BE THE GATE.** Fitting a comparator
against 74 molecules and then reporting agreement on those same 74 is
circular. Whatever configuration wins must be held to data the search never
saw: `check-smiles-rewrites.mjs` already re-writes the corpus **1,148 ways**,
and canonical ranks are by definition invariant to how a molecule is written -
so every re-writing must give the same ranks, and that is a held-out test of
the convention with no new oracle needed.

## Result: stage 1 not reached. STOPPING, per the rule above.

Searched, by score against the oracle rather than by intuition:

| axis | searched | best class order |
|---|---|---|
| initial invariant field order | greedy over all 9 fields, every position | 41/74 |
| neighbour multiset direction | ascending, descending | no change |
| degree inside the neighbour key | with, without | no change |
| neighbour signature | class / class+bond order / bond order+class / class+aromatic | no change |
| whole-order direction | ascending vs descending class->rank | descending **0/74** |

**Plateau at 41/74 on class order, against the 74/74 the gate requires.** The
decomposition at the stopping point:

```
contiguous blocks (partition)                74/74
class order ascending                         37     (the shipped configuration)
class order descending                         0
genuinely permuted                            37
exact ranks                                   33/74
```

🔴 **IT IS NOT A DIRECTION FLIP AND NOT A FIELD PERMUTATION.** Zero molecules
come out descending, so the convention is not reversed; and no ordering of the
nine per-atom invariants, with or without bond order in the neighbour
signature, gets past 41. For 37 molecules - glycerol, naphthalene, indole,
phenol, toluene, acetate, methanesulfonate, sulfate among them - our classes
sort into RDKit's blocks in a genuinely permuted order. Something in RDKit's
comparator is not a function of the fields dumped here.

🔴 **AND THE RIGHT NEXT STEP IS NOT MORE FITTING.** This was black-box
reverse-engineering from output while the source is open: RDKit's
`Code/GraphMol/new_canon.h` and `Canon.cpp` define `rankMolAtoms`, its
`AtomCompareFunctor`, and `BreakTies`, and they are not on this machine (a pip
wheel ships no C++). Reading them is bounded work with a definite answer, where
another sweep of the configuration space is not. Thirty-seven molecules whose
class order is permuted is exactly the data to read them against.

## What this leaves, and it is unchanged from before the attempt

- `shared/chem/canonical-rank.js` stays, **wired to nothing**, with its header
  stating it does not reproduce RDKit. It is 74/74 on the symmetry partition,
  which is the reusable half; a future attempt starts from there rather than
  from scratch.
- `tools/check-canonical-rank.mjs` reports both numbers and fails no build.
- The tool's labels stand as measured: `affinity_probability_binary` a weak
  ranking signal (rho 0.89, 2 of 15 pairs inverted),
  `affinity_pred_value` **not usable for potency or ranking** (rho -0.03).
- 🔴 **Adopting the 41/74 configuration was considered and declined.** A rank
  is right or it is not; a configuration that names 41 molecules correctly
  instead of 37 produces no better affinity, and swapping it in would be
  motion dressed as progress.

## Second attempt, from the source: 33/74 -> 59/74

`refs/rdkit-src/` holds `new_canon.h`, `new_canon.cpp` and `Canon.cpp` pinned at
`Release_2025_03_5`, the oracle's own RDKit version. Ported from them:

| what | source | effect |
|---|---|---|
| `basecomp`'s field order; `explicitValence`/`inRing`/`aromatic` REMOVED (not in it) | `new_canon.h:299` | part of 33->40 |
| charge read signed into an **unsigned** variable, so `0 < +1 < ... < -2 < -1` | `new_canon.h:391` | " |
| `bondholder::compare` - type, then stereo, then neighbour class | `new_canon.h:68` | " |
| bond lists sorted **descending**, walked in **parallel**, length last | `new_canon.cpp:687` | " |
| `BreakTies` takes the class's **last** atom and gives it the **highest** rank | `new_canon.h:791` | 37 -> 40 |
| atom map number compared **before** degree | `new_canon.h:326` | 57 -> 58 |
| `SpecialChirality` pass, `updateAtomNeighborNumSwaps`, `countSwapsToInterconvert` | `new_canon.cpp:702` | 58 -> 59 |

🔴 **THE LARGEST SINGLE FIX WAS A HARNESS BUG, NOT AN ALGORITHM ONE.**
`canon_atom.totalNumHs` is `getTotalNumHs()` on the **H-added** molecule, which
is **0 for every atom** because the hydrogens are explicit by then; the only
adjustment (`new_canon.cpp:663`) applies to fragments. This port had been
feeding the pre-`AddHs` count, inventing a base-level distinction RDKit does not
make - bromoethane's two carbons came out swapped by it. Correcting it was class
order **41 -> 62** and exact **40 -> 57**, more than every algorithmic fix
combined. The dump had RECORDED that field as 0 earlier and it was dismissed as
a dump artefact; it was the specification.

🔴 **AND AN ATTRIBUTION OF MINE WAS WRONG.** The 17 failures were first split by
whether the molecule HAS chirality or rings - a property of the input, not a
diagnosis. Implementing the chirality pass gained **one** molecule, because
those molecules fail on class ORDER and that pass refines WITHIN classes:
`RefinePartitions` sorts inside a partition and structurally cannot reorder
partitions. Attributing by correlation cost a pass's worth of work for one
molecule.

### Standing at 59/74, and what the last 15 need

```
symmetry partition   74/74       exact ranks   59/74
class order          63/74       tie-free       5/6
```

All 15 remaining are polycyclic or stereocentre-bearing: naphthalene, indole,
bicyclic, caffeine, porphine, coronene_frag, taxol_core, biotin, atp, glucose,
cholesterol, penicillin_core, nad_frag, cyclosporin_frag, erythromycin_frag.
The unported piece is the **`SpecialSymmetry` pass** (`new_canon.cpp:341`) with
`compareRingAtomsConcerningNumNeighbors` and the `revistedNeighbors`
machinery - gated on ring-symmetry fraction > 0.5 with a branching ring atom,
which is exactly this set. `getAtomRingNbrCode` was read and is inert here: it
keys on `isRingStereoAtom`, RDKit's cis/trans ring-stereo flag, not on
tetrahedral centres.

**The gate still requires 74/74 and is not met.** Nothing imports the module.

### Both special passes ported; the fault is the DRIVER

`SpecialChirality` (with `updateAtomNeighborNumSwaps` and
`countSwapsToInterconvert`) and `SpecialSymmetry` (with
`compareRingAtomsConcerningNumNeighbors` and its two level-profile vectors, plus
RDKit's own gate) are both ported from the pinned source. **Together they gain
one molecule: 58 -> 59.**

That is not a bad port, it is the algorithm's shape, and measuring it located
the fault:

```
basecomp-only initial partition, consistent with RDKit   74/74  (contiguous AND ordered)
class order after refinement                             63/74
exact ranks                                              59/74
```

🔴 **EVERY REMAINING FAILURE IS A CLASS-ORDER ERROR, AND NO REFINEMENT PASS CAN
FIX ONE.** `RefinePartitions` sorts within a partition: it splits classes and
never reorders them. So a comparator added after the main refinement cannot
change which class sorts first, however faithful it is. Verified rather than
reasoned: naphthalene meets the symmetry gate exactly (10 ring atoms, a
branching ring atom, ties present), the pass runs, and nothing changes.

🔴 **AND THE SAME MIS-ATTRIBUTION WAS MADE TWICE.** The failures were grouped
by whether a molecule HAS chirality or rings - a property of the input - and
each pass was then built on that basis. Both were the wrong tool, for a reason
visible in the source before either was written. Group failures by diagnosis.

🔴 **THE REMAINING WORK IS `RefinePartitions`' BOOKKEEPING.** RDKit carries a
persistent `order` array across sweeps, sorts each partition's slice in place
with a STABLE sort (`hanoisort`), and assigns `symclass = offset + i` from
positions in that array. This port re-ranks globally each sweep and breaks ties
by ATOM INDEX. Those agree on the first sweep and may diverge on every one
after, because a stable sort preserves the PREVIOUS order where index
tie-breaking does not. The initial partition being right on 74/74 while the
post-refinement order is right on 63 is exactly that signature.

---

# Two composition defects, found by a cold review, and the re-measurement

A fresh review pass - a subagent reading the tree with no memory of building it -
found two defects in the WIRING of the affinity pass. Both had survived every
gate in this file, and both were upstream of the atom-name difference this
document spends its length attributing the gap to. **Every number recorded above
was taken through them.**

## Defect 1: per-ATOM coordinates where the module needs per-TOKEN

`web/affinity-run.js` passed `best.sample.positions` - `fold.positions`, indexed
by dense **atom slot** (`token*dense + atom`, which is what
`shared/af3/structure/pdb.js` writes from, `tokens*dense*3` long). The module's
distogram indexes `positions[i*3]` for `i < tokens`. At a 43-token crop with
dense 24 it read dense slots 0..42 - the first ~2 tokens' atoms, masked slots
included - as if they were 43 token centres. **Every distance in the distogram
was wrong.**

🔴 **THE FLOAT64 REFERENCE ENFORCED THE CONTRACT AND THE GPU PATH DID NOT.**
`cpu/af3/affinity/affinity.js:309` asserts `need(positions, n * 3)` and throws
`"positions: expected 12 values, got 288"`; the GPU loop had no length check and
indexed happily past the tokens. The correct array was already captured and
unused one line away: `fold.pseudoBeta`, `tokens*3`, the disto gather, which is
what upstream's `token_to_rep_atom` selects. A guard is now in the GPU path.

🔴 **AND WHY NO GATE SAW IT.** `check-affinity-forward.js` and
`check-affinity-oracle.mjs` feed `oracle("input/x")` - upstream's OWN per-token
coordinates - so they test the kernel and never the wiring.
`test/affinity-run.test.js` stubs the module and its fake samples carry
`positions` of the CORRECT shape, which is exactly why a real shape mismatch was
invisible. And `dump-affinity-inputs.js` derives its reported shape FROM the
array's length, so its own shape check could never fire on this tensor.

## Defect 2: the trunk pair handed over unmasked

`web/affinity-passes.js` read:

```js
// upstream pre-masks the trunk pair before handing it over
zAffinity: Float32Array.from(z),
```

The comment stated the requirement and the line did not do it. upstream applies
`z * cross_pair_mask` (boltz2.py:620) BEFORE the module's `z_norm` and
projections, so every receptor-receptor pair upstream zeroes was being
normalised, projected and mixed across pairs by the triangle operations. The
dump tool masked it and the page did not - the two disagreed about the module's
input, which is why the dumps looked consistent.

## Re-measured, same six ligands, same upstream control

```
affinity_pred_value (lower better)   upstream    before     after
  propofol-ish                         1.3702    1.3596    1.1708
  ethanol                              1.4100    2.3878    1.9593
  ibuprofen                            0.9301    0.8564    0.8440
  aspirin                              1.5887    1.0201    0.8951
  nicotine-ish                         0.8530    1.5061    1.0966
  1-naphthol                           1.3867    1.2670    1.2081
  rho   BEFORE -0.0286 (8/15 discordant)    AFTER +0.3143 (5/15)

affinity_probability_binary (higher better)
  rho   BEFORE +0.8857 (2/15)               AFTER +0.8857 (2/15)
```

🔴 **THE VALUE HEAD IMPROVED FROM NO CORRELATION TO WEAK; THE CLASSIFIER DID NOT
MOVE AT ALL.** ρ −0.03 → +0.31 with discordant pairs 8 → 5, which at n = 6 is
not significant (p ≈ 0.54) and is consistent with noise as much as with the
fixes. Aspirin still inverts worst. The value column's verdict is unchanged: not
usable for potency or for ranking.

🔴 **AND THE CLASSIFIER'S ρ BEING IDENTICAL TO FOUR DECIMALS IS THE INFORMATIVE
PART.** A distogram reading entirely the wrong atoms, and an unmasked pair
track, changed its ordering not at all - the same two pairs invert. Its ranking
is therefore robust to both defects and driven by what reaches it through
`s_inputs`, which is where the atom-name difference lives. **So the atom-name
attribution survives.** It was not safe to assert before this re-measurement,
and this document asserted it anyway.

## What this says about the gates, which is the durable lesson

Both defects were in the COMPOSITION. Thirteen gates hold this port's stages to
upstream's own dumps at 1e-5 to 1e-8, and not one could see either: a gate that
supplies the reference's inputs tests the function, never the wiring that
produces them in the real path. `run-affinity-screen.js`'s own header names
"the pre-masking of the trunk pair" as a composition risk only running would
find - it ran, for the whole G8 campaign, with the pre-mask absent.

🔴 **AND THE STUB THAT MADE IT INVISIBLE WAS THE MORE EXPENSIVE MISTAKE.**
`test/affinity-run.test.js` feeds the module a fake sample whose `positions` has
the shape the module wants rather than the shape the real path supplies. A stub
that is more correct than production is a stub that certifies the bug.

### Final state of the ranker: 64/74, and the fault is isolated to class NUMBERING

Every function in `CanonicalRankAtoms`' call path has now been read from
`refs/rdkit-src/` (pinned at the oracle's own RDKit 2025.03.5) and ported:
`initCanonAtoms`, `getBonds`, `makeBondHolder`, `basecomp` with its field order
and its unsigned-charge quirk, `bondholder::compare`, the descending sort and
parallel walk, `RefinePartitions`' persistent `order`/`count`/`activeset`
bookkeeping with stable within-partition sorts, `BreakTies`, and both special
passes.

🔴 **AND THE DECISIVE DIAGNOSTIC WAS AVAILABLE ALL ALONG:
`CanonicalRankAtoms(mol, breakTies=False)` returns RDKit's OWN pre-tie-break
symmetry classes.** Measured against those rather than inferred from final
ranks:

```
identical PARTITION       74/74
identical class NUMBERS   64/74
exact ranks               64/74
```

So the refinement is **exactly right on every molecule** - this port's classes
ARE RDKit's - and the entire remaining fault is the ORDER in which 10
molecules' classes are numbered. Tie-breaking is correct wherever the numbering
is. Two sessions of this work were spent inferring an intermediate state that
one keyword argument hands over directly.

**What was searched and ruled out**, each scored on all 74 rather than argued:

| candidate | result |
|---|---|
| neighbour sort ascending instead of descending | 53/74 (worse) |
| neighbour class compared before bond type | 58/74 (worse) |
| bond-list length compared first | no change |
| the `div` / `ATNUM_CLASS_OFFSET` two-pass trick | **not in this path** - it belongs to `ChiralAtomCompareFunctor`, which `rankMolAtoms` does not use |
| `getAtomRingNbrCode` | inert - keys on `isRingStereoAtom`, RDKit's cis/trans ring flag, not tetrahedral centres |

The 10 remaining molecules are indole, naphthalene, caffeine, biotin, atp,
glucose, cholesterol, penicillin_core, nad_frag and one more - all polycyclic
or stereocentre-bearing. Traced on indole, which is TIE-FREE so tie-breaking
cannot be involved: exactly one adjacent pair is swapped, atoms 5 and 7, both
degree-3 aromatic carbons that tie through all of basecomp. Our parallel walk
decides at position 0 (its neighbour's class 15 against 14); RDKit's answer is
what you get if positions 0 and 1 TIE and position 2 decides, where
`single(1) < aromatic(12)`. No ordering of the keys read from the source
produces that, which is where this stops.

🔴 **NOT FINISHED, AND THE GATE SAYS SO.** 64/74 against a required 74/74.
Every structural piece is ported and the remaining difference is not visible by
reading the source or by searching the space the source defines. Finishing it
needs RDKit's own intermediate ORDER array, which `VERBOSE_CANON` prints and a
pip wheel cannot produce - a source build of RDKit, which is a different kind
of task from this one. Nothing imports the module.

## The atom names, closed — measured, not substituted (2026-11-10)

The "required, not optional" item above is done. RDKit's `CanonicalRankAtoms`
is reproduced exactly over the 74-molecule corpus and wired into the path the
page runs, and the term comparison was re-measured through it rather than by
substituting upstream's ten names by hand.

| term, G8's job (`MKTAY…`, `CCCc1ccccc1O`) | AF3's names | canonical names |
|---|---|---|
| `s_inputs`, whole | — | **4.3764e-2** |
| `s_inputs`, 33 protein tokens | 3.90e-2 | 3.91e-2 (unchanged: the int5 floor) |
| `s_inputs`, **10 ligand tokens** | **1.1419** | **4.8053e-2** |
| `z`, 760 live pairs | 2.10e-1 | 2.0033e-1 |

🔴 **THE PREDICTION WAS 4.38e-2 AND THE MEASUREMENT IS 4.3764e-2.** The
hand-substitution experiment implied an `s_inputs` of 4.38e-2 once the names
were upstream's; the production path - this port's own SMILES parser, its own
hydrogen-added graph, its own ranker, its own naming - lands there. The ligand
half is now at the protein half's value, which is the int5 quantisation floor,
so there is nothing left in `s_inputs` above the bundle's own precision.

🔴 **AND `z` IS STILL 2.0e-1, WHICH THIS DOES NOT FIX.** The trunk pair moved
2.10e-1 -> 2.0033e-1 - about 5% of the remaining gap - so the conditioning
`z` is now the single largest unexplained term in the affinity comparison by a
wide margin, where before it could be read as inheriting `s_inputs`. It cannot
be: `s_inputs` is at the quantisation floor and `z` is twenty times it. That is
the next question, and no claim here rests on it being small.

### What the ranking fault actually was

`basecomp` carries a chirality field, `getChiralRank`, that reads the **live**
classes - and this port had chirality only in the later `compareChirality`
pass, where `RefinePartitions` can subdivide a partition but never reorder one.
So the term could split a class and could not decide which side of its
neighbours it landed on, which is why adding both special passes earlier had
gained exactly one molecule.

It was found by diffing the two implementations' PARTITION PROCESSING ORDER,
not by reading comparators. Both sides pop the same six partitions in the same
order with the same members on glucose; the only difference in the whole trace
was one `offset`, 20 against 21, which is a single comparator call on a
partition both sides agree about. Printing RDKit's own verdict for that call
gave `CMP 2 vs 4 -> 1 base=1` with atom 2's bond list never even refreshed -
so the decision was made in `basecomp`, before any neighbour was consulted, by
a field this port did not have. `basecomp(2,4)` is **0** on the first pop and
**1** on the second, which is only possible for a field that reads the classes.

The term abstains the moment two of an atom's neighbours share a class (RDKit
`break`s out of the walk and returns 0), which is what makes it safe to apply
before the classes have settled. On glucose's second pop only carbon 2 has four
distinct neighbour classes, so it alone gets a verdict and sorts above the
other four stereocentres; the six oxygens then inherited that swap through
their own neighbour classes, which is why the fault presented as an oxygen
ordering two pops away from its cause.

### Two neighbour-ordering conventions found on the way

Both are in the production path only - the corpus gate feeds the ranker RDKit's
own dumped graph, so neither could appear there.

- **A ring-closure bond comes after every chain bond.** RDKit defers closures
  to `SmilesParseOps::CloseMolRings`, run once the string is read, so
  `GetNeighbors()` lists chain neighbours first whatever order the digits
  appeared in. 40 atoms of the corpus, and 7 stereocentre tags with them
  through `getChiralRank`'s permutation parity.
- **Among one atom's own closures, ring-digit order.** Those bookmarks live in
  a map keyed by the digit, so `c61` lists ring 1's partner before ring 6's -
  not the written order. Cholesterol's two closures agree either way; coronene's
  `c61` is what separates them.

### Gates

- `npm run test:canonical-rank` — **74/74** exact ranks, 74/74 partitions,
  against RDKit 2025.03.5. Was 66/74 before `getChiralRank`.
- `npm run test:canonical-names` — **72/74** atom names, through the whole
  production path. The names are what reach the model.

🔴 **AND THE TWO FAILURES ARE NOT RANKING FAULTS - THEY ARE PARSE-LEVEL, AND
BOTH ARE WORSE THAN A NAME.** Neither is fixed and neither is hidden:

- **`deuterium` — FIXED 2026-11-10.** This port folded a bracket hydrogen into
  its neighbour's hydrogen COUNT, so `[2H]C([2H])([2H])O` parsed to TWO atoms -
  a carbon carrying three plain hydrogens - where RDKit keeps five
  (`H2 C H2 H2 O`), and **the isotope was lost entirely**: a deuterated ligand
  featurised as the protiated one, wrong in `ref_element` and in the atom count
  for every model family, not merely in a name.

  `mergeExplicitHydrogens` now keeps a hydrogen singled out by an **isotope** or
  an **atom map**, which is what RDKit's `removeHs` keeps - and 🔴 that
  function's own comment already said "a charged or mapped hydrogen is being
  singled out deliberately" while the code tested only the charge. The rule was
  written down correctly and implemented halfway. Every field category in
  `test:canonical-names` is now 0 differing, plain `[H]` still folds as RDKit
  folds it, and the gate DEMANDED its declaration be deleted rather than
  carried - it fails when a pinned molecule comes good.
- **`porphine` — STILL OPEN, cause now exact.** This port TRUSTS the SMILES'
  lower case; RDKit RE-PERCEIVES aromaticity with its own ring model and
  demotes two of porphine's four meso bridges: atoms **0, 1, 10 and 11** become
  non-aromatic and their six bonds ordinary (2,1,1,2,1,1), where this port has
  all twenty carbons aromatic. 8 atoms' bond types differ, every per-atom field
  agrees, and the ranking of the molecule this port builds is correct FOR that
  molecule.

  🔴 **IT IS NOT A PATCH, WHICH IS WHY IT IS STILL OPEN.** Closing it means
  porting that model: kekulise, then electron-count each ring AND each fused
  ring SYSTEM with RDKit's per-atom contributions. Aromaticity feeds the
  bond-order plane (whose index was off by one until today), the conformer's
  planarity rules and `test:smiles-batch`'s exact agreement with AF3 across
  eight families - so a half-model that fixed porphine while quietly moving
  another molecule would be worse than a declared gap.
  `bondholder::compare` ranks the bond TYPE before the neighbour's class, so
  one differently-perceived ring reorders every atom around it.

Both affect every family's features for those molecules, not just boltz2's
names, so they are recorded here and left for their own change.

### Where it is wired

`smilesCanonicalAtomNames` in `shared/af3/dialect.js`, true for **boltz2
alone** and false for the other eight - `shared/chem/canon-input.js` builds the
hydrogen-added molecule and `shared/chem/component.js` chooses the rule. The
affinity path (`web/affinity-passes.js`) and the AF3-lineage fold path
(`web/af3-model.js`) read the flag. ESMFold2's ligand path in `web/app.js` and
`shared/bioemu2/inputs.js` are deliberately left on AF3's rule: they are not
boltz2 and upstream's naming is not theirs. `npm run test:smiles-batch` is
still exact, which is the check that AF3's own path did not move.

## Localising `z`: it is not quantisation, and it is not the pose (2026-11-10)

With `s_inputs` closed, `z` was the only large term left. Five candidates are
now excluded by measurement rather than by argument.

| candidate | how it was excluded |
|---|---|
| **recycles** | both sides run **5**. Upstream's `predict_affinity_args` in `main.py` sets `recycling_steps: 5`; `AFFINITY_PASS.recyclingSteps` is 5 |
| **weight source** | upstream's affinity stage loads the WHOLE model from `boltz2_aff.ckpt`, so its trunk is that checkpoint's - and that trunk is **byte-identical** to `boltz2_conf.ckpt`'s: relRMS exactly **0** over all 5019 shared tensors. (Independently reproduces G0, which `tools/export_boltz2_affinity.py`'s header already records.) Reading the structure bundle's trunk for the affinity pass is correct |
| **the template stage** | a real difference, now fixed, and worth **nothing**: see below |
| **method conditioning** | upstream's affinity data module passes `override_method="other"`, and `const.py`'s `method_types_ids["other"]` is **4** - the class this port sets. The 1 -> 4 fix is right for the affinity pass BY NAME, not by coincidence |
| **the crop / token set** | the identity on this job, 43 of 43, reported by both dumps |

### 🔴 int5 IS 3.04e-2 AND THIS PORT IS 2.00e-1, SO THE GAP IS 6.6x THE FLOOR

The trunk ships at 5 bits and `z` is four MSA blocks plus forty-eight
pairformer blocks deep where `s_inputs` is one layer, so "it is the
quantisation" was the obvious reading - and CLAUDE.md's own datum, boltz2 int5
**0.167** against f32 8.25e-7 at a trunk seam, is the same order as 0.200.

It is measurable without an f32 bundle, and without this port: put LocalFold's
exact scheme - asymmetric per group of 32, 5 bits, scale and zero float16, read
off `model-boltz2-int5/manifest.json` - into **upstream's own weights** and
re-dump. `dump_boltz2_affinity_inputs.py --quantise-trunk` does it: 2251 trunk
tensors round-tripped, 1468 norms and biases kept float32, and it raises rather
than continuing if no tensor was touched.

| | upstream int5 vs upstream f32 (**the floor**) | this port vs upstream f32 |
|---|---|---|
| `s_inputs` whole | 4.1388e-2 | 4.3764e-2 |
| `s_inputs` protein | 3.8794e-2 | 3.9128e-2 |
| `s_inputs` ligand | 4.3927e-2 | 4.8053e-2 |
| **`z` live pairs** | **3.0373e-2** | **2.0033e-1** |

Two conclusions, and the first is as useful as the second:

- **`s_inputs` is AT the floor** on all three rows, so the atom-name work is
  finished and nothing in the conditioning is left above the bundle's own
  precision. This is the control that makes the `z` row believable.
- **`z` is 6.6x the floor**, so the disagreement is real and enters AFTER the
  input embedder. 🔴 **AND QUANTISATION ATTENUATES THROUGH THE TRUNK RATHER
  THAN AMPLIFYING** - `z`'s floor (3.04e-2) is BELOW `s_inputs`' (4.14e-2),
  through 52 blocks - so "it is a deep stack" is not available as an excuse,
  and neither is the int5 reading that the 0.167 datum invited.

🔴 **AND THE TRUNK'S WEIGHTS ARE NOT UPSTREAM'S AT ALL, WHICH THIS COMPARISON
MADE PLAIN.** `model-boltz2-int5`'s tensors are HAIKU paths out of
af3-any-model's `boltz2.bin.zst` - its re-export of Boltz-2 - while the
affinity module alone is exported from `boltz2_aff.ckpt` by this project's own
tool. So `z` has three possible sources and only the first is now ruled out:
the 5-bit packing (excluded above), **af3-any-model's conversion** of the
boltz2 trunk, and this port's arithmetic. `check-bundle-vs-params.py` once
found FOUR NEGATED TENSORS in that very export, so the middle one is not
hypothetical.

### The template stage: a real difference worth exactly zero

`boltz2_aff.ckpt` carries no `template_module`, so `Boltz2.__init__` leaves
`use_templates` false and the affinity run never executes
`z = z + template_module(z, ...)`. This port ran it, with four empty slots,
because `trunk.js` gated the stage on nothing at all - and that file's own note
says an empty stage is "still a quarter of what enters the MSA stack".

Fixed (`noTemplate`, set by the affinity pass only), and **`z` is byte-identical
either way**: 2.0032767412975716e-1 before and after, `rms` 38.38218 both. 🔴
**THE ARM WAS PROVEN TO REACH THE BRANCH BEFORE THAT WAS BELIEVED** - a forced
`throw` in the skipped path failed the run - because byte-identical output is
exactly what a control that never ran looks like, which this repository has
recorded twice (`--f16=off` on the matrix kernels, `--free=off` on pooling). So
the empty-slot term is genuinely zero for boltz2 and the change is a stage of
work saved per recycle, not a fix.

### Next

The stage table: `z_init` and its five summands, then each recycle's MSA and
pairformer output, on both sides. Upstream's are forward hooks on submodules
(`dump_boltz2_affinity_inputs.py` writes `z-stages.npz`); this port already
taps `tap.z_init_generic`, `tap.z_after_template`, `tap.z_after_msa` and
`trunk_out_pair`. `rel_pos`, `token_bonds`, `token_bonds_type` and
`contact_conditioning` are FEATURISATION terms of the same kind as the atom
names, and none of them is visible in `s_inputs`.

### The stage table, each seam against its OWN int5 floor

Same method as above - upstream's weights round-tripped through this bundle's
5-bit scheme - but with the trunk's stages hooked, so every seam is judged
against the precision available AT THAT SEAM rather than against the output's.
Last recycle, live pairs only.

| seam | int5 floor | this port | x floor |
|---|---|---|---|
| `z_init` + recycle | 2.0878e-2 | 7.1652e-2 | **3.4x** |
| after the MSA term | 1.6865e-2 | 9.7108e-2 | **5.8x** |
| pairformer out (= `z`) | 3.0373e-2 | 2.0510e-1 | **6.8x** |

🔴 **SO IT IS ALREADY WRONG AT THE EMBEDDER'S PAIR, AND IT GROWS.** 3.4x the
floor before a single MSA or pairformer block runs, then 5.8x, then 6.8x. The
floor itself does NOT grow through the stack (2.1e-2 -> 1.7e-2 -> 3.0e-2), so
the growth is this port's and not the arithmetic's depth.

Two interpretation checks passed on the way, and both were worth taking:
`tap.z_init_generic` matches upstream's `z_init + z_recycle` at 7.17e-2 and
`z_init` ALONE at 2.99e-1, which is how one knows our embedder's output already
carries the recycle term; and `tap.template_term` is **exactly 0.000e+00**,
independently confirming the template result above from the port's side.

### 🔴 AND IT IS CONCENTRATED ON **BONDED** PAIRS - 10.7x THE FLOOR ON 20 OF 760

| `z_init` region | int5 floor | this port | x floor | pairs |
|---|---|---|---|---|
| all live | 2.0855e-2 | 7.1707e-2 | 3.4x | 760 |
| ligand-ligand | 2.4293e-2 | 1.3282e-1 | 5.5x | 100 |
| **bonded** | **2.9138e-2** | **3.1303e-1** | **10.7x** | **20** |
| ligand-ligand, unbonded | 2.3371e-2 | 6.3232e-2 | 2.7x | 80 |
| protein-ligand | 2.0253e-2 | 5.6126e-2 | 2.8x | 660 |

Those 20 pairs are the ligand's own 10 bonds, both directions, and they carry
**38% of the whole residual's energy**. Energy by `|i-j|` says the same thing
from the other side: 675.8 at `|i-j| = 1` against 25-50 everywhere else, which
is a chain-written ligand's adjacent atoms.

🔴 **THE OBVIOUS INNOCENT EXPLANATION IS REFUTED BY THE FLOOR, WHICH IS WHY THE
FLOOR HAD TO BE MEASURED PER REGION.** `z_init_1/2` are OUTER sums of
`s_inputs`, so a ligand-ligand pair inherits the ligand `s_inputs` residual on
BOTH indices and should be worse with no defect at all - and all 20 bonded
pairs are ligand-ligand. But the floor is FLAT across every region
(2.03e-2 to 2.91e-2): the inheritance does not raise it, so it cannot be what
raises ours 10.7x. Had the comparison stopped at "bonded pairs are worse" it
would have reported a real effect with the wrong cause.

So there are two things in `z_init`, not one: a **bonded-pair fault** at ~4x the
background, and a background of 2.7-2.8x the floor spread over every region.
`z_init`'s summands are `z_init_1/2` (from `s_inputs`, which is AT the floor),
`rel_pos`, `token_bonds`, `token_bonds_type` and `contact_conditioning`; only
the two bond planes are supported on exactly the bonded pairs, and
`token_bonds_type` is a large term (rms 1.3175 over live pairs against
`rel_pos`'s 1.7353 and `contact_conditioning`'s 0.2896).

Upstream's codes are `const.py`'s `bond_types` - OTHER 0, SINGLE 1, DOUBLE 2,
TRIPLE 3, AROMATIC 4, COVALENT 5 - written SYMMETRICALLY
(`bonds_type[t2, t1] = bond_type`), which matches the conventions this port
already records, so the enumeration is not the difference and the next step is
to dump our two bond planes and diff them entry by entry against upstream's
`token_bonds`/`type_bonds` on those 20 pairs. Nothing here rests on a guess
about which plane it is.

### 🔴 FOUND: THE BOND-TYPE CODE IS SHIFTED BY ONE, AND `z` CLOSES TO THE FLOOR

Dumping both bond planes and diffing them pair by pair settled it in one step.
The ADJACENCY is identical - the same 20 pairs, `token_bonds` matching as a set
- and **every type code is off by exactly +1**:

| | this port | upstream |
|---|---|---|
| the 8 single bonds | 1 | **2** |
| the 12 aromatic bonds | 4 | **5** |

Upstream's tokeniser writes `bond["type"] + 1` (`boltz/data/tokenize/boltz2.py`,
"Add atom-atom bonds from ligands") and leaves every unbonded pair at 0. 🔴
**WHICH IS WHY THE EMBEDDING HAS SEVEN ROWS FOR THE SIX NAMES IN `const.py`'s
`bond_types`** - row 0 is "not bonded", and that spare row is the tell that was
sitting in both the checkpoint `(7, 128)` and this bundle's
`token_bonds_type_embed [7, 128]` the whole time. This port wrote the unshifted
code, so every bonded pair read the row one too early: a SINGLE bond took the
OTHER row and an AROMATIC bond the TRIPLE row.

Fixed at the LOOKUP, in `cpu/af3/trunk/embedder.js` and the matching WGSL in
`webgpu/af3/trunk/embedder.js`, not in the featuriser - the plane itself carries
AF3's unshifted convention for the seven other families, and only boltz2 has
`tokenBondsTypeEmbed`, so the branch is already boltz2-only. 🔴 **AND THE SHIFT
IS CONDITIONAL ON THE ADJACENCY PLANE**: adding one everywhere would move
unbonded pairs off row 0, and the order plane alone cannot tell "no bond" from
an OTHER bond - both are 0.

| seam | int5 floor | before | after | x floor |
|---|---|---|---|---|
| `z_init` + recycle | 2.0855e-2 | 7.1707e-2 | **2.4341e-2** | 1.2x |
| after the MSA term | 1.6798e-2 | 9.6485e-2 | **2.9009e-2** | 1.7x |
| **pairformer out (= `z`)** | **3.0385e-2** | **2.0033e-1** | **3.1032e-2** | **1.0x** |
| `z_init`, the 20 bonded pairs | 2.9138e-2 | 3.1303e-1 | **3.0200e-2** | 1.0x |

**`z` is now AT the int5 floor**, 6.5x closer than it was, and so are both
conditioning inputs the affinity module reads. The background 2.7-2.8x that the
region split showed off the bonded pairs went with it, so it was this same term
leaking through `z_recycle` rather than a second fault.

🔴 **AND NO GATE HERE COULD HAVE SEEN IT, WHICH IS THE LESSON.** `test:batch`
compares the PLANE against af3-any-model's dump and the plane was never wrong;
the lookup is in the embedder, and every embedder oracle in this tree compares
against **af3-any-model**, which converts the same unshifted way. The reference
and the port shared the mistake - docs/SMILES.md records that exact shape from
the other end, "a writer and reader sharing a mistake agree perfectly". Only
genuine upstream could say otherwise, and only elementwise: this is the fourth
defect G8 has found that every self-consistent gate passed. `test:batch`,
`test:affinity-oracle`, `test:affinity-contract` and `npm test` are all still
green, which is the check that the fix did not move the plane.

## End to end, re-measured after the atom names and the bond code (2026-11-10)

Upstream was NOT re-run: nothing on its side changed, so its five-seed control
(`refs/g8-out-5`) and its per-ligand values (`refs/g8-rank`) are still its own
numbers, and only this port's arm was repeated - eleven runs, sequential, one
headless Chrome at a time (`tools/run/g8_rerun.sh`,
`tools/run/g8_rank_report.py`).

| one ligand, five seeds a side | upstream | ours BEFORE | ours AFTER |
|---|---|---|---|
| `affinity_pred_value` | 1.3541 [1.2920..1.4061] | 1.3651 | **1.3208** [1.3045..1.3454] |
| gap in upstream's own spreads | — | 0.19x | **0.58x** |
| `affinity_probability_binary` | 0.2243 [0.2167..0.2344] | 0.3391 | **0.2144** [0.2067..0.2195] |
| gap in upstream's own spreads | — | **12.96x** | **1.12x** |

🔴 **THE CLASSIFIER CLOSED 11.6x AND THE VALUE HEAD DID NOT MOVE USEFULLY ON
THIS LIGAND - SAID PLAINLY BECAUSE IT IS THE LESS FLATTERING HALF.** The binary
head went from 13 spreads out to just over one. The value head's gap GREW,
0.0110 -> 0.0333, from 0.19x to 0.58x of upstream's spread: still comfortably
inside its seed-to-seed variation, so both readings are "agrees as well as the
quantity admits", and on ONE ligand with five seeds the move is not evidence of
anything either way. The six-ligand table is what carries the value head.

| six ligands, one seed each | upstream value / binary | BEFORE | AFTER |
|---|---|---|---|
| propylphenol | 1.3702 / 0.2234 | 1.3596 / 0.3321 | 1.3045 / 0.2195 |
| ethanol | 1.4100 / 0.1378 | 2.3878 / 0.2236 | 1.4400 / 0.1215 |
| ibuprofen | 0.9301 / 0.5244 | 0.8564 / 0.5473 | 0.8110 / 0.5539 |
| aspirin | 1.5887 / 0.3442 | 1.0201 / 0.3561 | 1.5561 / 0.2854 |
| nicotine | 0.8530 / 0.1791 | 1.5061 / 0.3404 | 0.9977 / 0.1707 |
| naphthol | 1.3867 / 0.2413 | 1.2670 / 0.5356 | 1.3031 / 0.2488 |

| Spearman rho vs upstream | BEFORE | AFTER | inverted pairs |
|---|---|---|---|
| `affinity_pred_value` | **-0.029** | **+0.886** | 8 -> **2 of 15** |
| `affinity_probability_binary` | +0.886 | **+1.000** | 2 -> **0 of 15** |

🔴 **THE "VALUE HEAD IS UNCORRELATED, rho = -0.03" FINDING IS SUPERSEDED, NOT
EXPLAINED AWAY.** It was a correct measurement of the port as it then stood, and
it is what justified calling the canonical ranking required rather than
optional. With the names and the bond code fixed the same six ligands give
+0.886.

🔴 **AND THE TWO SURVIVING INVERSIONS ARE WEIGHED AGAINST UPSTREAM'S OWN
SPREAD RATHER THAN COUNTED.** Its half-range here is 0.0570 log units.
`propylphenol/naphthol` sit **0.0165 apart in upstream's numbers** - inside it,
so upstream does not separate them at one seed and neither implementation can
be said to have them the wrong way round. `ibuprofen/nicotine` are 0.0771
apart, outside it: a genuine inversion, and the one real ordering defect left.

🔴 **AND WHAT THIS IS NOT.** n = 6; the ranking control is ONE upstream seed a
ligand, so it has no spread of its own; one 33-residue target, single-sequence.
rho = +0.886 at six points sits essentially on the 5% two-tailed critical
value - a real but thin correlation - while +1.000 is p ~ 0.003. And all of it
is agreement with upstream Boltz-2, NOT with experiment: no measured affinity
has been compared to anything here.
