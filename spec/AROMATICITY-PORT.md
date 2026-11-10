# RDKit's aromaticity model, ported — DONE (2026-11-10)

`shared/chem/aromaticity.js`, called from `parseSmiles` after kekulisation
and the implicit hydrogens. **77/77 on atom flags and 77/77 on bond flags**
against RDKit 2025.03.5 (`npm run test:aromaticity`), and the canonical-name
gate went 73/74 to **77/77 with nothing declared**.

## What it took, beyond the algorithm below

Three faults, two of them mine and one a rule I had not read:

1. 🔴 **Hydrogens counted twice.** RDKit's `getValence(EXPLICIT)` does NOT
   include an implicit hydrogen; every hydrogen in this parser is a count, so
   adding it to a sum that already held it put benzene's carbons at valence 5
   against an allowed 4 and made EVERY aromatic ring non-aromatic.
2. 🔴 **`atom.z` is undefined on a parsed graph** — the parser calls it
   `element` and only the ranker's adapter calls it `z`. A silent failure,
   not an error: `DEFAULT_VALENCE.get(undefined)` is undefined, so every atom
   looked univalent and nothing was aromatic. It read 49/74, and 49 is just
   the number of molecules with no aromatic ring to get wrong.
3. 🔴 **Two rings are neighbours only if they share EXACTLY ONE bond.**
   `makeRingNeighborMap(brings, neighMap, 24, 1)` passes `maxOverlapSize = 1`
   and tests `inter.size() > 0 && inter.size() <= maxOverlapSize`. Each of
   porphine's pyrroles shares TWO bonds with the 16-ring, so every ring there
   is its own fused system and only single rings are ever tried. With "shares
   at least one bond" the five rings became one system, a two-ring combination
   passed, and porphine came out aromatic where RDKit leaves four carbons out.
   This is the rule the whole molecule turns on, and the gate is watched
   failing on it.

## The original plan and the open question it had to answer

`test:canonical-names` is 73/74. The last molecule is **porphine**, and the
cause is exact: this port TRUSTS the SMILES' lower case, where RDKit
RE-PERCEIVES aromaticity and demotes two of the four meso bridges — atoms 0, 1,
10, 11 become non-aromatic and their six bonds ordinary (2,1,1,2,1,1).

🔴 **THIS IS NOT THE ONLY THING THE MISSING MODEL COSTS, AND THE CORPUS HIDES
IT.** Every aromatic molecule in the 74 is written in lower case, so trusting
the input happens to agree. A user who writes Kekulé benzene — `C1=CC=CC=C1` —
gets a molecule this port marks NON-aromatic where RDKit marks it aromatic, and
nothing in the corpus can see that. The model is the fix for both directions.

## Where it runs

RDKit's pipeline is parse → sanitise → **kekulise** → `setAromaticity`, so
perception reads INTEGER bond orders, not the input's aromatic flags.
`shared/chem/kekulize.js` already does the middle step: it mutates `bond.order`
from 1.5 to 1 or 2 in place (benzene+OH becomes 2,1,2,1,2,1,1). So the port
slots in after it, and must set BOTH `atom.aromatic` and `bond.aromatic` from
what it computes rather than from the parse.

The default model is `aromaticityHelper(mol, srings, 0, 0, true)`
(Aromaticity.cpp:1122): **no minimum or maximum ring size**, fused systems
included. `maxFusedAromaticRingSize` is 24.

## The algorithm, from Code/GraphMol/Aromaticity.cpp

### 1. `countAtomElec(atom)` → electrons available to the π system

    dv = defaultValence(Z);           if (dv <= 1) return -1   // univalent: never aromatic
    degree = heavyNeighbours + totalNumHs
    if (degree > 3) return -1                                   // >3 coordinate: never aromatic
    nlp = max(outerElecs(Z) - dv - charge, 0)                    // lone pairs
    res = (dv - degree) + nlp - numRadicalElectrons
    if (res > 1 && (explicitValence - heavyDegree) > 1) res = 1  // triple-or-higher: 1 electron

### 2. `getAtomDonorTypeArom(atom)` → the donor type

Needs three predicates, all over the atom's bonds:

  * `incidentNonCyclicMultipleBond` — a bond NOT in any ring whose valence
    contribution is >= 2; returns the other atom (`who`) as well
  * `incidentCyclicMultipleBond`    — the same for a bond that IS in a ring
  * `incidentMultipleBond`          — `explicitValence != degree`, where degree
    counts neighbours plus explicit Hs minus bonds contributing nothing

Then, with `nelec = countAtomElec(atom)`:

  * `nelec < 0`  → None
  * `nelec == 0` → non-cyclic multiple bond: **Vacant**; else cyclic multiple
    bond: **One**; else **None**
  * `nelec == 1` → non-cyclic multiple bond: **Vacant** if the partner is MORE
    ELECTRONEGATIVE than this atom (this is the C=O case — the exocyclic bond
    steals the electron), else **One**. With no such bond: **One** if any
    multiple bond, else **Vacant** if charge is +1 (tropylium, cyclopropenyl).
  * `nelec > 1`  → decrement `nelec` once if a non-cyclic multiple bond goes to
    a more electronegative partner; then **One** if odd, **Two** if even.

### 3. `isAtomCandForArom(atom, edon)`

  * Z > 18 rejected unless Se (34) or Te (52)
  * `edon` must be Vacant / One / Two / OneOrTwo / Any
  * 🔴 rejected if `totalValence > defaultValence(Z - charge)` — note the
    periodic table is indexed by **Z minus the charge**, so N+ (Z 7, +1) is
    allowed a valence of 4 because `defaultValence(6)` is 4. A deliberate
    quirk; reproduce it rather than "fixing" it.
  * rejected on radical electrons unless neutral carbon
  * rejected if more than one double/triple bond (`C1=C=NC=N1`)

### 4. `applyHuckel(ringAtoms, edon)`

Sum per-atom electron BOUNDS, not a single count:

    Any | OneOrTwo -> [1,2]   One -> [1,1]   Two -> [2,2]   None | Vacant -> [0,0]

More than one `Any` donor in the ring returns false. Then:

    aromatic = rup >= 6 ? any rie in [rlw, rup] with (rie - 2) % 4 == 0
             : rup == 2

### 5. The fused search, `applyHuckelToFused`

Per fused system (rings sharing at least one bond), try combinations of rings
of increasing size, 1 up to `min(nrings, maxFused)`; skip a combination whose
rings are not connected (`checkFused`). For a combination:

  * count how many of its rings each atom appears in
  * 🔴 the Hückel set is the atoms appearing in **exactly 1 or 2** of them —
    this is RDKit issue #2895, acepentalene's central atom
  * if `applyHuckel` passes, mark every atom of those rings aromatic, and mark
    a bond aromatic only where it appears **once** across the combination's
    rings, so interior fusing bonds stay non-aromatic

## ✅ THE QUESTION IS ANSWERED — measured, not reasoned (2026-11-10)

`refs/canon-trace/arom.cpp` drives RDKit's own perception and
`Aromaticity.cpp`'s `aromaticityHelper` prints its rings, its donor types and
every combination it tries, under `AROM_TRACE=1`. For porphine:

```
SRINGS 5 (all), 5 candidate
  ring 0 size  5: 0 1 2 20 19
  ring 1 size 16: 3 4 23 7 8 9 22 12 13 14 21 17 18 19 20 2
  ring 2 size  5: 5 4 23 7 6
  ring 3 size  5: 10 9 22 12 11
  ring 4 size  5: 15 14 21 17 16
TRY rings 0 (sizes  5) -> no
TRY rings 1 (sizes 16) -> AROMATIC
TRY rings 2 (sizes  5) -> AROMATIC
TRY rings 3 (sizes  5) -> no
TRY rings 4 (sizes  5) -> AROMATIC
```

All 20 carbons are `type=1 elecs=[1,1]`; the nitrogens split **2, 1, 2, 1** —
N21 and N23 are the `[nH]` pair at `[2,2]`, N20 and N22 are `[1,1]`.

| ring | electrons | Huckel |
|---|---|---|
| 0 `0 1 2 19 20` | 4xC + N20(1) = **5** | `rup < 6` and `!= 2` → no |
| 1, the 16-ring | 12xC + 1+2+1+2 = **18** | (18-2)%4 == 0 → **aromatic** |
| 2 `4 5 6 7 23` | 4xC + N23(2) = **6** | aromatic |
| 3 `9 10 11 12 22` | 4xC + N22(1) = **5** | no |
| 4 `14 15 16 17 21` | 4xC + N21(2) = **6** | aromatic |

🔴 **THE ARITHMETIC IN THE SECTION ABOVE WAS RIGHT AND THE RING SET WAS WRONG.**
The macrocycle really does carry 18 electrons and really is aromatic. What was
wrong was calling atoms 0, 1, 10, 11 "meso bridges in the macrocycle": the SSSR
puts them in two FIVE-membered rings that fail on 5 electrons, and the
16-membered ring does not contain them at all. They are non-aromatic because
every ring they belong to failed - nothing to do with fusing, and no
combination larger than one ring is ever tried here (`doneBonds` covers the
system after the first pass).

**This is why the port was not written on the reasoning alone.** The conclusion
"the macrocycle is not aromatic" was false, and a model built to reproduce it
would have been wrong for porphine and wrong for every other molecule too.

### And our own SSSR already agrees, which was the real risk

Perception is only as good as the rings it is handed, and
`shared/chem/rings.js`'s `smallestRings` returns RDKit's exact five for
porphine - the four 5-rings and the 16-ring, same members. So the hard
dependency holds on the hardest case in the corpus.

## What is left to write

1. `countAtomElec` and `getAtomDonorTypeArom` per the sections above, needing a
   small periodic table: default valence and outer electrons for H, B, C, N, O,
   F, Si, P, S, Cl, Se, Br, I, plus an electronegativity comparison for the
   "exocyclic bond steals the electron" rule.
2. `isAtomCandForArom`, including the `defaultValence(Z - charge)` quirk.
3. `applyHuckel` on electron BOUNDS.
4. 🔴 **The fused search, which this corpus CANNOT justify skipping.** Every
   aromatic ring in the 74 passes on its own, so a size-1-only implementation
   would score 74/74 and still be wrong: azulene's fused 5+7 system has two
   rings that each fail and a union that passes. Add azulene (`c1ccc2cccc2c1`)
   and a Kekule-written benzene to the corpus BEFORE trusting the model, or the
   gate cannot see either gap.
5. Call it from the parser after `kekulize`, replacing the trust-the-input
   behaviour, and set both `atom.aromatic` and `bond.aromatic` from it.

The gate is `tools/check-canonical-names.mjs`, which already compares every
per-atom field and every bond type against RDKit over all 74: a correct model
keeps 73 and takes porphine to 74.
