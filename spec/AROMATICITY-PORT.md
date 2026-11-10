# Porting RDKit's aromaticity model — the algorithm, and the one open question

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

## 🔴 THE OPEN QUESTION, AND IT MUST BE ANSWERED BEFORE A LINE IS WRITTEN

By the above, porphine's 16-membered macrocycle should be perceived AROMATIC,
and RDKit does not perceive it so. Worked through: no size limit applies, all
16 atoms pass `isAtomCandForArom`, and the electron sum is

    2 NH nitrogens x 2  +  2 pyridine-type N x 1  +  12 carbons x 1  =  18

with (18 - 2) % 4 == 0, which `applyHuckel` accepts. RDKit's own answer
(oracle-dumps/rdkit-canon-ranks.json) is that atoms 0, 1, 10, 11 are NOT
aromatic, i.e. only the four pyrroles are, and the macrocycle is not.

So an assumption about the model's INPUTS is wrong, and there are two
candidates:

  1. `MolOps::symmetrizeSSSR` does not hand the helper that 16-membered ring at
     all. Porphine has 5 SSSR rings (28 bonds - 24 atoms + 1); the four
     pyrroles are four of them, and the fifth may not be the ring assumed here.
  2. An alpha or meso carbon's donor type is not `One` as derived.

🔴 **ANSWER IT BY DUMPING, NOT BY REASONING** — which is what settled the
canonical ranking after two sessions of argument failed. The RDKit build in
`refs/rdkit-build` is already configured, and `refs/canon-trace/trace.cpp` is a
working driver that links against it. Print, for porphine:

  * `mol.getRingInfo()->atomRings()` after sanitisation — the actual `srings`
  * `edon[i]` and `acands[i]` for every atom, from inside `aromaticityHelper`
  * every `curRs` combination tried and whether `applyHuckel` accepted it

Three `std::cerr` lines in `aromaticityHelper` give all of it. Until those
numbers exist, any port rests on the contradiction above — and a model that
fixed porphine while quietly moving one of the other 73 would be worse than the
declared gap, because aromaticity feeds the bond-order plane, the conformer's
planarity rules and `test:smiles-batch`'s exact agreement with AF3 across eight
families.

## The gate to build against

`tools/check-canonical-names.mjs` already compares every per-atom field AND
every bond type against RDKit over all 74 molecules, so it is the oracle: a
correct model keeps 73 and takes porphine to 74. Extend it with Kekulé-written
inputs (`C1=CC=CC=C1` beside `c1ccccc1`) before trusting the model, because the
present corpus cannot see a perception that is right only for lower case.
