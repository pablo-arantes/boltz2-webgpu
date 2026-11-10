"""RDKit's canonical atom ranks, as the reference a JS reimplementation is held to.

    <python-with-rdkit> tools/oracle/dump_rdkit_canon_ranks.py \
      --out=oracle-dumps/rdkit-canon-ranks.json

🔴 WHY THIS IS NEEDED AT ALL. Boltz-2 names a SMILES ligand's atoms
`element + (CanonicalRankAtoms + 1)` over the molecule WITH hydrogens, and that
name is 256 of the atom encoder's 388 input columns. G8 measured this port's
own naming (AF3's per-element counter) against upstream's on a real job: not
one of ten names agreed, the atom encoder's ligand half read relRMS 1.1556, and
substituting upstream's names took it to 4.48e-2 - the int5 floor. So the ranks
ARE the remaining gap, and reproducing them needs an exhaustive reference.

🔴 THE CORPUS IS THE ONE THE SMILES GATES ALREADY USE, reused deliberately: 74
molecules each chosen for a convention it reaches, with the intent written
beside it. A new corpus for this would be a corpus nobody has read.

🔴 AND THE RANKS ARE DUMPED FOR THE H-ADDED MOLECULE, because that is what
upstream ranks. `AddHs` changes every atom's degree and neighbour multiset, so
ranks computed on the heavy-atom graph are a different quantity - and the heavy
atoms' ranks come out non-contiguous precisely because the hydrogens occupy
ranks among them.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module

CORPUS = import_module("dump_rdkit_smiles").CORPUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="oracle-dumps/rdkit-canon-ranks.json")
    args = ap.parse_args()

    from rdkit import Chem
    from rdkit.Chem import AllChem

    records = []
    for name, smiles in CORPUS:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            records.append({"name": name, "smiles": smiles, "error": "unparsed"})
            continue
        withH = AllChem.AddHs(mol)
        # exactly upstream's call sequence, in its order
        ranks = list(AllChem.CanonicalRankAtoms(withH))
        Chem.AssignStereochemistry(withH, force=True, cleanIt=True)
        heavyNumHs = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        atoms = []
        for atom, rank in zip(withH.GetAtoms(), ranks):
            atoms.append({
                "idx": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "z": atom.GetAtomicNum(),
                "degree": atom.GetDegree(),
                "explicitValence": atom.GetExplicitValence(),
                # on the H-added molecule this is 0 for every heavy atom;
                # the original molecule's count is kept beside it because that
                # is the one an invariant would use
                "totalNumHs": atom.GetTotalNumHs(),
                "originalNumHs": (heavyNumHs[atom.GetIdx()]
                                  if atom.GetIdx() < len(heavyNumHs) else 0),
                "charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "inRing": atom.IsInRing(),
                "aromatic": atom.GetIsAromatic(),
                "chiralTag": str(atom.GetChiralTag()),
                # 🔴 basecomp COMPARES THE ATOM MAP NUMBER BEFORE DEGREE
                # (df_useAtomMaps, default true). Skipped on the first read of
                # the source, and the corpus's `atom_map` molecule is the one
                # that fails without it.
                "atomMapNumber": atom.GetAtomMapNum(),
                # 🔴 THE CHIRALITY PASS NEEDS THE TAG AS CW/CCW, not merely
                # present, and it needs each atom's neighbours in RDKit's OWN
                # order: `updateAtomNeighborNumSwaps` builds `ref` from
                # `nbrIds` (molecule order) and `probe` from the SORTED bond
                # list, then counts the swaps between them. Both orders are
                # load-bearing; reconstructing either would be guessing.
                "chiralCode": (1 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                               else 2 if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                               else 0),
                "nbrIds": [n.GetIdx() for n in atom.GetNeighbors()],
                # the SpecialSymmetry pass's gate needs how many rings an atom
                # is in, not merely whether it is in one
                "numAtomRings": withH.GetRingInfo().NumAtomRings(atom.GetIdx()),
                # 🔴 basecomp'S LAST TERM READS THESE, AND IT READS NEIGHBOURS'
                # LIVE CLASSES. `getAtomRingNbrCode` sums
                # `neighbour.index * 10000 + 1` over an atom's ring-stereo
                # neighbours - so basecomp is NOT a static per-atom key, and
                # this port treated it as one. `isRingStereoAtom` is a
                # tetrahedral tag PLUS the `_ringStereoAtoms` property, which
                # glucose's ring stereocentres carry; `hasRingNbr` is true when
                # any neighbour is such an atom.
                "isRingStereoAtom": bool(
                    atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                            Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    and atom.HasProp("_ringStereoAtoms")),
                "hasRingNbr": any(
                    n.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                         Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    and n.HasProp("_ringStereoAtoms")
                    for n in atom.GetNeighbors()),
                "rank": int(rank),
                # the name upstream would give it
                "name": atom.GetSymbol().upper() + str(int(rank) + 1),
            })
        # 🔴 THE BOND LIST OF THE H-ADDED MOLECULE, so the JS side needs to
        # reconstruct NOTHING. The first version of this dump omitted it and
        # the harness rebuilt connectivity from `totalNumHs` - which is 0 on
        # an H-added molecule, because the hydrogens are explicit atoms by
        # then. It placed no hydrogens and 71 of 74 molecules "disagreed" on
        # ranks that had never been computed on the right graph.
        # 🔴 WITH THE BOND ORDER, because the refinement's neighbour signature
        # needs it. A field-order search over per-atom invariants alone
        # plateaued at 41/74 on class order, and every molecule it could not
        # place - acetate, sulfate, methanesulfonate, phenol, toluene,
        # naphthalene - is one whose equivalent-looking atoms are told apart by
        # the ORDER of the bond reaching them, not by anything local to the atom.
        bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                  float(b.GetBondTypeAsDouble()),
                  1 if b.GetIsAromatic() else 0]
                 for b in withH.GetBonds()]
        heavy = AllChem.RemoveHs(withH, sanitize=False)
        records.append({
            "name": name, "smiles": smiles,
            "atomsWithH": withH.GetNumAtoms(),
            "heavyAtoms": heavy.GetNumAtoms(),
            "ranks": [int(r) for r in ranks],
            "bonds": bonds,
            "atoms": atoms,
            # 🔴 ALSO THE RANKS WITHOUT HYDROGENS, so a reimplementation can be
            # debugged on the simpler graph before facing the real one.
            "heavyOnlyRanks": [int(r) for r in AllChem.CanonicalRankAtoms(mol)],
            # 🔴 RDKit'S OWN PRE-TIE-BREAK CLASSES, which is the intermediate
            # state this port had been trying to infer from final ranks.
            # `breakTies=False` returns the symmetry classes refinement
            # settles on, so comparing against THESE separates a refinement
            # fault from a tie-breaking fault outright instead of by argument.
            "classesNoBreak": [int(r) for r in
                               AllChem.CanonicalRankAtoms(withH, breakTies=False)],
        })

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rdkit": Chem.rdBase.rdkitVersion, "records": records},
              open(out, "w"), indent=1)
    ok = [r for r in records if "error" not in r]
    print(f"{len(ok)} of {len(records)} molecules, RDKit {Chem.rdBase.rdkitVersion}"
          f" -> {out}")
    print(f"atoms with H: {sum(r['atomsWithH'] for r in ok)}")


if __name__ == "__main__":
    main()
