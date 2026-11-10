"""RDKit's molecular weight and heavy-atom count, as the reference G4 is held to.

    python3 tools/oracle/dump_rdkit_molwt.py
    node tools/check-molecular-weight.mjs

🔴 RDKit IS AN ORACLE, NOT A DEPENDENCY, and this file is the one place that
says so for the weight: nothing under shared/ imports it or anything derived
from it, and it does not ship to a browser.

🔴 AND THE CORPUS IS IMPORTED, NOT RETYPED. dump_rdkit_smiles.py already holds
the 76 molecules the SMILES gates run on - chosen by reading the parser for
branches nothing exercised - and a second copy here would be a second thing to
keep in step. CLAUDE.md's own lesson: one list for both loops.

WHAT IS COMPARED, AND WHY IT IS `MolWt` AND NOT `ExactMolWt`.
boltz/data/parse/schema.py computes the affinity feature as
`AllChem.Descriptors.MolWt(mol_no_h)` where `mol_no_h = AllChem.RemoveHs(mol)`,
so that - average weight, hydrogens included, isotope labels honoured - is the
quantity. `ExactMolWt` is the monoisotopic mass and is a different number;
both are dumped so a port that reached for the wrong one is visible rather
than merely wrong.

`molWtNoH` is dumped BESIDE `molWt` to keep honest the claim that `RemoveHs`
does not change the weight. If those two columns ever diverge, the port's
"count every hydrogen" rule is wrong and this gate will say so.
"""
import json
import pathlib
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dump_rdkit_smiles import CORPUS            # the one corpus  # noqa: E402

OUT = pathlib.Path("oracle-dumps/rdkit-molwt.json")


def one(name: str, smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"name": name, "smiles": smiles, "error": "RDKit refused it"}
    no_h = AllChem.RemoveHs(mol, sanitize=False)
    return {
        "name": name,
        "smiles": smiles,
        # the quantity upstream feeds to the affinity correction
        "molWt": Descriptors.MolWt(no_h),
        # ...and the same on the molecule before RemoveHs, to keep that claim honest
        "molWtWithH": Descriptors.MolWt(mol),
        # a different number, dumped so reaching for it is visible
        "exactMolWt": Descriptors.ExactMolWt(mol),
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "heavyAtoms": mol.GetNumHeavyAtoms(),
        "fragments": len(Chem.GetMolFrags(mol)),
        # does any atom carry an isotope label at all?
        "isotopes": sorted({a.GetIsotope() for a in mol.GetAtoms()
                            if a.GetIsotope()}),
        # ...and specifically on a hydrogen, which parseSmiles cannot represent
        "hydrogenIsotope": any(
            a.GetAtomicNum() == 1 and a.GetIsotope()
            for a in Chem.AddHs(mol).GetAtoms()),
    }


def main() -> None:
    rows = [one(name, smiles) for name, smiles in CORPUS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": f"rdkit {Chem.rdBase.rdkitVersion}",
        "quantity": "Descriptors.MolWt(RemoveHs(mol)) - boltz/data/parse/schema.py",
        "molecules": rows,
    }, indent=1) + "\n")
    bad = [r for r in rows if "error" in r]
    print(f"wrote {OUT}: {len(rows)} molecules, {len(bad)} RDKit refused")
    print(f"rdkit {Chem.rdBase.rdkitVersion}")


if __name__ == "__main__":
    main()
