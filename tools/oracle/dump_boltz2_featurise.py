"""Upstream Boltz-2's token and atom featurisers, as the reference for the crop's batch.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      python3 tools/oracle/dump_boltz2_featurise.py --boltz=<checkout> \
        --out=oracle-dumps/boltz2-featurise.json

🔴 THIS IS THE ORACLE `featuriseCrop` NEEDS AND NOTHING ELSE HERE COULD BE.
Pass 2 featurises a CROPPED token set, and upstream does that by cropping the
token table rather than by rebuilding a sequence - so gaps keep their `res_idx`
and no bond is implied across them. Porting it means re-deriving the atom
windows, the token-atom gathers and the atom subsets on a token subset, which
is the class of work docs/ records getting wrong in FOUR OF SEVEN MODELS
(`check-atom-windows.js`: "Three whole conventions lived in that gap - SLIDE,
CLAMP-and-mask, and IntelliFold-2's slide against a query-block-PADDED edge").
Building that without a reference would be the first unverified link in this
chain.

🔴 AND IT NEEDS NO CCD, NO PARSER AND NO 1.86 GB mols.tar. The featuriser reads
only six fields of `data.structure` - atoms, chains, residues, bonds, coords and
ensemble - so a synthetic `Tokenized` reaches it the same way the cropper's
oracle does. `molecules` is a dict of RDKit mols keyed by residue name, used for
`atom_name_to_ref` and one conformer, so synthetic mols with NAMED atoms satisfy
it.

🔴 WHAT IS COMPARED AND WHAT IS NOT, BECAUSE THE CONFORMERS ARE SYNTHETIC. The
mols below carry arbitrary conformer coordinates, so `ref_pos` from this dump is
MEANINGLESS and is not dumped as a comparable - docs/ already treats `ref_pos`
as "a reported floor, never a failure" even with real conformers, for the
separate reason that this port ships one idealised set. What IS dumped and
compared is everything that depends on atom COUNTS and ORDER rather than
geometry: the token-atom gathers, the query and key windows, the atom-to-token
map, the subset count, `ref_space_uid` as a PARTITION, and the token-level
index fields. Those are exactly what a crop re-derives and exactly what the
atom-window bug was in.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                             # noqa: E402

PROTEIN, NONPOLYMER = 0, 3
# the backbone-plus-CB atom sets the synthetic residues carry, in CCD order
RESIDUE_ATOMS = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "GLY": ["N", "CA", "C", "O"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
            "CE3", "CZ2", "CZ3", "CH2"],
}
CENTRE = {"ALA": 1, "GLY": 1, "SER": 1, "LYS": 1, "TRP": 1}     # CA
DISTO = {"ALA": 4, "GLY": 1, "SER": 4, "LYS": 4, "TRP": 4}      # CB, GLY -> CA
RES_TYPE = {"ALA": 1, "GLY": 8, "SER": 16, "LYS": 12, "TRP": 18}


def synthetic_mol(name: str, atom_names: list[str]):
    """An RDKit mol with NAMED atoms and one conformer.

    🔴 THE CONFORMER IS ARBITRARY AND THAT IS DECLARED. The featuriser reads it
    only for `ref_pos`, which this dump does not offer as a comparable; what it
    is needed for is `atom_name_to_ref` and a conformer EXISTING.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol()
    for atom_name in atom_names:
        atom = Chem.Atom(atom_name[0] if atom_name[0] in "CNOS" else "C")
        atom.SetProp("name", atom_name)
        mol.AddAtom(atom)
    for index in range(len(atom_names) - 1):
        mol.AddBond(index, index + 1, Chem.BondType.SINGLE)
    built = mol.GetMol()
    Chem.SanitizeMol(built, Chem.SanitizeFlags.SANITIZE_NONE)
    conformer = Chem.Conformer(built.GetNumAtoms())
    for index in range(built.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(index * 1.5, 0.0, 0.0))
    built.AddConformer(conformer, assignId=True)
    for atom in built.GetAtoms():
        atom.SetProp("name", atom_names[atom.GetIdx()])
    return built


def build(case: dict, dtypes: dict):
    """A synthetic Structure + token table for one case."""
    atoms, residues, chains, token_rows, bonds = [], [], [], [], []
    atom_index = 0
    residue_index = 0
    token_index = 0

    sequence = case["sequence"]
    # --- the protein chain ---------------------------------------------------
    chain_atom_start = atom_index
    chain_res_start = residue_index
    for position, code in enumerate(sequence):
        names = RESIDUE_ATOMS[code]
        for offset, atom_name in enumerate(names):
            atoms.append((atom_name,
                          (position * 3.8 + offset * 0.3, offset * 0.4, 0.0),
                          True, 0.0, 0.0))
        residues.append((code, RES_TYPE[code], position, atom_index, len(names),
                         CENTRE[code], DISTO[code], True, True))
        token_rows.append({
            "token_idx": token_index, "atom_idx": atom_index,
            "atom_num": len(names), "res_idx": position,
            "res_type": RES_TYPE[code], "res_name": code,
            "sym_id": 0, "asym_id": 0, "entity_id": 0, "mol_type": PROTEIN,
            "center_idx": atom_index + CENTRE[code],
            "disto_idx": atom_index + DISTO[code],
            "resolved_mask": True, "disto_mask": True, "affinity_mask": False,
        })
        atom_index += len(names)
        residue_index += 1
        token_index += 1
    chains.append(("A", PROTEIN, 0, 0, 0, chain_atom_start, atom_index - chain_atom_start,
                   chain_res_start, residue_index - chain_res_start, 0))

    # --- the ligand: one token an atom, one residue -------------------------
    ligand_names = [f"C{i + 1}" for i in range(case["ligand"])]
    ligand_atom_start = atom_index
    for offset, atom_name in enumerate(ligand_names):
        atoms.append((atom_name,
                      (len(sequence) * 1.9 + offset * 0.4, 2.5, offset * 0.2),
                      True, 0.0, 0.0))
    residues.append(("LIG", 20, 0, ligand_atom_start, len(ligand_names),
                     0, 0, False, True))
    for offset, atom_name in enumerate(ligand_names):
        token_rows.append({
            "token_idx": token_index, "atom_idx": ligand_atom_start + offset,
            "atom_num": 1, "res_idx": 0, "res_type": 20, "res_name": "LIG",
            "sym_id": 0, "asym_id": 1, "entity_id": 1, "mol_type": NONPOLYMER,
            "center_idx": ligand_atom_start + offset,
            "disto_idx": ligand_atom_start + offset,
            "resolved_mask": True, "disto_mask": True, "affinity_mask": True,
        })
        token_index += 1
    atom_index += len(ligand_names)
    chains.append(("B", NONPOLYMER, 1, 0, 1, ligand_atom_start,
                   len(ligand_names), residue_index, 1, 0))
    residue_index += 1

    structure_atoms = np.array(atoms, dtype=dtypes["atom"])
    coords = np.array([(a[1],) for a in atoms], dtype=dtypes["coords"])
    ensemble = np.array([(0, len(atoms))], dtype=dtypes["ensemble"])
    return {
        "atoms": structure_atoms,
        "residues": np.array(residues, dtype=dtypes["residue"]),
        "chains": np.array(chains, dtype=dtypes["chain"]),
        "bonds": np.zeros(0, dtype=dtypes["bond"]),
        "coords": coords, "ensemble": ensemble,
        "token_rows": token_rows,
        "ligand_names": ligand_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz", required=True)
    parser.add_argument("--out", default="oracle-dumps/boltz2-featurise.json")
    arguments = parser.parse_args()
    root = pathlib.Path(arguments.boltz).resolve()
    sys.path.insert(0, str(root / "src"))

    from boltz.data import const                                  # noqa: E402
    from boltz.data.types import (                                 # noqa: E402
        AtomV2, BondV2, Chain, Coords, Ensemble, Residue, StructureV2, Tokenized)
    from boltz.data.feature.featurizerv2 import (                  # noqa: E402
        process_atom_features, process_token_features)
    from boltz.data.tokenize.boltz2 import TokenV2                 # noqa: E402

    dtypes = {"atom": AtomV2, "bond": BondV2, "residue": Residue,
              "chain": Chain, "coords": Coords, "ensemble": Ensemble}

    CASES = [
        {"name": "contiguous", "sequence": "ALA GLY SER LYS TRP ALA SER GLY".split(),
         "ligand": 6},
        # 🔴 A GAP IN res_idx, WHICH IS WHAT A CROP LEAVES BEHIND and the whole
        # reason this oracle exists: the windows have to be derived over the
        # tokens that are PRESENT, not over a contiguous range.
        {"name": "cropped-gap", "sequence": "ALA GLY SER LYS TRP".split(),
         "ligand": 6, "drop": [2]},
        {"name": "cropped-two-gaps", "sequence": "ALA GLY SER LYS TRP ALA SER".split(),
         "ligand": 4, "drop": [1, 4, 5]},
        # a ligand big enough to cross an atom-window boundary (32 queries)
        {"name": "big-ligand", "sequence": "ALA GLY SER".split(), "ligand": 40},
        {"name": "window-edge", "sequence": "TRP TRP TRP".split(), "ligand": 2},
    ]

    out = {"source": f"upstream boltz featurizerv2, numpy {np.__version__}",
           "atomsPerWindowQueries": 32,
           "note": "synthetic conformers: ref_pos is NOT a comparable here;"
                   " see this dumper's header",
           "cases": []}

    for case in CASES:
        built = build(case, dtypes)
        rows = built["token_rows"]
        drop = set(case.get("drop", []))
        # the crop: drop whole protein residues, keep the ligand, RENUMBER nothing
        kept = [r for r in rows if not (r["mol_type"] == PROTEIN and r["res_idx"] in drop)]
        tokens = np.zeros(len(kept), dtype=TokenV2)
        for i, row in enumerate(kept):
            for field in tokens.dtype.names:
                if field in row:
                    tokens[i][field] = row[field]
        structure = StructureV2(
            atoms=built["atoms"], bonds=built["bonds"],
            residues=built["residues"], chains=built["chains"],
            interfaces=np.zeros(0, dtype=[("chain_1", "i4"), ("chain_2", "i4")]),
            mask=np.ones(len(built["chains"]), dtype=bool),
            ensemble=built["ensemble"], coords=built["coords"])
        data = Tokenized(tokens=tokens, bonds=np.zeros(0, dtype=dtypes["bond"]),
                         structure=structure, msa={})

        molecules = {code: synthetic_mol(code, names)
                     for code, names in RESIDUE_ATOMS.items()}
        molecules["LIG"] = synthetic_mol("LIG", built["ligand_names"])

        rng = np.random.default_rng(7)
        ensemble_features = {"ensemble_ref_idxs": np.array([0])}
        # 🔴 EMPTY LISTS, NOT THE DEFAULTS. `inference_pocket_constraints` and
        # `inference_contact_constraints` default to `False` and are ITERATED
        # ("for binder, contacts, max_distance, force in ..."), so calling this
        # function with its own defaults raises `'bool' object is not iterable`.
        # A caller has to pass sequences; the page passes none, which is [].
        token_features = process_token_features(
            data, rng, max_tokens=None,
            inference_pocket_constraints=[], inference_contact_constraints=[])
        atom_features = process_atom_features(
            data, rng, ensemble_features, molecules,
            atoms_per_window_queries=32, max_atoms=None, max_tokens=None)

        def listed(name, source):
            value = source[name]
            array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            return {"shape": list(array.shape),
                    "values": array.reshape(-1).tolist()}

        # 🔴 THE SEQUENCE IS RECORDED, NOT RECONSTRUCTED BY THE READER. The
        # first version of the gate rebuilt it from a hard-coded residue order
        # and silently compared the `window-edge` case (TRP TRP TRP) against
        # ALA GLY SER - which its own residue_index check caught as 2 of 5
        # differing. A reader that has to guess the input is a reader that can
        # be wrong about it.
        one_letter = {"ALA": "A", "GLY": "G", "SER": "S", "LYS": "K", "TRP": "W"}
        record = {"name": case["name"], "tokens": int(len(kept)),
                  "sequence": "".join(
                      one_letter[r["res_name"]] for r in kept
                      if r["mol_type"] == PROTEIN),
                  # the UNCROPPED sequence, so a reader can build the full
                  # batch and crop it the way pass 2 does
                  "fullSequence": "".join(
                      one_letter[r["res_name"]] for r in rows
                      if r["mol_type"] == PROTEIN),
                  "ligandAtoms": sum(1 for r in kept
                                     if r["mol_type"] == NONPOLYMER),
                  "droppedResidues": sorted(drop),
                  "keptTokenIdx": [int(r["token_idx"]) for r in kept],
                  "token": {}, "atom": {}}
        # token-level index fields, which a crop re-derives
        for field in ["residue_index", "token_index", "asym_id", "entity_id",
                      "sym_id", "mol_type", "token_pad_mask", "affinity_token_mask"]:
            if field in token_features:
                record["token"][field] = listed(field, token_features)
        # the gathers and windows - what the atom-window bug was in
        for field in ["atom_to_token", "token_to_rep_atom", "r_set_to_rep_atom",
                      "atom_pad_mask", "ref_space_uid", "atom_resolved_mask"]:
            if field in atom_features:
                record["atom"][field] = listed(field, atom_features)
        record["atomCount"] = int(atom_features["atom_pad_mask"].shape[0]) \
            if "atom_pad_mask" in atom_features else None
        out["cases"].append(record)
        print(f"{case['name']:<18} tokens {len(kept):>3}"
              f"  atoms {record['atomCount']}"
              f"  dropped {sorted(drop)}", file=sys.stderr)

    path = pathlib.Path(arguments.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(f"\nwrote {path}: {len(out['cases'])} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
