"""Upstream's AffinityCropper on deterministic pockets, as the reference for the port.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      python3 tools/oracle/dump_affinity_crop.py --boltz=<checkout> \
        --out=oracle-dumps/affinity-crop.json

🔴 THE REAL CROPPER RUNS. Nothing here re-implements the selection: it builds
the structured arrays upstream's `Tokenized` carries, calls
`AffinityCropper().crop(...)` with inferencev2.py's own arguments
(`max_tokens=256, max_atoms=2048`) and records which token indices came back.

🔴 AND THE CASES ARE CHOSEN TO REACH THE BRANCHES, not to be realistic. The
cropper has four decisions a careless draw never exercises:

  * the WHOLE-CHAIN branch, when a chain is no longer than the neighbourhood
  * the GROW loop, when the +/-10 residue window holds fewer than 10 tokens
  * the three BREAK conditions - token budget, atom budget, protein budget -
    which are a `break` and not a `continue`, so each one truncates the crop at
    a different point
  * TIES in the distance ranking, which every ligand token is at the front of
    (distance 0 from itself) and which numpy's stable argsort orders by index

So the corpus runs a small protein where everything fits, a long one that hits
the 200-token protein budget, a short-chain case for the whole-chain branch, one
with unresolved tokens (the ranking skips them but the whole-chain test counts
them), a multi-token-per-residue case for the grow loop, and a low atom budget.
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

NONPOLYMER = 3


def build(case: dict):
    """One synthetic Tokenized's tokens array, in upstream's dtype fields."""
    rng = np.random.default_rng(case["seed"])
    rows = []
    token = 0
    # the protein chain: one token a residue, laid on a line with a jitter so
    # the distance ranking is a real ordering rather than a tie.
    #
    # 🔴 `per_residue` AND `res_stride` EXIST BECAUSE THE FIRST CORPUS COULD NOT
    # SEE TWO RULES. With one token a residue and consecutive numbering, the
    # +/-neighbourhood window always holds far more tokens than the
    # neighbourhood, so the grow loop converges to the same set whatever
    # `neighborhood_size` is - the separation control reported "neighbourhood 9
    # not 10: 0 of 8 cases change". `per_residue > 1` makes a residue several
    # tokens, which is how a modified residue or an atomised ligand is
    # tokenised and which the grow loop's own comment is about ("we switch to
    # the res_idx instead of the token_idx to always include all tokens from
    # modified residues"); `res_stride > 1` leaves gaps so the window holds
    # FEWER tokens than it looks like it should.
    per_residue = case.get("per_residue", 1)
    stride = case.get("res_stride", 1)
    for residue in range(case["protein"]):
        for sub in range(per_residue):
            rows.append({
                "token_idx": token, "atom_num": case.get("protein_atoms", 5),
                "res_idx": residue * stride, "asym_id": 0, "mol_type": 0,
                "center": np.array([residue * 1.6, 0.0, sub * 0.9])
                + rng.normal(0, 0.25, 3),
                "resolved": residue not in set(case.get("unresolved", [])),
                "affinity": False,
            })
            token += 1
    # ...and a second protein chain, so `asym_id` actually selects
    for residue in range(case.get("protein2", 0)):
        rows.append({
            "token_idx": token, "atom_num": case.get("protein_atoms", 5),
            "res_idx": residue, "asym_id": 2, "mol_type": 0,
            "center": np.array([residue * 1.6, 9.0, 0.0]) + rng.normal(0, 0.25, 3),
            "resolved": True, "affinity": False,
        })
        token += 1
    # the ligand: one token an atom, clustered near the middle of chain 0
    middle = case["protein"] / 2 * 1.6
    for atom in range(case["ligand"]):
        rows.append({
            "token_idx": token, "atom_num": 1,
            # 🔴 ONE RESIDUE FOR THE WHOLE LIGAND, as a non-polymer chain is
            # tokenised: every heavy atom is a token of res_idx 0.
            "res_idx": 0, "asym_id": 1, "mol_type": NONPOLYMER,
            "center": np.array([middle, 2.5, 0.0]) + rng.normal(0, 1.1, 3),
            "resolved": True, "affinity": True,
        })
        token += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz", required=True)
    parser.add_argument("--out", default="oracle-dumps/affinity-crop.json")
    arguments = parser.parse_args()
    sys.path.insert(0, str(pathlib.Path(arguments.boltz).resolve() / "src"))
    from boltz.data.crop.affinity import AffinityCropper            # noqa: E402
    from boltz.data.types import Tokenized                          # noqa: E402

    CASES = [
        {"name": "small-fits", "protein": 40, "ligand": 6, "seed": 1},
        {"name": "protein-budget", "protein": 400, "ligand": 8, "seed": 2},
        {"name": "short-chain", "protein": 7, "ligand": 4, "seed": 3},
        {"name": "two-chains", "protein": 60, "protein2": 40, "ligand": 6, "seed": 4},
        {"name": "unresolved", "protein": 50, "ligand": 5, "seed": 5,
         "unresolved": [10, 11, 12, 13, 14, 25, 26]},
        {"name": "atom-budget", "protein": 300, "ligand": 6, "seed": 6,
         "protein_atoms": 14},
        {"name": "token-budget", "protein": 150, "ligand": 120, "seed": 7},
        {"name": "big-ligand", "protein": 80, "ligand": 60, "seed": 8},
        # 🔴 ADDED AFTER THE FIRST EIGHT ALL PASSED AND THE SEPARATION CONTROL
        # REPORTED TWO RULES IT COULD NOT SEE. docs/ records the same lesson on
        # the SMILES corpus: "a corpus that agrees everywhere proves only that
        # the cases somebody thought of are right."
        # several tokens a residue - the grow loop counts TOKENS and steps by
        # RESIDUE, so this is where neighborhood_size actually decides
        {"name": "multi-token-residue", "protein": 120, "ligand": 6, "seed": 9,
         "per_residue": 4},
        # 🔴 A `res_stride` CASE IS DELIBERATELY ABSENT, AND NOT BECAUSE IT WAS
        # AWKWARD. One was written and it HANGS upstream's cropper: with
        # residues numbered every third index the +/-10 residue window holds 7
        # tokens, `while new_tokens.size < 10` has no other exit, and
        # `max_token_set` caps what growing can reach - so the condition is
        # never satisfied. Isolated separately and confirmed.
        #
        # It is unreachable from a real input, because `res_idx` is the
        # tokeniser's own per-chain index and therefore contiguous, which makes
        # the window at least 21 tokens wherever that branch runs. So the case
        # does not belong in an oracle corpus - an oracle that cannot answer is
        # not a reference - but the PORT carries a terminating guard anyway
        # (see shared/af3/affinity/crop.js), because hanging where upstream
        # hangs is not bug-compatibility on the path of every ligand in a screen.
        # a residue heavy enough that the atom budget binds within one window,
        # where the earlier atom-budget case overshot by 12 and an off-by-one
        # on the budget changed nothing
        {"name": "atom-budget-tight", "protein": 200, "ligand": 4, "seed": 11,
         "protein_atoms": 24},
        {"name": "atom-budget-tight-2", "protein": 200, "ligand": 4, "seed": 12,
         "protein_atoms": 37},
        # both at once: atomised residues AND a tight atom budget
        {"name": "multi-token-tight", "protein": 100, "ligand": 8, "seed": 13,
         "per_residue": 3, "protein_atoms": 9},
    ]

    dtype = np.dtype([
        ("token_idx", "i4"), ("atom_idx", "i4"), ("atom_num", "i4"),
        ("res_idx", "i4"), ("res_type", "i4"), ("sym_id", "i4"),
        ("asym_id", "i4"), ("entity_id", "i4"), ("mol_type", "i4"),
        ("center_idx", "i4"), ("disto_idx", "i4"),
        ("center_coords", "3f4"), ("disto_coords", "3f4"),
        ("resolved_mask", "?"), ("disto_mask", "?"), ("modified", "?"),
        ("cyclic_period", "i4"), ("affinity_mask", "?"),
    ])
    bond_dtype = np.dtype([("token_1", "i4"), ("token_2", "i4"), ("type", "i4")])

    out = {"source": "upstream boltz AffinityCropper",
           "maxTokens": 256, "maxAtoms": 2048,
           "neighborhoodSize": 10, "maxTokensProtein": 200, "cases": []}

    for case in CASES:
        rows = build(case)
        tokens = np.zeros(len(rows), dtype=dtype)
        for i, row in enumerate(rows):
            tokens[i]["token_idx"] = row["token_idx"]
            tokens[i]["atom_num"] = row["atom_num"]
            tokens[i]["res_idx"] = row["res_idx"]
            tokens[i]["asym_id"] = row["asym_id"]
            tokens[i]["mol_type"] = row["mol_type"]
            tokens[i]["center_coords"] = row["center"]
            tokens[i]["disto_coords"] = row["center"]
            tokens[i]["resolved_mask"] = row["resolved"]
            tokens[i]["disto_mask"] = row["resolved"]
            tokens[i]["affinity_mask"] = row["affinity"]
        bonds = np.zeros(0, dtype=bond_dtype)
        data = Tokenized(tokens=tokens, bonds=bonds, structure=None, msa={})
        cropped = AffinityCropper().crop(data, max_tokens=256, max_atoms=2048)
        kept = sorted(int(v) for v in cropped.tokens["token_idx"])
        protein = [i for i in kept if tokens[i]["mol_type"] != NONPOLYMER]
        out["cases"].append({
            "name": case["name"],
            "tokens": [{
                "tokenIdx": int(t["token_idx"]), "atomNum": int(t["atom_num"]),
                "resIdx": int(t["res_idx"]), "asymId": int(t["asym_id"]),
                "molType": int(t["mol_type"]),
                "centre": [float(v) for v in t["center_coords"]],
                "resolved": bool(t["resolved_mask"]),
                "affinity": bool(t["affinity_mask"]),
            } for t in tokens],
            "kept": kept,
            "keptCount": len(kept),
            "proteinKept": len(protein),
            "atoms": int(sum(int(tokens[i]["atom_num"]) for i in kept)),
        })
        print(f"{case['name']:<16} {len(rows):>4} tokens -> kept {len(kept):>3}"
              f"  protein {len(protein):>3}"
              f"  atoms {sum(int(tokens[i]['atom_num']) for i in kept):>5}",
              file=sys.stderr)

    path = pathlib.Path(arguments.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(f"\nwrote {path}: {len(out['cases'])} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
