"""This port's affinity numbers against upstream's, after today's fixes.

Upstream's arm is READ FROM DISK rather than re-run: nothing on its side
changed, so its five-seed control (refs/g8-out-5) and its per-ligand values
(refs/g8-rank) are still its own numbers. Only this port's arm was re-run.

🔴 THE RANKING IS THE PRODUCT AND THE GAP IS NOT. A screening tool's whole
output is an ORDER, so Spearman and the count of INVERTED PAIRS are reported
beside the per-ligand values - a tool can sit a constant offset away from
upstream and still rank perfectly, and it can match on the mean and invert the
two ligands a reader cares about.
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path("/Users/pabloarantes/Downloads/Boltz2_webGPU")
LIGANDS = ['CCCc1ccccc1O', 'CCO', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
           'CC(=O)Oc1ccccc1C(=O)O', 'CN1CCC[C@H]1c1cccnc1', 'c1ccc2c(c1)cccc2O']
NAMES = ['propylphenol', 'ethanol', 'ibuprofen', 'aspirin', 'nicotine', 'naphthol']


def spearman(a, b):
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def discordant(a, b):
    bad = []
    for i in range(len(a)):
        for k in range(i + 1, len(a)):
            if (a[i] - a[k]) * (b[i] - b[k]) < 0:
                bad.append((i, k))
    return bad


def read(path, key):
    j = json.loads(pathlib.Path(path).read_text())
    return [r[key] for r in j[key.split("@")[0] and "ours"]] if False else j


def main():
    print("=" * 72)
    print("MAIN COMPARISON - one ligand, five seeds each side")
    print("=" * 72)
    up = json.loads((ROOT / "refs/g8-out-5/g8-result.json").read_text())["upstream"]
    for label, path in (("before today", "refs/g8-out-fixed"),
                        ("after today", "refs/g8-today")):
        f = ROOT / path / "g8-result.json"
        if not f.exists():
            print(f"{label:14s} (missing {f})")
            continue
        ours = json.loads(f.read_text())["ours"]
        for field in ("affinity_pred_value", "affinity_probability_binary"):
            u = [r[field] for r in up]
            o = [r[field] for r in ours]
            unit = (max(u) - min(u)) / 2
            gap = abs(statistics.fmean(u) - statistics.fmean(o))
            print(f"{label:14s} {field:28s} upstream {statistics.fmean(u):.4f}"
                  f" [{min(u):.4f}..{max(u):.4f}]  ours {statistics.fmean(o):.4f}"
                  f" [{min(o):.4f}..{max(o):.4f}]  gap {gap:.4f}"
                  f" = {gap / unit:.2f} x spread")
        print()

    print("=" * 72)
    print("SIX-LIGAND RANKING - upstream one seed each (its own stored values)")
    print("=" * 72)
    old_dirs = ["CCCc1ccccc1O_", "CCO_", "CC_C_Cc1ccc_cc1_C_C_C__O_O_",
                "CC__O_Oc1ccccc1C__O_O_", "CN1CCC_C_H_1c1cccnc1_",
                "c1ccc2c_c1_cccc2O_"]
    rows = []
    for index, (smiles, name) in enumerate(zip(LIGANDS, NAMES)):
        old = json.loads((ROOT / "refs/g8-rank" / old_dirs[index]
                          / "g8-result.json").read_text())
        new_file = ROOT / f"refs/g8-rank-today/lig{index + 1}/g8-result.json"
        new = json.loads(new_file.read_text()) if new_file.exists() else None
        rows.append({
            "name": name, "smiles": smiles,
            "up_v": old["upstream"][0]["affinity_pred_value"],
            "up_b": old["upstream"][0]["affinity_probability_binary"],
            "old_v": old["ours"][0]["affinity_pred_value"],
            "old_b": old["ours"][0]["affinity_probability_binary"],
            "new_v": new["ours"][0]["affinity_pred_value"] if new else None,
            "new_b": new["ours"][0]["affinity_probability_binary"] if new else None,
        })
    print(f"{'ligand':14s} {'upstream':>18s} {'ours BEFORE':>18s} {'ours AFTER':>18s}")
    print(f"{'':14s} {'value   binary':>18s} {'value   binary':>18s} {'value   binary':>18s}")
    for r in rows:
        na = "   n/a" if r["new_v"] is None else f"{r['new_v']:7.4f}"
        nb = "   n/a" if r["new_b"] is None else f"{r['new_b']:7.4f}"
        print(f"{r['name']:14s} {r['up_v']:9.4f} {r['up_b']:8.4f}"
              f" {r['old_v']:9.4f} {r['old_b']:8.4f} {na} {nb}")
    print()
    for field, label in (("v", "affinity_pred_value (lower = tighter)"),
                         ("b", "affinity_probability_binary (higher = binder)")):
        u = [r[f"up_{field}"] for r in rows]
        print(label)
        for which in ("old", "new"):
            o = [r[f"{which}_{field}"] for r in rows]
            if any(x is None for x in o):
                print(f"   {which:5s} n/a")
                continue
            bad = discordant(u, o)
            pairs = ", ".join(f"{rows[i]['name']}/{rows[k]['name']}" for i, k in bad)
            print(f"   {which:5s} rho {spearman(u, o):+.3f}"
                  f"   {len(bad)} of 15 pairs inverted"
                  + (f"  [{pairs}]" if bad else ""))
        print()


if __name__ == "__main__":
    sys.exit(main())
