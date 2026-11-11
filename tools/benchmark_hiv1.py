"""Enrichment of the nine approved HIV-1 protease inhibitors among 100 drugs.

    python3 -I tools/benchmark_hiv1.py --screen=<run-affinity-screen output.json>

🔴 THE FIRST CHECK HERE AGAINST EXPERIMENT RATHER THAN AGAINST UPSTREAM. Every
other number in this project is agreement with upstream PyTorch Boltz-2 - that
the port computes what Boltz-2 computes. This asks a different and harder
question: does the ranking put molecules that are KNOWN to inhibit this target
above molecules that do not. A port can be numerically perfect and still fail
it, because then the answer is about Boltz-2 rather than about this code.

🔴 AND THE SIZE BASELINE IS NOT OPTIONAL, IT IS THE POINT. The nine actives are
peptidomimetics of 40-55 heavy atoms and the decoys are ordinary drugs with a
median of 24, so HEAVY-ATOM COUNT ALONE is a ranking. If it scores as well as
the model, the model's number says nothing about recognition - it says the
actives are bigger. Reported beside every model metric, never omitted; the
ligand file's own header warns about exactly this.

🔴 BOTH HEADS, SEPARATELY, NEVER MERGED. `affinity_probability_binary` is a
classifier (higher = likelier binder) and `affinity_pred_value` is an IC50-like
strength on a log10 scale (LOWER = tighter). They come from heads trained on
different data with different supervision; averaging them would invent a score
neither reports.
"""
import argparse
import json
import math
import pathlib
import subprocess


def heavy_atoms(root, smiles):
    """Heavy-atom counts from this project's own SMILES parser."""
    script = (
        'const { parseSmiles } = await import("./shared/chem/smiles.js");'
        'const list = JSON.parse(process.argv[1]);'
        'console.log(JSON.stringify(list.map((s) => parseSmiles(s).atoms.length)));'
    )
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script, json.dumps(smiles)],
        cwd=str(root), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def roc_auc(scores, labels):
    """AUC by rank sum (Mann-Whitney U), ties averaged.

    🔴 TIES MUST BE AVERAGED RATHER THAN BROKEN. A head that returns the same
    value for two ligands has not ordered them, and breaking the tie by input
    order would credit or penalise the model for the order of a file.
    """
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def enrichment(order, labels, fraction):
    """Hit rate in the top `fraction`, over the hit rate in the whole set."""
    take = max(1, round(len(order) * fraction))
    found = sum(1 for index in order[:take] if labels[index])
    expected = sum(labels) * take / len(labels)
    return (found / expected if expected else float("nan")), found, take


def bootstrap_auc(scores, labels, draws=4000, seed=0):
    """A percentile interval for the AUC.

    🔴 NINE ACTIVES CANNOT SUPPORT A BARE THREE-DIGIT AUC. Resampling the
    ligands says how much of a gap between two scores is real: without it,
    0.762 against 0.958 reads as a settled ordering when the intervals may
    overlap. The actives and the decoys are resampled SEPARATELY so each draw
    keeps nine actives - resampling the pool together would sometimes give a
    draw with two actives and an undefined AUC.
    """
    import random
    rng = random.Random(seed)
    positives = [s for s, label in zip(scores, labels) if label]
    negatives = [s for s, label in zip(scores, labels) if not label]
    out = []
    for _ in range(draws):
        p = [rng.choice(positives) for _ in positives]
        n = [rng.choice(negatives) for _ in negatives]
        out.append(roc_auc(p + n, [1] * len(p) + [0] * len(n)))
    out.sort()
    return out[int(0.025 * draws)], out[int(0.975 * draws)]


def report(name, scores, labels, lower_is_better, names):
    # a single orientation convention: build a "higher is better" score
    signed = [-value if lower_is_better else value for value in scores]
    auc = roc_auc(signed, labels)
    order = sorted(range(len(signed)), key=lambda i: -signed[i])
    ranks = [position + 1 for position, index in enumerate(order) if labels[index]]
    print(f"\n{name}")
    low, high = bootstrap_auc(signed, labels)
    print(f"  ROC AUC                 {auc:.3f}   95% CI {low:.3f}-{high:.3f}"
          f"   (0.5 = indistinguishable)")
    for fraction in (0.01, 0.05, 0.10, 0.20):
        factor, found, take = enrichment(order, labels, fraction)
        print(f"  EF at {fraction*100:4.0f}% (top {take:3d})    {factor:5.2f}x"
              f"   {found} of {sum(labels)} actives")
    print(f"  ranks of the actives    {ranks}")
    print(f"  median active rank      {sorted(ranks)[len(ranks)//2]} of {len(labels)}")
    best = order[:5]
    print("  top 5: " + ", ".join(
        f"{names[i]}{'*' if labels[i] else ''}" for i in best))
    return {"auc": auc, "ci": [low, high], "ranks": ranks}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--smi", default="tools/fixtures/hiv1-protease-100.smi")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    rows = [line.split("\t") for line in
            (root / args.smi).read_text().splitlines()
            if line and not line.startswith("#")]
    by_smiles = {row[0]: row[1].split("#")[0].strip() for row in rows}
    actives = {row[0] for row in rows if "PR inhibitor" in row[1]}

    screen = json.loads(pathlib.Path(args.screen).read_text())
    results = [r for r in screen["results"] if r.get("affinity_pred_value") is not None]
    print(f"screened {len(results)} of {len(rows)} ligands"
          f" in {screen.get('elapsedMs', 0) / 60000:.1f} min"
          f" ({screen.get('throughput', {}).get('medianMs', 0) / 1000:.1f} s a ligand)")

    smiles = [r["smiles"] for r in results]
    names = [by_smiles.get(s, "?") for s in smiles]
    labels = [1 if s in actives else 0 for s in smiles]
    print(f"actives present: {sum(labels)} of {len(actives)}")

    heads = report("P(binder)  - affinity_probability_binary, higher is better",
                   [r["affinity_probability_binary"] for r in results],
                   labels, False, names)
    value = report("pIC50      - affinity_pred_value, LOWER log10 IC50 is better",
                   [r["affinity_pred_value"] for r in results],
                   labels, True, names)

    # 🔴 THE CONTROL, AND IT IS COUNTED WITH THE PAGE'S OWN PARSER. The screen
    # output carries no heavy-atom count, and the baseline is the point of this
    # whole file - so it is computed here rather than skipped, through
    # shared/chem/smiles.js, which is the same definition the ranked table's
    # HEAVY column shows.
    heavy = heavy_atoms(root, smiles)
    size = report("heavy atom count - THE BASELINE, not a prediction",
                  heavy, labels, False, names)

    out = {"n": len(results), "actives": sum(labels),
           "binary": heads, "value": value, "size": size}
    (root / "spec/HIV1-BENCHMARK.json").write_text(json.dumps(out, indent=1))
    print("\nwritten spec/HIV1-BENCHMARK.json")


if __name__ == "__main__":
    main()
