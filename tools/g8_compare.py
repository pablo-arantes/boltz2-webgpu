"""G8: this port's affinity numbers against UPSTREAM BOLTZ running the same job.

    python3 tools/g8_compare.py --seeds=3 [--sequence=... --smiles=...]
    python3 tools/g8_compare.py --yaml=refs/boltz/examples/affinity.yaml --seeds=2

🔴 WHAT MAKES THIS DIFFERENT FROM EVERY OTHER AFFINITY GATE HERE. The others
hold one stage to a dump of upstream's own module: the f64 reference seam by
seam (test:affinity-oracle), the whole module forward (test:affinity-forward),
the crop as an exact index set (test:affinity-crop), the batch against
upstream's featuriser (test:boltz2-featurise). Each of those compares a
FUNCTION to a FUNCTION on identical input. None of them can see an error in how
the pieces are composed, because none of them runs the pipeline - and the
number a reader ranks on comes out of the composition.

🔴 AND IT CANNOT BE A TOLERANCE COMPARISON, WHICH IS THE WHOLE DIFFICULTY. The
affinity is read off a SAMPLED pose: upstream's structure pass is a diffusion
model with a random start, so upstream disagrees with ITSELF from seed to seed,
and so do we. A bound picked by hand would be measuring the author's optimism.

So the comparison is against upstream's own spread:

  1. run upstream N times with N different seeds -> its mean and its spread.
     THIS IS THE CONTROL, and it is the only honest unit for the gap.
  2. run this port N times with the same seeds -> our mean and our spread.
  3. report the gap between the means IN UNITS OF upstream's own spread.

A gap inside upstream's seed-to-seed spread is as much agreement as the
quantity admits. A gap of many spreads is a real difference in the pipeline,
and the per-seed table says whether it is a shift (every seed the same way) or
scatter. docs/ records this exact distinction being got wrong the other way -
"a difference in both arms is a difference in the CONFIGURATION, not in the
sampler".

🔴 SEEDS ARE NOT SHARED BETWEEN THE TWO IMPLEMENTATIONS AND MUST NOT BE READ AS
IF THEY WERE. Seed 0 here and seed 0 in PyTorch are different noise; the seeds
exist to sample each implementation's own spread, not to pair runs. The per-seed
table is printed in seed order for inspection, never differenced row by row.

🔴 NOT A CI GATE. Upstream runs on CPU here - minutes to an hour a seed - and
needs 6.1 GB of its own assets (tools/fetch_boltz_cache.sh). It is a bench to
be run deliberately and its numbers recorded.
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
# the repository root IS the app tree: tools/ lives inside it
APP = ROOT
DEFAULT_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
DEFAULT_SMILES = "CCCc1ccccc1O"


def write_yaml(path, sequence, smiles, a3m=None):
    """upstream's own job format, single-sequence.

    🔴 `msa: empty` IS NOT A DEFAULT. Without it `boltz predict` refuses the job
    asking for an MSA or --use_msa_server, and the comparison must be
    single-sequence on both sides or it is comparing alignments.
    """
    # 🔴 THE SAME ALIGNMENT ON BOTH SIDES, OR THERE IS NOTHING TO COMPARE.
    # Measured single-sequence on a 33-residue target: the two implementations'
    # structures are 9.43 A CA rms apart with the ligand centroid 16.69 A away,
    # so their affinity numbers differ because their POSES differ. `msa: empty`
    # is kept as an explicit arm rather than a default, because "no alignment"
    # is a legitimate thing to compare and a misleading thing to assume.
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {sequence}",
        # 🔴 `msa:` IS NOT OPTIONAL AND THE TWO VALUES MEAN DIFFERENT THINGS.
        # Without the key `boltz predict` refuses the job, asking for an MSA or
        # --use_msa_server. `empty` is a single-sequence fold; a PATH is the
        # alignment, and it must be the SAME file this port is given or the two
        # are not folding the same problem.
        f"      msa: {a3m}" if a3m else "      msa: empty",
        "  - ligand:",
        "      id: B",
        f"      smiles: '{smiles}'",
        "properties:",
        "  - affinity:",
        "      binder: B",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def run_upstream(yaml_path, out_dir, cache, seed, args):
    """One `boltz predict`, returning its affinity JSON."""
    python = ROOT / "refs/venv-boltz/bin/boltz"
    command = [
        str(python), "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--cache", str(cache),
        "--accelerator", "cpu",
        "--seed", str(seed),
        "--recycling_steps", str(args.recycles),
        "--sampling_steps", str(args.steps),
        "--diffusion_samples", str(args.samples),
        "--sampling_steps_affinity", str(args.affinity_steps),
        "--diffusion_samples_affinity", str(args.affinity_samples),
        "--output_format", "pdb",
        "--num_workers", "0",
    ]
    started = time.time()
    # 🔴 CWD IS **NOT** THE CACHE OR THE CHECKOUT. Downloaded archives are
    # untrusted input and an interpreter loads modules from its working
    # directory; this runs from the project root with every path absolute.
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - started
    name = yaml_path.stem
    found = list(pathlib.Path(out_dir).rglob(f"affinity_{name}.json"))
    if not found:
        tail = (result.stdout + result.stderr)[-2500:]
        raise SystemExit(f"upstream wrote no affinity JSON for seed {seed}.\n{tail}")
    return json.loads(found[0].read_text()), elapsed


def run_ours(sequence, smiles, seed, args):
    """One run of this port, through the same loop the page drives."""
    command = [
        "node", "tools/gpu-chrome.mjs", "tools/gpu/run-affinity-screen.js",
        f"--sequence={sequence}", f"--ligands={smiles}",
        # 🔴 THE TWO PASSES SET SEPARATELY, to upstream's own split: its
        # structure pass takes --recycling_steps/--sampling_steps, and its
        # affinity pass fixes recycles at 5 and takes
        # --sampling_steps_affinity/--diffusion_samples_affinity. A comparison
        # driven from one step count is not a comparison at upstream's
        # settings.
        f"--struct-recycles={args.recycles}", f"--struct-steps={args.steps}",
        "--affinity-recycles=5",
        f"--affinity-steps={args.affinity_steps}",
        f"--affinity-samples={args.affinity_samples}",
        f"--seed={seed}",
    ] + ([f"--a3m=/{os.path.relpath(os.path.abspath(args.a3m), APP)}"]
         if args.a3m else [])
    started = time.time()
    result = subprocess.run(command, cwd=str(APP), capture_output=True, text=True)
    elapsed = time.time() - started
    at = result.stdout.find("\n{\n")
    if at < 0:
        raise SystemExit("this port returned no JSON:\n"
                         + (result.stdout + result.stderr)[-2500:])
    doc = json.loads(result.stdout[at:])
    if not doc.get("results"):
        raise SystemExit(f"this port scored nothing: {doc}")
    return doc["results"][0], elapsed


def spread(values):
    """Mean and the spread to judge a gap against.

    🔴 THE RANGE, NOT THE STANDARD DEVIATION, AT THESE SAMPLE SIZES. Three
    seeds give an sd with about one significant figure and it is biased low;
    the half-range is what three numbers actually support, and it is the
    quantity a reader can check by eye against the per-seed table.
    """
    if len(values) == 1:
        return {"mean": values[0], "halfRange": float("nan"), "n": 1}
    return {"mean": statistics.fmean(values),
            "halfRange": (max(values) - min(values)) / 2,
            "sd": statistics.stdev(values) if len(values) > 2 else float("nan"),
            "lo": min(values), "hi": max(values), "n": len(values)}


FIELDS = ["affinity_pred_value", "affinity_probability_binary"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    ap.add_argument("--smiles", default=DEFAULT_SMILES)
    ap.add_argument("--yaml", default=None,
                    help="use an existing job file instead (its own sequence"
                         " and ligand; this port is driven from them)")
    ap.add_argument("--a3m", default=None,
                    help="an .a3m given to BOTH implementations; without it"
                         " both run single-sequence, which on a small target"
                         " makes the poses - and so the affinity - diverge")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--affinity-steps", type=int, default=200)
    ap.add_argument("--affinity-samples", type=int, default=5)
    ap.add_argument("--cache", default=str(ROOT / "refs/boltz-cache"))
    ap.add_argument("--out", default=str(ROOT / "refs/g8-out"))
    ap.add_argument("--skip-upstream", action="store_true")
    ap.add_argument("--skip-ours", action="store_true")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    for needed in ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols.tar"):
        if not (cache / needed).exists():
            raise SystemExit(f"{cache / needed} is missing - run"
                             f" tools/fetch_boltz_cache.sh {cache}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.yaml is not None:
        yaml_path = pathlib.Path(args.yaml)
        text = yaml_path.read_text()
        sequence = next(l.split(":", 1)[1].strip() for l in text.splitlines()
                        if l.strip().startswith("sequence:"))
        smiles = next(l.split(":", 1)[1].strip().strip("'\"")
                      for l in text.splitlines() if l.strip().startswith("smiles:"))
    else:
        sequence, smiles = args.sequence, args.smiles
        yaml_path = write_yaml(out / "g8.yaml", sequence, smiles,
                               a3m=os.path.abspath(args.a3m) if args.a3m else None)

    print(f"target   {len(sequence)} residues")
    print(f"ligand   {smiles}")
    print(f"settings structure {args.recycles} recycles / {args.steps} steps /"
          f" {args.samples} samples; affinity {args.affinity_steps} steps /"
          f" {args.affinity_samples} samples")
    print(f"seeds    {args.seeds} per implementation (NOT paired - see the header)")
    print()

    upstream, ours = [], []
    for seed in range(args.seeds):
        if not args.skip_upstream:
            record, elapsed = run_upstream(yaml_path, out / f"up{seed}", cache,
                                           seed, args)
            upstream.append(record)
            print(f"upstream seed {seed}: "
                  + "  ".join(f"{f}={record[f]:.4f}" for f in FIELDS)
                  + f"   {elapsed:.0f} s")
        if not args.skip_ours:
            record, elapsed = run_ours(sequence, smiles, seed, args)
            ours.append(record)
            print(f"ours     seed {seed}: "
                  + "  ".join(f"{f}={record[f]:.4f}" for f in FIELDS)
                  + f"   {elapsed:.0f} s")

    print()
    verdicts = {}
    for field in FIELDS:
        up = spread([r[field] for r in upstream]) if upstream else None
        mine = spread([r[field] for r in ours]) if ours else None
        if up is None or mine is None:
            continue
        gap = abs(up["mean"] - mine["mean"])
        unit = up["halfRange"]
        # 🔴 A ZERO SPREAD IS NOT A TIGHT BOUND, IT IS AN UNMEASURED ONE. With
        # one seed, or with a quantity upstream happens to reproduce exactly,
        # the gap has no unit and the verdict says so rather than dividing.
        ratio = (gap / unit) if (unit and unit > 0) else float("nan")
        verdicts[field] = {"upstream": up, "ours": mine, "gap": gap,
                           "gapInUpstreamSpreads": ratio}
        print(f"{field}")
        print(f"  upstream  mean {up['mean']:.4f}  range"
              f" {up.get('lo', float('nan')):.4f}..{up.get('hi', float('nan')):.4f}"
              f"  (+/- {unit:.4f})")
        print(f"  ours      mean {mine['mean']:.4f}  range"
              f" {mine.get('lo', float('nan')):.4f}..{mine.get('hi', float('nan')):.4f}"
              f"  (+/- {mine['halfRange']:.4f})")
        print(f"  gap       {gap:.4f}"
              + ("" if ratio != ratio else f"  = {ratio:.2f} x upstream's own spread"))
    print()
    json.dump({"sequence": sequence, "smiles": smiles,
               "settings": vars(args), "upstream": upstream, "ours": ours,
               "verdicts": verdicts},
              open(out / "g8-result.json", "w"), indent=2)
    print(f"written {out / 'g8-result.json'}")


if __name__ == "__main__":
    main()
