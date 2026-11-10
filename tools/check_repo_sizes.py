#!/usr/bin/env python3
"""Tracked files over the size the publishing brief sets, as a GATE.

    python3 -I tools/check_repo_sizes.py [--limit-mib 10]

🔴 WHY THIS IS A GATE AND NOT A ONE-OFF `du`. The brief for publishing this
work says "Report anything over ~10 MB before committing", and a rule that
lives only in a conversation is a rule that holds exactly once. Model weights
are the reason it exists - they stay on HuggingFace and are fetched at runtime
from a pinned revision, never committed - but the file this actually catches is
an ORACLE DUMP, which is the shape nobody watches for: not a weight, not a
build artefact, just a reference tensor set that grew.

🔴 AND THE ONE KNOWN FILE IS DECLARED RATHER THAN EXCLUDED BY PATTERN. A
glob would silently swallow the next 18 MiB dump dropped beside it. Each entry
states what it is and why it is still here, and a file that LEAVES the tree
fails this too - so the declaration has to be deleted rather than quietly
carried, which is the same treatment `ref_pos` and the canonical-name corpus
get elsewhere in this tree.
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# path -> why it is over the limit and why it is still tracked
DECLARED = {
    "oracle-dumps/af3-6mrr.json":
        "AlphaFold 3's own featurised batch for 6MRR, inherited from LocalFold."
        " Four tools read it (tools/gpu/fold.js --target=6mrr,"
        " tools/oracle/check_af3_featurise.js, tools/oracle/dump_af3_batch.py,"
        " shared/af3/featurise/batch-from-dump.js) and it is what lets a fold"
        " be compared against the reference's OWN features rather than against"
        " this port's featuriser - docs/AF3.md's `--dump=` numbers all rest on"
        " it. It is NOT published: tools/build_site.py copies no oracle dump"
        " into dist/, verified by this gate's companion check below.",
}


def tracked() -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [pathlib.Path(name) for name in out.stdout.split("\0") if name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-mib", type=float, default=10.0)
    args = ap.parse_args()
    limit = args.limit_mib * 1048576

    over = {}
    total = 0
    for path in tracked():
        full = ROOT / path
        if not full.is_file():
            continue
        size = full.stat().st_size
        total += size
        if size > limit:
            over[str(path)] = size

    print(f"{len(tracked())} tracked files, {total / 1048576:.1f} MiB")
    print(f"over {args.limit_mib:g} MiB: {len(over)}")
    for path, size in sorted(over.items(), key=lambda kv: -kv[1]):
        state = "declared" if path in DECLARED else "🔴 UNDECLARED"
        print(f"  {size / 1048576:6.1f} MiB  {path}   [{state}]")

    # 🔴 THE COMPANION CHECK: being large is tolerable, being PUBLISHED is not.
    # GitHub Pages caps a site at a gigabyte and a reader of the page has no use
    # for a reference tensor set, so a dump that reaches dist/ is wasted
    # bandwidth on every visit.
    dist = ROOT / "dist"
    published = sorted(str(p.relative_to(dist)) for p in dist.rglob("*")
                       if p.is_file() and p.stat().st_size > limit) if dist.is_dir() else []
    if dist.is_dir():
        print(f"dist/ files over {args.limit_mib:g} MiB: {len(published)}")
        for name in published:
            print(f"  🔴 {name}")
    else:
        print("dist/ not built - run tools/build_site.py to check the published half")

    undeclared = sorted(set(over) - set(DECLARED))
    gone = sorted(set(DECLARED) - set(over))
    for path in undeclared:
        print(f"\n🔴 {path} is over the limit and not declared in this file.")
        print("   Weights belong on HuggingFace, fetched from a pinned revision."
              " If it genuinely must be tracked, declare it here with its reason.")
    for path in gone:
        print(f"\n{path} is no longer over the limit - delete its entry from"
              " DECLARED in this file.")
    ok = not undeclared and not gone and not published
    print(f"\nverdict: {len(over)} over the limit, {len(DECLARED)} declared,"
          f" {len(published)} published"
          + (" - as the brief requires" if ok else " - NOT as the brief requires"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
