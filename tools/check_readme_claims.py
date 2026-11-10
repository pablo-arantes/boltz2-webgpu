"""Does the project README still describe this tree?

    python3 tools/check_readme_claims.py

🔴 A README IS THE ONE DOCUMENT NOBODY RE-READS AND EVERYBODY TRUSTS, and this
project's own working notes record the cost of prose that has drifted: a table
row that "said 8 OF 14 AND 6 REFUSALS AND WAS STALE", a figure "read as a
defect" that was two commits out of date, and the standing complaint that "a doc
that disagrees with itself costs more than a missing one". So every claim in the
README that CAN be checked mechanically is checked here: the three pinned SHAs
against spec/PINNED_COMMIT.txt, every source path it names, every npm gate it
lists, the licence text, the registry entry and the affinity bundle's own
manifest.

🔴 WHAT IT CANNOT CHECK IS THE MEASUREMENTS, and it does not pretend to. The
timings, the memory figures, the bond lengths and the superposition residuals
are measurements on one machine; re-taking them is the named tool's job
(`npm run test:affinity-memory`, `tools/gpu/run-affinity-screen.js`,
`tools/screen-in-page.py`), and this one deliberately says nothing about them
rather than asserting a number it cannot reproduce.

It found one on its first run: the README said the affinity bundle was 6.5 MiB
where its manifest says 6.4.
"""
import json, math, pathlib, re, subprocess, sys
# run from anywhere: the README is beside this tool's parent
# the repository root: tools/ lives inside the app tree, which IS the repo
root = pathlib.Path(__file__).resolve().parent.parent
# 🔴 THE README **PLUS** THE DOCS SPLIT OUT OF IT. The README was 506 lines and
# is now 172: the measured tables went to docs/MEASURED.md and the provenance to
# docs/PROVENANCE.md. The claims did not stop applying because the prose moved,
# so this reads all three - and a claim that says "stated once" still means once
# ACROSS them, which is why they are concatenated rather than searched in turn.
t = "\n".join((root / name).read_text() for name in
               ("README.md", "docs/MEASURED.md", "docs/PROVENANCE.md"))
bad = []

def claim(ok, what):
    print(("ok    " if ok else "WRONG ") + what)
    if not ok: bad.append(what)

# pinned SHAs must match the spec file
pin = (root / "spec/PINNED_COMMIT.txt").read_text()
for sha in re.findall(r"\b[0-9a-f]{40}\b", t):
    claim(sha in pin, f"SHA {sha[:12]} is in spec/PINNED_COMMIT.txt")

# every file path named in the "reverse-engineered" table exists under app/
paths = set(re.findall(r"`((?:cpu|webgpu|shared|web|tools|test)/[A-Za-z0-9_./-]+\.(?:js|mjs|py))`", t))
for path in sorted(paths):
    here = (root / path).exists()
    claim(here, f"path {path}")

scripts = json.loads((root / "package.json").read_text())["scripts"]
for gate in re.findall(r"\| `(test:[a-z0-9-]+)` \|", t):
    claim(gate in scripts, f"npm script {gate}")

# screen.html is what the page instructions say
claim((root / "screen.html").exists(), "app/screen.html exists")
claim("8080" in (root / "tools/serve.py").read_text(), "serve.py default port 8080")

# the licence claims
# The three licences, each as the README NAMES it, against the file it names.
lic = (root / "LICENSE").read_text()
claim("BEER-WARE" in lic and "Ovchinnikov" in lic,
      "app/LICENSE really is Beerware by Ovchinnikov")
claim(re.search(r"LocalFold's code\*\* [-—] \[Beerware\]", t) is not None,
      "the README calls LocalFold's code Beerware, which is what app/LICENSE says")
claim(re.search(r"Boltz-2's weights\*\* [-—] \*\*MIT\*\*", t) is not None,
      "the README calls the Boltz-2 weights MIT")
claim("Prohibited Use Policy" in t and "Do not redistribute those" in t,
      "the README carries DeepMind's terms and the do-not-redistribute line")

# the registry entry the README promises
reg = (root / "shared/bundles/manifests/index.js").read_text()
claim('"boltz2-affinity"' in reg, "MODEL_BUNDLES has boltz2-affinity")
# 🔴 PINNED TO A SHA, AND THE CLAIM SAYS SO RATHER THAN JUST "HAS A REMOTE".
# `resolve/main/` would satisfy a looser test and is exactly what the brief
# forbids: it re-points on any re-upload, so a visitor's download could change
# without a commit here.
import re as _re
_aff = _re.search(r'"boltz2-affinity":\s*\{.*?\n  \},', reg, _re.DOTALL)
_remote = _re.search(r'remote:\s*"([^"]+)"', _aff.group(0) if _aff else "")
claim(_remote is not None, "boltz2-affinity has a remote")
claim(bool(_remote) and _re.search(r"/resolve/[0-9a-f]{40}/", _remote.group(1)) is not None,
      "its remote is pinned to a 40-character revision, not a branch")

affinity = root / "refs/export/boltz2-affinity-int5/manifest.json"
if not affinity.exists():
    print("skip  the affinity export is not on this machine")
    print("WRONG: %d" % len(bad) if bad else "every checkable claim holds")
    sys.exit(1 if bad else 0)
man = json.loads(affinity.read_text())
# 🔴 READ THE README'S OWN WORDS, NOT THE TREE'S FACTS. Two claims here used to
# assert things about the manifest and the LICENSE file - `shards == 8`,
# `"BEER-WARE" in LICENSE` - which are true no matter WHAT the README says, so
# the falsification arm changed the README's shard count and its licence name
# and the checker passed both. A claim that does not read the document it is
# checking is not checking the document.
for pattern, actual, what in [
    (r"\*\*(\d+) tensors", len(man["tensors"]), "tensor count"),
    (r"(\d+) tensors, [\d.]+ MiB", len(man["tensors"]), "tensor count"),
    # 🔴 SCOPED TO THE AFFINITY BUNDLE. A bare `\((\d+) shards` also matched the
    # STRUCTURE bundle's "(24 shards)" and compared it against the affinity
    # manifest, so the clean tree failed with "shard count 24 matches the
    # manifest's 8" - a checker reporting a disagreement between two different
    # bundles as an error in the document.
    (r"MiB affinity \((\d+) shards", man["bundle"]["shards"],
     "affinity shard count"),
]:
    found = re.findall(pattern, t)
    claim(len(found) > 0, f"the README states the {what} ({pattern})")
    for given in found:
        claim(int(given) == actual,
              f"{what} {given} matches the manifest's {actual}")
mib = man["bundle"]["bytes"] / 1048576
# 🔴 EACH SENTENCE, NOT "DOES THE RIGHT NUMBER APPEAR SOMEWHERE". The first
# version asked whether `f"{mib:.1f} MiB"` was anywhere in the file, and the
# falsification arm walked straight past it: the README gives the bundle's size
# TWICE, so changing one to 9.9 MiB left the other saying 6.4 and the check
# passed. A claim that one correct instance can satisfy is not a check on the
# others. Both sentences are matched where they stand.
sizes = re.findall(r"520 tensors, ([\d.]+) MiB", t) \
    + re.findall(r"([\d.]+) MiB affinity \(", t)
claim(len(sizes) == 2, f"both bundle-size sentences are present (found {len(sizes)})")
for given in sizes:
    claim(abs(float(given) - mib) < 0.05,
          f"bundle size {given} MiB matches the manifest's {mib:.1f}")

# ...and the decoded size, which is a different quantity and separately wrong-able
decoded = sum(4 * math.prod(record["shape"])
              for record in man["tensors"].values()) / 1048576
# 🔴 `\s+` AND NOT A SPACE: the README is hard-wrapped, so "33 MiB" and
# "decoded" sit on different lines and a literal space matched neither. The
# first version of this check reported the figure as missing when it was there.
written = re.findall(r"([\d.]+) MiB\s+decoded", t)
claim(len(written) == 1, "the decoded size is stated once")
for given in written:
    claim(abs(float(given) - decoded) < 1.0,
          f"decoded size {given} MiB matches the computed {decoded:.1f}")
# The structure bundle's figures come from the Phase 1 verification, not from a
# manifest on this machine - spec/PINNED_COMMIT.txt records "all 24 shards,
# 365 MiB" - so they are checked against that.
for pattern, what in [(r"(\d+) MiB structure \(", "MiB"),
                      (r"MiB structure \((\d+) shards", "shards")]:
    for given in re.findall(pattern, t):
        claim(f"{given} {'shards' if what == 'shards' else 'MiB'}" in pin
              or given in pin,
              f"structure bundle {given} {what} is in spec/PINNED_COMMIT.txt")

print()
print("WRONG: %d" % len(bad) if bad else "every checkable claim holds")
sys.exit(1 if bad else 0)
