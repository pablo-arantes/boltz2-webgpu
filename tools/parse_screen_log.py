"""The per-ligand table from run-affinity-screen.js, as the JSON it would have
returned.

🔴 WHY THIS EXISTS. The runner THROWS when any ligand fails - `${n} ligand(s)
failed` - so a 50-minute screen of 100 ligands returns nothing at all because
one of them was a salt. The 99 that worked were printed line by line as they
finished, so the numbers are not lost; this recovers them. The throw is right
(a partial screen must not look complete) but it should not also discard the
part that succeeded.
"""
import json
import re
import sys

ROW = re.compile(
    r"^(lig\d+)\t(\S+)\tvalue (-?[\d.]+)\tpIC50 (-?[\d.]+)\tprob ([\d.]+)"
    r"\tcrop (\d+)t/(\d+)p\tipTM ([\d.]+)\t([\d.]+)s")

results, failures = [], []
for line in open(sys.argv[1], errors="replace"):
    found = ROW.match(line)
    if found:
        results.append({
            "id": found.group(1), "smiles": found.group(2),
            "affinity_pred_value": float(found.group(3)),
            "pIC50": float(found.group(4)),
            "affinity_probability_binary": float(found.group(5)),
            "croppedTokens": int(found.group(6)),
            "croppedProteinTokens": int(found.group(7)),
            "iptm": float(found.group(8)),
            "runtimeMs": float(found.group(9)) * 1000,
        })
    elif "FAILED" in line and line.startswith("FAILED"):
        failures.append(line.strip())

total = sum(r["runtimeMs"] for r in results)
times = sorted(r["runtimeMs"] for r in results)
out = {"results": results, "failures": failures,
       "elapsedMs": total,
       "throughput": {"medianMs": times[len(times) // 2], "samples": len(times)}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"recovered {len(results)} results, {len(failures)} failure(s)")
for failure in failures:
    print("  " + failure[:120])
