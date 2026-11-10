"""G1 - is the affinity bundle's float32Tensors list actually protecting the norms?

    python3 tools/check-affinity-quantisation.py \
      --f32=<boltz2-affinity-f32> --int5=<boltz2-affinity-int5>

🔴 THIS GATE EXISTS BECAUSE quantize_af3.py's OWN REGEX CANNOT SEE OUR NAMES.
`KEEP_FLOAT32` is `/(scale|offset|bias)$|_bias$|_weight$|/output_b$` - SLASH
separated, because every other bundle here carries Haiku paths. The affinity
export carries upstream's PyTorch names, which are DOT separated
(`affinity_module1.z_norm.weight`), so that regex matches NONE of them and
every LayerNorm scale and bias would be group-quantised - which
quantize_af3.py's own header calls "the worst thing to group-quantise", since a
128-wide LayerNorm scale is four groups and that tensor's job is to set the
scale of everything after it.

It survives only because line 245 is a union with the manifest's own list:
`if KEEP_FLOAT32.search(name) or name in named_float32 or here == 32`. So
`float32Tensors` is LOAD-BEARING for this bundle in a way it is not for any
other, and a future re-export that drops or truncates it would quantise the
norms with nothing failing. Hence a gate rather than a comment.

WHAT IT MEASURES. It decodes the int5 bundle on the host - codes, float16
scales, float16 zero points - and compares every tensor against the float32
export it came from. The protected groups must come back EXACTLY (relRMS 0,
because they were never encoded); the large projections must come back
quantised. `--falsify` strips the list, re-quantises into a temporary
directory, and requires the protected groups to DEGRADE - because a gate
nobody has watched fail is a gate nobody has checked.

🔴 AND `boundaries` IS THE ONE WHOSE DAMAGE IS STRUCTURAL RATHER THAN NOISY.
It is the distogram's 63 bin edges, `linspace(2, 22, 63)`, and the module
buckets every token-pair distance against it. Quantising a monotone ramp moves
the edges, so distances land in the wrong bin - an error of a different kind
from a slightly noisy weight. It is a 1-D tensor and so protected by the list's
own rule, and this gate names it separately so that stays true.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

CODES_PER_GROUP = 32
BYTES_PER_GROUP = 20          # 32 codes x 5 bits = 160 bits exactly


def decode_bundle(directory: pathlib.Path) -> tuple[dict, dict]:
    """Every tensor as float32: int5 decoded as code * scale + zero."""
    manifest = json.loads((directory / "manifest.json").read_text())
    files = {name["file"] for name in manifest["tensors"].values()}
    blobs = {f: (directory / f).read_bytes() for f in files}
    out: dict[str, np.ndarray] = {}
    for name, record in manifest["tensors"].items():
        count = int(np.prod(record["shape"])) if record["shape"] else 1
        blob = blobs[record["file"]]
        if record["dtype"] == "float32":
            out[name] = np.frombuffer(blob, "<f4", count=count,
                                      offset=record["byteOffset"]).copy()
            continue
        if record["dtype"] != "int5":
            raise SystemExit(f"{name}: unexpected dtype {record['dtype']}")
        groups = (count + CODES_PER_GROUP - 1) // CODES_PER_GROUP
        raw = np.frombuffer(blob, np.uint8, count=groups * BYTES_PER_GROUP,
                            offset=record["byteOffset"])
        bits = np.unpackbits(raw.reshape(groups, BYTES_PER_GROUP), axis=1,
                             bitorder="little")
        codes = (bits.reshape(groups, CODES_PER_GROUP, 5)
                 * np.array([1, 2, 4, 8, 16], np.uint16)).sum(2).astype(np.float32)
        scales = np.frombuffer(blob, "<f2", count=groups,
                               offset=record["scaleOffset"]).astype(np.float32)
        zeros = np.frombuffer(blob, "<f2", count=groups,
                              offset=record["zeroOffset"]).astype(np.float32)
        out[name] = (codes * scales[:, None]
                     + zeros[:, None]).reshape(-1)[:count]
    return manifest, out


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    """relRMS, and it REFUSES rather than returning a perfect score.

    🔴 An unguarded relRMS over a non-array returns exactly 0 - "a perfect
    score from comparing nothing", which this repository records as a trap in
    tools/gpu/relative-rms.js.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.ndim != 1 or a.size == 0 or a.size != b.size:
        raise ValueError(f"not comparable: {a.shape} against {b.shape}")
    denominator = float(np.sqrt(np.mean(a * a)))
    if denominator == 0.0:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denominator)


NORM_SUFFIXES = ("norm.weight", "norm_in.weight", "norm_out.weight",
                 "layer_norm.weight", "z_norm.weight")
OUTPUT_SUFFIXES = ("to_affinity_pred_value.4.weight",
                   "to_affinity_pred_score.4.weight",
                   "to_affinity_logits_binary.weight")


def grouping(reference: dict) -> dict[str, list[str]]:
    return {
        "LayerNorm weights": [n for n in reference if n.endswith(NORM_SUFFIXES)],
        "biases": [n for n in reference if n.endswith(".bias")],
        "boundaries (distogram edges)":
            [n for n in reference if n.endswith(".boundaries")],
        "dist_bin embedding":
            [n for n in reference if n.endswith("dist_bin_pairwise_embed.weight")],
        "output projections": [n for n in reference if n.endswith(OUTPUT_SUFFIXES)],
        "large projections":
            [n for n in reference if reference[n].size >= 1024
             and not n.endswith(".bias")],
    }


PROTECTED = ("LayerNorm weights", "biases", "boundaries (distogram edges)",
             "dist_bin embedding", "output projections")


def worst(reference: dict, decoded: dict, names: list[str]) -> float:
    return max(relative_rms(reference[n], decoded[n]) for n in names)


def requantise_without_list(f32_dir: pathlib.Path, root: pathlib.Path) -> dict:
    """The falsification arm: the same export with float32Tensors emptied."""
    work = pathlib.Path(tempfile.mkdtemp(prefix="g1-nolist-"))
    try:
        source = work / "f32"
        shutil.copytree(f32_dir, source)
        path = source / "manifest.json"
        manifest = json.loads(path.read_text())
        removed = len(manifest["float32Tensors"])
        manifest["float32Tensors"] = []
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        out = work / "int5"
        subprocess.run(
            [sys.executable, "-I", str(root / "tools" / "quantize_af3.py"),
             f"--source={source}", f"--out={out}"],
            check=True, capture_output=True)
        _, decoded = decode_bundle(out)
        return {"removed": removed, "decoded": decoded}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--f32", required=True)
    parser.add_argument("--int5", required=True)
    parser.add_argument("--falsify", action="store_true")
    arguments = parser.parse_args()

    f32_dir = pathlib.Path(arguments.f32)
    manifest, reference = decode_bundle(f32_dir)
    _, shipped = decode_bundle(pathlib.Path(arguments.int5))

    if set(reference) != set(shipped):
        raise SystemExit("the two bundles name different tensors")

    groups = grouping(reference)
    listed = set(manifest.get("float32Tensors", []))
    print(f"tensors          : {len(reference)}")
    print(f"float32Tensors   : {len(listed)}")
    print()

    failures = 0
    rows = {}
    for label, names in groups.items():
        if not names:
            continue
        rows[label] = worst(reference, shipped, names)

    print(f"{'tensor group':<30} {'relRMS':>11}   expected")
    for label, value in rows.items():
        protected = label in PROTECTED
        expect = "EXACT (kept float32)" if protected else "quantised"
        bad = (protected and value != 0.0) or (not protected and value == 0.0)
        failures += bad
        print(f"{label:<30} {value:>11.3e}   {expect}"
              + ("   <-- FAIL" if bad else ""))
    print()

    if arguments.falsify:
        arm = requantise_without_list(f32_dir, root)
        print(f"falsify arm: float32Tensors emptied ({arm['removed']} names removed)")
        print(f"{'tensor group':<30} {'relRMS':>11}   verdict")
        degraded = 0
        for label, names in groups.items():
            if not names:
                continue
            value = worst(reference, arm["decoded"], names)
            if label in PROTECTED:
                ok = value > 1e-4
                degraded += ok
                print(f"{label:<30} {value:>11.3e}   "
                      + ("DEGRADED, as expected" if ok
                         else "unchanged   <-- the list is not what protects it"))
            else:
                same = abs(value - rows[label]) < 1e-9
                print(f"{label:<30} {value:>11.3e}   "
                      + ("unchanged, as expected (control)" if same
                         else "MOVED   <-- the list reached a tensor it does not name"))
        print()
        if degraded == len(PROTECTED):
            print(f"FALSIFY OK: all {degraded} protected groups degrade without the list")
        else:
            print(f"FALSIFY FAILED: only {degraded} of {len(PROTECTED)} degraded")
            failures += 1

    print("G1 PASS: float32Tensors is what keeps the norms exact"
          if failures == 0 else f"G1 FAIL: {failures}")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
