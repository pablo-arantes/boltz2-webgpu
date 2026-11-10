"""Are the affinity export's tensors the checkpoint's bytes, at the right offsets?

    python3 tools/check-affinity-export-fidelity.py --f32=<boltz2-affinity-f32>

🔴 THIS CLOSES THE "ONE BUNDLE TO BOTH SIDES" HOLE BEFORE G2 OPENS IT.
check-bundle-vs-params.py's header states the trap: every parity tool feeds ONE
bundle to both sides, so a tensor that is the wrong sign, the wrong width or at
the wrong offset is invisible to all of them - both sides read it and both are
wrong together.

G2 compares a JavaScript reference against upstream's PyTorch module. If the
oracle loaded its weights from this export, an export that put the right bytes
at the wrong NAME would produce two forwards that agree perfectly and are both
wrong. So either the oracle reads the checkpoint independently (33 MiB over the
wire, every run), or this runs once and establishes that the export IS the
checkpoint - after which loading the oracle from the export is sound.

HOW, FOR ABOUT A MEGABYTE. The exporter already verified each fetched payload's
CRC-32 against the checkpoint's ZIP central directory, so the BYTES are known
good. What that did not pin is the LAYOUT: which tensor's bytes landed at which
manifest offset. This re-reads every tensor out of the shard at the offset the
manifest gives, CRC-32s it, and compares against the central-directory entry
for the member that tensor's storage key names. A name pointing at another
tensor's bytes fails here, and so does an off-by-one offset.

Only the central directory and one data.pkl are fetched - a couple of MB rather
than the 1.92 GiB file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import ckpt_crc                                                   # noqa: E402

AFF_URL = ("https://huggingface.co/boltz-community/boltz-2/resolve/"
           "main/boltz2_aff.ckpt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f32", required=True)
    parser.add_argument("--url", default=AFF_URL)
    arguments = parser.parse_args()

    directory = pathlib.Path(arguments.f32)
    manifest = json.loads((directory / "manifest.json").read_text())
    tensors = manifest["tensors"]
    blobs = {f: (directory / f).read_bytes()
             for f in {r["file"] for r in tensors.values()}}

    print("reading the checkpoint's central directory and tensor map ...",
          file=sys.stderr)
    size = ckpt_crc.total_size(arguments.url)
    members = ckpt_crc.members(arguments.url, size)
    remote = ckpt_crc.tensor_map(arguments.url, members)
    remote = {k: v for k, v in remote.items() if k.startswith("affinity_module")}
    print(f"  checkpoint {size:,} B; affinity tensors there {len(remote)}",
          file=sys.stderr)

    missing = sorted(set(tensors) - set(remote))
    extra = sorted(set(remote) - set(tensors))
    if missing or extra:
        for n in missing[:10]:
            print(f"  EXPORTED BUT NOT IN THE CHECKPOINT: {n}")
        for n in extra[:10]:
            print(f"  IN THE CHECKPOINT BUT NOT EXPORTED: {n}")
        print(f"FAIL: {len(missing)} extra, {len(extra)} missing")
        return 1

    same = 0
    bad = []
    for name, record in sorted(tensors.items()):
        count = int(np.prod(record["shape"])) if record["shape"] else 1
        if record["dtype"] != "float32":
            bad.append((name, f"dtype {record['dtype']} in a float32 export"))
            continue
        start = record["byteOffset"]
        chunk = blobs[record["file"]][start:start + count * 4]
        if len(chunk) != count * 4:
            bad.append((name, f"shard holds {len(chunk)} B, needs {count * 4}"))
            continue
        member = members.get(remote[name]["member"])
        if member is None:
            bad.append((name, "no such member in the checkpoint"))
            continue
        if member["usize"] != count * 4:
            bad.append((name, f"member is {member['usize']} B, shape needs {count * 4}"))
            continue
        got = zlib.crc32(chunk) & 0xFFFFFFFF
        if got != member["crc"]:
            bad.append((name, f"crc {got:#010x} != checkpoint {member['crc']:#010x}"))
            continue
        if list(record["shape"]) != list(remote[name]["shape"]):
            bad.append((name, f"shape {record['shape']} != {remote[name]['shape']}"))
            continue
        same += 1

    for name, why in bad[:20]:
        print(f"  MISMATCH {name}: {why}")
    print()
    print(f"tensors checked      : {len(tensors)}")
    print(f"byte-identical       : {same}")
    print(f"mismatched           : {len(bad)}")
    print(f"false-same probability: ~{len(tensors) / 2 ** 32:.2e} (CRC-32)")
    print()
    print("EXPORT FIDELITY PASS - the export is the checkpoint, name by name,"
          " so G2 may load its oracle from it" if not bad
          else f"EXPORT FIDELITY FAIL: {len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
