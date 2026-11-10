#!/usr/bin/env python3
"""Write a LocalFold float32 model directory for Boltz-2's AFFINITY modules.

    python3 -I tools/export_boltz2_affinity.py --out refs/export/boltz2-affinity-f32

Output is the shape tools/quantize_af3.py consumes - manifest.json plus
weights-NN.f32.bin shards, every tensor float32 - so the int5 packing, the
shard limits and the digest bookkeeping stay in LocalFold's existing tooling.

WHY THIS IS NOT export_af3_model.py. That reads a ColabDesign2 blob of
(scope, name, array) records, which is what af3-any-model emits for the
AF3-lineage checkpoints. af3-any-model has NO affinity module, so there is no
blob and no Haiku path to match. This reads boltz-community/boltz-2's
boltz2_aff.ckpt directly.

🔴 THE TENSOR NAMES ARE UPSTREAM'S PyTORCH PATHS, UNCHANGED - for exactly the
reason export_af3_model.py keeps the Haiku ones: "the names ARE the
description... a tensor that is wrong can be found by grepping for it in both
places." For the affinity module the oracle IS upstream PyTorch, so matching
`affinity_module1.pairformer_stack.layers.0.tri_mul_out.norm_in.weight`
preserves that property. Inventing a Haiku path would destroy it.

🔴 AND THAT IS WHY float32Tensors MUST BE COMPLETE HERE. quantize_af3.py's
KEEP_FLOAT32 is `/(scale|offset|bias)$|_bias$|_weight$|/output_b$` - SLASH
separated and Haiku shaped - and our names are DOT separated, so that regex
matches NONE of them. Every LayerNorm scale and bias would be group-quantised,
which that tool's own header calls "the worst thing to group-quantise". It is
survivable only because line 245 is a union with the manifest's own list:
`if KEEP_FLOAT32.search(name) or name in named_float32 or here == 32`.
The list below is therefore load-bearing, not documentation.

🔴 ONLY 8.74 M OF THE CHECKPOINT'S 529 M PARAMETERS ARE READ. G0
(tools/g0_verify.py) established that the 5,019 tensors boltz2_aff.ckpt shares
with boltz2_conf.ckpt are byte-identical, so LocalFold's shipped boltz2-int5
bundle already carries the trunk, structure and confidence weights the affinity
pass needs. Exporting them again would be 365 MiB of duplicate download.

Nothing is downloaded whole: each tensor's ZIP member is fetched by HTTP range
and its CRC-32 checked against the central directory, so a short or corrupted
read fails here rather than inside a dequantisation later.
"""
import argparse, json, pathlib, re, sys, time, zlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ckpt_crc

AFF_URL = ("https://huggingface.co/boltz-community/boltz-2/resolve/"
           "main/boltz2_aff.ckpt")
# The repo revision G0 was run against. Recorded in the manifest so the export
# names the bytes it came from.
AFF_REPO_SHA = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
SHARD_LIMIT = 48 * 1024 * 1024          # export_af3_model.py's cap


def keep_float32_reason(name: str, shape: list[int]) -> str | None:
    """Why this tensor escapes quantisation, or None if it is quantised.

    Follows quantize_af3.py's stated reasoning rather than its regex: norms,
    offsets and biases are 0.09% of the parameters and the worst thing to
    group-quantise, because a 128-wide LayerNorm scale is four groups and that
    tensor's job is to set the scale of everything after it.
    """
    if name.endswith(".bias"):
        return "bias"
    if re.search(r"(^|\.)(norm|norm_in|norm_out|layer_norm|act_norm|z_norm|"
                 r"pre_norm_[sz]|s_norm)[0-9]*\.weight$", name):
        return "norm.weight"
    if len(shape) <= 1:
        # `boundaries` (63 floats, a derived buffer) and any 1-D scale.
        return "1-D buffer"
    if name.endswith("dist_bin_pairwise_embed.weight"):
        # An embedding table feeding every block. ESM-C's is why
        # quantize_af3.py learned to honour float32Tensors at all.
        return "embedding table"
    if shape[0] == 1:
        # The scalar output projections: `to_affinity_pred_*.4.weight` [1,384]
        # and `to_affinity_logits_binary.weight` [1,1]. These ARE the number a
        # user ranks on, and together they are 1.5 KiB.
        return "output projection"
    return None


class ShardWriter:
    """Lay tensors into float32 shards, four-byte aligned, none over the limit.

    Deliberately the same shape as export_af3_model.py's, so the downstream
    tools cannot tell which exporter produced the directory.
    """

    def __init__(self, out_dir: pathlib.Path):
        self.out_dir = out_dir
        self.records: dict[str, dict] = {}
        self.chunks: list[bytes] = []
        self.offset = 0
        self.index = 0

    def _flush(self) -> None:
        if not self.chunks:
            return
        path = self.out_dir / f"weights-{self.index:02d}.f32.bin"
        with path.open("wb") as handle:
            for chunk in self.chunks:
                handle.write(chunk)
        self.chunks, self.offset, self.index = [], 0, self.index + 1

    def add(self, name: str, values: np.ndarray) -> None:
        flat = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
        if self.offset + flat.nbytes > SHARD_LIMIT and self.chunks:
            self._flush()
        self.records[name] = {
            "file": f"weights-{self.index:02d}.f32.bin",
            "shape": list(values.shape),
            "byteOffset": self.offset,
            "dtype": "float32",
        }
        self.chunks.append(flat.tobytes())
        self.offset += flat.nbytes

    def close(self) -> None:
        self._flush()


def affinity_tensor_map(url, members):
    """name -> record, for the affinity_module* tensors only.

    Captures storage_offset and stride as well as shape, because a non-zero
    offset or a non-contiguous stride means the ZIP member is NOT simply this
    tensor's bytes - and reading it as though it were would be a different
    tensor of the right size.
    """
    import io, struct
    pkl = [n for n in members if n.endswith("data.pkl")][0]
    root = pkl.rsplit("/", 1)[0]
    e = members[pkl]
    hdr = ckpt_crc.ranged(url, e["lho"], e["lho"] + 29)
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    at = e["lho"] + 30 + nlen + elen
    obj = ckpt_crc.Restricted(
        io.BytesIO(ckpt_crc.ranged(url, at, at + e["usize"] - 1))).load()
    out = {}
    for name, call in ckpt_crc._walk(obj):
        if not name.startswith("state_dict.affinity_module"):
            continue
        a = call.args
        storage, storage_offset, shape, stride = a[0], a[1], a[2], a[3]
        pid = storage.pid
        width, storage_name = 4, None
        for p in pid:
            n2 = getattr(p, "name", None)
            if isinstance(n2, str) and n2 in ckpt_crc.WIDTH:
                width, storage_name = ckpt_crc.WIDTH[n2], n2
        short = name[len("state_dict."):]
        out[short] = {
            "member": f"{root}/data/{pid[2]}",
            "shape": list(shape), "stride": list(stride),
            "storage_offset": int(storage_offset),
            "width": width, "storage": storage_name,
        }
    return out


def contiguous_stride(shape):
    s, acc = [], 1
    for d in reversed(shape):
        s.append(acc); acc *= d
    return list(reversed(s))


def fetch_with_retry(url, members, member, attempts=6):
    last = None
    for i in range(attempts):
        try:
            blob = ckpt_crc.fetch_member(url, members, member)
            want = members[member]["crc"]
            got = zlib.crc32(blob) & 0xFFFFFFFF
            if got != want:
                raise OSError(f"CRC mismatch on {member}: "
                              f"{got:#010x} != {want:#010x}")
            return blob
        except Exception as error:                       # noqa: BLE001
            last = error
            if i < attempts - 1:
                time.sleep(2 * 2 ** i)
    raise SystemExit(f"failed to fetch {member}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refs/export/boltz2-affinity-f32")
    ap.add_argument("--url", default=AFF_URL)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("reading central directory and tensor table ...", file=sys.stderr)
    size = ckpt_crc.total_size(args.url)
    members = ckpt_crc.members(args.url, size)
    tensors = affinity_tensor_map(args.url, members)
    print(f"  checkpoint {size:,} B; affinity tensors {len(tensors)}",
          file=sys.stderr)

    # --- refuse anything whose member is not simply its own bytes -----------
    problems = []
    for name, rec in sorted(tensors.items()):
        if rec["storage_offset"] != 0:
            problems.append(f"{name}: storage_offset {rec['storage_offset']}")
        if rec["stride"] != contiguous_stride(rec["shape"]):
            problems.append(f"{name}: stride {rec['stride']} is not contiguous "
                            f"for shape {rec['shape']}")
        if rec["storage"] != "FloatStorage":
            problems.append(f"{name}: storage {rec['storage']}, expected FloatStorage")
    if problems:
        for p in problems[:20]:
            print("  REFUSED " + p, file=sys.stderr)
        raise SystemExit(f"{len(problems)} tensors are not plain contiguous "
                         "float32 storages; reading them as such would be a "
                         "different tensor of the right size")
    print("  all tensors: contiguous, storage_offset 0, FloatStorage",
          file=sys.stderr)

    # --- fetch and write ----------------------------------------------------
    writer = ShardWriter(out_dir)
    keep_float32, reasons = [], {}
    fetched = 0
    for name in sorted(tensors):
        rec = tensors[name]
        blob = fetch_with_retry(args.url, members, rec["member"])
        count = int(np.prod(rec["shape"])) if rec["shape"] else 1
        if len(blob) != count * 4:
            raise SystemExit(f"{name}: member is {len(blob)} B, "
                             f"expected {count * 4}")
        values = np.frombuffer(blob, dtype="<f4", count=count).reshape(rec["shape"])
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"{name}: contains non-finite values")
        writer.add(name, values)
        why = keep_float32_reason(name, rec["shape"])
        if why:
            keep_float32.append(name)
            reasons[why] = reasons.get(why, 0) + 1
        fetched += len(blob)
    writer.close()

    parameters = sum(int(np.prod(r["shape"]))
                     for r in writer.records.values())
    manifest = {
        "formatVersion": 1,
        "source": f"Boltz-2 affinity modules from boltz2_aff.ckpt "
                  f"(boltz-community/boltz-2 @ {AFF_REPO_SHA})",
        "model": {"name": "boltz2-affinity", "recycles": 0},
        "bundle": {"purpose": "browser-inference", "model": "boltz2-affinity",
                   "encoding": "float32-le"},
        # What is deliberately NOT here: the trunk, structure, confidence and
        # diffusion scopes. G0 established they are byte-identical to
        # boltz2_conf.ckpt's, which LocalFold's boltz2-int5 bundle already
        # carries, so re-exporting them would be 365 MiB of duplicate download.
        "coverage": {
            "included": ["affinity_module1", "affinity_module2"],
            "excludedScopes": ["pairformer_module", "msa_module",
                               "input_embedder", "structure_module",
                               "confidence_module", "diffusion_conditioning",
                               "distogram_module", "contact_conditioning"],
            "sharedWithBoltz2Bundle":
                "5019 tensors, byte-identical per tools/g0_verify.py",
        },
        # 🔴 LOAD-BEARING: quantize_af3.py's own regex matches none of these
        # dot-separated names. See this file's header.
        "float32Tensors": sorted(keep_float32),
        "tensors": writer.records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    written = sum(p.stat().st_size for p in out_dir.glob("*.f32.bin"))
    print(f"\nfetched  {fetched / 1048576:.2f} MiB of {size / 2**30:.2f} GiB "
          f"({100 * fetched / size:.2f}% of the checkpoint)", file=sys.stderr)
    print(f"wrote    {len(writer.records)} tensors, {parameters:,} parameters, "
          f"{written / 1048576:.2f} MiB float32 across {writer.index} shard(s)",
          file=sys.stderr)
    print(f"kept f32 {len(keep_float32)} tensors: "
          + ", ".join(f"{v} {k}" for k, v in sorted(reasons.items())),
          file=sys.stderr)


if __name__ == "__main__":
    main()
