"""Upstream Boltz-2's AffinityModule, stage by stage, as the reference G2 is held to.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
      python3 tools/oracle/dump_boltz2_affinity.py \
        --boltz=<checkout of jwohlwend/boltz> \
        --f32=<boltz2-affinity-f32 export> \
        --out=oracle-dumps/boltz2-affinity

🔴 THE REAL MODULE RUNS, AND THE TAPS ARE HOOKS. Nothing here re-implements
AffinityModule.forward: it is instantiated from upstream's own source,
`torch.nn.Module` forward hooks record each seam, and a re-implementation that
drifted from upstream could not make this dump wrong - which is the whole point
of calling it an oracle.

🔴 AND THE WEIGHTS COME FROM OUR EXPORT, WHICH IS ONLY SOUND BECAUSE
check-affinity-export-fidelity.py ESTABLISHED THAT THE EXPORT IS THE
CHECKPOINT. check-bundle-vs-params.py's header names the trap this would
otherwise be: feeding ONE bundle to both sides makes a tensor at the wrong name
invisible, because both forwards read it and both are wrong together. That gate
compares every one of the 520 tensors' bytes, at the manifest's own offsets,
against the checkpoint's ZIP CRC-32 - so the export's fidelity is established
INDEPENDENTLY of this dump, and loading from it here costs nothing.

🔴 THE ORACLE IS FLOAT32, AND NOT BY CHOICE. Both triangle multiplications
in upstream's `layers/triangular_mult.py` do
`a, b = torch.chunk(x.float(), 2, dim=-1)` before the O(L^3) einsum - a
deliberate precision decision, commented "Split input and cast to float" - so
the projection is f32 whatever the surrounding dtype is. A `.double()` module
therefore dies in the very next line, where `norm_out`'s double parameters meet
an f32 activation. There is no float64 arm to be had without editing upstream,
which would stop it being the oracle.

So the dump is f32, which is what upstream actually computes, and the f64
reference G2 compares against it cannot agree better than f32 accumulation
through eight triangle layers allows. docs/ records the same shape for the AF3
denoise checker: it "holds the GPU to the ORACLE and the CPU to the GPU,
because boltz2's token transformer amplifies its input by ~2.2e4 and this f64
reference cannot agree with an f32 oracle to better than 1e-2 however right the
port is". The gate therefore carries a SEPARATION CONTROL: a bound that cannot
distinguish a correct port from a broken one is not a bound.

🔴 AND THIS IS A WIRING AND ARITHMETIC GATE, NOT A GEOMETRY ONE. docs records
the AF3 denoise oracle passing at 1.30e-5 while side chains folded at 6.7x the
reference's bond error, "because both sides read the same wrong tensor and
neither needs real geometry to agree about it". So what this dump can say is
that the port computes upstream's FUNCTION of its inputs. What it cannot say is
that a real fold's trunk produces inputs like these. The inputs are
nevertheless built to exercise the parts that a careless random draw would
miss:

  * COORDINATES SPREAD so the distogram's 64 bins are actually used, and the
    bin histogram is written into the dump. A draw that put every distance in
    the top bin would test one row of a 64-row embedding table.
  * NO DISTANCE WITHIN `--margin` OF A BIN EDGE. The bucketing is a strict `>`
    against 63 boundaries, and upstream reaches it through `torch.cdist`, whose
    matmul expansion of |a-b|^2 is accurate to ~1e-14 off-diagonal and 4.8e-07
    ON it. A distance sitting on an edge would bucket differently between the
    two arithmetics and read as a port defect. The dump records the achieved
    margin so an unlucky input says so instead of failing mysteriously.
  * A PROTEIN CHAIN AND A LIGAND CHAIN, with real masks, because all three of
    the module's masks are built from `mol_type` and `affinity_token_mask` and
    a single-chain input makes two of them identical.
  * PADDING, because `token_pad_mask` multiplies into every mask and an
    all-ones draw cannot see a pad bug.

Both ensemble members are dumped: module1 has 8 pairformer layers and module2
has 4, so a port that hard-coded a depth passes one and fails the other.

Seams are written as raw little-endian float64 beside an index.json, because
JSON floats would be both lossy and enormous: the pair seam alone is
tokens^2 x 128.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                             # noqa: E402

TOKEN_S = 384
TOKEN_Z = 128
NUM_DIST_BINS = 64
MAX_DIST = 22


def build_input(protein: int, ligand: int, padded: int, seed: int,
                margin: float, boundaries: np.ndarray):
    """A deterministic, realistic-shaped input that exercises the bins.

    Returns float64 numpy arrays; the caller casts for torch.
    """
    rng = np.random.default_rng(seed)
    real = protein + ligand
    tokens = real + padded

    mol_type = np.zeros(tokens, np.int64)
    mol_type[protein:real] = 3                 # NONPOLYMER; PROTEIN is 0
    mol_type[real:] = 0                        # padded slots carry restype 0
    affinity_token_mask = np.zeros(tokens, np.int64)
    affinity_token_mask[protein:real] = 1      # the binder chain's tokens
    token_pad_mask = np.zeros(tokens, np.float64)
    token_pad_mask[:real] = 1.0

    # One representative atom a token, as token_to_rep_atom gathers: a
    # one-hot [tokens, atoms]. Upstream multiplies it as a float matrix.
    atoms = tokens
    token_to_rep_atom = np.eye(tokens, atoms, dtype=np.float64)

    # Coordinates: a protein blob of radius ~14 A with the ligand inside it, so
    # pair distances cover 2-22 A rather than piling into one bin.
    for attempt in range(200):
        draw = np.random.default_rng(seed + attempt)
        prot = draw.normal(0.0, 8.0, (protein, 3))
        lig = draw.normal(0.0, 3.0, (ligand, 3)) + np.array([3.0, 1.0, -2.0])
        pad = draw.normal(0.0, 8.0, (padded, 3))
        x = np.concatenate([prot, lig, pad], 0)
        diff = x[:, None, :] - x[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))       # computed DIRECTLY, not by matmul
        off = ~np.eye(tokens, dtype=bool)
        gap = np.abs(d[off][:, None] - boundaries[None, :]).min()
        bins = (d[..., None] > boundaries).sum(-1)
        covered = len(np.unique(bins))
        if gap >= margin and covered >= 40:
            return {
                "x": x, "d": d, "bins": bins, "margin": float(gap),
                "binsCovered": int(covered),
                "mol_type": mol_type,
                "affinity_token_mask": affinity_token_mask,
                "token_pad_mask": token_pad_mask,
                "token_to_rep_atom": token_to_rep_atom,
                "s_inputs": draw.normal(0.0, 0.6, (real + padded, TOKEN_S)),
                "z_trunk": draw.normal(0.0, 0.4, (tokens, tokens, TOKEN_Z)),
                "attempt": attempt,
            }
    raise SystemExit(
        f"no draw in 200 attempts kept every distance {margin} A from a bin edge"
        " with 40+ bins covered; widen --margin or change --seed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz", required=True)
    parser.add_argument("--f32", required=True)
    parser.add_argument("--out", default="oracle-dumps/boltz2-affinity")
    parser.add_argument("--protein", type=int, default=24)
    parser.add_argument("--ligand", type=int, default=8)
    parser.add_argument("--padded", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--margin", type=float, default=1e-4)
    arguments = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(arguments.boltz).resolve() / "src"))
    import torch                                               # noqa: E402
    from boltz.model.modules.affinity import AffinityModule     # noqa: E402

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    export = pathlib.Path(arguments.f32)
    manifest = json.loads((export / "manifest.json").read_text())
    blobs = {f: (export / f).read_bytes()
             for f in {r["file"] for r in manifest["tensors"].values()}}

    def tensor(name: str) -> np.ndarray:
        record = manifest["tensors"][name]
        count = int(np.prod(record["shape"])) if record["shape"] else 1
        flat = np.frombuffer(blobs[record["file"]], "<f4", count=count,
                             offset=record["byteOffset"])
        return flat.reshape(record["shape"]).astype(np.float64)

    boundaries = np.linspace(2, MAX_DIST, NUM_DIST_BINS - 1)
    built = build_input(arguments.protein, arguments.ligand, arguments.padded,
                        arguments.seed, arguments.margin, boundaries)
    tokens = built["z_trunk"].shape[0]
    print(f"input: {arguments.protein} protein + {arguments.ligand} ligand"
          f" + {arguments.padded} padded = {tokens} tokens", file=sys.stderr)
    print(f"  bins covered {built['binsCovered']}/64,"
          f" closest distance to a bin edge {built['margin']:.3e} A"
          f" (draw {built['attempt']})", file=sys.stderr)

    out_dir = pathlib.Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}

    def write(name: str, array) -> None:
        a = np.ascontiguousarray(np.asarray(array, np.float64))
        path = out_dir / f"{name}.f64.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(a.tobytes())
        index[name] = {"file": f"{name}.f64.bin", "shape": list(a.shape)}

    # the inputs, so the port reads exactly what the oracle read
    for key in ("s_inputs", "z_trunk", "x", "d", "token_to_rep_atom",
                "token_pad_mask"):
        write(f"input/{key}", built[key])
    for key in ("mol_type", "affinity_token_mask", "bins"):
        write(f"input/{key}", built[key].astype(np.float64))

    for which, layers in (("affinity_module1", 8), ("affinity_module2", 4)):
        module = AffinityModule(
            token_s=TOKEN_S, token_z=TOKEN_Z,
            pairformer_args={"num_blocks": layers, "dropout": 0.25,
                             "activation_checkpointing": False},
            transformer_args={"num_blocks": 12, "num_heads": 8,
                              "token_s": TOKEN_S,
                              "activation_checkpointing": False},
            num_dist_bins=NUM_DIST_BINS, max_dist=MAX_DIST,
        ).float().eval()

        state = module.state_dict()
        loaded, absent = 0, []
        for key in state:
            full = f"{which}.{key}"
            if full in manifest["tensors"]:
                state[key] = torch.from_numpy(
                    tensor(full).astype(np.float32).copy())
                loaded += 1
            else:
                absent.append(key)
        if absent:
            # 🔴 A MISSING WEIGHT MUST STOP THIS, not default to the random
            # initialisation - a module that is 90% loaded produces a
            # plausible dump the port can never match.
            for key in absent[:10]:
                print(f"  ABSENT {which}.{key}", file=sys.stderr)
            raise SystemExit(f"{len(absent)} weights absent from the export")
        module.load_state_dict(state, strict=True)
        print(f"{which}: {loaded} weights loaded, {layers} pairformer layers",
              file=sys.stderr)

        taps: dict[str, np.ndarray] = {}

        def record(tag):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                taps[tag] = value.detach().numpy().copy()
            return hook

        def record_inputs(tag):
            def hook(_module, inputs, kwargs=None):
                for i, value in enumerate(inputs):
                    if torch.is_tensor(value):
                        taps[f"{tag}/in{i}"] = value.detach().numpy().copy()
                for key, value in (kwargs or {}).items():
                    if torch.is_tensor(value):
                        taps[f"{tag}/{key}"] = value.detach().numpy().copy()
            return hook

        handles = [
            module.z_norm.register_forward_hook(record("z_norm")),
            module.z_linear.register_forward_hook(record("z_linear")),
            module.s_to_z_prod_in1.register_forward_hook(record("s_to_z_1")),
            module.s_to_z_prod_in2.register_forward_hook(record("s_to_z_2")),
            module.dist_bin_pairwise_embed.register_forward_hook(
                record("distogram_embed")),
            # the conditioner's own inputs ARE z-after-the-s-sum and the
            # distogram, so this one pre-hook pins both seams
            module.pairwise_conditioner.register_forward_pre_hook(
                record_inputs("conditioner"), with_kwargs=True),
            module.pairwise_conditioner.register_forward_hook(
                record("conditioner_out")),
            module.affinity_heads.affinity_out_mlp.register_forward_pre_hook(
                record_inputs("pooled")),
            module.affinity_heads.affinity_out_mlp.register_forward_hook(
                record("out_mlp")),
            module.affinity_heads.to_affinity_pred_value.register_forward_hook(
                record("pred_value")),
            module.affinity_heads.to_affinity_pred_score.register_forward_hook(
                record("pred_score")),
            module.affinity_heads.to_affinity_logits_binary.register_forward_hook(
                record("logits_binary")),
        ]
        for i, layer in enumerate(module.pairformer_stack.layers):
            handles.append(layer.register_forward_hook(record(f"pairformer{i:02d}")))

        feats = {
            "token_to_rep_atom": torch.from_numpy(
                built["token_to_rep_atom"][None].astype(np.float32)),
            "token_pad_mask": torch.from_numpy(
                built["token_pad_mask"][None].astype(np.float32)),
            "mol_type": torch.from_numpy(built["mol_type"][None]),
            "affinity_token_mask": torch.from_numpy(
                built["affinity_token_mask"][None]),
        }
        # upstream pre-masks the trunk pair before handing it over
        rec = (built["mol_type"] == 0) * built["token_pad_mask"]
        lig = (built["affinity_token_mask"] > 0) * built["token_pad_mask"]
        cross = (lig[:, None] * rec[None, :] + rec[:, None] * lig[None, :]
                 + lig[:, None] * lig[None, :])
        z_affinity = built["z_trunk"] * cross[:, :, None]
        write("input/cross_pair_mask", cross)
        write("input/z_affinity", z_affinity)

        result = module(
            s_inputs=torch.from_numpy(
                built["s_inputs"][None].astype(np.float32)),
            z=torch.from_numpy(z_affinity[None].astype(np.float32)),
            # 🔴 FLOAT32 ON PURPOSE, AND ONLY HERE. Upstream's forward does
            # `torch.bmm(token_to_rep_atom.float(), x_pred)`, which forces f32
            # and refuses a double x_pred outright. That confines f32 to two
            # steps - the one-hot representative-atom gather (a selection, so
            # exact either way) and `torch.cdist` - while the embedding lookup
            # and everything after it stay float64, because `boundaries` and
            # the parameters are double.
            #
            # It cannot move a bin: f32 costs ~2.4e-6 A on a 20 A distance and
            # `build_input` keeps every distance at least `--margin` (8.7e-4 A
            # by default) from a bin edge, which is ~360x the error. The gate
            # asserts the bins agree EXACTLY rather than trusting that sum.
            x_pred=torch.from_numpy(built["x"][None, None]).float(),
            feats=feats, multiplicity=1, use_kernels=False,
        )
        for handle in handles:
            handle.remove()

        for tag, value in taps.items():
            write(f"{which}/{tag}", value)
        for key, value in result.items():
            write(f"{which}/out/{key}", value.detach().numpy())
        write(f"{which}/out/affinity_probability_binary",
              torch.sigmoid(result["affinity_logits_binary"]).numpy())

    (out_dir / "index.json").write_text(json.dumps({
        "source": f"upstream boltz AffinityModule, torch {torch.__version__}",
        "precision": "float32 - upstream's triangle einsum forces it; see the dumper header",
        "note": "a WIRING and ARITHMETIC reference, not a geometry one;"
                " see this dumper's header",
        "tokens": tokens,
        "protein": arguments.protein, "ligand": arguments.ligand,
        "padded": arguments.padded, "seed": arguments.seed,
        "tokenS": TOKEN_S, "tokenZ": TOKEN_Z,
        "numDistBins": NUM_DIST_BINS, "maxDist": MAX_DIST,
        "binsCovered": built["binsCovered"],
        "binEdgeMargin": built["margin"],
        "layers": {"affinity_module1": 8, "affinity_module2": 4},
        "tensors": index,
    }, indent=1) + "\n")
    total = sum(int(np.prod(v["shape"])) for v in index.values())
    print(f"\nwrote {out_dir}: {len(index)} tensors, {total:,} float64 values",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
