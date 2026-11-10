"""Capture the REAL inputs upstream hands its affinity module, on a real job.

    refs/venv-boltz/bin/python tools/oracle/dump_boltz2_affinity_inputs.py \
      --yaml=refs/g8-out-5/g8.yaml --cache=refs/boltz-cache \
      --out=refs/g8-inputs --seed=0

🔴 WHY THIS EXISTS AND WHY dump_boltz2_affinity.py IS NOT ENOUGH. That dumper
feeds upstream's AffinityModule SYNTHETIC tensors - `rng.normal(...)` - and
records every seam, which is what established that this port's module computes
upstream's function to ~1e-7. G8 then measured a 0.343 log-unit gap end to end,
six times upstream's own seed-to-seed spread, with the module's arithmetic
already excluded at eight orders of magnitude. So the fault is in what the
module is GIVEN, and nothing had ever captured that from a real run.

🔴 WHAT IS CAPTURED IS THE CALL SITE'S OWN KWARGS, NOT A RECONSTRUCTION.
`boltz2.py` calls

    affinity_module1(s_inputs=s_inputs.detach(), z=z_affinity.detach(),
                     x_pred=coords_affinity, feats=feats, multiplicity=1, ...)

and this wraps `AffinityModule.forward` to write those four out before calling
through. A reconstruction - building `z * cross_pair_mask` here from a saved
trunk output - could differ from what the module actually received, which is
the whole thing being tested.

🔴 AND BOTH MEMBERS ARE CAPTURED SEPARATELY. G8 found the two ensemble members
wrong in OPPOSITE directions (member 2's value out by 0.88 log units, member
1's probability by a factor of 4.6) with the mean cancelling most of it, so a
dump that merged them would hide exactly the signal.

🔴 `s_inputs` AND `z` ARE POSE-INDEPENDENT ON A TARGET THE CROP DOES NOT CUT,
which is what makes an elementwise comparison legitimate at all. The trunk
reads features, not coordinates; `s_inputs` is the input embedder over the same
features. On the G8 target the crop keeps 43 of 43 tokens - it is the identity -
so both sides featurise the same tokens in the same order and the only
pose-dependent input is `x_pred`. On a target the crop DOES cut, the two sides'
crops differ with their poses and these tensors are not comparable elementwise;
the dump records the token count so that condition can be checked rather than
assumed.
"""
import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# 🔴 FOUR COPIES OF libomp ON THIS BOX ABORT torch ON IMPORT, and the offered
# escape hatch warns of "silently incorrect results" - so the threads are
# pinned to one above rather than duplicate loading being allowed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--affinity-steps", type=int, default=200)
    ap.add_argument("--affinity-samples", type=int, default=5)
    # 🔴 THE int5 FLOOR FOR `z`, MEASURED INSIDE UPSTREAM. This port's trunk is
    # a 5-bit bundle, and `z` is 4 MSA blocks plus 48 pairformer blocks deep
    # where `s_inputs` is one layer - so a weight error of a few percent can
    # arrive at the affinity module as tens of percent, and the only honest way
    # to know is to put the SAME quantisation into upstream's own arithmetic
    # and compare its z against its own f32 z. Nothing of this port is involved,
    # which is the point: it separates "the bundle's precision" from "the port".
    ap.add_argument("--quantise-trunk", action="store_true",
                    help="round-trip every quantisable trunk tensor through"
                         " int5 asymmetric group-32 with f16 scale and zero -"
                         " LocalFold's own scheme - before running")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from boltz.main import predict
    from boltz.model.modules.affinity import AffinityModule

    original = AffinityModule.forward
    seen = {"n": 0}

    def capture(self, *call_args, **kwargs):
        # The call site passes everything by keyword; positional would mean
        # upstream changed its call and this dump would be of the wrong thing.
        if call_args:
            raise SystemExit("upstream now calls AffinityModule positionally -"
                             " this dumper reads kwargs and must be updated")
        seen["n"] += 1
        index = seen["n"]
        record = {}
        for name in ("s_inputs", "z", "x_pred"):
            tensor = kwargs.get(name)
            if tensor is not None:
                record[name] = tensor.detach().float().cpu().numpy()
        feats = kwargs.get("feats", {})
        # 🔴 `profile_affinity` AND `deletion_mean_affinity` ARE SEPARATE
        # FEATURES, not the ordinary ones modified. `InputEmbedder.forward`
        # branches on its `affinity` flag and reads those two keys instead -
        # so a dump that captured only `profile`/`deletion_mean`, as the first
        # version of this file did, recorded the features the affinity pass
        # does NOT use.
        for name in ("token_pad_mask", "mol_type", "affinity_token_mask",
                     "res_type", "token_index", "asym_id", "entity_id",
                     "sym_id", "residue_index", "profile", "deletion_mean",
                     "profile_affinity", "deletion_mean_affinity",
                     "method_feature", "modified", "cyclic_period",
                     "token_to_rep_atom", "atom_pad_mask",
                     # 🔴 THE REFERENCE CONFORMER, which is what is left after
                     # the method fix: `a` (the atom encoder) still disagrees
                     # at relRMS 1.16 on LIGAND tokens and 4.4e-2 on protein,
                     # and `perAtomConditioning` reads these four directly. A
                     # conformer that is a different shape - or the same shape
                     # in a different FRAME, since the positions go through a
                     # linear layer rather than a distance - changes the
                     # embedding.
                     "ref_pos", "ref_element", "ref_charge",
                     "ref_atom_name_chars", "ref_space_uid", "atom_to_token",
                     "atom_name",
                     # 🔴 THE TWO BOND PLANES, because `z_init`'s residual is
                     # 10.7x the int5 floor on exactly the 20 BONDED pairs and
                     # these are the only summands supported there. `token_bonds`
                     # is the 0/1 adjacency and `type_bonds` the code per pair
                     # (const.py: OTHER 0, SINGLE 1, DOUBLE 2, TRIPLE 3,
                     # AROMATIC 4, COVALENT 5), written symmetrically.
                     "token_bonds", "type_bonds"):
            value = feats.get(name)
            if value is not None and hasattr(value, "detach"):
                record[f"feats.{name}"] = value.detach().float().cpu().numpy()
        # 🔴 THE LAYER COUNT IDENTIFIES WHICH MEMBER THIS IS, from the module
        # itself rather than from the call order: member 1 has 8 pairformer
        # layers and member 2 has 4, so a dump cannot be mislabelled by a
        # change in the order boltz2.py calls them.
        record["layers"] = np.array(
            [len(self.pairformer_stack.layers)], dtype=np.int32)
        path = out / f"affinity-inputs-member{index}.npz"
        np.savez(path, **record)
        print(f"captured member {index}"
              f" ({record['layers'][0]} pairformer layers) -> {path}", flush=True)
        for key, value in record.items():
            print(f"  {key:28s} {tuple(value.shape)}")
        return original(self, **kwargs)

    # --- the target_feat / s_inputs summands, one module deeper -----------
    #
    # 🔴 `s_inputs` DISAGREES AT relRMS 9.0e-1 AND IT IS A SUM, so the useful
    # question is which summand. upstream's InputEmbedder.forward is
    #
    #     s = a                                           (atom encoder)
    #       + res_type_encoding(res_type)
    #       + msa_profile_encoding(cat([profile, deletion_mean]))
    #       + method/modified/cyclic/mol_type conditionings
    #
    # so each submodule's OUTPUT is recorded by a forward hook. Hooks rather
    # than a re-implementation: recomputing `a` here from saved weights would
    # compare this script's arithmetic with upstream's rather than upstream's
    # with the port's.
    from boltz.model.modules.trunkv2 import InputEmbedder

    terms = {}

    def term_hook(label):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if hasattr(tensor, "detach"):
                terms[label] = tensor.detach().float().cpu().numpy()
        return hook

    embedder_calls = {"n": 0}
    original_embedder = InputEmbedder.forward

    def embedder_capture(self, feats, affinity=False):
        # 🔴 ONLY THE affinity=True CALL. The same embedder runs for the
        # structure pass with the ordinary profile, and recording that one
        # would overwrite the affinity call's terms with a different tensor
        # under the same name.
        if not affinity:
            return original_embedder(self, feats, affinity=affinity)
        embedder_calls["n"] += 1
        handles = []
        for label, module in (
            ("a_atom_encoder", getattr(self, "atom_attention_encoder", None)),
            ("res_type_term", getattr(self, "res_type_encoding", None)),
            ("profile_term", getattr(self, "msa_profile_encoding", None)),
            ("method_term", getattr(self, "method_conditioning_init", None)),
            ("modified_term", getattr(self, "modified_conditioning_init", None)),
            ("cyclic_term", getattr(self, "cyclic_conditioning_init", None)),
            ("mol_type_term", getattr(self, "mol_type_conditioning_init", None)),
        ):
            if module is not None:
                handles.append(module.register_forward_hook(term_hook(label)))
        try:
            out = original_embedder(self, feats, affinity=affinity)
        finally:
            for handle in handles:
                handle.remove()
        terms["s_inputs"] = out.detach().float().cpu().numpy()
        np.savez(out_path_terms, **terms)
        print(f"captured {len(terms)} s_inputs terms -> {out_path_terms}",
              flush=True)
        for key in sorted(terms):
            print(f"  {key:22s} {tuple(terms[key].shape)}")
        return out

    out_path_terms = out / "s-inputs-terms.npz"
    InputEmbedder.forward = embedder_capture
    AffinityModule.forward = capture

    # 🔴 THE SCHEME IS THE BUNDLE'S OWN, READ OFF ITS MANIFEST: asymmetric per
    # group of 32, 5 bits, scale and zero both float16. Norms and biases stay
    # float32, which is what `float32Tensors` holds and what quantize_af3.py's
    # own header calls "the worst thing to group-quantise".
    def int5_round_trip(x):
        flat = x.detach().reshape(-1).float()
        pad = (-flat.numel()) % 32
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        groups = flat.reshape(-1, 32)
        lo = groups.min(dim=1, keepdim=True).values.half().float()
        hi = groups.max(dim=1, keepdim=True).values.half().float()
        scale = ((hi - lo) / 31.0).half().float()
        safe = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = torch.clamp(torch.round((groups - lo) / safe), 0, 31)
        back = (codes * scale + lo).reshape(-1)[: x.numel()]
        return back.reshape(x.shape).to(x.dtype)

    if args.quantise_trunk:
        from boltz.model.models.boltz2 import Boltz2
        # 🔴 LIGHTNING'S `load_from_checkpoint` IS A PLAIN FUNCTION ON THE
        # CLASS, NOT A CLASSMETHOD DESCRIPTOR, so `.__func__` does not exist
        # and the replacement must not be wrapped in `classmethod` either.
        original_load = Boltz2.load_from_checkpoint

        # the trunk only - the affinity modules are exported from this very
        # checkpoint at float32 and are not what is under test
        TRUNK = ("input_embedder", "s_init", "z_init_1", "z_init_2", "rel_pos",
                 "token_bonds", "token_bonds_type", "contact_conditioning",
                 "s_norm", "z_norm", "s_recycle", "z_recycle", "msa_module",
                 "pairformer_module")

        def quantised_load(*a, **kw):
            model = original_load(*a, **kw)
            touched = skipped = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not name.startswith(TRUNK):
                        continue
                    leaf = name.split(".")[-1]
                    if leaf == "bias" or "norm" in name.lower() or param.dim() < 2:
                        skipped += 1
                        continue
                    param.copy_(int5_round_trip(param))
                    touched += 1
            print(f"quantised {touched} trunk tensors to int5,"
                  f" kept {skipped} at float32", flush=True)
            if touched == 0:
                raise SystemExit("no trunk tensor was quantised - the arm did"
                                 " not reach the weights")
            return model

        Boltz2.load_from_checkpoint = quantised_load

    # 🔴 THE TRUNK'S OWN STAGES, BECAUSE `z` IS 6.6x THE int5 FLOOR AND SO THE
    # GAP IS REAL. Measured: quantising upstream's own trunk to this port's
    # 5-bit scheme moves its `z` by 3.04e-2 where this port is 2.00e-1 away, and
    # `s_inputs` is AT the floor - so the disagreement enters after the input
    # embedder and before the module. Each summand of `z_init` and each
    # recycle's stage output is a submodule, so a forward hook records it; the
    # last call wins, which is the recycle the module is handed.
    from boltz.model.models.boltz2 import Boltz2 as _B2

    stages = {}

    def stage_hook(label, index=None):
        def hook(_module, _inputs, output):
            tensor = output[index] if index is not None else output
            tensor = tensor[0] if isinstance(tensor, tuple) else tensor
            if hasattr(tensor, "detach"):
                stages[label] = tensor.detach().float().cpu().numpy()
        return hook

    _loaded = {"n": 0}
    _prev_load = Boltz2_load = getattr(_B2, "load_from_checkpoint")

    def staged_load(*a, **kw):
        model = _prev_load(*a, **kw)
        # 🔴 THE AFFINITY STAGE'S MODEL, NOT THE STRUCTURE STAGE'S. `boltz
        # predict` runs two `trainer.predict` calls with two checkpoints, and
        # only `boltz2_aff.ckpt` carries `affinity_module1` - which is also why
        # it has no `template_module` and runs no template stage at all.
        if not hasattr(model, "affinity_module1"):
            return model
        _loaded["n"] += 1
        for label, attr, idx in (
            ("z_init_1", "z_init_1", None),
            ("z_init_2", "z_init_2", None),
            ("rel_pos", "rel_pos", None),
            ("token_bonds", "token_bonds", None),
            ("token_bonds_type", "token_bonds_type", None),
            ("contact_conditioning", "contact_conditioning", None),
            ("z_recycle", "z_recycle", None),
            ("msa_module", "msa_module", None),
            ("pairformer_z", "pairformer_module", 1),
        ):
            module = getattr(model, attr, None)
            if module is not None:
                module.register_forward_hook(stage_hook(label, idx))
        return model

    setattr(_B2, "load_from_checkpoint", staged_load)

    torch.manual_seed(args.seed)

    argv = [
        str(pathlib.Path(args.yaml).resolve()),
        "--out_dir", str(out / "predict"),
        "--cache", str(pathlib.Path(args.cache).resolve()),
        "--accelerator", "cpu",
        "--seed", str(args.seed),
        "--recycling_steps", str(args.recycles),
        "--sampling_steps", str(args.steps),
        "--diffusion_samples", str(args.samples),
        "--sampling_steps_affinity", str(args.affinity_steps),
        "--diffusion_samples_affinity", str(args.affinity_samples),
        "--output_format", "pdb",
        "--num_workers", "0",
    ]
    # 🔴 THE CLICK COMMAND, CALLED DIRECTLY. Running the CLI as a subprocess
    # would put the patch in the wrong process; `standalone_mode=False` keeps
    # click from calling sys.exit on the way out so this script can report.
    predict.main(argv, standalone_mode=False)
    if embedder_calls["n"] == 0:
        raise SystemExit("the input embedder was never called with"
                         " affinity=True - nothing to compare term by term")
    if seen["n"] == 0:
        raise SystemExit("the affinity module was never called - did the job"
                         " carry `properties: affinity: binder:`?")
    if stages:
        np.savez(out / "z-stages.npz", **stages)
        print(f"\ncaptured {len(stages)} trunk stages -> {out / 'z-stages.npz'}")
        for key in sorted(stages):
            print(f"  {key:22s} {tuple(stages[key].shape)}")
    print(f"\n{seen['n']} member call(s) captured in {out}")


if __name__ == "__main__":
    main()
