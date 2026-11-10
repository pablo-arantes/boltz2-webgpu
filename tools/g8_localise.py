"""Where the G8 gap is: compare the affinity module's INPUTS, elementwise.

    refs/venv-boltz/bin/python tools/oracle/dump_boltz2_affinity_inputs.py ...
    node tools/gpu-chrome.mjs tools/gpu/dump-affinity-inputs.js ... > ours.json
    python3 tools/g8_localise.py --upstream=refs/g8-inputs --ours=ours.json

🔴 THE QUESTION. G8 measured this port 0.343 log units from upstream end to
end - 6x upstream's own seed spread - with the module's arithmetic verified to
1e-5..1e-7 against upstream's dumps and the pose measured not to be the cause.
That leaves the inputs. This compares them.

🔴 THE COMPARISON IS ONLY LEGITIMATE WHERE THE CROP IS THE IDENTITY, and both
dumps report whether it is. The trunk reads features rather than coordinates,
so `s_inputs` and `z` are pose-independent given the same token set; the token
set comes from the crop, and the crop comes from the pose. On a target small
enough that nothing is cut (43 of 43 here) both sides featurise the same tokens
in the same order. On a target the crop cuts they do not, and this refuses
rather than reporting a meaningless number.

🔴 AND PAIR ENTRIES ARE COMPARED ONLY WHERE THE CROSS-PAIR MASK IS LIVE. Both
sides hand the module `z * cross_pair_mask`, so every receptor-receptor entry
is an exact zero on both sides by construction - including them would average
hundreds of thousands of matching zeros into the residual and make any real
disagreement look small. This port's own notes record that trap from the other
side: "an unguarded relRMS over a non-array returns exactly 0 - a perfect score
from comparing nothing".
"""
import argparse
import base64
import json
import pathlib

import numpy as np


def rel_rms(a, b):
    """relRMS, the measure every differential in this project uses."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or a.size != b.size:
        raise SystemExit(f"cannot compare {a.size} against {b.size} values")
    scale = np.sqrt(np.mean(b * b))
    if scale == 0:
        scale = 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="the dumper's output directory")
    ap.add_argument("--ours", required=True, help="dump-affinity-inputs.js JSON")
    ap.add_argument("--member", type=int, default=1)
    args = ap.parse_args()

    up = np.load(pathlib.Path(args.upstream)
                 / f"affinity-inputs-member{args.member}.npz")
    raw = pathlib.Path(args.ours).read_text()
    ours = json.loads(raw[raw.index("\n{\n"):])

    def mine(name):
        values = np.frombuffer(base64.b64decode(ours["data"][name]),
                               dtype=np.float32)
        return values.reshape(ours["tensors"][name]["shape"])

    if not ours["cropIsIdentity"]:
        raise SystemExit("this port's crop is not the identity on this target,"
                         " so the two sides' token sets differ with their poses"
                         " and these tensors are not comparable elementwise")
    n = ours["tokens"]
    if up["s_inputs"].shape[1] != n:
        raise SystemExit(f"upstream has {up['s_inputs'].shape[1]} tokens and"
                         f" this port has {n}: not the same problem")
    print(f"tokens            {n} on both sides, crop is the identity")
    print(f"upstream member   {args.member} ({int(up['layers'][0])} pairformer layers)")
    print()

    # --- the features, first: a difference here explains everything after it
    pairs = [("mol_type", "feats.mol_type", "mol_type"),
             ("affinity_token_mask", "feats.affinity_token_mask",
              "affinity_token_mask")]
    for label, upName, ourName in pairs:
        u = np.asarray(up[upName][0]).ravel()
        o = mine(ourName).ravel()
        differ = int(np.count_nonzero(u != o))
        print(f"{label:22s} {differ} of {u.size} differ"
              + ("" if differ == 0 else f"   upstream {u.tolist()}\n"
                 f"{'':22s}   ours     {o.tolist()}"))

    # --- s_inputs: pose-independent, so a plain elementwise comparison
    u_s = np.asarray(up["s_inputs"][0])              # (n, 384)
    o_s = mine("s_inputs")
    print()
    print(f"s_inputs          upstream {u_s.shape}  ours {o_s.shape}")
    print(f"  rms             upstream {np.sqrt(np.mean(u_s**2)):.5f}"
          f"   ours {np.sqrt(np.mean(o_s**2)):.5f}")
    print(f"  relRMS          {rel_rms(o_s, u_s):.4e}")
    # 🔴 PER TOKEN KIND, because the ligand is what the affinity module is
    # about and 10 of 43 tokens could be wholly wrong inside a respectable
    # whole-tensor average.
    lig = mine("affinity_token_mask").ravel() > 0
    print(f"  relRMS protein  {rel_rms(o_s[~lig], u_s[~lig]):.4e}"
          f"   ({int((~lig).sum())} tokens)")
    print(f"  relRMS ligand   {rel_rms(o_s[lig], u_s[lig]):.4e}"
          f"   ({int(lig.sum())} tokens)")

    # --- the summands, because `s_inputs` is a SUM and 9.0e-1 on the total
    # says nothing about which term carries it.
    #
    # 🔴 THE SPLIT IS TWO-WAY ON PURPOSE: the atom-encoder term `a`, and
    # everything else (`s - a`: the restype one-hot, the profile, the deletion
    # mean and boltz2's four conditionings). Two terms either side, computed by
    # SUBTRACTION from each side's own total rather than re-summed here, so the
    # split cannot introduce arithmetic of its own.
    terms = pathlib.Path(args.upstream) / "s-inputs-terms.npz"
    if terms.exists() and "a_atom_encoder" in ours["tensors"]:
        up_terms = np.load(terms)
        u_a = np.asarray(up_terms["a_atom_encoder"])
        u_a = u_a.reshape(-1, u_a.shape[-1])
        u_rest = u_s - u_a
        o_a = mine("a_atom_encoder")
        o_rest = mine("s_minus_a")
        print()
        print("s_inputs, term by term")
        print(f"  a (atom encoder)      rms upstream {np.sqrt(np.mean(u_a**2)):.5f}"
              f"   ours {np.sqrt(np.mean(o_a**2)):.5f}"
              f"   relRMS {rel_rms(o_a, u_a):.4e}")
        print(f"    ... ligand tokens                              "
              f"            relRMS {rel_rms(o_a[lig], u_a[lig]):.4e}")
        print(f"  s - a (everything else) rms upstream {np.sqrt(np.mean(u_rest**2)):.5f}"
              f"   ours {np.sqrt(np.mean(o_rest**2)):.5f}"
              f"   relRMS {rel_rms(o_rest, u_rest):.4e}")
        # and the profile the affinity branch actually reads, which is a
        # DIFFERENT feature key upstream (`profile_affinity`)
        if "feats.profile_affinity" in up and "profile_affinity" in ours["tensors"]:
            u_p = np.asarray(up["feats.profile_affinity"][0])
            o_p = mine("profile_affinity")
            print(f"  profile_affinity      upstream {u_p.shape}  ours {o_p.shape}")
            shared = min(u_p.shape[1], o_p.shape[1])
            print(f"    first {shared} columns                            "
                  f"      relRMS {rel_rms(o_p[:, :shared], u_p[:, :shared]):.4e}")
            extra = np.abs(u_p[:, shared:]).max() if u_p.shape[1] > shared else 0.0
            print(f"    upstream's extra {u_p.shape[1] - shared} columns: max |v| {extra:.3e}"
                  "   (nonzero would mean this port cannot represent them)")

    # --- z, masked exactly as both sides mask it
    u_z = np.asarray(up["z"][0])                     # (n, n, 128)
    o_z = mine("z")
    live = mine("cross_pair_mask") > 0
    print()
    print(f"z                 upstream {u_z.shape}  ours {o_z.shape}")
    print(f"  live pairs      {int(live.sum())} of {live.size}")
    print(f"  rms (live)      upstream {np.sqrt(np.mean(u_z[live]**2)):.5f}"
          f"   ours {np.sqrt(np.mean(o_z[live]**2)):.5f}")
    print(f"  relRMS (live)   {rel_rms(o_z[live], u_z[live]):.4e}")
    # the masked-out half must be zero on both sides; if it is not, the
    # pre-mask convention differs and that is a finding of its own
    dead = ~live
    print(f"  masked half     upstream max |z| {np.max(np.abs(u_z[dead])) if dead.any() else 0:.3e}"
          f"   ours {np.max(np.abs(o_z[dead])) if dead.any() else 0:.3e}")

    print()
    verdict = {
        "s_inputs_relRMS": rel_rms(o_s, u_s),
        "s_inputs_ligand_relRMS": rel_rms(o_s[lig], u_s[lig]),
        "z_live_relRMS": rel_rms(o_z[live], u_z[live]),
    }
    print(json.dumps(verdict, indent=2))
    # 🔴 NO PASS/FAIL HERE ON A BOUND NOBODY HAS MEASURED. What these numbers
    # are for is attribution: whichever of the two is large is where the 0.343
    # log-unit gap comes from, and a threshold invented now would only dress
    # that judgement up as a test.


if __name__ == "__main__":
    main()
