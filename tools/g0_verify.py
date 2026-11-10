"""G0 - are boltz2_aff.ckpt's shared tensors the same weights as boltz2_conf.ckpt's?

    python3 -I tools/g0_verify.py --out spec/G0-RESULT.md            # full (needs network)
    python3 -I tools/g0_verify.py --offline --aff <dir> --conf <dir>  # structural half only

WHY THIS GATE EXISTS. The whole Phase 2 plan rests on the claim that LocalFold's
existing boltz2 bundle already carries the weights the affinity pass's trunk
needs - i.e. that the 5,019 tensor names the two checkpoints share hold
IDENTICAL values. A sample of 33 supported it. A sample is not the claim.

If one shared tensor differs, the affinity pass needs its own ~500 M-parameter
trunk export and the plan's size, cost and registry sections all change. So this
runs before anything is built on it.

HOW, WITHOUT MOVING 4 GB. torch.save stores tensor payloads UNCOMPRESSED, so
every member's CRC-32 is already in the ZIP central directory. Two central
directories plus two data.pkl files is a few MB and covers every tensor.
CRC-32 is a checksum, not a hash: the chance two DIFFERENT payloads collide is
about 2^-32 each, so over ~5,000 tensors a false "same" runs at ~1.2e-6. That is
reported, not hidden - and any tensor whose CRC or size disagrees is escalated to
a full sha256 byte comparison, so a REPORTED difference is always exact.

tools/ckpt_crc.py's central-directory parser is validated against a synthetic
stored zip with a positive control (identical payloads -> equal CRC) and a
negative one (one bit apart -> different CRC).
"""
import argparse, collections, hashlib, io, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ckpt_crc

AFF = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt"
CONF = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"


def structural(aff_t, conf_t):
    """The half that needs no payload: names, shapes, element counts."""
    shared = sorted(set(aff_t) & set(conf_t))
    only_a = sorted(set(aff_t) - set(conf_t))
    only_c = sorted(set(conf_t) - set(aff_t))
    shape_mismatch = [k for k in shared if aff_t[k]["shape"] != conf_t[k]["shape"]]
    width_mismatch = [k for k in shared
                      if aff_t[k].get("width") != conf_t[k].get("width")]
    return {"shared": shared, "only_aff": only_a, "only_conf": only_c,
            "shape_mismatch": shape_mismatch, "width_mismatch": width_mismatch}


def tensors_from_pkl(path):
    """The tensor map out of a saved data.pkl, as ckpt_crc.tensor_map returns it."""
    blob = pathlib.Path(path).read_bytes()
    obj = ckpt_crc.Restricted(io.BytesIO(blob)).load()
    out = {}
    for name, call in ckpt_crc._walk(obj):
        if not name.startswith("state_dict."): continue
        a = call.args
        if len(a) < 4 or not isinstance(a[0], ckpt_crc.Storage): continue
        pid = a[0].pid
        if not isinstance(pid, (list, tuple)) or len(pid) < 3: continue
        w = 4
        for p in pid:
            n2 = getattr(p, "name", None)
            if isinstance(n2, str) and n2 in ckpt_crc.WIDTH: w = ckpt_crc.WIDTH[n2]
            elif isinstance(p, str) and p in ckpt_crc.WIDTH: w = ckpt_crc.WIDTH[p]
        shape = list(a[2]) if isinstance(a[2], (list, tuple)) else []
        cnt = 1
        for d in shape: cnt *= d
        out[name[len("state_dict."):]] = {
            "member": f"data/{pid[2]}", "shape": shape, "elements": cnt, "width": w}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="structural half only, from saved data.pkl files")
    ap.add_argument("--aff-pkl", default="refs/boltz-ckpt/aff/data.pkl")
    ap.add_argument("--conf-pkl", default="refs/boltz-ckpt/conf/data.pkl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = []
    def say(s=""):
        print(s); lines.append(s)

    if args.offline:
        say("# G0 - structural half (offline)")
        say()
        aff_t = tensors_from_pkl(args.aff_pkl)
        conf_t = tensors_from_pkl(args.conf_pkl)
        st = structural(aff_t, conf_t)
        say(f"aff  state_dict tensors : {len(aff_t):,}")
        say(f"conf state_dict tensors : {len(conf_t):,}")
        say(f"shared names            : {len(st['shared']):,}")
        say(f"only in aff             : {len(st['only_aff']):,}")
        say(f"only in conf            : {len(st['only_conf']):,}")
        say(f"shape mismatches        : {len(st['shape_mismatch'])}")
        say(f"dtype-width mismatches  : {len(st['width_mismatch'])}")
        for k in st["shape_mismatch"][:20]:
            say(f"   SHAPE {k}: aff {aff_t[k]['shape']} conf {conf_t[k]['shape']}")
        for k in st["width_mismatch"][:20]:
            say(f"   WIDTH {k}: aff {aff_t[k]['width']} conf {conf_t[k]['width']}")
        say()
        c = collections.Counter(k.split(".")[0] for k in st["only_aff"])
        say("scopes only in aff : " + ", ".join(f"{k}({v})" for k, v in c.most_common()))
        c = collections.Counter(k.split(".")[0] for k in st["only_conf"])
        say("scopes only in conf: " + ", ".join(f"{k}({v})" for k, v in c.most_common()))
        say()
        shared_params = sum(aff_t[k]["elements"] for k in st["shared"])
        say(f"shared parameters       : {shared_params:,}")
        say()
        verdict = (not st["shape_mismatch"]) and (not st["width_mismatch"])
        say(f"STRUCTURAL: {'PASS' if verdict else 'FAIL'}")
        say()
        say("VALUE HALF NOT RUN - needs the two ZIP central directories "
            "(a few MB from huggingface.co).")
        say("Re-run without --offline to complete G0.")
    else:
        say("# G0 - full verification")
        say()
        say("opening aff ...")
        a = ckpt_crc.open_ckpt(AFF)
        say("opening conf ...")
        b = ckpt_crc.open_ckpt(CONF)
        aff_t, conf_t = a["tensors"], b["tensors"]
        st = structural(aff_t, conf_t)
        say(f"aff  {len(aff_t):,} tensors / conf {len(conf_t):,} tensors")
        say(f"shared {len(st['shared']):,}  only-aff {len(st['only_aff']):,}  "
            f"only-conf {len(st['only_conf']):,}")
        say(f"shape mismatches {len(st['shape_mismatch'])}  "
            f"width mismatches {len(st['width_mismatch'])}")
        say()

        same = []
        suspect = []
        for k in st["shared"]:
            ma, mb = aff_t[k]["member"], conf_t[k]["member"]
            ea = a["members"].get(ma); eb = b["members"].get(mb)
            if ea is None or eb is None:
                suspect.append((k, "member missing")); continue
            if ea["usize"] != eb["usize"]:
                suspect.append((k, f"size {ea['usize']} vs {eb['usize']}")); continue
            if ea["crc"] != eb["crc"]:
                suspect.append((k, f"crc {ea['crc']:#010x} vs {eb['crc']:#010x}")); continue
            same.append(k)
        say(f"CRC-32 + size identical : {len(same):,} / {len(st['shared']):,}")
        say(f"needing byte escalation : {len(suspect):,}")
        say()

        confirmed_diff = []
        for k, why in suspect:
            da = ckpt_crc.fetch_member(AFF, a["members"], aff_t[k]["member"])
            db = ckpt_crc.fetch_member(CONF, b["members"], conf_t[k]["member"])
            ha = hashlib.sha256(da).hexdigest(); hb = hashlib.sha256(db).hexdigest()
            if ha != hb:
                confirmed_diff.append((k, why, ha[:16], hb[:16]))
                say(f"   DIFFER {k}  ({why})  {ha[:16]} vs {hb[:16]}")
            else:
                say(f"   same-after-escalation {k}  ({why})")
        say()
        n = len(st["shared"])
        say(f"false-same probability over {n:,} CRC-32 comparisons: "
            f"~{n / 2**32:.2e}")
        say()
        ok = (not confirmed_diff and not st["shape_mismatch"]
              and not st["width_mismatch"])
        say(f"G0: {'PASS' if ok else 'FAIL'} - "
            f"{len(same):,} identical, {len(confirmed_diff)} differing")
        if not ok:
            say()
            say("CONSEQUENCE: the affinity pass needs its own trunk export; "
                "revise PHASE2-PORT-PLAN.md sections 1.1, 8 and 9.")

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
