"""Byte-compare named tensors between two remote PyTorch checkpoints.

    python3 -I tools/ckpt_compare.py --a <url> --b <url> --pattern pairformer_module --n 12

Resolves each tensor's ZIP member from the pickle's storage key and fetches
only that member's bytes by HTTP range. Same restricted unpickler as
ckpt_index.py: nothing from the file is executed.
"""
import argparse, hashlib, io, pickle, struct, sys, urllib.request, random

def ranged(url, start, end):
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{end}", "User-Agent": "ckpt-compare/1"})
    with urllib.request.urlopen(req, timeout=180) as r: return r.read()

def total_size(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "ckpt-compare/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])

def members(url, size):
    tail = ranged(url, max(0, size - 65_600), size - 1)
    at = tail.rfind(b"PK\x05\x06")
    count, cd_size, cd_off = struct.unpack_from("<HII", tail, at + 10)
    z64 = tail.rfind(b"PK\x06\x06")
    if z64 >= 0 and (cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or count == 0xFFFF):
        cd_size = struct.unpack_from("<Q", tail, z64 + 40)[0]
        cd_off = struct.unpack_from("<Q", tail, z64 + 48)[0]
    cd = ranged(url, cd_off, cd_off + cd_size - 1)
    out, pos = {}, 0
    while pos + 46 <= len(cd) and cd[pos:pos+4] == b"PK\x01\x02":
        usize = struct.unpack_from("<I", cd, pos + 24)[0]
        nlen, elen, clen = struct.unpack_from("<HHH", cd, pos + 28)
        lho = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos+46: pos+46+nlen].decode("utf-8", "replace")
        extra = cd[pos+46+nlen: pos+46+nlen+elen]
        if usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, e)
                if hid == 0x0001:
                    vals, off = [], e + 4
                    while off + 8 <= e + 4 + hsz:
                        vals.append(struct.unpack_from("<Q", extra, off)[0]); off += 8
                    it = iter(vals)
                    if usize == 0xFFFFFFFF: usize = next(it, usize)
                    if lho == 0xFFFFFFFF: lho = next(it, lho)
                    break
                e += 4 + hsz
        out[name] = (lho, usize)
        pos += 46 + nlen + elen + clen
    return out

class Call:
    def __init__(self, what, args, kwargs=None): self.what, self.args, self.kwargs = what, args, kwargs or {}
class Storage:
    def __init__(self, pid): self.pid = pid
class StubBase:
    qualname = "?"
    def __new__(cls, *a, **k):
        s = object.__new__(cls); s.args, s.kwargs, s.state = a, k, None; return s
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return Call(type(self).qualname, a, k)
    def __setstate__(self, st): self.state = st
_S = {}
def stub_type(m, n):
    if (m, n) not in _S:
        _S[(m, n)] = type(f"Stub_{n}", (StubBase,), {"qualname": f"{m}.{n}", "name": n})
    return _S[(m, n)]
def _rec(what):
    def mk(*a, **k): return Call(what, a, k)
    return mk
class Restricted(pickle.Unpickler):
    def find_class(self, m, n):
        if n.startswith("_rebuild"): return _rec(f"{m}.{n}")
        return stub_type(m, n)
    def persistent_load(self, pid): return Storage(pid)

def walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): yield from walk(v, f"{pre}.{i}")
    elif isinstance(o, Call): yield pre, o

WIDTH = {"FloatStorage": 4, "HalfStorage": 2, "BFloat16Storage": 2,
         "DoubleStorage": 8, "LongStorage": 8, "IntStorage": 4,
         "ShortStorage": 2, "CharStorage": 1, "ByteStorage": 1, "BoolStorage": 1}

def table(url):
    """name -> (zip member, element count, bytes-per-element)"""
    size = total_size(url)
    mem = members(url, size)
    pkl = [n for n in mem if n.endswith("data.pkl")][0]
    root = pkl.rsplit("/", 1)[0]
    lho, usize = mem[pkl]
    hdr = ranged(url, lho, lho + 29)
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    at = lho + 30 + nlen + elen
    obj = Restricted(io.BytesIO(ranged(url, at, at + usize - 1))).load()
    out = {}
    for name, call in walk(obj):
        if not name.startswith("state_dict."): continue
        a = call.args
        if len(a) < 4 or not isinstance(a[0], Storage): continue
        pid = a[0].pid
        if not isinstance(pid, (list, tuple)) or len(pid) < 3: continue
        key, w = None, 4
        for part in pid:
            nm = getattr(part, "name", None)
            if isinstance(nm, str) and nm in WIDTH: w = WIDTH[nm]
            elif isinstance(part, str) and part in WIDTH: w = WIDTH[part]
        # torch pid is ('storage', storage_type, key, location, numel):
        # the KEY is index 2. numel is also digit-like, so never scan for it.
        if len(pid) >= 3 and isinstance(pid[2], (str, int)):
            key = str(pid[2])
        if key is None: continue
        shape = tuple(a[2]) if isinstance(a[2], (list, tuple)) else ()
        n = 1
        for d in shape: n *= d
        out[name[len("state_dict."):]] = (f"{root}/data/{key}", n, w, shape)
    return url, mem, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--pattern", default=""); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-bytes", type=int, default=4_000_000)
    args = ap.parse_args()

    print("indexing A…", file=sys.stderr); _, mem_a, ta = table(args.a)
    print("indexing B…", file=sys.stderr); _, mem_b, tb = table(args.b)
    shared = sorted(set(ta) & set(tb))
    cand = [k for k in shared if args.pattern in k
            and ta[k][1] * ta[k][2] <= args.max_bytes and ta[k][1] > 0]
    random.Random(0).shuffle(cand)
    cand = cand[:args.n]
    print(f"\nshared {len(shared)} tensors; comparing {len(cand)} matching "
          f"{args.pattern!r}\n", file=sys.stderr)
    same = diff = 0
    for k in cand:
        ma, n, w, shape = ta[k]; mb, n2, w2, _ = tb[k]
        nb = n * w
        lha, ua = mem_a[ma]; lhb, ub = mem_b[mb]
        ha = ranged(args.a, lha, lha + 29); hb = ranged(args.b, lhb, lhb + 29)
        na, ea = struct.unpack_from("<HH", ha, 26); nbn, eb = struct.unpack_from("<HH", hb, 26)
        aa = lha + 30 + na + ea; bb = lhb + 30 + nbn + eb
        da = ranged(args.a, aa, aa + nb - 1); db = ranged(args.b, bb, bb + nb - 1)
        sa = hashlib.sha256(da).hexdigest()[:16]; sb = hashlib.sha256(db).hexdigest()[:16]
        ok = sa == sb
        same += ok; diff += (not ok)
        print(f"  {'SAME' if ok else 'DIFF'}  {k}  {list(shape)}  {nb:,}B  {sa} {sb}")
    print(f"\n=== identical {same} / differing {diff} ===")

if __name__ == "__main__":
    main()
