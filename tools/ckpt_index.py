"""Read a remote PyTorch .ckpt's tensor table without downloading the weights.

    python3 -I tools/ckpt_index.py <url> --out <dir>

A torch.save checkpoint is a ZIP. The central directory is at the end, and
`*/data.pkl` holds the pickled state_dict whose rebuild calls carry every
tensor's dtype, shape and stride. Fetching the EOCD, the central directory and
that one member is a few MB instead of gigabytes.

SAFETY: the pickle comes off the internet, so it is read with a RESTRICTED
Unpickler that resolves nothing. Every global becomes an inert stub and every
persistent id a placeholder, so no code from the file can run. We only ever
want the metadata the opcodes carry.
"""
import argparse, io, json, pathlib, pickle, struct, sys, urllib.request

def ranged(url, start, end):
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{end}", "User-Agent": "ckpt-index/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def total_size(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "ckpt-index/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])

def central_directory(url, size):
    tail = ranged(url, max(0, size - 65_600), size - 1)
    at = tail.rfind(b"PK\x05\x06")
    if at < 0:
        raise SystemExit("no end-of-central-directory found")
    count, cd_size, cd_off = struct.unpack_from("<HII", tail, at + 10)
    # zip64 if any field is saturated
    z64 = tail.rfind(b"PK\x06\x06")
    if z64 >= 0 and (cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF
                     or count == 0xFFFF):
        count, cd_size, cd_off = struct.unpack_from("<QQQ", tail, z64 + 32)[0], \
            struct.unpack_from("<Q", tail, z64 + 40)[0], \
            struct.unpack_from("<Q", tail, z64 + 48)[0]
    cd = ranged(url, cd_off, cd_off + cd_size - 1)
    entries, pos = [], 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        comp, usize, nlen, elen, clen = struct.unpack_from("<I", cd, pos + 20)[0], \
            struct.unpack_from("<I", cd, pos + 24)[0], \
            struct.unpack_from("<H", cd, pos + 28)[0], \
            struct.unpack_from("<H", cd, pos + 30)[0], \
            struct.unpack_from("<H", cd, pos + 32)[0]
        lho = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos + 46: pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen: pos + 46 + nlen + elen]
        # zip64 extended information, if the 32-bit fields are saturated
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
                    if comp == 0xFFFFFFFF: comp = next(it, comp)
                    if lho == 0xFFFFFFFF: lho = next(it, lho)
                    break
                e += 4 + hsz
        entries.append({"name": name, "size": usize, "local_header": lho})
        pos += 46 + nlen + elen + clen
    return entries

class Call:
    """A recorded construction. Nothing is executed."""
    def __init__(self, what, args, kwargs=None):
        self.what, self.args, self.kwargs = what, args, kwargs or {}
    def __repr__(self): return f"{self.what}(...)"

class Storage:
    def __init__(self, pid): self.pid = pid
    def __repr__(self): return f"<storage {self.pid}>"

class StubBase:
    """Inert instance of a global the pickle named. Absorbs state silently."""
    qualname = "?"
    def __new__(cls, *a, **k):
        self = object.__new__(cls)
        self.args, self.kwargs, self.state = a, k, None
        return self
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return Call(type(self).qualname, a, k)
    def __setstate__(self, state): self.state = state
    def __repr__(self): return f"<{type(self).qualname}>"

_STUBS = {}

def stub_type(module, name):
    """A real `type` (NEWOBJ needs one) standing in for module.name."""
    key = (module, name)
    if key not in _STUBS:
        _STUBS[key] = type(f"Stub_{name}", (StubBase,),
                           {"qualname": f"{module}.{name}",
                            "module": module, "name": name})
    return _STUBS[key]

class Restricted(pickle.Unpickler):
    """Resolves NOTHING: every global becomes an inert local type."""
    def find_class(self, module, name):
        # the rebuild helpers are the ones whose ARGUMENTS we want recorded
        if name.startswith("_rebuild") or name.endswith("Storage") \
                or name in ("OrderedDict", "_load_from_bytes"):
            t = stub_type(module, name)
            return _recorder(f"{module}.{name}") if name.startswith("_rebuild") else t
        return stub_type(module, name)
    def persistent_load(self, pid): return Storage(pid)

def _recorder(what):
    def make(*a, **k): return Call(what, a, k)
    return make

def walk(obj, prefix=""):
    """Yield (name, Call) for every rebuild call in a nested state_dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(obj, Call):
        yield prefix, obj

DTYPES = {"FloatStorage": "float32", "HalfStorage": "float16",
          "BFloat16Storage": "bfloat16", "DoubleStorage": "float64",
          "LongStorage": "int64", "IntStorage": "int32",
          "ShortStorage": "int16", "CharStorage": "int8",
          "ByteStorage": "uint8", "BoolStorage": "bool"}

def describe(call):
    """A rebuild call's (dtype, shape). torch._utils._rebuild_tensor_v2 is
    (storage, storage_offset, size, stride, requires_grad, backward_hooks)."""
    a = call.args
    if len(a) < 4: return None, None
    storage, shape = a[0], a[2]
    dtype = None
    if isinstance(storage, Storage) and isinstance(storage.pid, (list, tuple)):
        for part in storage.pid:
            nm = getattr(part, "name", None) or getattr(part, "__name__", None)
            if isinstance(nm, str):
                nm = nm.replace("Stub_", "")
                if nm in DTYPES: dtype = DTYPES[nm]
            elif isinstance(part, str) and part in DTYPES:
                dtype = DTYPES[part]
    shape = tuple(shape) if isinstance(shape, (list, tuple)) else None
    return dtype, shape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    size = total_size(args.url)
    print(f"remote size: {size:,} bytes ({size/2**30:.2f} GiB)", file=sys.stderr)
    entries = central_directory(args.url, size)
    print(f"zip members: {len(entries):,}", file=sys.stderr)

    pkls = [e for e in entries if e["name"].endswith("data.pkl")]
    if not pkls: raise SystemExit("no data.pkl member")
    e = pkls[0]
    print(f"reading {e['name']} ({e['size']:,} bytes)", file=sys.stderr)
    # local header: 30 bytes + name + extra, then the (stored) data
    hdr = ranged(args.url, e["local_header"], e["local_header"] + 29)
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    data_at = e["local_header"] + 30 + nlen + elen
    blob = ranged(args.url, data_at, data_at + e["size"] - 1)
    (out / "data.pkl").write_bytes(blob)

    obj = Restricted(io.BytesIO(blob)).load()
    top = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
    print(f"top-level keys: {top}", file=sys.stderr)

    rows = []
    for name, call in walk(obj):
        dtype, shape = describe(call)
        if shape is None: continue
        n = 1
        for d in shape: n *= d
        rows.append({"name": name, "dtype": dtype, "shape": list(shape),
                     "elements": n})
    rows.sort(key=lambda r: r["name"])
    (out / "tensors.json").write_text(json.dumps(rows, indent=1))
    print(f"tensors: {len(rows):,}  params: {sum(r['elements'] for r in rows):,}",
          file=sys.stderr)
    # also dump any non-tensor scalars that look like config
    cfg = {k: v for k, v in obj.items() if k != "state_dict"} if isinstance(obj, dict) else {}
    (out / "nonstate.txt").write_text(repr(cfg)[:400_000])

if __name__ == "__main__":
    main()
