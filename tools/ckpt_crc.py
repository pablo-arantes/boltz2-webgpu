"""Central-directory CRC-32 + size for every ZIP member, and the tensor map.

A library for G0: torch.save stores tensor payloads UNCOMPRESSED, so each
member's CRC-32 sits in the central directory. Fetching two central directories
and two data.pkl files is a few MB and yields an exhaustive comparison.

Same restricted unpickler as ckpt_index.py: nothing from the file is executed.
"""
import io, pickle, struct, urllib.request

UA = {"User-Agent": "ckpt-crc/1"}

def ranged(url, start, end):
    r = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(r, timeout=180) as f: return f.read()

def total_size(url):
    r = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(r, timeout=120) as f:
        return int(f.headers["Content-Length"])

def members(url, size):
    """name -> dict(crc, usize, method, local_header)"""
    tail = ranged(url, max(0, size - 65_600), size - 1)
    at = tail.rfind(b"PK\x05\x06")
    count, cd_size, cd_off = struct.unpack_from("<HII", tail, at + 10)
    z64 = tail.rfind(b"PK\x06\x06")
    if z64 >= 0 and (cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or count == 0xFFFF):
        cd_size = struct.unpack_from("<Q", tail, z64 + 40)[0]
        cd_off = struct.unpack_from("<Q", tail, z64 + 48)[0]
    cd = ranged(url, cd_off, cd_off + cd_size - 1)
    out, pos = {}, 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", cd, pos + 10)[0]
        crc = struct.unpack_from("<I", cd, pos + 16)[0]
        csize = struct.unpack_from("<I", cd, pos + 20)[0]
        usize = struct.unpack_from("<I", cd, pos + 24)[0]
        nlen, elen, clen = struct.unpack_from("<HHH", cd, pos + 28)
        lho = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos + 46: pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen: pos + 46 + nlen + elen]
        if usize == 0xFFFFFFFF or lho == 0xFFFFFFFF or csize == 0xFFFFFFFF:
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, e)
                if hid == 0x0001:
                    vals, off = [], e + 4
                    while off + 8 <= e + 4 + hsz:
                        vals.append(struct.unpack_from("<Q", extra, off)[0]); off += 8
                    it = iter(vals)
                    if usize == 0xFFFFFFFF: usize = next(it, usize)
                    if csize == 0xFFFFFFFF: csize = next(it, csize)
                    if lho == 0xFFFFFFFF: lho = next(it, lho)
                    break
                e += 4 + hsz
        out[name] = {"crc": crc, "usize": usize, "method": method, "lho": lho}
        pos += 46 + nlen + elen + clen
    return out

# --- restricted unpickler -----------------------------------------------------
class Call:
    def __init__(self, what, args, kwargs=None):
        self.what, self.args, self.kwargs = what, args, kwargs or {}
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
def _stub(m, n):
    if (m, n) not in _S:
        _S[(m, n)] = type(f"Stub_{n}", (StubBase,), {"qualname": f"{m}.{n}", "name": n})
    return _S[(m, n)]
def _rec(what):
    def mk(*a, **k): return Call(what, a, k)
    return mk
class Restricted(pickle.Unpickler):
    def find_class(self, m, n):
        return _rec(f"{m}.{n}") if n.startswith("_rebuild") else _stub(m, n)
    def persistent_load(self, pid): return Storage(pid)

def _walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from _walk(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): yield from _walk(v, f"{pre}.{i}")
    elif isinstance(o, Call): yield pre, o

WIDTH = {"FloatStorage": 4, "HalfStorage": 2, "BFloat16Storage": 2, "DoubleStorage": 8,
         "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1,
         "ByteStorage": 1, "BoolStorage": 1}

def tensor_map(url, mem, scope="state_dict."):
    """tensor name -> dict(member, shape, elements, width)"""
    pkl = [n for n in mem if n.endswith("data.pkl")][0]
    root = pkl.rsplit("/", 1)[0]
    e = mem[pkl]
    hdr = ranged(url, e["lho"], e["lho"] + 29)
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    at = e["lho"] + 30 + nlen + elen
    obj = Restricted(io.BytesIO(ranged(url, at, at + e["usize"] - 1))).load()
    out = {}
    for name, call in _walk(obj):
        if scope and not name.startswith(scope): continue
        a = call.args
        if len(a) < 4 or not isinstance(a[0], Storage): continue
        pid = a[0].pid
        if not isinstance(pid, (list, tuple)) or len(pid) < 3: continue
        w = 4
        for p in pid:
            n2 = getattr(p, "name", None)
            if isinstance(n2, str) and n2 in WIDTH: w = WIDTH[n2]
            elif isinstance(p, str) and p in WIDTH: w = WIDTH[p]
        key = str(pid[2])
        shape = tuple(a[2]) if isinstance(a[2], (list, tuple)) else ()
        cnt = 1
        for d in shape: cnt *= d
        out[name[len(scope):] if scope else name] = {
            "member": f"{root}/data/{key}", "shape": list(shape),
            "elements": cnt, "width": w}
    return out

def open_ckpt(url):
    size = total_size(url)
    mem = members(url, size)
    return {"url": url, "size": size, "members": mem,
            "tensors": tensor_map(url, mem)}

def fetch_member(url, mem, name):
    e = mem[name]
    hdr = ranged(url, e["lho"], e["lho"] + 29)
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    at = e["lho"] + 30 + nlen + elen
    return ranged(url, at, at + e["usize"] - 1)
