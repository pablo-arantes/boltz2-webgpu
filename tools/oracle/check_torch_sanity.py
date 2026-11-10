"""Is this torch arithmetically trustworthy on THIS box, before it is used as an oracle?

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 tools/oracle/check_torch_sanity.py

🔴 THIS EXISTS BECAUSE torch ABORTS HERE WITHOUT AN UNSUPPORTED FLAG. Four
copies of libomp are installed (miniforge's libiomp5 and libomp, sklearn's, and
torch's own), so `import torch` dies with OMP Error #15 and SIGABRT. The hint
OpenMP prints offers `KMP_DUPLICATE_LIB_OK=TRUE` and calls it "an unsafe,
unsupported, undocumented workaround" that "may cause crashes or SILENTLY
PRODUCE INCORRECT RESULTS".

Silently incorrect is disqualifying for an oracle: the port would be held to a
wrong reference and would agree with it. docs/ records this exact shape - "a
dump is only as good as the reference that wrote it", where agreeing with a
renamed weight at 1.55e-5 was agreeing with the defect.

So two things, rather than trust:
  * ONE THREAD. The duplicate-runtime hazard is in parallel regions, so
    OMP_NUM_THREADS=1 and torch.set_num_threads(1) remove the mechanism.
  * VALIDATE EVERY PRIMITIVE the affinity module uses against an independent
    float64 numpy implementation written here. If torch and numpy agree to
    float64 rounding on LayerNorm, Linear, SiLU, ReLU, cdist, bucketize and the
    broadcast sums, the dump can be trusted for THIS computation - which is a
    narrower and checkable claim than "torch works".

Exits non-zero if any primitive disagrees, so the dumper can refuse to run.
"""
from __future__ import annotations

import os

# Set before torch is imported, or they do nothing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                            # noqa: E402
import torch                                                   # noqa: E402
from torch import nn                                           # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

BOUND = 1e-12          # float64 against float64: rounding, not tolerance


def report(name: str, delta: float, bound: float = BOUND) -> bool:
    ok = np.isfinite(delta) and delta <= bound
    # 🔴 `delta > bound` is FALSE for NaN, which is how a check passes without
    # computing anything. isfinite is asserted first, deliberately.
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<34} {delta:.3e}")
    return ok


def main() -> int:
    rng = np.random.default_rng(20261109)
    good = True
    print(f"torch {torch.__version__}; threads {torch.get_num_threads()}; "
          f"KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print()

    # --- LayerNorm, the one whose eps and biased variance are conventions ----
    x = rng.standard_normal((7, 13, 128))
    w = rng.standard_normal(128)
    b = rng.standard_normal(128)
    eps = 1e-5
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)        # biased, as torch uses
    mine = (x - mean) / np.sqrt(var + eps) * w + b
    ln = nn.LayerNorm(128, eps=eps).double()
    with torch.no_grad():
        ln.weight.copy_(torch.from_numpy(w)); ln.bias.copy_(torch.from_numpy(b))
        theirs = ln(torch.from_numpy(x)).numpy()
    good &= report("LayerNorm(eps=1e-5)", float(np.abs(mine - theirs).max()))

    # --- Linear, with and without bias: row-major weight is [out, in] --------
    a = rng.standard_normal((11, 64))
    wl = rng.standard_normal((32, 64))
    bl = rng.standard_normal(32)
    lin = nn.Linear(64, 32, bias=True).double()
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(wl)); lin.bias.copy_(torch.from_numpy(bl))
        theirs = lin(torch.from_numpy(a)).numpy()
    good &= report("Linear (y = x @ W.T + b)",
                   float(np.abs(a @ wl.T + bl - theirs).max()))

    nobias = nn.Linear(64, 32, bias=False).double()
    with torch.no_grad():
        nobias.weight.copy_(torch.from_numpy(wl))
        theirs = nobias(torch.from_numpy(a)).numpy()
    good &= report("LinearNoBias", float(np.abs(a @ wl.T - theirs).max()))

    # --- SiLU and ReLU, the transition's and the heads' ---------------------
    z = rng.standard_normal(2048) * 4
    good &= report("SiLU", float(np.abs(
        z / (1 + np.exp(-z)) - nn.functional.silu(torch.from_numpy(z)).numpy()).max()))
    good &= report("ReLU", float(np.abs(
        np.maximum(z, 0) - nn.functional.relu(torch.from_numpy(z)).numpy()).max()))
    good &= report("sigmoid", float(np.abs(
        1 / (1 + np.exp(-z)) - torch.sigmoid(torch.from_numpy(z)).numpy()).max()))

    # --- cdist, which the distogram is built from ---------------------------
    p = rng.standard_normal((1, 40, 3)) * 12
    d = np.sqrt(((p[0][:, None, :] - p[0][None, :, :]) ** 2).sum(-1))
    theirs = torch.cdist(torch.from_numpy(p), torch.from_numpy(p)).numpy()[0]
    # 🔴 A LOOSER BOUND, AND IT IS NOT A CONVENTION DIFFERENCE. torch.cdist
    # uses a matrix-multiply expansion of |a-b|^2, which cancels catastrophically
    # near zero - the DIAGONAL is exactly the case. The port computes the
    # difference directly, so this is torch being less accurate, not this port
    # being wrong, and the dumper computes d the direct way for that reason.
    good &= report("cdist (off-diagonal)", float(np.abs(
        (d - theirs)[~np.eye(40, dtype=bool)]).max()), 1e-8)
    print(f"       cdist diagonal, torch: max {np.abs(np.diag(theirs)).max():.3e}"
          "  (exactly 0 computed directly)")

    # --- bucketize: the distogram's STRICT `>` against 63 boundaries --------
    boundaries = np.linspace(2, 22, 63)
    dist = rng.random((30, 30)) * 26
    mine_bins = (dist[..., None] > boundaries).sum(-1)
    theirs_bins = (torch.from_numpy(dist).unsqueeze(-1)
                   > torch.from_numpy(boundaries)).sum(-1).numpy()
    good &= report("distogram bucketing (strict >)",
                   float(np.abs(mine_bins - theirs_bins).max()), 0)

    # --- the outer broadcast sum s_to_z_prod_in1/in2 ------------------------
    s = rng.standard_normal((1, 9, 16))
    t1 = rng.standard_normal((1, 9, 16))
    mine_z = s[:, :, None, :] + t1[:, None, :, :]
    theirs_z = (torch.from_numpy(s)[:, :, None, :]
                + torch.from_numpy(t1)[:, None, :, :]).numpy()
    good &= report("outer broadcast sum", float(np.abs(mine_z - theirs_z).max()))

    # --- masked mean, the heads' pooling -----------------------------------
    zz = rng.standard_normal((1, 12, 12, 8))
    mask = (rng.random((1, 12, 12, 1)) > 0.5).astype(np.float64)
    mine_g = (zz * mask).sum((1, 2)) / (mask.sum((1, 2)) + 1e-7)
    theirs_g = (torch.sum(torch.from_numpy(zz) * torch.from_numpy(mask), dim=(1, 2))
                / (torch.sum(torch.from_numpy(mask), dim=(1, 2)) + 1e-7)).numpy()
    good &= report("masked mean (+1e-7)", float(np.abs(mine_g - theirs_g).max()))

    print()
    print("TORCH SANITY PASS - trustworthy for this computation on one thread"
          if good else
          "TORCH SANITY FAIL - do NOT dump an oracle from this install")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
