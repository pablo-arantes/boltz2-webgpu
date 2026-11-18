# Screening on a Colab GPU

The page screens in your own browser, which caps the target at roughly 2,000
residues: one pair tensor is `residues² × 128 × 4` bytes and a WebGPU binding
cannot exceed `maxStorageBufferBindingSize`, so the ceiling is
`sqrt(limit / 512)` — about 2,048 where an adapter reports 2 GiB and 2,896
where it reports 4 (docs/MEASURED.md, web/target-size.js).

`notebooks/boltz2_webgpu.ipynb` moves the screen onto a Colab GPU instead. One
cell, one link, and the page you click is this repository served from that
machine — so the target it can hold is that card's, which the link box prints.

## How it is arranged

```
   your browser                 the Colab machine
   ────────────                 ─────────────────
   bench.html                   tools/colab_backend.py
     ?backend=colab   ──/in──▶    serves this checkout
     (the READER)     ◀─/down─    brokers two mailboxes
   no GPU at all                        │
   draws what arrives            opens headless Chrome on
                                 bench.html?role=runtime
                                   (the RUNTIME)
                                 the card, the weights,
                                 web/screen.js — it screens
                                        │
                                 /up ◀──┘  /out ──▶
```

Five routes, all token-checked (`hmac.compare_digest`):

| route | who | what |
| --- | --- | --- |
| `GET /health` | reader | the card, the residue ceiling, whether a screen is running |
| `POST /in` | reader | `{op: "screen"\|"stop"\|"ping"\|"shutdown", payload}` |
| `GET /out?since=N` | runtime | the commands it has not obeyed |
| `POST /up` | runtime | what it has said: status, each result, the finish |
| `GET /down?since=N&wait=` | reader | the same, held until there is something |

Both mailboxes are **append-only and read by watermark**: a caller says what it
has already applied and gets what came after, so an overlapping poll or a
reloaded page repeats itself rather than losing anything. `head=1` means "I have
just started" — on `/down` it gives the watermark without the stream (so a
reader does not replay the previous screen's hundred results), and on `/out` it
means the runtime is not owed the past (a reload otherwise re-obeys every
command the session ever carried).

## 🔴 There is no second implementation of a screen

The runtime half fills **this page's own controls** and presses **its own Run
button**; the reader half hands each result to `acceptResult`, the same function
the local run loop calls. That is deliberate: a remote feed with its own copy of
either would assemble the scene, the plate and the table a second way, and an
hour-long run is the one nobody re-checks. `tools/screen-in-page.py` already
proves the control path works by driving exactly those elements.

The controller's two seams are small and inert on an ordinary page:

- `installScreenTap(sink)` — every `say()`, every accepted result and failure,
  and one `finished` event go to the sink. With no tap installed it is one
  comparison per call and `screen.html` never knows it is there.
- `acceptResult` / `acceptFailure` — the two functions the local loop and the
  remote feed share.

## What is ours and what is LocalFold's

The arrangement, the five routes, the watermark, the token, the liveness check
and the notebook's setup script are **ported from LocalFold** at
`af474628e0` — Beerware, Sergey Ovchinnikov. docs/PROVENANCE.md records it.

🔴 **Those files no longer exist upstream.** They were removed on 2026-11-01
(`1725c392ef`) when every fold there went to a native CUDA worker, so
LocalFold's own notebook now starts a file that is not in its tree. We cannot
follow: there is no native port of the affinity module here, the screen is
WebGPU-only, and a headless runtime running our own page is what works.
`web/native-bridge.js` — already in this tree, the reader half of the newer
design — supplies `remoteCommand`, `remoteEvents` and `remoteHead` rather than
having them written twice.

What differs, because the **unit of work** differs:

- Upstream brokers one fold. A screen is N ligands, so the events are the run
  loop's own (`status`, `result`, `failed`, `finished`).
- 🔴 **The one-screen-at-a-time flag is lowered by `finished`, not by a
  result.** A port that lowered it on a result would free the GPU after the
  first ligand of a hundred; one that never sent it would refuse every later
  screen for the session.
- `/health` carries **`residueCeiling`**, the number this whole arrangement
  exists for, so the reader's page can say what the runtime can hold instead of
  what the laptop could.
- `/down?wait=` is honoured, because this tree's reader already asks for it.
  Without it the reader degrades to a 400 ms poll — correct, and a request every
  400 ms for an hour.

## Measured

On this developer machine (macOS, Apple GPU through the same headless Chrome the
Colab runtime uses), 2026-11-18:

```
$ python3 -I tools/colab_backend.py --port 8710 --token testtoken
serving … on 127.0.0.1:8710 · no Colab runtime here, Disconnect stops the service
BACKEND {"token": "testtoken", "gpu": {"webgpu": true, "vendor": "apple",
  "architecture": "metal-3", "shaderF16": true,
  "maxStorageBufferBindingSize": 4294967292}, "residueCeiling": 2896}
```

A reader's two-ligand screen driven over the routes:

| | |
| --- | --- |
| watermark at open | 1133, and the reader started there rather than replaying it |
| `screen-started` | 0.5 s after the command |
| first result | **2.4 s**, with its PDB (22,104 bytes) |
| second result | 4.2 s |
| `finished` | 4.2 s — `{done: 2, attempted: 2, seconds: "4"}` |
| broker `busy` after | `False` |

...and the reader **page**, driven in a browser against the same runtime:

| | |
| --- | --- |
| status line | `screening on the runtime's apple metal-3 · it can hold a target up to about 2896 residues · shader-f16` |
| the limits box | reports the **runtime's** ceiling, not the browser's |
| live feed | the runtime's own `[2/2] propofolish · about 2 s left` appeared on the reader |
| table / plate / scatter | 2 rows, 2 wells, 2 points |
| the 3D scene | **46 coordinates**, rebuilt from the streamed PDBs |
| download | enabled |
| console errors | **none** |

## What has NOT been verified here

🔴 **The Colab setup itself.** The `apt`, the unpacked `libnvidia-gl`, the
Vulkan ICD and the zipped Chrome are Linux-and-NVIDIA steps that cannot run on
this machine, and they are the part most likely to need a version bump — the
Chrome build is pinned to `153.0.8010.52` in the cell. They are LocalFold's own,
measured by it on a Colab T4, and carried across unchanged for that reason.
**Run the notebook once on Colab before trusting it**; if the setup fails the
cell prints its whole log, which is what that log is kept for.

Also unverified: a target large enough to need the A100 at all. The ceiling is
arithmetic and the broker reports it, but nothing here has folded 4,000
residues; what the measurement above shows is that the *path* carries a screen
both ways.
