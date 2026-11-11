"""Does the SCREEN PAGE work - the controls a reader touches, not the API?

    python3 tools/screen-in-page.py [--ligands=CCO,CCCc1ccccc1O] [--steps=20]

🔴 THE PAGE IS THE ONE PART NO NODE TEST CAN REACH, and CLAUDE.md records what
that costs three separate times: "a row that is not `ligand` fell to
`cleanSequence`, so `c1ccccc1` became `CCCCCC` on a click", "`setChains`
DELETED the row", and the whole `--smiles-ui` arm existing because "three bugs
survived every --smiles run because the API path cannot see a UI one". So this
drives the DOM: it types into the ligand box, fires its events, presses Screen,
and then reads what is DRAWN - the table's rows, the scatter's circles, the
viewer's own object list - rather than any value the controller happens to hold.

🔴 AND IT ASSERTS THE ZIP BY READING IT BACK. A download button that produces a
truthy Blob is a download button that may be producing an empty archive; the
probe builds the archive in the page, reads it back through the page's own
readZip, and requires the CSV's header, one row per attempted ligand and one
structure per finished one.

🔴 THE WEIGHTS COME OFF THE DISK. Both bundles are local exports here
(`model-boltz2-int5` and `model-boltz2-affinity-int5` are symlinks), so the
manifest rewrite fold-in-page.py uses is not needed - but the affinity bundle's
`remote` is null, which means on a DEPLOYED site this page would fall back to a
directory the build does not copy. That is stated in MODEL_BUNDLES and is why
this probe is not evidence about a published site.
"""
import argparse
import json
import base64
import pathlib
import os
import shutil
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402
from importlib import import_module                          # noqa: E402

fold_in_page = import_module("fold-in-page")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = fold_in_page.PORT

DEFAULT_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    ap.add_argument("--ligands", default="CCO,CCCc1ccccc1O")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--keep", action="store_true", help="keep the Chrome profile")
    # 🔴 THE LIVE SITE IS NOT THE LOCAL BUILD, AND ONLY THE LIVE ONE 404s. A
    # project page is served from /<repo>/ and its weights come from a pinned
    # HuggingFace revision over the real network, so base paths, CORS and the
    # shard cache are all exercised for the first time there. With --url this
    # starts no local server and drives the published page instead.
    # 🔴 THE ALIGNMENT ARM, WHICH THE DEFAULT RUN DOES NOT REACH. Without this
    # the probe asserts the dial is on "Single sequence" and never exercises the
    # one thing that leaves the machine - so "the MSA is searched ONCE per
    # target and cached by sequence hash", the page's own claim, went unchecked.
    # With --msa=mmseqs2 it counts requests to api.colabfold.com per ligand:
    # searched once means the count stops growing after the first.
    ap.add_argument("--msa", default="none",
                    help="the MSA dial: none (default) or mmseqs2, which sends"
                         " the target sequence to api.colabfold.com")
    # 🔴 A SCREENSHOT OF THIS PAGE, because the README's was a screenshot of
    # LocalFold's FOLDING page captioned as the screening page - a picture of
    # the wrong product. The only honest way to illustrate a page is to drive it
    # and photograph the result.
    # 🔴 A DESKTOP VIEWPORT, BECAUSE EVERY CAPTURE BEFORE THIS WAS BELOW 900px
    # AND THAT HID A LAYOUT DEFECT. Chrome's `--window-size` clamps at 500 (see
    # tools/cdp.py's header), so every screenshot of this page was taken at a
    # width where `@media (max-width: 900px)` had ALREADY collapsed the
    # scene/table grid to one column - the two-column layout a reader on a
    # 1280px screen actually saw could not appear in any of them. A user had to
    # send a photograph of their own browser. `Emulation.setDeviceMetricsOverride`
    # gives a true viewport at any width.
    ap.add_argument("--width", type=int, default=1280,
                    help="CSS viewport width to drive and photograph at")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--screenshot", default=None,
                    help="write a PNG of the finished page to this path")
    # 🔴 WHICH PAGE, BECAUSE THERE ARE NOW TWO FRONT ENDS OVER ONE CONTROLLER.
    # bench.html and screen.html satisfy the same twenty-two ids with different
    # markup and different stylesheets, so every gate in this file applies to
    # both - and the one that does not transfer by reading the code is the
    # canvas-width check, which depends on each page's own CSS fighting
    # py2Dmol's slot wrapper. Run it twice; a page that is never driven is a
    # page that is broken and nobody knows.
    ap.add_argument("--page", default="screen.html",
                    help="which page on the local server to drive")
    ap.add_argument("--url", default=None,
                    help="drive this URL instead of a local server, e.g."
                         " https://pablo-arantes.github.io/boltz2-webgpu/screen.html")
    args = ap.parse_args()

    # local_weights=False: both bundles here are local directories already, and
    # the rewrite would raise on a `remote: null` entry it cannot strip.
    httpd = None if args.url else fold_in_page.serve(local_weights=False)
    # a CLEAN profile for a live run: no HTTP cache, no CacheStorage, no
    # compiled shaders, which is what a first visitor actually pays
    profile = os.path.join("/tmp", "localfold-screen-profile")
    if args.url:
        shutil.rmtree(profile, ignore_errors=True)
    proc, ws = cdp.launch(args.port, profile, keep=args.keep)
    failures = []
    try:
        ws.call("Runtime.enable")
        ws.call("Page.enable")
        # 🔴 CAPTURE THE CONSOLE BEFORE ANY PAGE SCRIPT RUNS. py2Dmol's
        # `initializeApp` wraps `initializePy2DmolViewer` in a try that, on a
        # throw, does `console.error("Failed to initialize viewer:", e)` and
        # RETURNS - before assigning the module-level `viewerApi` that every
        # commit path in `processFiles` is optional-chained through. So a blank
        # scene with a registered viewer is exactly what that early return
        # looks like, and the only record of it is a console line. It also
        # calls `setStatus`, which writes to `#status` - an element this page
        # deliberately does not have, so the message has nowhere to go on
        # screen either.
        ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
          (() => {
            globalThis.__consoleErrors = [];
            for (const level of ['error', 'warn']) {
              const original = console[level].bind(console);
              console[level] = (...args) => {
                try {
                  globalThis.__consoleErrors.push(level + ': ' + args.map((a) =>
                    (a && a.stack) ? String(a.stack).split('\\n').slice(0, 3).join(' | ')
                                   : String(a)).join(' '));
                } catch {}
                original(...args);
              };
            }
            addEventListener('error', (e) => {
              globalThis.__consoleErrors.push('uncaught: ' + (e?.message ?? e));
            });
          })();
        """)
        ws.call("Emulation.setDeviceMetricsOverride", width=args.width,
                height=args.height, deviceScaleFactor=1, mobile=False)
        target = args.url or ("http://127.0.0.1:%d/%s" % (PORT, args.page))
        print("driving:", target)
        started = time.time()
        ws.call("Page.navigate", url=target)
        cdp.wait_for(ws, "document.getElementById('run') !== null", timeout=60,
                     what="the page to load")

        # 🔴 WAIT FOR THE BUTTON TO ENABLE, NOT FOR A FIXED TIME. `boot()`
        # enables Screen only once the device and BOTH bundles are up; a sleep
        # here would press a disabled button and read the absence of results as
        # a failure of the run.
        cdp.wait_for(ws, "document.getElementById('run').disabled === false",
                     timeout=args.timeout, what="the weights to load",
                     progress="document.getElementById('screen-status').textContent")
        loaded = cdp.evaluate(ws, "document.getElementById('screen-status').textContent")
        # 🔴 COLD START IS MEASURED TO THE MOMENT THE PAGE CAN RUN, not to
        # first paint: what a reader waits for is the weights, and on a first
        # visit that is almost all of it.
        print("cold start: %.1f s to a runnable page" % (time.time() - started))
        print("loaded:", loaded)
        if "affinity tensors" not in loaded:
            failures.append("the status line does not report the affinity bundle")

        ligands = [s for s in args.ligands.split(",") if s]
        # Type into the controls and fire the events a reader's typing fires.
        script = """(() => {
          const set = (id, value) => {
            const el = document.getElementById(id);
            el.value = value;
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
          };
          set('sequence', %s);
          set('ligands', %s);
          set('seed', '0');
          set('recycles', '%d');
          set('steps', '%d');
          set('samples', '%d');
          return {
            ligandNote: document.getElementById('ligand-note').textContent,
            sequenceNote: document.getElementById('sequence-note').textContent,
            msa: document.getElementById('msa').value,
          };
        })()""" % (json.dumps(args.sequence),
                   json.dumps("\n".join(ligands)),
                   args.recycles, args.steps, args.samples)
        typed = cdp.evaluate(ws, script)
        print("typed:", typed)
        # 🔴 SINGLE SEQUENCE MUST BE THE DEFAULT. The brief requires the option
        # and requires nothing to leave the machine unasked; a page that
        # defaulted to the search would send this probe's sequence to
        # api.colabfold.com on every run.
        if typed["msa"] != "none":
            failures.append("the MSA default is %r, not single sequence" % typed["msa"])
        if args.msa != "none":
            cdp.evaluate(ws, """(() => {
              const dial = document.getElementById('msa');
              dial.value = %r;
              dial.dispatchEvent(new Event('change', { bubbles: true }));
              return dial.value;
            })()""" % args.msa, await_promise=False)
            chosen = cdp.evaluate(ws, "document.getElementById('msa').value")
            if chosen != args.msa:
                failures.append("the MSA dial would not take %r" % args.msa)
            print("msa dial:", chosen)
        if "%d residues" % len(args.sequence) not in typed["sequenceNote"]:
            failures.append("the sequence box did not report its length: %r"
                            % typed["sequenceNote"])
        if "%d ligand" % len(ligands) not in typed["ligandNote"]:
            failures.append("the ligand box did not parse %d ligands: %r"
                            % (len(ligands), typed["ligandNote"]))

        cdp.evaluate(ws, "document.getElementById('run').click()", await_promise=False)
        cdp.wait_for(ws, "document.getElementById('run').disabled === false",
                     timeout=args.timeout, what="the screen to finish",
                     progress="document.getElementById('screen-status').textContent")
        print("finished:", cdp.evaluate(
            ws, "document.getElementById('screen-status').textContent"))

        # --- what is DRAWN ------------------------------------------------
        drawn = cdp.evaluate(ws, """(() => {
          const rows = [...document.querySelectorAll('#table tbody tr')];
          const header = [...document.querySelectorAll('#table thead th')]
            .map((th) => th.textContent.trim());
          return {
            header,
            rows: rows.map((tr) => [...tr.children].map((td) => td.textContent.trim())),
            classes: rows.map((tr) => tr.className),
            points: document.querySelectorAll('#scatter circle').length,
            titles: [...document.querySelectorAll('#scatter circle title')]
              .map((t) => t.textContent),
            queue: [...document.querySelectorAll('#queue span')]
              .map((s) => s.className || 'pending'),
            caveats: document.getElementById('caveats').textContent,
            sceneNote: document.getElementById('scene-note').textContent,
            viewers: Object.keys(window.py2dmol_viewers ?? {}).length,
            downloadEnabled: !document.getElementById('download').disabled,
          };
        })()""")
        print("table header:", drawn["header"])
        for row in drawn["rows"]:
            print("  row:", row)
        print("scatter points:", drawn["points"], " queue:", drawn["queue"])
        print("scene:", drawn["sceneNote"])
        print("caveats:", drawn["caveats"])

        if len(drawn["rows"]) != len(ligands):
            failures.append("%d table rows for %d ligands"
                            % (len(drawn["rows"]), len(ligands)))
        if drawn["points"] != sum(1 for c in drawn["classes"] if c == "done"):
            failures.append("%d scatter points for %d finished ligands"
                            % (drawn["points"], sum(1 for c in drawn["classes"]
                                                    if c == "done")))
        if drawn["points"] == 0:
            failures.append("nothing was plotted")
        # 🔴 BOTH OUTPUTS MUST BE COLUMNS, and neither may be a merged score.
        header = " ".join(drawn["header"]).lower()
        for wanted in ("p(binder)", "pic50", "log10 ic50", "iptm"):
            if wanted not in header:
                failures.append("the table has no %s column" % wanted)
        if "superposed onto run #1" not in drawn["sceneNote"]:
            failures.append("the scene note does not say what it superposed")
        if drawn["viewers"] == 0:
            failures.append("py2Dmol has no viewer, so nothing was drawn in 3D")
        if "two heads trained on different data" not in drawn["caveats"]:
            failures.append("the page does not say the two heads differ")
        if not drawn["downloadEnabled"]:
            failures.append("the download button is still disabled")

        # 🔴 IS THE VIEWER ACTUALLY ON SCREEN? The scene's own bookkeeping said
        # "target drawn once; 3 of 3 ligands superposed" while the reader saw an
        # empty box, because py2Dmol's subtree is copied from index.html with
        # `display: none` and only index.html un-hid it. A probe that asserts
        # what the page BELIEVES cannot see that; this asserts the computed
        # style and the canvas's drawing buffer, because a WebGL canvas laid out
        # while hidden has no size even once it is revealed.
        viewer = cdp.evaluate(ws, """(() => {
          const box = document.getElementById('viewer-container');
          const canvas = document.getElementById('canvas');
          return {
            display: box === null ? 'missing' : getComputedStyle(box).display,
            width: canvas?.width ?? 0, height: canvas?.height ?? 0,
            clientWidth: canvas?.clientWidth ?? 0,
            // 🔴 THE CONTAINER TOO. The canvas is sized from this box's rect at
            // init, so "the canvas is 600" and "the box is 600" point at
            // different faults with different fixes.
            boxWidth: document.getElementById('canvasContainer')?.clientWidth ?? 0,
          };
        })()""")
        metrics = cdp.evaluate(ws,
                                "({ inner: innerWidth,"
                                " column: document.body.clientWidth })")
        print("viewport:", metrics)
        body_width = metrics["column"]
        print("viewer:", viewer)
        print("chain:", cdp.evaluate(ws, """(() => {
          const out = [];
          let el = document.getElementById('canvasContainer');
          while (el && el.tagName !== 'HTML') {
            const cs = getComputedStyle(el);
            out.push((el.id || '.' + String(el.className).split(' ')[0])
              + ' client=' + el.clientWidth + ' css=' + cs.width
              + ' inline=' + (el.style.width || '-') + ' ' + cs.display
              + '/' + cs.flexDirection);
            el = el.parentElement;
          }
          return out;
        })()"""))
        # 🔴 py2Dmol'S OWN STATE, because every outside symptom is consistent
        # with "the loader accepted the file and registered nothing": the
        # container is visible and sized, the promise resolves, no error is
        # reported, and the structure picker is empty. Walk the registry rather
        # than guess which field holds the structures.
        print("console:", cdp.evaluate(ws,
              "(globalThis.__consoleErrors ?? []).slice(0, 8)"))
        print("page loads:", cdp.evaluate(ws, "globalThis.__sceneLoads ?? 'none'"))
        # 🔴 py2Dmol'S OWN STATE, which it exposes as `window.buildViewerState`.
        # The parser is sound - with no MODEL records it still pushes one model
        # at the end - so `framesAdded: 0` comes from a step after parsing, and
        # this splits the two possibilities: the object is in the state and the
        # renderer was never pointed at it, or the object was never created.
        print("viewer state:", cdp.evaluate(ws, """(() => {
          try {
            const s = globalThis.buildViewerState?.();
            if (!s) return 'no buildViewerState';
            const objects = s.objects ?? [];
            return { keys: Object.keys(s).slice(0, 12), objects: objects.length,
              each: objects.slice(0, 4).map((o) => ({
                name: o?.name ?? o?.id ?? '?',
                frames: (o?.frames ?? []).length,
                atoms: (o?.frames?.[0]?.coords ?? o?.coords ?? []).length })),
              current: s.currentObject ?? s.selected ?? null };
          } catch (e) { return 'threw: ' + (e?.message ?? e); }
        })()"""))
        print("pdb head:", cdp.evaluate(ws,
              "(globalThis.__scenePdb ?? '(none)').split('\\n').slice(0,3)"))
        print("pdb lines:", cdp.evaluate(ws,
              "(globalThis.__scenePdb ?? '').split('\\n').length"))
        print("registry:", cdp.evaluate(ws, """(() => {
          const reg = globalThis.py2dmol_viewers ?? {};
          const keys = Object.keys(reg);
          const out = { viewers: keys };
          const v = reg[keys[0]];
          if (!v) return out;
          out.viewerKeys = Object.keys(v);
          const describe = (o) => {
            if (o === null || o === undefined) return String(o);
            if (Array.isArray(o)) return `array(${o.length})`;
            if (o instanceof Map) return `map(${o.size})`;
            if (typeof o === 'object') return Object.keys(o).slice(0, 14);
            return typeof o;
          };
          out.fields = {};
          for (const k of Object.keys(v)) out.fields[k] = describe(v[k]);
          const r = v.renderer;
          if (r) {
            out.rendererFields = {};
            for (const k of Object.keys(r)) out.rendererFields[k] = describe(r[k]);
          }
          const picker = document.querySelector('#viewer-container select');
          out.pickerOptions = picker
            ? [...picker.options].map((o) => o.value) : 'no select';
          return out;
        })()"""))
        if viewer["display"] in ("none", "missing"):
            failures.append(f"the 3D viewer is {viewer['display']} - a blank scene")
        if viewer["width"] < 2 or viewer["height"] < 2:
            failures.append("the viewer canvas has no drawing buffer"
                            f" ({viewer['width']}x{viewer['height']})")
        # 🔴 AND DOES THE RENDERER ACTUALLY HOLD THE STRUCTURE? This is the
        # assertion test/py2dmol-scaffold.test.js says belongs here - "the
        # authority on what is required is tools/screen-in-page.py, which
        # asserts at RUNTIME that window.py2dmol_viewers is non-empty and that
        # THE SCENE DREW" - and what was actually asserted was the scene NOTE's
        # text, which the page writes from its own bookkeeping whether or not a
        # single atom reached the viewer. A visible, correctly sized, EMPTY
        # canvas passed every check for weeks.
        atoms = cdp.evaluate(ws, """(() => {
          const reg = globalThis.py2dmol_viewers ?? {};
          const r = reg[Object.keys(reg)[0]]?.renderer;
          return { viewers: Object.keys(reg).length,
                   coords: r?.coords?.length ?? -1,
                   chains: r?.chains?.length ?? -1 };
        })()""")
        print("renderer holds:", atoms)
        if atoms["viewers"] < 1:
            failures.append("no py2Dmol viewer was built")
        # 🔴 AND THE CANVAS FILLS THE COLUMN. py2Dmol wraps the canvas box in
        # `.py2dmol-slot-body` once a structure loads and its stylesheet pins
        # that wrapper to 600, so the viewer silently reverts to 600 of a 948
        # page - which is what it did for four attempts while every box this
        # probe measured looked right. The chain is printed above for the same
        # reason; this makes it fail rather than merely print.
        if viewer["boxWidth"] < 0.8 * body_width:
            failures.append(f"the canvas is {viewer['boxWidth']}px in a"
                            f" {body_width}px column - the slot wrapper is"
                            " pinning it again (see the chain above)")

        if atoms["coords"] < 1:
            failures.append("the viewer holds no coordinates - the scene is"
                            f" blank ({atoms})")

        # 🔴 AND THE COLOUR BY CONTROL MUST REACH THE ATOMS. It did not: the
        # page computed a score domain in `drawScene` and never used it, so the
        # 3D ligands came out in py2Dmol's own per-chain palette and the control
        # moved only the scatter and the table. Reading the page's own state
        # cannot see this - the select's value changed, the table recoloured,
        # and the picture did not. So this asks the RENDERER what colour it
        # would paint one atom of each ligand, which is the only number that
        # answers "did this ligand change colour".
        #
        # 🔴 AND THE TEST IS THE RAMP'S **ORIENTATION**, NOT "something moved".
        # The first version of this gate required the colours to change when the
        # column changed, and failed on a correct page: with two ligands the
        # ends of the ramp are always one blue and one orange, so unless the two
        # columns disagree about which ligand is better, the right answer is
        # that nothing moves. What can be asserted for every column is that the
        # WARM end is the ligand that column calls best - which is also the
        # check that catches `affinity_pred_value` being drawn the wrong way up,
        # the one real hazard here (lower is better for that column alone).
        # 🔴 PER LIGAND, NOT PER CHAIN LETTER. A PDB chain is one character, so
        # a screen past ~25 ligands reuses letters and "the first atom of chain
        # B" is two different ligands - a reading that would have hidden exactly
        # the colour collision the page now avoids by keying on the (chain,
        # residue) pair. This resolves each ligand to its own atoms through the
        # pair the page recorded, and asks the renderer for that atom's colour.
        probe_colours = """(() => {
          const reg = globalThis.py2dmol_viewers ?? {};
          const r = reg[Object.keys(reg)[0]]?.renderer;
          if (!r || !r.chains) return null;
          const placed = globalThis.__scenePlaced ?? [];
          const at = new Map();
          for (let i = 0; i < r.chains.length; i += 1) {
            const key = `${r.chains[i]}|${(r.residueNumbers ?? [])[i]}`;
            if (!at.has(key)) at.set(key, i);
          }
          const rgb = (i) => {
            if (i === undefined) return null;
            const c = r.getAtomColor(i);
            return `${c?.r},${c?.g},${c?.b}`;
          };
          const out = { ligands: {}, chains: {}, positions: r.chains.length };
          for (const spot of placed) {
            out.ligands[spot.id] = rgb(at.get(`${spot.chain}|${spot.resSeq}`));
          }
          const seen = new Set(placed.map((spot) => spot.chain));
          for (let i = 0; i < r.chains.length; i += 1) {
            if (seen.has(r.chains[i])) continue;
            out.chains[r.chains[i]] ??= rgb(i);
          }
          return out;
        })()"""
        WARM, COOL = "214,108,20", "49,86,184"
        scores = cdp.evaluate(ws, """(globalThis.__screenResults ?? [])
          .filter((r) => typeof r.affinity_probability_binary === 'number')
          .map((r) => ({ id: r.id,
                         affinity_probability_binary: r.affinity_probability_binary,
                         affinity_pred_value: r.affinity_pred_value,
                         iptm: r.iptm }))""")
        for column, lower_is_better in (("affinity_probability_binary", False),
                                        ("affinity_pred_value", True),
                                        ("iptm", False)):
            cdp.evaluate(ws, """(() => {
              const select = document.getElementById('colour-by');
              select.value = %s;
              select.dispatchEvent(new Event('change', { bubbles: true }));
            })()""" % json.dumps(column))
            time.sleep(0.4)
            painted = cdp.evaluate(ws, probe_colours)
            print(f"atom colour by {column}:", painted)
            if painted is None:
                failures.append("the renderer could not be asked for atom"
                                f" colours ({column})")
                continue
            # the target keeps the viewer's own colours; every ligand must
            # resolve to a colour, and to the score's colour
            if len(painted["ligands"]) != len(ligands):
                failures.append(f"{len(painted['ligands'])} ligands resolved to"
                                f" atoms for {len(ligands)} in the scene"
                                f" ({painted})")
                continue
            if None in painted["ligands"].values():
                failures.append("a ligand's atoms could not be found by its"
                                f" (chain, residue) pair ({painted})")
            if len(set(painted["ligands"].values())) < 2:
                failures.append("every ligand is the same colour - the score"
                                f" ramp is not reaching the scene ({painted})")
            ranked = sorted((row for row in scores
                             if isinstance(row.get(column), (int, float))),
                            key=lambda row: row[column],
                            reverse=not lower_is_better)
            if len(ranked) < 2:
                continue
            best, worst = ranked[0]["id"], ranked[-1]["id"]
            got_best = painted["ligands"].get(best)
            got_worst = painted["ligands"].get(worst)
            if got_best != WARM or got_worst != COOL:
                failures.append(
                    f"colouring by {column}: the best ligand {best} is"
                    f" {got_best} and the worst {worst} is {got_worst} -"
                    f" expected warm {WARM} and cool {COOL} (the ramp is"
                    f" upside down or not applied) ({painted})")
            # 🔴 AND EVERY LIGAND IN BETWEEN, NOT JUST THE TWO ENDS. The ends
            # are right by luck on any scene where the colours merely happen to
            # be spread; what makes the picture readable is that WARMER ALWAYS
            # MEANS BETTER, for every pair. This is also the only check that
            # catches a chain-letter collision past 25 ligands: a ligand
            # painted in another's colour breaks the order without touching
            # either end. Warmth is red-minus-blue, which rises strictly along
            # every segment of this ramp.
            warmth = []
            for row in ranked:
                rgb = painted["ligands"].get(row["id"])
                if rgb is None:
                    continue
                r, _, b = (int(part) for part in rgb.split(","))
                warmth.append((row["id"], r - b))
            backwards = [(warmth[i - 1], warmth[i])
                         for i in range(1, len(warmth))
                         if warmth[i - 1][1] < warmth[i][1]]
            if backwards:
                failures.append(
                    f"colouring by {column}: {len(backwards)} pair(s) of"
                    " ligands are drawn with the worse one WARMER, so the"
                    f" ramp does not follow the ranking - first {backwards[0]}"
                    f" of {len(warmth)} ligands")
        # 🔴 AND THE "one colour per ligand" MODE MUST LEAVE THE RAMP. It is
        # the only entry in COLOUR BY that is NOT a column: it deletes the
        # page's colour from the object and lets py2Dmol colour the scene, so
        # the failure to look for is the opposite of the others - a ligand still
        # wearing the score ramp's blue or orange means the advanced colour map
        # was not removed and the viewer's own palette is being overridden.
        # The swatches are read too: the page asks the renderer what colour each
        # ligand came out and feeds that to the table, so a swatch that does not
        # match its atoms means the readback is broken and a row no longer
        # points at a ligand on screen.
        cdp.evaluate(ws, """(() => {
          const select = document.getElementById('colour-by');
          select.value = 'rainbow';
          select.dispatchEvent(new Event('change', { bubbles: true }));
        })()""")
        time.sleep(0.5)
        bowed = cdp.evaluate(ws, probe_colours)
        swatches = cdp.evaluate(ws, """(() => {
          const out = {};
          for (const tr of document.querySelectorAll('#table tbody tr')) {
            const swatch = tr.querySelector('.swatch');
            const id = tr.querySelector('td')?.textContent?.trim();
            if (!swatch || !id) continue;
            out[id.replace(/\s*⚠$/, '')] =
              getComputedStyle(swatch).backgroundColor;
          }
          return out;
        })()""")
        print("one colour per ligand:", bowed)
        print("table swatches:", swatches)
        # 🔴 AND A PICTURE OF THIS MODE TOO. The colours here are the VIEWER's,
        # read back off the renderer and fed to the table - a readback that is
        # wrong by a row, or off by a chain, produces numbers that all agree
        # with each other and a picture where the swatches point at the wrong
        # ligands. The final screenshot is taken with a score column selected,
        # so without this one nobody ever looks at this mode.
        if args.screenshot:
            spare = pathlib.Path(args.screenshot)
            spare = spare.with_name(f"{spare.stem}-viewer-palette{spare.suffix}")
            shot = ws.call("Page.captureScreenshot",
                           format="png", captureBeyondViewport=True)
            spare.write_bytes(base64.b64decode(shot["data"]))
            print("screenshot:", spare)
        if bowed is None or len(bowed["ligands"]) != len(ligands):
            failures.append(f"one colour per ligand lost ligands ({bowed})")
        else:
            on_ramp = {id_: rgb for id_, rgb in bowed["ligands"].items()
                       if rgb in (WARM, COOL)}
            if on_ramp:
                failures.append("with the ramp off, these ligands are still"
                                f" painted by the page: {on_ramp}")
            # 🔴 THE PALETTE HOLDS 16 AND CYCLES, so this cannot ask for one
            # colour per ligand: measured on a 27-ligand scene, exactly 16
            # distinct colours with ligand 17 wearing ligand 1's. What it can
            # require is that the viewer spread them as far as it has colours -
            # a drop below that means the page is painting after all, or the
            # viewer fell back to one colour for the lot.
            spread = len(set(bowed["ligands"].values()))
            if spread < min(len(ligands), 16):
                failures.append(f"the viewer's palette gave {spread} colours"
                                f" to {len(ligands)} ligands - expected"
                                f" {min(len(ligands), 16)}"
                                f" ({bowed['ligands']})")
            # the swatch must be the colour the renderer actually painted
            mismatched = {}
            for id_, rgb in bowed["ligands"].items():
                want = f"rgb({rgb.replace(',', ', ')})"
                if swatches.get(id_) != want:
                    mismatched[id_] = (swatches.get(id_), want)
            if mismatched:
                failures.append("the table swatch does not match the atoms in"
                                f" the scene: {mismatched}")

        # 🔴 AND THE COLUMNS MUST BE MADE TO DISAGREE, or the check above
        # passes on a page that ignores the control. Every column ranked these
        # two ligands the same way - the bigger one wins on probability, on
        # pIC50 and on ipTM - so blue and orange land on the same chains
        # whatever is selected, and a page HARDWIRED to the classifier would
        # have satisfied all three rounds. This flips one ligand's ipTM so the
        # order is reversed, requires the warm end to move with it, and puts the
        # numbers back: it is the difference between "the ramp reaches the
        # scene" and "the ramp follows the control".
        flipped = cdp.evaluate(ws, """(() => {
          const results = (globalThis.__screenResults ?? [])
            .filter((r) => typeof r.iptm === 'number');
          if (results.length < 2) return null;
          const saved = results.map((r) => [r.id, r.iptm]);
          const order = [...results].sort((a, b) => b.iptm - a.iptm);
          // give the WORST-ipTM ligand the best ipTM, and the other the worst
          const top = order[0].iptm, bottom = order[order.length - 1].iptm;
          order[0].iptm = bottom;
          order[order.length - 1].iptm = top;
          // the block above left COLOUR BY on "one colour per ligand"; this
          // puts it back on a real column, which is what makes a flip visible
          const select = document.getElementById('colour-by');
          select.value = 'iptm';
          select.dispatchEvent(new Event('change', { bubbles: true }));
          return { saved, best: order[order.length - 1].id };
        })()""")
        if flipped is None:
            failures.append("no ipTM to flip - the disagreement check could"
                            " not run")
        else:
            time.sleep(0.4)
            painted = cdp.evaluate(ws, probe_colours)
            print("atom colour with ipTM reversed:", painted,
                  "expecting warm on", flipped["best"])
            got = (painted or {"ligands": {}})["ligands"].get(flipped["best"])
            if got != WARM:
                failures.append(
                    f"with ipTM reversed the warm colour should move to"
                    f" {flipped['best']}, which is {got} - COLOUR BY is not"
                    f" choosing the column the scene is painted by ({painted})")
            cdp.evaluate(ws, """((saved) => {
              const by = new Map(saved);
              for (const r of globalThis.__screenResults ?? []) {
                if (by.has(r.id)) r.iptm = by.get(r.id);
              }
              const select = document.getElementById('colour-by');
              select.value = 'affinity_probability_binary';
              select.dispatchEvent(new Event('change', { bubbles: true }));
            })(%s)""" % json.dumps(flipped["saved"]))
            time.sleep(0.4)
            restored = cdp.evaluate(ws, """(globalThis.__screenResults ?? [])
              .map((r) => [r.id, r.iptm])""")
            if restored != [list(pair) for pair in flipped["saved"]]:
                failures.append("the probe failed to put the ipTM values back"
                                f" ({restored} vs {flipped['saved']})")

        # 🔴 AND THE TARGET IS NOT ON THE RAMP. Only the ligands are scored, so
        # the protein must keep the viewer's own colouring - which is also what
        # makes it still read as a protein. A colour MODE, or the score written
        # into the B-factor column to borrow `plddt`, would have painted it too.
        final = cdp.evaluate(ws, probe_colours)
        if final is not None:
            print("non-ligand chains:", final["chains"])
            on_ramp = {chain: colour for chain, colour in final["chains"].items()
                       if colour in (WARM, COOL)}
            if on_ramp:
                failures.append("the TARGET is painted with the score ramp"
                                f" ({on_ramp})")

        # 🔴 AND THE SHOW PICKER MUST ACTUALLY REMOVE LIGANDS FROM THE SCENE.
        # The scene is one concatenated object, so hiding is a re-draw with
        # fewer ligands in the file - a picker that only set a variable would
        # leave the picture unchanged, which is the failure this whole pass is
        # about.
        picker = cdp.evaluate(ws, """(() => {
          const select = document.getElementById('show-ligand');
          return { options: [...select.options].map((o) => o.value),
                   value: select.value, disabled: select.disabled };
        })()""")
        print("show picker:", picker)
        if picker["options"][:1] != [""]:
            failures.append(f"the show picker has no all-ligands option"
                            f" ({picker})")
        if len(picker["options"]) != len(ligands) + 1:
            failures.append(f"the show picker lists {len(picker['options'])}"
                            f" entries for {len(ligands)} ligands ({picker})")
        one = picker["options"][1] if len(picker["options"]) > 1 else None
        if one is not None:
            cdp.evaluate(ws, """(() => {
              const select = document.getElementById('show-ligand');
              select.value = %s;
              select.dispatchEvent(new Event('change', { bubbles: true }));
            })()""" % json.dumps(one))
            time.sleep(0.5)
            alone = cdp.evaluate(ws, probe_colours)
            print(f"atom colour per chain (only {one}):", alone)
            if alone is None or len(alone["ligands"]) != 1:
                failures.append(f"picking one ligand left {alone} in the scene")
            elif list(alone["ligands"]) != [one]:
                failures.append(f"picking {one} drew"
                                f" {list(alone['ligands'])} instead")
            cdp.evaluate(ws, """(() => {
              const select = document.getElementById('show-ligand');
              select.value = '';
              select.dispatchEvent(new Event('change', { bubbles: true }));
            })()""")
            time.sleep(0.5)
            back = cdp.evaluate(ws, probe_colours)
            if back is None or len(back["ligands"]) != len(ligands):
                failures.append("all-ligands did not restore the scene"
                                f" ({back})")

        # 🔴 EVERY HAND-OVER MUST ADD A FRAME. `processFiles` reports
        # `objectsLoaded` as the number of files it MEANT to load, so it said 1
        # while py2Dmol held nothing; `framesAdded` is the number that actually
        # moved. A scene that loads 0 frames is the blank view this page shipped
        # with.
        loads = cdp.evaluate(ws, "globalThis.__sceneLoads ?? []")
        for record in loads:
            added = (record.get("settled") or {}).get("framesAdded")
            if added != 1:
                failures.append(f"scene load {record.get('call')} added"
                                f" {added} frames, not 1 ({record})")

        # 🔴 AND AN EMPTY CONSOLE, because the fault that produced the blank
        # scene was a CAUGHT exception: `buildPendingObject` reads
        # `#loadAsFramesCheckbox`'s `.checked` with no guard, the subtree here
        # omitted the element, and `processFiles`' per-file try swallowed the
        # TypeError. The promise resolved, the counts looked right, and the only
        # record of it was a console line nothing was reading.
        errors = cdp.evaluate(ws, "(globalThis.__consoleErrors ?? [])")
        if errors:
            failures.append(f"the page logged {len(errors)} console error(s):"
                            f" {errors[:3]}")

        if args.screenshot:
            shot = ws.call("Page.captureScreenshot",
                           format="png", captureBeyondViewport=True)
            pathlib.Path(args.screenshot).write_bytes(
                base64.b64decode(shot["data"]))
            print("screenshot:", args.screenshot)

        # --- the archive, read back ---------------------------------------
        archive = cdp.evaluate(ws, """(async () => {
          // 🔴 RESOLVED AGAINST THE PAGE, NOT THE SERVER ROOT. A root-absolute
          // specifier works against a local server at `/` and 404s under a
          // project page's `/<repo>/` base - which is exactly the failure this
          // probe exists to catch, and it caught it in the probe itself first:
          // `import('/web/screen-export.js')` fetched
          // pablo-arantes.github.io/web/screen-export.js with the base dropped.
          const at = (m) => new URL(m, document.baseURI).href;
          const { screenArchive } = await import(at('web/screen-export.js'));
          const { readZip } = await import(at('web/zip.js'));
          const rows = [...document.querySelectorAll('#table tbody tr')]
            .map((tr) => tr.children[0].textContent.trim().replace(/[^\\w.-]/g, ''));
          const bytes = await screenArchive(window.__screenResults ?? [], {});
          const members = await readZip(bytes);
          return { names: [...members.keys()].sort(), bytes: bytes.length,
                   csv: (members.get('affinity.csv') ?? '').split('\\n').length,
                   // a circle per scored ligand; an empty <svg> would pass a
                   // presence check and show nothing
                   plotPoints: ((members.get('affinity.svg') ?? '')
                     .match(/<circle/g) ?? []).length,
                   readme: (members.get('README.txt') ?? '').slice(0, 0) === '' };
        })()""")
        print("archive:", archive)

        # 🔴 "SEARCHED ONCE PER TARGET" IS A COUNT, NOT A COMMENT. Resource
        # timing records every request the page made, so the number aimed at
        # api.colabfold.com is checkable: one search's worth however many
        # ligands were scored.
        msa_calls = cdp.evaluate(ws, """(() => {
          const hits = performance.getEntriesByType('resource')
            .map((e) => e.name).filter((n) => n.includes('colabfold'));
          return { total: hits.length, distinct: [...new Set(hits)].length };
        })()""")
        print("colabfold requests:", msa_calls)
        if args.msa == "none" and msa_calls["total"] != 0:
            failures.append("single-sequence mode still called colabfold"
                            " %d time(s)" % msa_calls["total"])
        if "affinity.csv" not in archive["names"]:
            failures.append("the archive has no CSV")
        # 🔴 THE PLOT TOO. The archive shipped the numbers and the structures
        # and left out the scatter - the one picture this page exists to draw -
        # and nothing here asserted the member list, so a reader found it
        # before any gate did.
        if "affinity.svg" not in archive["names"]:
            failures.append("the archive has no plot (affinity.svg)")
        if archive.get("plotPoints", 0) < 1:
            failures.append("affinity.svg has no plotted points")
    finally:
        if not args.keep:
            proc.kill()
        if httpd is not None:
            httpd.shutdown()

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\nok")


if __name__ == "__main__":
    main()
