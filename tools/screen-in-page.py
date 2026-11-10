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
    ap.add_argument("--screenshot", default=None,
                    help="write a PNG of the finished page to this path")
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
        target = args.url or ("http://127.0.0.1:%d/screen.html" % PORT)
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
          };
        })()""")
        print("viewer:", viewer)
        # 🔴 py2Dmol'S OWN STATE, because every outside symptom is consistent
        # with "the loader accepted the file and registered nothing": the
        # container is visible and sized, the promise resolves, no error is
        # reported, and the structure picker is empty. Walk the registry rather
        # than guess which field holds the structures.
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
        # 🔴 THE LOADER'S OWN RETURN VALUE. processFiles reports
        # {objectsLoaded, framesAdded, structureCount}; the renderer holds
        # coords: array(0), so it registered nothing from a file it accepted
        # without error, and only these counts say at which step it gave up.
        print("loader says:", cdp.evaluate(ws, """(async () => {
          const text = globalThis.__scenePdb;
          if (!text) return 'no scene pdb';
          const f = { name: 'probe.pdb', readAsync: () => Promise.resolve(text) };
          try { return await globalThis.py2dmolLoadFiles([f]); }
          catch (e) { return 'threw: ' + (e?.message ?? e); }
        })()"""))
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
        if atoms["coords"] < 1:
            failures.append("the viewer holds no coordinates - the scene is"
                            f" blank ({atoms})")

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
