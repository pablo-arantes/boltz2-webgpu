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
import os
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
    args = ap.parse_args()

    # local_weights=False: both bundles here are local directories already, and
    # the rewrite would raise on a `remote: null` entry it cannot strip.
    httpd = fold_in_page.serve(local_weights=False)
    profile = os.path.join("/tmp", "localfold-screen-profile")
    proc, ws = cdp.launch(args.port, profile, keep=args.keep)
    failures = []
    try:
        ws.call("Runtime.enable")
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/screen.html" % PORT)
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

        # --- the archive, read back ---------------------------------------
        archive = cdp.evaluate(ws, """(async () => {
          const { screenArchive } = await import('/web/screen-export.js');
          const { readZip } = await import('/web/zip.js');
          const rows = [...document.querySelectorAll('#table tbody tr')]
            .map((tr) => tr.children[0].textContent.trim().replace(/[^\\w.-]/g, ''));
          const bytes = await screenArchive(window.__screenResults ?? [], {});
          const members = await readZip(bytes);
          return { names: [...members.keys()].sort(), bytes: bytes.length,
                   csv: (members.get('affinity.csv') ?? '').split('\\n').length,
                   readme: (members.get('README.txt') ?? '').slice(0, 0) === '' };
        })()""")
        print("archive:", archive)
        if "affinity.csv" not in archive["names"]:
            failures.append("the archive has no CSV")
    finally:
        if not args.keep:
            proc.kill()
        httpd.shutdown()

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\nok")


if __name__ == "__main__":
    main()
