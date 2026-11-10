/**
 * The screen page: one target, a .smi file, a ranked table and one scene.
 *
 * 🔴 THIS FILE IS WIRING AND NOTHING ELSE. Every decision it looks like it is
 * making is made somewhere that can be tested without a browser: the run loop
 * and its stop/resume in web/affinity-run.js, the crop and the featurisation in
 * shared/af3/affinity/, the two passes in web/affinity-passes.js, the
 * superposition in web/screen-scene.js, the scatter and the table in
 * web/screen-view.js, the CSV and the ZIP in web/screen-export.js, the MSA
 * cache in shared/input/msa-cache.js. What is here is DOM, and it is here
 * because a page is the one part no node test can reach.
 *
 * 🔴 NOTHING LEAVES THE MACHINE EXCEPT THE MSA SEARCH, AND THAT IS OPT-IN. The
 * page loads with "Single sequence" selected; MMseqs2 is a choice the reader
 * makes, it goes to api.colabfold.com once per target, and the a3m is cached so
 * a second run of the same target does not repeat it. The weights come from the
 * registry's pinned revisions and are cached in CacheStorage. There is no
 * server in this tool.
 */
import { getDevice } from "./model.js";
import { settleReleasedMemory } from "../webgpu/runtime/allocator.js";
import { loadStructureWeights, loadAffinityWeights } from "./screen-weights.js";
import { makeFoldStructure, makeFoldAffinityPass, makeAffinityModule }
  from "./affinity-passes.js";
import { runAffinityScreen, AFFINITY_PASS } from "./affinity-run.js";
import { parseSmiles } from "../shared/chem/smiles.js";
import { parseSmi, uniqueIds, describeSmi } from "../shared/input/smi.js";
import { alignmentForScreen } from "../shared/input/msa-cache.js";
import { generateMmseqs2Msa } from "../shared/input/mmseqs2-api.js";
import { ScreenScene, scoreDomain, renumberLigand } from "./screen-scene.js";
import { drawScatter, drawTable, DEFAULT_SORT, startsDescending }
  from "./screen-view.js";
import { screenArchive, statusOf } from "./screen-export.js";

const $ = (id) => document.getElementById(id);
const state = {
  results: [],
  jobs: [],
  scene: new ScreenScene(),
  visible: new Set(),
  sort: { ...DEFAULT_SORT },
  colourBy: "affinity_probability_binary",
  running: false,
  controller: null,
  loaded: null,
  settings: null,
  sequence: "",
};

function say(message, bad = false) {
  const box = $("screen-status");
  box.textContent = message;
  box.classList.toggle("bad", bad === true);
}

/**
 * 🔴 THE LIGAND LIST IS PARSED BY shared/input/smi.js AND NOT HERE. A `.smi`
 * file is SMILES, whitespace, then an optional name - and the three bugs that
 * module's own tests record (a leading tab read as a SMILES, ids deduplicated
 * before sanitising, whitespace substituted before the trailing dot was
 * stripped) are exactly what a second ad-hoc split in a page would reproduce.
 */
function readLigands() {
  const text = $("ligands").value;
  const parsed = parseSmi(text);
  // 🔴 `uniqueIds` RETURNS THE IDS, NOT THE ENTRIES. It is a parallel array of
  // strings in entry order - deduplicated and made filename-safe - and reading
  // it as the entries gave every job `id: undefined`, which the run loop
  // happily carried into a table of rows called "undefined" and a describeSmi
  // that counted every ligand as unnamed. The ids are zipped back on.
  const ids = uniqueIds(parsed.entries);
  const entries = parsed.entries.map((entry, index) => ({
    ...entry, id: ids[index],
  }));
  $("ligand-note").textContent = entries.length === 0
    ? "a .smi file, or one SMILES a line below"
    : describeSmi(parsed);
  // 🔴 AND WHAT COULD NOT BE READ IS SAID OUT LOUD. parseSmi collects
  // `problems` and `skipped`; dropping them silently is how a 300-ligand file
  // screens 280 and reads as complete.
  if (parsed.problems.length > 0 || parsed.skipped.length > 0) {
    $("ligand-note").textContent += ` \u00b7 ${parsed.problems.length} problem`
      + `${parsed.problems.length === 1 ? "" : "s"}`
      + (parsed.skipped.length > 0 ? `, ${parsed.skipped.length} skipped` : "");
  }
  return entries;
}

/**
 * 🔴 A 300-LIGAND SCREEN IS AN HOUR OF SOMEBODY'S BROWSER, AND A RELOAD LOST
 * ALL OF IT. Stop and in-page failures keep their results (the run loop
 * returns what finished), but nothing survived a refresh or a crash. The rows
 * are written to sessionStorage as they complete - not localStorage, because
 * this is crash recovery for one tab's work rather than a history, and a
 * screen of a different target in a new tab must not inherit it.
 *
 * 🔴 AND THE STRUCTURES ARE DROPPED FROM WHAT IS STORED. A PDB per ligand is
 * tens of kilobytes and sessionStorage is a few megabytes; keeping them would
 * fail the quota partway through a long run and lose everything rather than
 * the 3D scene. The numbers and the CSV survive; the scene does not.
 */
const RESUME_KEY = "localfold-screen-results-v1";

function persistResults() {
  try {
    sessionStorage.setItem(RESUME_KEY, JSON.stringify({
      sequence: state.sequence, seed: state.seed, settings: state.settings,
      msa: state.msa, msaDepth: state.msaDepth,
      results: state.results.map(({ pdb, ...rest }) => rest),
    }));
  } catch {
    // quota, a private window, or storage disabled: recovery is a courtesy,
    // never a dependency, and losing it must not interrupt the run.
  }
}

/** Offer whatever a previous tab left, without touching the live state. */
function recoverResults() {
  try {
    const raw = sessionStorage.getItem(RESUME_KEY);
    if (raw === null) return null;
    const saved = JSON.parse(raw);
    return Array.isArray(saved?.results) && saved.results.length > 0
      ? saved : null;
  } catch {
    return null;
  }
}

function refresh() {
  // 🔴 PUBLISHED FOR THE PROBE, AND ONLY THAT. tools/screen-in-page.py builds
  // the archive from what the page actually holds; without a handle it would
  // have to rebuild the results from the table's text, which is a different
  // thing from what the ZIP is made of.
  globalThis.__screenResults = state.results;
  persistResults();
  drawScatter($("scatter"), state.results,
    { colourBy: state.colourBy, highlight: state.highlight });
  drawTable($("table"), state.results, {
    sort: state.sort, colourBy: state.colourBy, visible: state.visible,
    onSort: (key) => {
      state.sort = state.sort.key === key
        ? { key, ascending: !state.sort.ascending }
        : { key, ascending: !startsDescending(key) };
      refresh();
    },
    onToggle: (id, visible) => {
      if (visible) state.visible.add(id); else state.visible.delete(id);
      state.scene.setVisible(id, visible);
      drawScene();
      refresh();
    },
    onPick: (id) => { state.highlight = id; refresh(); },
  });
  import("./screen-view.js").then(({ caveatsFor }) => {
    $("caveats").textContent = caveatsFor(state.results);
  });
  $("download").disabled = state.results.length === 0;
  renderQueue();
}

function renderQueue() {
  const queue = $("queue");
  queue.replaceChildren();
  for (const job of state.jobs) {
    const found = state.results.find((result) => result.id === job.id);
    const chip = document.createElement("span");
    chip.textContent = job.id;
    chip.title = job.smiles;
    chip.className = found === undefined
      ? (state.runningId === job.id ? "running" : "")
      : statusOf(found);
    queue.append(chip);
  }
}

/**
 * Un-hide py2Dmol's viewer, and resize its canvas.
 *
 * 🔴 WITHOUT THIS THE SCENE IS BLANK AND NOTHING REPORTS A FAULT. The vendor
 * subtree is copied from index.html with `style="display: none;"` on
 * `#viewer-container`, because index.html reveals it when a fold lands - and
 * this page never did. So the structure WAS loaded, py2Dmol held it, the note
 * read "target drawn once; 3 of 3 ligands superposed", and the reader saw an
 * empty box. A probe that asserts the scene's own bookkeeping cannot see this;
 * only looking at the page can, which is how it was found.
 *
 * 🔴 AND THE RESIZE IS NOT OPTIONAL. A WebGL canvas laid out inside
 * `display: none` has no size, so revealing the container alone leaves a 0x0
 * drawing buffer and the box stays blank. `_updateCanvasDimensions` is what
 * index.html calls for the same reason.
 */
function revealViewer() {
  const container = document.getElementById("viewer-container");
  if (container !== null && getComputedStyle(container).display === "none") {
    container.style.display = "flex";
  }
  try {
    const registry = globalThis.py2dmol_viewers ?? {};
    const renderer = registry[Object.keys(registry)[0]]?.renderer;
    // 🔴 RESIZE **AND** RENDER, which is the pair the vendor bundle itself
    // uses: its own ResizeObserver runs `_updateCanvasDimensions(),
    // render("ResizeObserver")`. A resize alone changes the buffer and draws
    // nothing into it.
    renderer?._updateCanvasDimensions?.();
    renderer?.render?.("screen-reveal");
  } catch { /* a resize is not worth losing the scene over */ }
}

/**
 * Hand the scene to py2Dmol.
 *
 * 🔴 ONE OBJECT PER UPDATE, NOT ONE PER LIGAND, AND THE COST IS NAMED. py2Dmol
 * ingests virtual files and keeps an object per file; adding each ligand as its
 * own object would let it be hidden without redrawing, but every ligand would
 * then be a separate object in the viewer's own list and the camera is reset by
 * its loader either way. So the whole visible scene is concatenated and handed
 * over as one structure, and show/hide is a redraw. For a few hundred small
 * ligands that is cheap; it is the wrong shape for thousands, and that is a
 * limit rather than a defect to find later.
 */
function drawScene() {
  const load = globalThis.py2dmolLoadFiles;
  if (typeof load !== "function" || state.scene.target === null) return;
  const done = state.results.filter((result) => statusOf(result) === "done");
  const domain = scoreDomain(done, state.colourBy);
  const target = state.scene.target.replace(/\nEND\s*$/, "");
  const parts = [target];
  // 🔴 EVERY LIGAND IS RENUMBERED INTO THE SCENE'S OWN SERIAL/CHAIN/RESIDUE
  // SPACE. Each comes from a separate fold and restarts its serials, and
  // py2Dmol resolves CONECT by serial with the last atom winning - so
  // concatenating them raw puts every ligand's bonds on the last ligand and
  // leaves the others as loose spheres. See renumberLigand.
  let serial = 1 + (target.match(/^(ATOM|HETATM)/gm) ?? []).length;
  let resSeq = 1;
  const letters = "BCDEFGHIJKLMNOPQRSTUVWXYZ";
  for (const entry of state.scene.visible()) {
    const chain = letters[(resSeq - 1) % letters.length];
    const renumbered = renumberLigand(entry.ligand, serial, chain, resSeq);
    parts.push(renumbered.pdb);
    serial = renumbered.nextSerial;
    resSeq += 1;
  }
  parts.push("END");
  const pdb = parts.join("\n");
  // 🔴 EXPOSED FOR THE PROBE, which re-loads this exact text through
  // `py2dmolLoadFiles` and prints the loader's own counters. That is what
  // showed the scene is an object with no FRAME - see the blank-scene note in
  // docs/MEASURED.md - and the diagnosis is not reproducible without it.
  globalThis.__scenePdb = pdb;
  const file = (name, text) => ({ name, readAsync: () => Promise.resolve(text) });
  try {
    // 🔴 REVEALED **BEFORE** THE LOAD, AND THAT ORDER IS THE WHOLE FIX. py2Dmol
    // fits its camera to the structure as it ingests it, reading the canvas's
    // own size; a canvas laid out inside `display: none` is 0x0, so loading
    // first and revealing afterwards left a correctly sized canvas showing
    // nothing - the container was visible, the buffer was 707x707, and the box
    // was white. Revealing first means the fit happens against a real
    // viewport.
    //
    // 🔴 AND IT IS RE-ASSERTED AFTERWARDS, because the vendor bundle finishes
    // building asynchronously and re-applies the hidden state it was handed:
    // measured `before: none -> after: flex` on the synchronous call and
    // `display: none` again by the end of the run.
    revealViewer();
    const loaded = load([file("screen.pdb", pdb)]);
    Promise.resolve(loaded)
      .then(() => { revealViewer(); requestAnimationFrame(revealViewer); })
      // 🔴 AN ASYNC REFUSAL MUST BE REPORTED, NOT SWALLOWED. `py2dmolLoadFiles`
      // is `processFiles`, which is async: the synchronous `catch` below cannot
      // see it reject, so an empty handler here turns a failed load into a
      // blank box with no message - which is exactly the symptom this whole
      // fix started from.
      .catch((error) => {
        say(`the viewer refused the scene: ${error?.message ?? error}`, true);
      });
  } catch (error) {
    say(`the viewer refused the scene: ${error?.message ?? error}`, true);
    return;
  }
  // The legend under the viewer carries the colour meaning, since the viewer
  // itself is drawing one structure and knows nothing about scores.
  const rms = state.scene.ligands
    .map((entry) => entry.rms).filter((value) => Number.isFinite(value));
  // 🔴 A LIGAND WHOSE FIT WAS DECLINED IS NOT SUPERPOSED, AND WAS COUNTED AS
  // IF IT WERE. `superposeRun` returns `on: "none"` when the CA counts differ
  // or there are fewer than three, and `rms: null` with it - so a ligand drawn
  // tens of angstrom away in its own frame was being reported inside "N of M
  // superposed onto run #1".
  const unfitted = state.scene.ligands
    .filter((entry) => entry.on === "none").map((entry) => entry.id);
  $("scene-note").textContent =
    `target drawn once; ${state.scene.visible().length} of `
    + `${state.scene.ligands.length} ligands superposed onto run #1`
    + (rms.length > 1
      ? ` (target CA rms ${Math.max(...rms).toFixed(1)} A at worst - the`
        + " target itself folds differently per ligand)"
      : "")
    + (unfitted.length === 0 ? ""
      : `  ·  ⚠ ${unfitted.length} NOT superposed`
        + ` (${unfitted.join(", ")}) - drawn in its own frame`)
    // 🔴 NO COLOUR CLAIM. This printed the score range as though the scene were
    // coloured by it; nothing passes a colour to py2Dmol, so the viewer uses
    // its own default and the claim was false. The table swatch and the
    // scatter do share a colour; the 3D scene does not, and saying so beats
    // asserting a mapping that was never applied.
    + "  ·  scene colours are the viewer's own, not the score";
}

async function boot() {
  say("Asking for a WebGPU device…");
  let device;
  try {
    device = await getDevice();
  } catch (error) {
    say(String(error?.message ?? error), true);
    return;
  }
  say("Fetching the Boltz-2 structure weights…");
  const structure = await loadStructureWeights((progress) => {
    const done = (progress.loadedBytes / 1048576).toFixed(0);
    const total = (progress.totalBytes / 1048576).toFixed(0);
    say(`Boltz-2 weights ${done} / ${total} MiB`);
  });
  say("Fetching the affinity checkpoint…");
  const affinity = await loadAffinityWeights((progress) => {
    const done = (progress.loadedBytes / 1048576).toFixed(1);
    say(`affinity weights ${done} MiB`);
  });
  state.loaded = { device, structure, affinity };
  say(`Ready. Boltz-2 ${structure.depths.pairformerBlocks} trunk blocks,`
    + ` ${affinity.tensors} affinity tensors`
    + ` (${(affinity.bytes / 1048576).toFixed(0)} MiB decoded).`
    + " Paste a target and some SMILES.");
  $("run").disabled = false;
  const recovered = recoverResults();
  if (recovered !== null) {
    const done = recovered.results.filter((r) => statusOf(r) === "done").length;
    say(`Ready. A previous run in this tab left ${done} scored ligand(s);`
      + " they are in the table and downloadable. Screening again replaces"
      + " them.");
    state.results = recovered.results;
    state.sequence = recovered.sequence ?? "";
    state.settings = recovered.settings ?? null;
    state.seed = recovered.seed;
    state.msa = recovered.msa;
    state.msaDepth = recovered.msaDepth;
  }
  refresh();
}

async function screen() {
  if (state.running) return;
  const sequence = $("sequence").value.replace(/\s+/g, "").toUpperCase();
  if (sequence.length < 4) { say("paste a protein sequence first", true); return; }
  const entries = readLigands();
  if (entries.length === 0) { say("no ligands to screen", true); return; }

  state.running = true;
  $("run").disabled = true;
  $("stop").disabled = false;
  state.controller = new AbortController();
  // 🔴 A NEW RUN CLEARS THE SCENE, because run #1 is the frame everything else
  // is in: keeping the old target and superposing a new run's ligands onto it
  // would put them in a frame derived from a fold that is no longer shown.
  state.scene.clear();
  state.visible.clear();
  state.results = [];
  state.sequence = sequence;
  state.jobs = entries.map((entry) => ({ id: entry.id, smiles: entry.smiles }));
  renderQueue();

  let alignment = null;
  if ($("msa").value === "mmseqs2") {
    try {
      const found = await alignmentForScreen(sequence, {
        onStatus: say,
        search: async (query) => {
          const searched = await generateMmseqs2Msa(query,
            { signal: state.controller.signal });
          return typeof searched === "string" ? searched : searched?.a3m ?? searched?.text;
        },
      });
      alignment = found.a3m;
      state.msa = "mmseqs2";
      state.msaDepth = found.depth;
      state.msaSource = found.source;
    } catch (error) {
      // 🔴 STOP IS NOT AN MSA FAILURE, AND SAYING SO WAS A LIE THAT HID THE
      // TRUTH. The MMseqs2 client retries a rate-limit forever with jitter and
      // no cap; pressing Stop aborts that wait, which arrived here as an
      // error, was reported as "screening as single sequence", and then
      // `runAffinityScreen` saw the already-aborted signal and returned with
      // ZERO results. The fallback never happened and the message was false.
      if (error?.name === "AbortError" || state.controller?.signal.aborted) {
        state.running = false;
        $("run").disabled = false;
        $("stop").disabled = true;
        say("stopped while waiting for the MSA search - nothing was screened."
          + " Choose Single sequence to screen without an alignment.");
        return;
      }
      say(`the MSA search failed (${error?.message ?? error}) - screening as`
        + " single sequence", true);
    }
  }

  const { device, structure, affinity } = state.loaded;
  const settings = {
    recyclingSteps: Number($("recycles").value),
    samplingSteps: Number($("steps").value),
    diffusionSamples: Number($("samples").value),
  };
  state.settings = settings;
  const seed = Number($("seed").value);
  state.seed = seed;
  if ($("msa").value !== "mmseqs2") { state.msa = "none"; state.msaDepth = 0; }
  const deps = {
    parseLigand: async (smiles) => parseSmiles(smiles),
    foldStructure: makeFoldStructure(device, structure.weights, {
      sequence, alignment, seed,
      recycles: settings.recyclingSteps, steps: settings.samplingSteps,
    }),
    foldAffinityPass: makeFoldAffinityPass(device, structure.weights, { seed }),
    affinityModule: makeAffinityModule(device, affinity.read, { precision: "f32" }),
    // 🔴 THE PER-LIGAND TEARDOWN THE RUN LOOP DOCUMENTS AND CALLS, which was
    // never supplied: `web/affinity-run.js` invokes `deps.freeRunBuffers?.()`
    // in a `finally` for every ligand, failures included, and the page passed
    // nothing. Residency measures flat without it (test:affinity-memory), so
    // this is the drain rather than a leak fix - but a loop that never drains
    // is one whose teardown has never run.
    freeRunBuffers: async () => { await settleReleasedMemory(device); },
  };

  const startedAt = Date.now();
  const run = await runAffinityScreen(state.jobs, deps, {
    ...settings,
    signal: state.controller.signal,
    mwCorrection: false,
    onProgress: ({ phase, id, index, total, result, eta, failure }) => {
      if (phase === "start") {
        state.runningId = id;
        say(`[${index + 1}/${total}] ${id}`
          + (eta.seconds === null ? ""
            : ` · about ${Math.round(eta.seconds)} s left`
              + (eta.confident ? "" : " (one sample so far)")));
        $("eta").textContent = eta.seconds === null ? ""
          : `~${Math.round(eta.seconds)} s remaining`;
        renderQueue();
      } else if (phase === "done") {
        state.runningId = null;
        state.results = [...state.results, result];
        if (typeof result.pdb === "string" && result.pdb.length > 0) {
          // 🔴 A DRAWING FAILURE MUST NOT COST THE LIGAND ITS RESULT. This
          // handler runs inside the run loop's try, so a throw here was caught
          // as a LIGAND failure: each ligand got a result row AND a failure
          // row, and the probe read 4 rows for 2 ligands. It was a real throw
          // (a legend referencing deleted variables), and the scene is a
          // picture - the numbers are the work.
          try {
            state.scene.add(result.id, result.pdb);
            state.visible.add(result.id);
            drawScene();
          } catch (error) {
            console.error("the scene could not be drawn", error);
          }
        }
        refresh();
      } else if (phase === "failed") {
        state.runningId = null;
        // 🔴 A LOST DEVICE OR A FATAL OOM ENDS THE RUN; it does not produce N
        // identical failure rows. The weights live on the dead device, so
        // every remaining ligand would fail the same way and bury the one
        // message that matters. Earlier results stay downloadable.
        if (/device is lost|device was lost|GPUDevice|out of memory|budget/i
            .test(String(failure.error))) {
          state.controller?.abort();
          state.deviceLost = true;
        }
        state.results = [...state.results,
          { id: failure.id, smiles: failure.smiles, error: failure.error,
            name: failure.name }];
        // 🔴 A FAILURE IS SHOWN, NOT ONLY COUNTED. The run loop catches every
        // ligand's error so one bad SMILES cannot end a screen - which also
        // means a real bug arrives as a row with a dash in it unless the page
        // says what happened.
        say(`${failure.id}: ${failure.error}`, true);
        if (failure.stack !== undefined) console.error(failure.id, failure.stack);
        refresh();
      }
    },
  });

  // 🔴 THE FAILURES ARE MERGED IN, not reported separately: a screen's record
  // is every ligand that was attempted. runAffinityScreen returns them apart
  // so a caller can refuse to carry on; this page carries on and keeps them.
  const seen = new Set(state.results.map((result) => result.id));
  state.results = [...state.results,
    ...run.failures.filter((failure) => !seen.has(failure.id)),
    ...run.results.filter((result) => result.skipped !== undefined
      && !seen.has(result.id))];

  state.running = false;
  state.controller = null;
  $("run").disabled = false;
  $("stop").disabled = true;
  $("eta").textContent = "";
  const done = state.results.filter((result) => statusOf(result) === "done").length;
  const seconds = ((Date.now() - startedAt) / 1000).toFixed(0);
  if (state.deviceLost === true) {
    say(`the GPU device was lost after ${done} ligand(s) - the weights went`
      + " with it, so the run stopped. Your finished results are still"
      + " downloadable; reload the page to screen the rest.", true);
    $("run").disabled = true;
    refresh();
    return;
  }
  say(`${done} of ${state.jobs.length} ligands scored in ${seconds} s`
    + (run.stopped ? " (stopped)" : "")
    + (run.throughput?.medianMs
      ? ` · median ${(run.throughput.medianMs / 1000).toFixed(1)} s a ligand`
      : "")
    + ` · ${settings.recyclingSteps} recycles / ${settings.samplingSteps}`
    + ` steps / ${settings.diffusionSamples} samples`
    + (settings.samplingSteps < AFFINITY_PASS.samplingSteps
      ? " - below upstream's 200, so read the ranking rather than the numbers"
      : ""));
  refresh();
}

function wire() {
  $("run").addEventListener("click", () => { screen().catch((error) => {
    state.running = false;
    $("run").disabled = false;
    $("stop").disabled = true;
    say(String(error?.message ?? error), true);
    // 🔴 AND IT IS LOGGED, not only shown. A one-line message in a status bar
    // is all a reader gets; a real bug needs a stack, and the console is where
    // it can be had.
    console.error(error);
  }); });
  // 🔴 STOP ABORTS THE CURRENT LIGAND AND KEEPS EVERY FINISHED ONE. The run
  // loop returns `{results, stopped: true}` on an AbortError rather than
  // throwing, so the table and the ZIP stay downloadable - which the brief asks
  // for by name ("keep partial results downloadable if a run fails").
  $("stop").addEventListener("click", () => {
    state.controller?.abort();
    say("stopping after this ligand…");
  });
  $("download").addEventListener("click", async () => {
    const bytes = await screenArchive(state.results, {
      target: `${state.sequence.length} residues`,
      sequence: state.sequence, settings: state.settings ?? {},
    });
    const url = URL.createObjectURL(new Blob([bytes],
      { type: "application/zip" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `screen-${state.results.length}-ligands.zip`;
    link.click();
    URL.revokeObjectURL(url);
  });
  $("ligands").addEventListener("input", readLigands);
  $("smi-file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (file === undefined) return;
    $("ligands").value = await file.text();
    readLigands();
  });
  $("sequence").addEventListener("input", () => {
    const length = $("sequence").value.replace(/\s+/g, "").length;
    $("sequence-note").textContent = length === 0
      ? "one protein chain, amino acids" : `${length} residues`;
  });
  $("colour-by").addEventListener("change", () => {
    state.colourBy = $("colour-by").value;
    drawScene();
    refresh();
  });
}

wire();
refresh();
boot().catch((error) => say(String(error?.message ?? error), true));
