/**
 * This screen, on somebody else's GPU: both halves of it.
 *
 * `tools/colab_backend.py` serves this checkout and opens a headless Chrome on
 * `bench.html?role=runtime`. That page is the RUNTIME: it has the card, the
 * weights and this controller, and it screens. The link the notebook prints
 * opens `bench.html?backend=colab`, which is the READER: it has no device at
 * all, it asks the runtime to screen and it draws what comes back.
 *
 * 🔴 WHY THIS EXISTS AT ALL: THE TARGET THAT WILL NOT FIT. A pair tensor is
 * `residues^2 x 128 x 4` bytes, so the ceiling is
 * `sqrt(maxStorageBufferBindingSize / 512)` residues - about 2,000 on a laptop
 * (web/target-size.js, docs/MEASURED.md). A Colab A100 reports a much larger
 * binding, and `/health` carries the number so the reader's page can say what
 * this runtime can hold rather than what their own machine could.
 *
 * 🔴 AND THERE IS NO SECOND IMPLEMENTATION OF A SCREEN. The runtime half fills
 * this page's own inputs and presses its own Run button; the reader half hands
 * each result to `acceptResult`, the same function the local run loop calls. A
 * remote feed with its own copy of that would assemble the scene, the plate and
 * the table a second way, and the hour-long run is the one nobody re-checks.
 *
 * 🔴 PORTED FROM LocalFold, in shape and in most of its comments:
 * `web/colab-bridge.js` at sokrypton/localfold `af474628e0` (Beerware, Sergey
 * Ovchinnikov), removed there on 2026-11-01 when every fold went to a native
 * CUDA worker. We have no CUDA port of the affinity module - the screen is
 * WebGPU-only - so the headless-runtime arrangement is the one that works here,
 * and `web/native-bridge.js` (already in this tree, the reader side of the
 * NEWER design) is where `remoteCommand`, `remoteEvents` and `remoteHead` come
 * from rather than being written twice. docs/PROVENANCE.md records both.
 *
 * Absent `?role=runtime` and `?backend=colab`, every function here is inert and
 * the page is what it was: the screen runs in the reader's own browser.
 */

import { remoteCommand, remoteEvents, remoteHead } from "./native-bridge.js";

const PARAMS = new URLSearchParams(location.search);
/**
 * 🔴 THE TOKEN IS TAKEN ONCE, AT LOAD. web/native-bridge.js's header records
 * why: a `door()` that re-reads `location.search` asks with `t=` empty once the
 * query string has been rewritten, which showed up on a real runtime as a
 * console full of `403 (Forbidden)`. The URL is where the token ARRIVES; it is
 * not where it lives.
 */
const TOKEN = PARAMS.get("t") ?? "";

/** "runtime", "reader", or null for an ordinary page. */
export function colabRole() {
  if (PARAMS.get("role") === "runtime") return "runtime";
  // `backend=colab` is this page's; `native` is what the newer LocalFold design
  // calls the same thing, and a reader who pastes that link should still work.
  const backend = PARAMS.get("backend");
  if (backend === "colab" || backend === "native") return "reader";
  return null;
}

const door = (route, extra = "") =>
  `${route}?t=${encodeURIComponent(TOKEN)}${extra}`;

const post = (route, body) => fetch(door(route), {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

/** What the runtime is, and how big a target it can hold. */
export async function colabHealth() {
  const answer = await fetch(door("/health"));
  if (!answer.ok) throw new Error(`the broker said ${answer.status}`);
  return answer.json();
}

/* ------------------------------------------------------------------ */
/* the runtime half: it screens, and says so                           */
/* ------------------------------------------------------------------ */

/**
 * 🔴 EVENTS GO OUT IN BATCHES ON A TIMER, NOT ONE REQUEST EACH. A screen's
 * status line moves several times a ligand and a result carries a whole PDB, so
 * a POST per event is hundreds of round trips an hour for no benefit. A batch
 * every 150 ms is still live to a reader's eye and is one connection.
 *
 * 🔴 AND A LOST BATCH IS A LOST PICTURE, NOT A LOST SCREEN. The screen is
 * running on this machine; if a push fails the ligands keep being folded and
 * the reader catches up from its watermark on the next one. So a failure here
 * is reported and dropped rather than retried forever.
 */
const outbox = [];
let flushing = null;

function flushSoon() {
  if (flushing !== null) return;
  flushing = setTimeout(async () => {
    flushing = null;
    if (outbox.length === 0) return;
    const going = outbox.splice(0, outbox.length);
    try {
      await post("/up", { events: going });
    } catch (cause) {
      console.error("the runtime could not push its events", cause);
    }
  }, 150);
}

function emit(kind, payload) {
  outbox.push({ kind, at: Date.now(), ...payload });
  flushSoon();
}

/**
 * Fill this page's own controls from what the reader typed, and press Run.
 *
 * 🔴 THE CONTROLS, NOT THE CONTROLLER. `screen()` is not exported and does not
 * need to be: driving the page the way a reader does is what makes this the
 * same screen rather than a second entry point, and tools/screen-in-page.py
 * already proves that path works by driving exactly these elements. Each one
 * gets its `input` event because the page's notes and its ligand parse hang off
 * those, not off the value.
 */
function runHere(payload, hooks) {
  const set = (id, value) => {
    const element = document.getElementById(id);
    if (element === null || value === undefined || value === null) return;
    element.value = String(value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  };
  set("sequence", payload.sequence);
  set("ligands", payload.ligands);
  set("msa", payload.msa ?? "none");
  for (const [id, key] of [["seed", "seed"], ["recycles", "recycles"],
                           ["steps", "steps"], ["samples", "samples"]]) {
    if (payload[key] !== undefined) set(id, payload[key]);
  }
  const run = document.getElementById("run");
  if (run === null) return;
  // 🔴 A DISABLED RUN IS NOT A REFUSAL, IT IS "THE WEIGHTS ARE STILL COMING".
  // `boot()` enables it once the device and both bundles are up, which on a
  // cold runtime is tens of seconds - so a command that arrives first waits for
  // it rather than being dropped, and the reader is told it is waiting.
  if (run.disabled) {
    hooks.say("the runtime is still loading its weights; starting when ready…");
    const waiting = setInterval(() => {
      const button = document.getElementById("run");
      if (button === null || button.disabled) return;
      clearInterval(waiting);
      button.click();
    }, 500);
    return;
  }
  run.click();
}

function serveRuntime(hooks) {
  let since = 0;
  let head = true;
  emit("runtime-ready", { href: location.href });

  const obey = (command) => {
    const { op, payload } = command;
    if (op === "ping") return emit("pong", { seq: command.seq });
    if (op === "stop") {
      document.getElementById("stop")?.click();
      return emit("status", { message: "stopped by the reader", bad: true });
    }
    if (op === "screen") {
      emit("screen-started", { payload });
      return runHere(payload ?? {}, hooks);
    }
    return emit("status", { message: `unknown op "${op}"`, bad: true });
  };

  const poll = async () => {
    for (;;) {
      try {
        // 🔴 FROM WHERE THE QUEUE STANDS NOW, NOT FROM ZERO. This page can be
        // re-created by a reload, and a poll from zero obeys every command the
        // session ever carried - upstream measured a reload re-running a fold
        // from ten minutes earlier. `head=1` says "I have just started".
        const answer = await fetch(door("/out",
          head ? "&head=1" : `&since=${since}`));
        if (answer.ok) {
          const said = await answer.json();
          since = said.n ?? since;
          head = false;
          for (const command of said.commands ?? []) obey(command);
        }
      } catch (cause) {
        // the broker is on this machine; a refused connection means it has gone
        console.error("the runtime lost the broker", cause);
      }
      await new Promise((done) => setTimeout(done, 400));
    }
  };
  poll();
}

/* ------------------------------------------------------------------ */
/* the reader half: it asks, and draws                                 */
/* ------------------------------------------------------------------ */

async function serveReader(hooks) {
  let since = 0;
  try {
    const head = await remoteHead();
    // 🔴 START AT THE WATERMARK, NOT AT ZERO. Opening the page mid-session
    // otherwise replays the previous screen's hundred results into this table.
    since = head?.n ?? 0;
  } catch {
    // a broker that is not answering yet is not an error: the poll retries
  }

  const apply = (event) => {
    const kind = event?.kind;
    if (kind === "status") hooks.say(event.message ?? "", event.bad === true);
    else if (kind === "result" && event.result) hooks.acceptResult(event.result);
    else if (kind === "failed" && event.failure) hooks.acceptFailure(event.failure);
    else if (kind === "finished") hooks.finish(event);
    else if (kind === "screen-started") hooks.remoteStarted(event.payload ?? {});
    else if (kind === "runtime-ready") hooks.say("the runtime is ready");
  };

  const poll = async () => {
    for (;;) {
      try {
        // 🔴 A HELD POLL, SO THE FEED IS LIVE. A screen is an hour of ligands
        // arriving one at a time; a 400 ms loop would be fine and a long poll
        // is better, and web/native-bridge.js already speaks `wait=`.
        const said = await remoteEvents(since, undefined, 20000);
        if (said) {
          // `from` says where the answer starts, which differs from `since`
          // only for a reader that fell behind the broker's cap.
          for (const event of said.events ?? []) apply(event);
          since = said.n ?? since;
          hooks.remoteState(said);
        }
      } catch {
        await new Promise((done) => setTimeout(done, 2000));
      }
    }
  };
  poll();
}

/* ------------------------------------------------------------------ */

/**
 * Wire whichever half this page is. Returns the role, or null.
 *
 * @param {{say: Function, acceptResult: Function, acceptFailure: Function,
 *          finish: Function, remoteStarted: Function, remoteState: Function,
 *          installTap: Function}} hooks the controller's own functions
 */
export function installColabBridge(hooks) {
  const role = colabRole();
  if (role === null) return null;
  if (role === "runtime") {
    // every status line, result and failure this page produces goes out
    hooks.installTap(emit);
    serveRuntime(hooks);
    return "runtime";
  }
  serveReader(hooks);
  return "reader";
}

/** The reader's Run: ask the runtime instead of this machine's GPU. */
export const askRuntimeToScreen = (payload) => remoteCommand("screen", payload);

/**
 * ...and the reader's Stop, which is the RUNTIME's Stop pressed from here.
 *
 * 🔴 NOT THIS PAGE'S ABORT. A reader has no run loop and no AbortController -
 * the screen is happening on another machine - so the local Stop handler
 * aborts nothing and says "stopping after this ligand…" about a ligand it is
 * not folding. The command goes over the broker and the runtime presses its own.
 */
export const askRuntimeToStop = () => remoteCommand("stop", null);
