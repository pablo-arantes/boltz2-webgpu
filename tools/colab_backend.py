"""This screen, on somebody else's GPU, reached over one URL.

    python3 tools/colab_backend.py --port 8710            # serve and broker
    python3 tools/colab_backend.py --port 8710 --token t  # a token of your own

It serves this checkout, opens a headless Chrome on it, and brokers between two
copies of the same page: the one it opened - `bench.html?role=runtime`, which
screens - and the one a reader opens, `bench.html?backend=colab`, which asks.
Five routes carry that, all of them token-checked:

    GET  /health            what the card is, and whether a screen is running
    POST /in                the reader asks: {op: "screen"|"stop"|"ping", payload}
    GET  /out?since=N       the runtime page collects what has been asked
    POST /up                the runtime page pushes what it says and draws
    GET  /down?since=N      the reader receives it

🔴 PORTED FROM LocalFold, NOT INVENTED HERE. `tools/colab_backend.py` and
`web/colab-bridge.js` at sokrypton/localfold `af474628e0` are where this
arrangement, these five routes, the watermark, the token and most of the
comments below come from - Beerware, Sergey Ovchinnikov, and docs/PROVENANCE.md
records it with the rest. They were REMOVED from that repository on 2026-11-01
(`1725c392ef`, "The WebGPU page from opus-55-opt, without the native, Metal or
design work`), so its own notebook now starts a file that is not there; this is
the last version that worked, adapted rather than re-imagined.

🔴 WHAT IS OURS IS THE UNIT OF WORK. Upstream brokers ONE FOLD. A screen is N
ligands, each a co-fold plus an affinity pass, and the answer is a table that
builds up - so the events this carries are the run loop's own
(`start`/`done`/`failed` per ligand, plus the status line), and the flag that
says "busy" is lowered by the screen FINISHING rather than by one structure
arriving. `op` is `screen`, and its payload is what the reader typed.

🔴 THE POINT IS THAT THERE IS NO SECOND IMPLEMENTATION. The screen that runs
here is web/screen.js's own, in a real browser, from this checkout - the same
code a visitor's laptop runs, on a card the laptop does not have. A Python
re-implementation would be a second answer to every question this repository
has already answered once, and the two would part company on the first
disconnected ligand.

🔴 AND THIS PROCESS IS A POST OFFICE, NOT A DRIVER. What is left for CDP is the
two things only it can do: start the browser, and say what card it got.

🔴 THE TOKEN IS NOT OPTIONAL. Colab's proxy makes the port reachable to whoever
holds the notebook's URL, and anything that reaches it can spend the GPU behind
it - so every route checks the token before it reads a body, with
`hmac.compare_digest` rather than `==`, which is the one line that stops the
comparison leaking its answer in its timing.

🔴 AND CORS STAYS WIDE, WHICH IS ONLY SAFE BECAUSE OF THE TOKEN. Both pages are
same-origin with this server today, so nothing here needs it; it is kept for the
page a reader might point at a runtime from their own laptop, and an allow-list
of origins cannot be written for a URL that changes every session.
"""
import argparse
import hmac
import http.server
import json
import math
import os
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 🔴 TWO MAILBOXES AND ONE SEQUENCE EACH, WHICH IS THE WHOLE BROKER. `EVENTS` is
# what the runtime page has said - status writes, each ligand's result, the
# finish - and `COMMANDS` is what readers have asked of it. Both are append-only
# and both are read by WATERMARK: a caller says what it has already applied and
# gets what came after, so a poll that overlaps another, or a page reloaded
# mid-screen, repeats itself rather than losing anything. Nothing is ever
# removed on read.
EVENTS = []
COMMANDS = []
MAIL_LOCK = threading.Lock()
# 🔴 AND THE OLDEST EVENTS DO GO, because a session is not one screen. A
# hundred-ligand run is a hundred results carrying a PDB each - tens of
# megabytes - and a notebook stays open for hours. `EVENT_BASE` is how many have
# been dropped, so a watermark keeps meaning the same thing across the drop and
# a reader that has fallen behind is told where the stream now starts instead of
# being handed the wrong events.
EVENT_BASE = 0
EVENT_CAP = 4000

# 🔴 ONE GPU, ONE SCREEN. Two at once is not twice the throughput, it is two
# screens that both take longer against a memory ceiling neither expected.
# Raised when a screen command is accepted and lowered by the runtime page's own
# `finished` event, which is what it sends however the screen ends - including
# stopped and failed.
SCREENING = {"on": False}
# 🔴 AND WHEN THE RUNTIME PAGE LAST ASKED FOR ITS COMMANDS, which is the only
# sign of life there is. A Colab runtime is recycled when the notebook is closed
# or left idle, and a reader whose screen was mid-flight then polls a broker
# that will never have another event for it - forever, because `SCREENING` is
# raised by the broker and lowered by the page. The page's own poll is the
# heartbeat; no second mechanism and nothing extra on the wire.
LAST_SEEN = {"at": 0.0}
# A reader who is done with the runtime wants its GPU back, and the only thing
# that frees it is this process going away: the browser it started holds the
# card for as long as it lives.
STOPPING = threading.Event()
# 🔴 AND THE MACHINE ITSELF CAN BE RELEASED, WHICH IS NOT THE SAME THING.
# Stopping this service frees the CARD; the Colab VM stays assigned until the
# notebook lets it go. `google.colab.runtime.unassign()` is a POST to a plain
# HTTP address in the environment (`TBE_RUNTIME_ADDR`) rather than a call over
# the kernel's channel - so a subprocess of the cell can do it, which is this.
RUNTIME_ADDR = os.environ.get("TBE_RUNTIME_ADDR")


def unassign_runtime():
    """Hand the Colab machine back. False where there is no machine to hand."""
    if not RUNTIME_ADDR:
        return False
    try:
        request = urllib.request.Request(f"http://{RUNTIME_ADDR}/unassign",
                                         data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status == 200
    except Exception as cause:                                # noqa: BLE001
        print(f"the runtime refused to unassign: {cause}", flush=True)
        return False


ADAPTER_JS = """(async () => {
  if (!navigator.gpu) return { webgpu: false, why: 'no navigator.gpu' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { webgpu: false, why: 'no adapter' };
  const info = adapter.info
    ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  return {
    webgpu: true,
    vendor: info.vendor ?? null, architecture: info.architecture ?? null,
    device: info.device ?? null, description: info.description ?? null,
    shaderF16: adapter.features.has('shader-f16'),
    subgroupMatrix: adapter.features.has('chromium-experimental-subgroup-matrix'),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
})()"""


def residue_ceiling(adapter):
    """The largest target this card can hold, which is why anyone comes here.

    🔴 THE ONE NUMBER THIS WHOLE ARRANGEMENT EXISTS FOR. web/target-size.js has
    the arithmetic - one f32 pair tensor is `residues^2 x 128 x 4` bytes and a
    WebGPU binding cannot exceed `maxStorageBufferBindingSize` - so the ceiling
    is `sqrt(limit / 512)` residues, and it is the difference between a laptop's
    ~2,000 and whatever an A100 reports. Computed here so `/health` can say it
    before a reader pastes anything.
    """
    limit = (adapter or {}).get("maxStorageBufferBindingSize")
    if not isinstance(limit, (int, float)) or limit <= 0:
        return None
    return int(math.floor(math.sqrt(limit / (128 * 4))))


class Backend:
    """The headless Chrome this serves from, started once."""

    def __init__(self, port, cdp_port, profile, token):
        self.port = port
        self.cdp_port = cdp_port
        self.profile = profile
        self.token = token
        self.proc = None
        self.ws = None
        self._adapter = None

    def alive(self):
        """Is the BROWSER still there - which is not "is the page answering".

        🔴 A BUSY PAGE STOPS POLLING, AND THAT IS NOT A DEAD RUNTIME. The
        heartbeat is the page's own `/out` poll, so a ligand that holds the main
        thread - shader compilation, a long upload - stops it, and a screen's
        ligands are seconds each. A reader that gave up on that alone would
        abort a screen that was working. The DevTools endpoint is served by the
        browser PROCESS rather than by the page, so it answers while the page is
        blocked and stops answering when the runtime is gone - which is the
        difference the reader actually needs.
        """
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.cdp_port}/json/version",
                    timeout=2) as answer:
                return answer.status == 200
        except Exception:                                     # noqa: BLE001
            return False

    def start(self):
        # cdp.py already carries the Linux flags this needs - Vulkan, the
        # sandbox off (a runtime is root in a container), and shader-f16.
        # Headless is opt-in there because a developer's box runs headed; here
        # there is no display at all.
        os.environ.setdefault("LOCALFOLD_HEADLESS", "1")
        # 🔴 `--disable-vulkan-surface`, WHICH THE SHARED FLAGS DO NOT CARRY.
        # A surface is a thing you present TO, and this container has no
        # display: Chrome's own Colab recipe passes it, and upstream measured
        # that without it the first runtime came back on SwiftShader - the CPU
        # wearing the card's clothes, which every later number would have been.
        self.proc, self.ws = cdp.launch(self.cdp_port, self.profile,
                                        extra_args=["--disable-vulkan-surface"])
        self.ws.call("Page.enable")
        self.ws.call("Runtime.enable")
        # 🔴 `role=runtime` IS THE PAGE BEING TOLD WHICH HALF IT IS, and the
        # token rides beside it because every route this page calls checks one.
        # web/colab-bridge.js reads both out of its own URL.
        self.ws.call("Page.navigate", url=(
            f"http://127.0.0.1:{self.port}/bench.html"
            f"?role=runtime&t={urllib.parse.quote(self.token)}"))
        # 🔴 THE CONTROLLER'S OWN SIGN OF LIFE, NOT THE MARKUP'S. `#run` is in
        # the HTML, so waiting for an element succeeds before web/screen.js has
        # attached a single listener - the trap tools/screen-in-page.py's
        # `--too-long` arm fell into. `__screenResults` is published by
        # `refresh()` at module load, which is after the wiring.
        cdp.wait_for(self.ws, "globalThis.__screenResults !== undefined", 180,
                     "the page")
        return self

    def adapter(self):
        """What the card is - the whole question this backend rests on.

        🔴 ASKED ONCE AND REMEMBERED, AND NEVER ALLOWED TO THROW. It is read
        over CDP from the page, so with the browser gone `/health` died
        mid-response and a reader trying to find out WHETHER the runtime was
        still there got an error that looked like the broker had gone too. The
        card does not change while the browser lives, so one answer is the
        answer.
        """
        if self._adapter is not None:
            return self._adapter
        try:
            self._adapter = cdp.evaluate(self.ws, ADAPTER_JS)
        except Exception as cause:                            # noqa: BLE001
            return {"webgpu": None, "why": f"the page did not answer: {cause}"}
        return self._adapter

    def wait_for_bridge(self, seconds=60):
        """...and that the page can REACH us, which is the other half.

        🔴 A STARTUP CHECK THAT USES THE REAL ROUTE. The bridge announces itself
        by POSTing `runtime-ready` to /up, so waiting for that event to appear
        in the mailbox proves the page loaded, read its token and can push - the
        three things every later event depends on. Waiting on a page internal
        over CDP proves only the first.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            with MAIL_LOCK:
                if any(e.get("kind") == "runtime-ready" for e in EVENTS):
                    return True
            time.sleep(0.25)
        return False


def serve(port, backend, token, host="127.0.0.1"):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers",
                             "content-type, x-localfold-token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            given = self.headers.get("X-LocalFold-Token", "")
            if not given:
                query = urllib.parse.urlparse(self.path).query
                given = urllib.parse.parse_qs(query).get("t", [""])[0]
            return hmac.compare_digest(given, token)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def _seen(self):
            """Milliseconds since the runtime page last asked for commands."""
            with MAIL_LOCK:
                at = LAST_SEEN["at"]
            return None if at == 0 else int((time.time() - at) * 1000)

        def _alive(self, seen):
            """Whether the BROWSER answers - asked only when the page is quiet.

            None while the page is answering, because then the question is
            already settled and the check costs a request.
            """
            if seen is not None and seen < 5000:
                return None
            return backend.alive()

        def _since(self):
            asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                return max(0, int(asked.get("since", ["0"])[0]))
            except ValueError:
                return 0

        def do_GET(self):
            route = urllib.parse.urlparse(self.path).path
            if route in ("/down", "/out", "/health") and not self._authorised():
                return self._json(403, {"error": "token"})
            # 🔴 A WATERMARK, NOT A QUEUE THE READER DRAINS. Two polls can
            # overlap and a page can be re-created by a reload, so nothing is
            # ever removed on read: `since` is what the caller has already
            # applied, which makes a repeated poll idempotent.
            if route == "/down":
                # 🔴 `head=1` IS THE WATERMARK WITHOUT THE STREAM. A reader
                # opening a screen needs to know where the stream stands so it
                # can ignore the last screen's events - and asking for that with
                # `since=0` hands it every result of the last run to throw away,
                # which on a hundred ligands is tens of megabytes.
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if asked.get("head", [""])[0] == "1":
                    # 🔴 `_seen()` TAKES THE SAME LOCK, AND IT IS NOT
                    # REENTRANT. Upstream deadlocked its whole broker this way:
                    # the first `head=1` request never returned AND never
                    # released. Read it before taking the lock.
                    seen = self._seen()
                    alive = self._alive(seen)
                    with MAIL_LOCK:
                        head = {"events": [], "n": EVENT_BASE + len(EVENTS),
                                "screening": SCREENING["on"], "runtimeSeen": seen,
                                "browserAlive": alive}
                    return self._json(200, head)
                since = self._since()
                # 🔴 HELD UNTIL THERE IS SOMETHING, BECAUSE A SCREEN IS AN HOUR
                # OF LIGANDS ARRIVING ONE AT A TIME. web/native-bridge.js - the
                # reader side already in this tree - asks with `wait=`, and the
                # version of this broker that was ported knew nothing about it:
                # the reader would have fallen back to a 400 ms loop, which
                # works and is a request every 400 ms for an hour. The hold is
                # capped at what the caller asked and at 25 s, under every
                # proxy's idle timeout.
                asked_wait = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query).get("wait", ["0"])[0]
                try:
                    wait_ms = max(0, min(25000, int(asked_wait)))
                except ValueError:
                    wait_ms = 0
                if wait_ms:
                    deadline = time.time() + wait_ms / 1000
                    while time.time() < deadline:
                        with MAIL_LOCK:
                            if EVENT_BASE + len(EVENTS) > since:
                                break
                        time.sleep(0.05)
                with MAIL_LOCK:
                    first = max(0, since - EVENT_BASE)
                    events = EVENTS[first:]
                    n = EVENT_BASE + len(EVENTS)
                    screening = SCREENING["on"]
                # `from` says where the answer actually starts, which is only
                # different from `since` for a caller that fell behind the cap.
                seen = self._seen()
                return self._json(200, {"events": events, "n": n,
                                        "from": EVENT_BASE + first,
                                        "screening": screening,
                                        # so a reader can tell a broker that
                                        # held its poll from one that cannot
                                        "waits": True,
                                        "runtimeSeen": seen,
                                        "browserAlive": self._alive(seen)})
            if route == "/out":
                # 🔴 `head=1` IS A RUNTIME PAGE SAYING IT HAS JUST STARTED, and
                # it exists because a reload replayed the SESSION: the page
                # polled from zero, so every command the notebook had ever sent
                # was obeyed again. A page that has just loaded is not owed the
                # past.
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                head = asked.get("head", [""])[0] == "1"
                since = self._since()
                with MAIL_LOCK:
                    LAST_SEEN["at"] = time.time()
                    commands = [] if head else COMMANDS[since:]
                    n = len(COMMANDS)
                return self._json(200, {"commands": commands, "n": n})
            if route == "/health":
                with MAIL_LOCK:
                    screening = SCREENING["on"]
                seen = self._seen()
                card = backend.adapter()
                return self._json(200, {"ok": True, "busy": screening,
                                        "runtimeSeen": seen,
                                        "browserAlive": self._alive(seen),
                                        # ...so the page can say whether
                                        # Disconnect releases the MACHINE or
                                        # only stops the service on it.
                                        "colabRuntime": bool(RUNTIME_ADDR),
                                        "gpu": card,
                                        # the number a reader came here for
                                        "residueCeiling": residue_ceiling(card)})
            return super().do_GET()

        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self):
            route = urllib.parse.urlparse(self.path).path
            if route not in ("/up", "/in"):
                return self._json(404, {"error": "no such route"})
            if not self._authorised():
                return self._json(403, {"error": "token"})
            try:
                body = self._body()
            except ValueError as cause:
                return self._json(400, {"error": f"not JSON: {cause}"})

            # THE RUNTIME PAGE SPEAKING. Every event is stamped on arrival
            # beside the page's own `at`, so a reader can say whether a ligand
            # was slow or the feed was.
            if route == "/up":
                got = int(time.time() * 1000)
                arrived = body.get("events") or []
                global EVENT_BASE
                with MAIL_LOCK:
                    for event in arrived:
                        if not isinstance(event, dict):
                            continue
                        event["got"] = got
                        EVENTS.append(event)
                        # 🔴 THE SCREEN FINISHING LOWERS THE FLAG, NOT A LIGAND
                        # ARRIVING. This is where a port of upstream's `result`
                        # would have been wrong: a screen sends one of those per
                        # ligand, so the first one would free the GPU while
                        # ninety-nine were still to run.
                        if event.get("kind") == "finished":
                            SCREENING["on"] = False
                        # ...and a page that has just loaded is not screening.
                        # The flag is raised by the broker and lowered by the
                        # page, so a runtime that died mid-screen took it with
                        # it and every later screen was refused for the session.
                        if event.get("kind") == "runtime-ready":
                            SCREENING["on"] = False
                    if len(EVENTS) > EVENT_CAP:
                        drop = len(EVENTS) - EVENT_CAP // 2
                        del EVENTS[:drop]
                        EVENT_BASE += drop
                    n = EVENT_BASE + len(EVENTS)
                return self._json(200, {"ok": True, "n": n})

            # ...AND THE READER ASKING.
            op = body.get("op")
            # 🔴 SHUTDOWN IS THE BROKER'S OWN, NOT THE PAGE'S. Every other op is
            # forwarded to the runtime page and obeyed there; this one ends the
            # service - the browser, the GPU it holds and this process - so it
            # is answered here, after the answer has been written.
            if op == "shutdown":
                self._json(200, {"ok": True, "stopping": True,
                                 "unassign": bool(RUNTIME_ADDR)})
                threading.Thread(target=lambda: (time.sleep(0.3),
                                                 STOPPING.set()), daemon=True).start()
                return None
            if op not in ("screen", "stop", "ping"):
                return self._json(400, {"error": f'unknown op "{op}"'})
            with MAIL_LOCK:
                if op == "screen" and SCREENING["on"]:
                    return self._json(429,
                                      {"error": "one GPU, one screen: try again"})
                if op == "screen":
                    SCREENING["on"] = True
                COMMANDS.append({"seq": len(COMMANDS), "op": op,
                                 "payload": body.get("payload")})
                return self._json(200, {"ok": True, "seq": len(COMMANDS) - 1})

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # 🔴 LOOPBACK, NOT EVERY INTERFACE. The tunnel client runs on this same
    # machine and reaches the port over 127.0.0.1, so binding wider buys nothing
    # and offers the runtime's own network a GPU with an HTTP API on it.
    # `--host 0.0.0.0` is there for whoever has a reason; the default has none.
    httpd = socketserver.ThreadingTCPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--cdp-port", type=int, default=9333)
    parser.add_argument("--token", default=None,
                        help="the shared secret; one is generated when absent")
    parser.add_argument("--profile", default="/tmp/boltz2-colab-backend")
    parser.add_argument("--host", default="127.0.0.1",
                        help="what to bind; the tunnel reaches loopback")
    arguments = parser.parse_args()

    token = arguments.token or secrets.token_urlsafe(24)
    backend = Backend(arguments.port, arguments.cdp_port, arguments.profile, token)
    httpd = serve(arguments.port, backend, token, arguments.host)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"serving {REPO} on {arguments.host}:{arguments.port}"
          + (" · Disconnect will release this Colab machine" if RUNTIME_ADDR
             else " · no Colab runtime here, Disconnect stops the service"),
          flush=True)
    # 🔴 THE SERVER FIRST, THE PAGE SECOND. The page announces itself to /up the
    # moment it loads, so a browser started before the socket is listening
    # announces into a refused connection and the wait below times out on a
    # runtime that is working perfectly.
    backend.start()
    if not backend.wait_for_bridge():
        print("the page never announced itself: the bridge cannot reach /up",
              flush=True)
    # One line, machine-readable, for the notebook cell that prints the handle.
    card = backend.adapter()
    print("BACKEND " + json.dumps({"token": token, "gpu": card,
                                   "residueCeiling": residue_ceiling(card)}),
          flush=True)
    try:
        # Woken by Ctrl-C, or by a reader pressing Disconnect - see STOPPING.
        while not STOPPING.wait(timeout=3600):
            pass
        print("stopped by the page", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        # 🔴 THE BROWSER IS THIS PROCESS'S, AND IT DOES NOT DIE WITH IT.
        # `cdp.launch` starts a headless Chrome that outlives a Ctrl-C and a
        # SIGINT from a gate. In a Colab runtime the container takes them; on a
        # developer's machine they are the "another browser on the machine" that
        # makes the next measurement somebody else's.
        if backend.proc is not None:
            backend.proc.terminate()
            try:
                backend.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.proc.kill()
        # 🔴 THE BROWSER FIRST, THE MACHINE SECOND. Unassigning pulls the VM out
        # from under this process, so anything that has to happen on the way out
        # has to have happened already.
        if STOPPING.is_set() and unassign_runtime():
            print("the Colab runtime has been unassigned", flush=True)


if __name__ == "__main__":
    main()
