/**
 * The two pages' py2Dmol scaffolds, held against each other.
 *
 * 🔴 THE VENDOR BUNDLE INITIALISES BY `getElementById` AND FAILS SILENTLY. It
 * queries fifteen ids and builds its viewer inside `#viewer-container`; given a
 * page without them it creates no viewer, raises nothing, and
 * `window.py2dmol_viewers` is simply empty. screen.html's first probe run
 * failed exactly that way, and the only symptom was an empty box.
 *
 * So screen.html carries index.html's viewer subtree verbatim, and this is what
 * stops the copy rotting: a control added to one page's viewer and not the
 * other is a control the vendor queries on one page and not the other.
 *
 * 🔴 AND THE ONE DELIBERATE DIFFERENCE IS ASSERTED, NOT EXEMPTED. `#status` is
 * py2Dmol's own element - it writes "Ready. Upload a file or fetch an ID." into
 * it - so the screen page's own status line is `#screen-status`. A test that
 * simply allowed any difference would have let that collision back in.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (name) => readFileSync(new URL(`../${name}`, import.meta.url), "utf8");

/** The ids inside the `#viewer-container` subtree of a page. */
function viewerIds(html) {
  const start = html.indexOf('id="viewer-container"');
  assert.notEqual(start, -1, "no #viewer-container on this page");
  const from = html.lastIndexOf("<div", start);
  let depth = 0;
  let at = from;
  let end = -1;
  while (at < html.length) {
    const open = html.indexOf("<div", at);
    const close = html.indexOf("</div>", at);
    if (close === -1) break;
    if (open !== -1 && open < close) { depth += 1; at = open + 4; continue; }
    depth -= 1;
    at = close + 6;
    if (depth === 0) { end = at; break; }
  }
  assert.ok(end > from, "the #viewer-container subtree does not close");
  const subtree = html.slice(from, end);
  return new Set([...subtree.matchAll(/id="([A-Za-z0-9_-]+)"/g)]
    .map((match) => match[1]));
}

/** Every id the vendor bundle looks up by a literal name. */
function vendorIds() {
  const source = read("web/vendor/py2Dmol.full.min.js");
  return new Set([...source.matchAll(/getElementById\("([A-Za-z0-9_-]+)"\)/g)]
    .map((match) => match[1]));
}

test("🔴 both pages give py2Dmol the same viewer subtree", () => {
  const fold = viewerIds(read("index.html"));
  const screen = viewerIds(read("screen.html"));
  const missing = [...fold].filter((id) => !screen.has(id));
  const extra = [...screen].filter((id) => !fold.has(id));
  assert.deepEqual(missing, [],
    `screen.html's viewer is missing ${missing.join(", ")} - copy the subtree`
    + " from index.html again");
  assert.deepEqual(extra, [],
    `screen.html's viewer has ${extra.join(", ")} that index.html does not`);
});

/**
 * 🔴 THERE IS NO "EVERY ID THE VENDOR LOOKS UP MUST EXIST" TEST HERE, AND THE
 * FIRST VERSION OF THIS FILE HAD ONE. It failed naming 53 ids - darkCheckbox,
 * msaView, useGpuCheckbox, colorSelect and the rest - on a page whose viewer
 * demonstrably works, because py2Dmol GUARDS most of its lookups: they belong
 * to panels (settings, MSA, sequence, selection) that this page does not have,
 * and a missing one is a feature that is absent rather than a viewer that is
 * broken. Counting `getElementById` calls cannot tell a guarded lookup from a
 * required one, so the authority on what is required is
 * tools/screen-in-page.py, which asserts at RUNTIME that
 * `window.py2dmol_viewers` is non-empty and that the scene drew. A static test
 * that asserts more than is true is a test that has to be deleted the first
 * time it is right about nothing.
 */

test("🔴 the screen page's own status line is NOT #status", () => {
  const html = read("screen.html");
  // py2Dmol owns #status and writes into it on load; sharing the id let the
  // vendor overwrite the screen's status the moment it initialised.
  assert.match(html, /id="screen-status"/);
  assert.ok(vendorIds().has("status"),
    "if the vendor no longer uses #status this split can be undone");
  assert.doesNotMatch(read("web/screen.js"), /\$\("status"\)/,
    "the controller must write to #screen-status");
});
