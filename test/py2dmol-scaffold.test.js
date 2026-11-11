/**
 * Every page's py2Dmol scaffold, held against index.html's.
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
 * 🔴 AND IT IS A LIST OF PAGES RATHER THAN A PAIR, because bench.html made it
 * three. A test written as "the two pages" is a test that silently stops
 * covering the newest copy of the subtree - which is the copy most likely to
 * have been typed rather than spliced.
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

/** Every page that embeds py2Dmol's viewer, index.html being the original. */
const PAGES = ["screen.html", "bench.html"];

test("🔴 every page gives py2Dmol the same viewer subtree", () => {
  const fold = viewerIds(read("index.html"));
  for (const page of PAGES) {
    const ids = viewerIds(read(page));
    const missing = [...fold].filter((id) => !ids.has(id));
    const extra = [...ids].filter((id) => !fold.has(id));
    assert.deepEqual(missing, [],
      `${page}'s viewer is missing ${missing.join(", ")} - copy the subtree`
      + " from index.html again");
    assert.deepEqual(extra, [],
      `${page}'s viewer has ${extra.join(", ")} that index.html does not`);
  }
});

/**
 * 🔴 THE SECOND FRONT END SHARES THE CONTROLLER, SO IT MUST SHARE ITS DOM
 * CONTRACT. web/screen.js reaches for its elements by id and fails at the
 * first missing one - `$("queue")` returning null throws inside `refresh`,
 * which runs before anything is on screen, so the page dies blank with one
 * console line. There is no way to notice by looking at the markup, and the
 * ids are spread over 600 lines of two files.
 */
/**
 * 🔴 AND ONE ID IS DELIBERATELY OPTIONAL, WHICH IS NOT THE SAME AS EXEMPT.
 * `#gpu-help` is the panel bench.html gives the WebGPU diagnosis; screen.html
 * does not have it and is not going to, because the whole point of keeping that
 * page is that nothing edits it. So the rule is not "every id exists
 * everywhere" but "every id the controller reads UNGUARDED exists everywhere,
 * and an optional one is really guarded" - which this checks, rather than
 * taking the exemption on trust. A `$("gpu-help")` whose null check was ever
 * dropped would make screen.html throw on the one path it is there to survive.
 */
const OPTIONAL_IDS = ["gpu-help"];

test("🔴 every page supplies every id the controller reaches for", () => {
  const controller = read("web/screen.js");
  const wanted = new Set([...controller
    .matchAll(/\$\("([a-z0-9-]+)"\)/g)].map((match) => match[1]));
  assert.ok(wanted.size > 15, "the controller's id list did not parse");
  for (const id of OPTIONAL_IDS) {
    assert.ok(wanted.has(id), `${id} is listed optional and never read`);
    const at = controller.indexOf(`$("${id}")`);
    assert.match(controller.slice(at, at + 160), /null/,
      `${id} is read without a null check within a few lines - it cannot be`
      + " optional, so either guard it or add it to every page");
  }
  const required = [...wanted].filter((id) => !OPTIONAL_IDS.includes(id));
  for (const page of PAGES) {
    const html = read(page);
    const missing = required
      .filter((id) => !html.includes(`id="${id}"`)).sort();
    assert.deepEqual(missing, [],
      `${page} is missing ${missing.join(", ")}, which web/screen.js reads -`
      + " the page will throw in refresh() and render nothing");
  }
  // the optional one still has to exist somewhere, or it is dead code
  for (const id of OPTIONAL_IDS) {
    assert.ok(PAGES.some((page) => read(page).includes(`id="${id}"`)),
      `${id} is read by the controller and is on no page at all`);
  }
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

/**
 * 🔴 AN ID IS NOT THE WHOLE CONTRACT. `#table` has to arrive with a `thead` and
 * a `tbody` already in it, because `drawTable` calls
 * `table.querySelector("thead").replaceChildren()` with no guard - and
 * `drawScatter` needs `#scatter` to be an SVG it can append SVG nodes to.
 * bench.html shipped `<table id="table"></table>`, passed the id check above,
 * and died on "Loading…" with every id present and one console line: the throw
 * happens in `refresh()`, before anything is drawn. Reading the markup cannot
 * show it; this can.
 */
test("🔴 the ranked table arrives with the sections drawTable writes into", () => {
  assert.match(read("web/screen-view.js"), /querySelector\("thead"\)/,
    "if drawTable no longer needs a thead, this test can go");
  for (const page of PAGES) {
    const html = read(page);
    const table = html.slice(html.indexOf('<table id="table"'));
    const end = table.indexOf("</table>");
    assert.ok(end > 0, `${page} has no <table id="table"> that closes`);
    const inside = table.slice(0, end);
    assert.match(inside, /<thead\b/, `${page}'s #table has no thead`);
    assert.match(inside, /<tbody\b/, `${page}'s #table has no tbody`);
  }
});

test("🔴 the scatter is an SVG with the coordinate space drawScatter draws in", () => {
  for (const page of PAGES) {
    const svg = read(page).match(/<svg[^>]*id="scatter"[^>]*>/);
    assert.ok(svg !== null, `${page} has no <svg id="scatter">`);
    // drawScatter lays out in a fixed 440x240 space and relies on the viewBox
    // to scale it; without one the plot is drawn at 440px whatever the column.
    assert.match(svg[0], /viewBox="0 0 440 240"/,
      `${page}'s #scatter must carry drawScatter's own 440x240 viewBox`);
  }
});

test("🔴 the screening pages' own status line is NOT #status", () => {
  // py2Dmol owns #status and writes into it on load; sharing the id let the
  // vendor overwrite the screen's status the moment it initialised.
  for (const page of PAGES) assert.match(read(page), /id="screen-status"/);
  assert.ok(vendorIds().has("status"),
    "if the vendor no longer uses #status this split can be undone");
  assert.doesNotMatch(read("web/screen.js"), /\$\("status"\)/,
    "the controller must write to #screen-status");
});
