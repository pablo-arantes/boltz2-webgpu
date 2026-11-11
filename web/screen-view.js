/**
 * The two pictures a screen is read through: a scatter and a ranked table.
 *
 * 🔴 WHY A SCATTER AND NOT A BAR CHART OR A HEATMAP, in one line as the brief
 * asks: the two affinity outputs are the two axes, so a scatter is the only
 * representation that shows both without merging them - and it puts the one
 * thing a reader should distrust (a high score at low confidence) in the eye as
 * a pale or small point rather than as a number in a column they have to think
 * about.
 *
 * x is pIC50 (higher = more potent, so right is better), y is
 * `affinity_probability_binary` (higher = more likely a binder, so up is
 * better). Both axes therefore point the same way and the top right is the
 * corner to look at - which is NOT true of the raw columns, where
 * `affinity_pred_value` is lower-is-better.
 *
 * 🔴 CONFIDENCE IS SIZE **AND** OPACITY, not colour. Colour is already spoken
 * for by whatever column the scene is coloured by, and a reader comparing the
 * scene to the scatter must see the same ligand in the same colour in both. A
 * low-ipTM pose is a guess and so is its affinity, so it is drawn small and
 * faint; nothing is hidden, because a hidden point is a decision made on the
 * reader's behalf.
 *
 * 🔴 AND IT IS INLINE SVG WITH NO LIBRARY. The fold page's own header records
 * what a CDN script costs when its host is unreachable: "a network that drops
 * this host answers nothing at all, and a head waits for what it asked - a
 * blank page for 34 s". A scatter of a few hundred points is a `for` loop.
 */
import { scoreColour, scoreDomain } from "./screen-scene.js";
import { statusOf } from "./screen-export.js";

const PAD = { left: 44, right: 10, top: 10, bottom: 30 };
const svgNs = "http://www.w3.org/2000/svg";

const node = (name, attributes = {}, text = undefined) => {
  const element = document.createElementNS(svgNs, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  if (text !== undefined) element.textContent = text;
  return element;
};

/** A nice-ish axis range: never zero-width, padded by a tenth. */
function axis(values, fallback) {
  const live = values.filter((value) => Number.isFinite(value));
  if (live.length === 0) return fallback;
  let lo = Math.min(...live);
  let hi = Math.max(...live);
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.1;
  return [lo - pad, hi + pad];
}

/**
 * Draw the scatter.
 *
 * @param {SVGElement} svg
 * @param {object[]} results
 * @param {{colourBy?: string, highlight?: string}} [options]
 */
export function drawScatter(svg, results, options = {}) {
  const width = 440;
  const height = 240;
  svg.replaceChildren();
  const done = results.filter((result) => statusOf(result) === "done"
    && Number.isFinite(result.pIC50)
    && Number.isFinite(result.affinity_probability_binary));

  const plotWidth = width - PAD.left - PAD.right;
  const plotHeight = height - PAD.top - PAD.bottom;
  const frame = node("g");
  // axes
  frame.append(node("line", { x1: PAD.left, y1: PAD.top, x2: PAD.left,
    y2: PAD.top + plotHeight, stroke: "#d1d5db" }));
  frame.append(node("line", { x1: PAD.left, y1: PAD.top + plotHeight,
    x2: PAD.left + plotWidth, y2: PAD.top + plotHeight, stroke: "#d1d5db" }));
  frame.append(node("text", { x: PAD.left + plotWidth / 2, y: height - 6,
    "text-anchor": "middle", fill: "#6b7280", "font-size": 10 },
  "pIC50  (6 - affinity_pred_value; IC50-like, for ranking)"));
  frame.append(node("text", { x: 11, y: PAD.top + plotHeight / 2,
    "text-anchor": "middle", fill: "#6b7280", "font-size": 10,
    transform: `rotate(-90 11 ${PAD.top + plotHeight / 2})` },
  "P(binder), higher is better"));
  svg.append(frame);

  if (done.length === 0) {
    svg.append(node("text", { x: PAD.left + plotWidth / 2,
      y: PAD.top + plotHeight / 2, "text-anchor": "middle", fill: "#9ca3af",
      "font-size": 11 }, "no finished ligands yet"));
    return;
  }

  const [xLo, xHi] = axis(done.map((result) => result.pIC50), [0, 10]);
  // 🔴 THE PROBABILITY AXIS IS 0..1 AND NOT THE DATA'S RANGE. It is a
  // probability: 0.31 and 0.39 being at opposite ends of the plot would make a
  // meaningless difference look decisive. The pIC50 axis scales because it has
  // no natural bounds.
  const [yLo, yHi] = [0, 1];
  const x = (value) => PAD.left + ((value - xLo) / (xHi - xLo)) * plotWidth;
  const y = (value) => PAD.top + plotHeight - ((value - yLo) / (yHi - yLo)) * plotHeight;

  for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(node("line", { x1: PAD.left - 3, y1: y(tick), x2: PAD.left,
      y2: y(tick), stroke: "#d1d5db" }));
    svg.append(node("text", { x: PAD.left - 6, y: y(tick) + 3,
      "text-anchor": "end", fill: "#6b7280", "font-size": 9 }, tick.toFixed(2)));
  }
  for (let step = 0; step <= 4; step += 1) {
    const value = xLo + ((xHi - xLo) * step) / 4;
    svg.append(node("line", { x1: x(value), y1: PAD.top + plotHeight,
      x2: x(value), y2: PAD.top + plotHeight + 3, stroke: "#d1d5db" }));
    svg.append(node("text", { x: x(value), y: height - 18,
      "text-anchor": "middle", fill: "#6b7280", "font-size": 9 },
    value.toFixed(1)));
  }

  const column = options.colourBy ?? "affinity_probability_binary";
  const domain = scoreDomain(done, column);
  // 🔴 AND AN OVERRIDE, FOR THE MODE WHERE THE COLOUR IS NOT A SCORE. "one
  // colour per ligand" is the VIEWER's palette, which this module cannot
  // compute - it belongs to py2Dmol. The page reads it back off the renderer
  // and passes it here, so the swatch and the point keep matching the atoms on
  // screen in every mode rather than only in the ones with a ramp.
  const fillFor = (result) => options.colours?.get(result.id)
    ?? scoreColour(result[column], domain.lo, domain.hi);
  for (const result of done) {
    const confidence = Number.isFinite(result.iptm) ? result.iptm : 0;
    // 🔴 THE RADIUS FLOOR IS NOT COSMETIC: a zero-confidence ligand must still
    // be a visible point, or the scatter silently drops the predictions a
    // reader most needs to see are bad.
    const radius = 3 + 5 * Math.max(0, Math.min(1, confidence));
    const point = node("circle", {
      cx: x(result.pIC50), cy: y(result.affinity_probability_binary), r: radius,
      fill: fillFor(result),
      "fill-opacity": (0.35 + 0.55 * Math.max(0, Math.min(1, confidence))).toFixed(2),
      stroke: result.id === options.highlight ? "#111827" : "#ffffff",
      "stroke-width": result.id === options.highlight ? 2 : 1,
    });
    point.append(node("title", {},
      `${result.id}  ${result.smiles}\nP(binder) `
      + `${result.affinity_probability_binary.toFixed(3)}  pIC50 `
      + `${result.pIC50.toFixed(2)}  ipTM ${confidence.toFixed(2)}`
      + (result.outsideTrainingRange ? "\noutside the training range" : "")));
    point.dataset.id = result.id;
    svg.append(point);
  }
  // the confidence legend, because size means nothing unlabelled
  svg.append(node("text", { x: PAD.left + plotWidth, y: PAD.top + 10,
    "text-anchor": "end", fill: "#9ca3af", "font-size": 9 },
  "point size & opacity = interface ipTM"));
}

/** The columns the table offers, with how each sorts. */
export const TABLE_COLUMNS = [
  { key: "id", label: "Ligand", left: true, text: true },
  { key: "smiles", label: "SMILES", left: true, text: true, cell: "smiles" },
  { key: "affinity_probability_binary", label: "P(binder)", digits: 3 },
  { key: "pIC50", label: "pIC50", digits: 2 },
  { key: "affinity_pred_value", label: "log10 IC50", digits: 3 },
  { key: "iptm", label: "ipTM", digits: 2 },
  { key: "heavyAtoms", label: "Heavy", digits: 0 },
  { key: "runtimeMs", label: "Seconds", digits: 1, scale: 1 / 1000 },
];

/**
 * 🔴 THE DEFAULT SORT IS THE CLASSIFIER, DESCENDING, and that is a choice with
 * a reason: a screen's first question is "what should I look at", which is the
 * binder/decoy head. Sorting by `affinity_pred_value` by default would rank a
 * library by a regression trained for comparing analogues of a known binder -
 * the right column later and the wrong one first.
 */
export const DEFAULT_SORT = { key: "affinity_probability_binary", ascending: false };

/** Lower is better for this column, so "best first" is ascending. */
const LOWER_IS_BETTER = new Set(["affinity_pred_value", "runtimeMs"]);

export function sortResults(results, sort) {
  const column = TABLE_COLUMNS.find((entry) => entry.key === sort.key);
  const rows = [...results];
  rows.sort((a, b) => {
    // 🔴 A FAILED LIGAND SORTS TO THE BOTTOM WHICHEVER WAY THE COLUMN POINTS.
    // Treating its missing score as 0 would put it first in an ascending sort
    // of log10(IC50) - the strongest binder in the table being a ligand that
    // never ran.
    const aDone = statusOf(a) === "done";
    const bDone = statusOf(b) === "done";
    if (aDone !== bDone) return aDone ? -1 : 1;
    const left = a[sort.key];
    const right = b[sort.key];
    if (column?.text === true) {
      const order = String(left ?? "").localeCompare(String(right ?? ""));
      return sort.ascending ? order : -order;
    }
    const x = Number.isFinite(left) ? left : (sort.ascending ? Infinity : -Infinity);
    const z = Number.isFinite(right) ? right : (sort.ascending ? Infinity : -Infinity);
    return sort.ascending ? x - z : z - x;
  });
  return rows;
}

/** Whether clicking this column should start descending (best first). */
export function startsDescending(key) {
  const column = TABLE_COLUMNS.find((entry) => entry.key === key);
  if (column?.text === true) return false;
  return !LOWER_IS_BETTER.has(key);
}

/**
 * Draw the ranked table.
 *
 * @param {HTMLTableElement} table
 * @param {object[]} results
 * @param {{sort: {key: string, ascending: boolean}, colourBy?: string,
 *          onPick?: (id: string) => void, onToggle?: (id: string) => void,
 *          visible?: Set<string>}} options
 */
export function drawTable(table, results, options) {
  const head = table.querySelector("thead");
  const body = table.querySelector("tbody");
  head.replaceChildren();
  body.replaceChildren();

  const headRow = document.createElement("tr");
  for (const column of TABLE_COLUMNS) {
    const cell = document.createElement("th");
    cell.textContent = column.label;
    if (column.left) cell.classList.add("left");
    if (options.sort.key === column.key) {
      cell.classList.add("sorted");
      cell.textContent = `${column.label} ${options.sort.ascending ? "↑" : "↓"}`;
    }
    cell.addEventListener("click", () => options.onSort?.(column.key));
    headRow.append(cell);
  }
  head.append(headRow);

  const done = results.filter((result) => statusOf(result) === "done");
  const domain = scoreDomain(done, options.colourBy ?? "affinity_probability_binary");
  for (const result of sortResults(results, options.sort)) {
    const status = statusOf(result);
    const row = document.createElement("tr");
    row.className = status;
    for (const column of TABLE_COLUMNS) {
      const cell = document.createElement("td");
      if (column.left) cell.classList.add("left");
      if (column.cell) cell.classList.add(column.cell);
      let value = result[column.key];
      if (column.key === "id") {
        // the swatch ties the row to the point and to the scene
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = status === "done"
          ? (options.colours?.get(result.id)
             ?? scoreColour(result[options.colourBy
                                   ?? "affinity_probability_binary"],
                            domain.lo, domain.hi))
          : "transparent";
        cell.append(swatch);
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = options.visible?.has(result.id) ?? true;
        box.disabled = status !== "done";
        box.title = "show this ligand in the scene";
        box.addEventListener("change", () => options.onToggle?.(result.id, box.checked));
        cell.append(box);
        cell.append(document.createTextNode(` ${result.id}`));
        if (result.outsideTrainingRange === true) {
          const flag = document.createElement("span");
          flag.className = "warn";
          flag.textContent = " ⚠";
          flag.title = result.warning ?? "outside the affinity training range";
          cell.append(flag);
        }
      } else if (status !== "done" && column.key === "smiles") {
        cell.textContent = String(value ?? "");
        cell.title = result.error ?? result.skipped ?? "";
      } else if (status !== "done") {
        // 🔴 A DASH, NOT A ZERO. A zero sorts and reads as a measurement.
        cell.textContent = column.key === "iptm" ? status : "–";
      } else if (column.text) {
        cell.textContent = String(value ?? "");
        cell.title = String(value ?? "");
      } else {
        if (column.scale !== undefined) value *= column.scale;
        cell.textContent = Number.isFinite(value)
          ? value.toFixed(column.digits) : "–";
      }
      row.append(cell);
    }
    row.addEventListener("click", () => options.onPick?.(result.id));
    body.append(row);
  }
}

/**
 * The sentence under the table: what a reader must not conclude.
 *
 * 🔴 THIS IS NOT DECORATION. The brief requires the two heads' different
 * training to be explicit in the UI, and it is the ensemble SPLIT that a mean
 * hides - measured on this port at 1.751 against 0.189 on one ligand. A run
 * where the members disagree is a run whose ranking is weak, and the page says
 * so with the number rather than in general.
 */
export function caveatsFor(results) {
  const done = results.filter((result) => statusOf(result) === "done");
  if (done.length === 0) return "";
  const parts = [];
  const splits = done
    .map((result) => Math.abs((result.affinity_pred_value1 ?? NaN)
      - (result.affinity_pred_value2 ?? NaN)))
    .filter((value) => Number.isFinite(value));
  if (splits.length > 0) {
    const worst = Math.max(...splits);
    const median = [...splits].sort((a, b) => a - b)[Math.floor(splits.length / 2)];
    parts.push(`Ensemble members differ by ${median.toFixed(2)} log units`
      + ` (median) and ${worst.toFixed(2)} at worst: a wide split is a reason`
      + " to distrust the mean, not to average it away.");
  }
  const weak = done.filter((result) => (result.iptm ?? 0) < 0.5).length;
  if (weak > 0) {
    parts.push(`${weak} of ${done.length} poses have ipTM below 0.5 - the`
      + " interface is a guess there, and so is the affinity.");
  }
  const outside = done.filter((result) => result.outsideTrainingRange).length;
  if (outside > 0) {
    parts.push(`${outside} ligand${outside === 1 ? " exceeds" : "s exceed"}`
      + " 56 heavy atoms, the largest seen in training: an extrapolation, not"
      + " a prediction.");
  }
  // 🔴 THE PAPER'S OWN FRAMING, NOT A PARAPHRASE. B.5.1: the value head is
  // "a general measure of binding strength that supports ranking" and is to
  // be "approximately interpreted as an IC50-like value" - it is trained on a
  // MIXTURE of Ki, Kd, IC50, AC50, EC50 and XC50. Calling the column
  // log10(IC50) overstated what the number is.
  parts.push("P(binder) and pIC50 come from two heads trained on different"
    + " data with different supervision; there is no single score. The value"
    + " head is trained on mixed Ki/Kd/IC50 assays - its authors call it an"
    + " IC50-LIKE measure that supports ranking, not a potency to quote.");
  return parts.join(" ");
}
