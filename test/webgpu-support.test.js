/**
 * The WebGPU diagnosis, held to the facts it asserts.
 *
 * 🔴 THIS IS THE MOST-READ TEXT ON THE SITE FOR ANYONE IT FAILS, and it is the
 * one piece of copy nobody here can check by running it: the machine this is
 * written on has WebGPU, so every branch below is for a platform that cannot be
 * reproduced locally. The report that started it was a photograph of one
 * sentence - "This browser has no WebGPU. It ships in current Chrome, Edge,
 * Safari and Firefox." - shown to two readers who were using current Chrome.
 * It named no cause, no check and no fix, and on Linux it was simply wrong.
 *
 * Every version claim here was checked against its source on 2026-11-11; the
 * sources are listed in web/webgpu-support.js's header. These tests exist so
 * that a later edit cannot quietly reintroduce a claim nobody verified.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { webgpuAdvice, readAgent, WEBGPU_BY_DEFAULT }
  from "../web/webgpu-support.js";

const UA = {
  chromeLinux: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML,"
    + " like Gecko) Chrome/143.0.0.0 Safari/537.36",
  chromeWindows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    + " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
  chromeOld: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    + " (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
  edgeWindows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    + " (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
  firefoxLinux: "Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101"
    + " Firefox/141.0",
  firefoxWindows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0)"
    + " Gecko/20100101 Firefox/139.0",
  safari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15"
    + " (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
  androidChrome: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36"
    + " (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
};

const pastes = (advice) => advice.steps
  .map((step) => step.paste).filter((paste) => paste !== undefined);

test("🔴 the agent is read browser-first, and Edge is not a Chrome", () => {
  // Edge's UA contains "Chrome/143" too; reading Chrome first names it wrong
  // and prints a version from the wrong product.
  assert.deepEqual(readAgent(UA.edgeWindows),
    { browser: "Edge", os: "Windows", version: 143 });
  assert.deepEqual(readAgent(UA.chromeWindows),
    { browser: "Chrome", os: "Windows", version: 143 });
  assert.deepEqual(readAgent(UA.firefoxLinux),
    { browser: "Firefox", os: "Linux", version: 141 });
  assert.deepEqual(readAgent(UA.safari),
    { browser: "Safari", os: "macOS", version: 18 });
});

test("🔴 ANDROID IS NOT DESKTOP LINUX, though its UA says Linux", () => {
  // Every Android browser's UA contains "Linux". Testing Linux before Android
  // sends a phone the Vulkan-flag procedure, which is not its problem and
  // cannot be carried out.
  assert.equal(readAgent(UA.androidChrome).os, "Android");
  const advice = webgpuAdvice("missing", { userAgent: UA.androidChrome });
  assert.doesNotMatch(advice.summary, /Linux/);
  assert.ok(!pastes(advice).includes("chrome://flags/#enable-vulkan"));
});

test("🔴 Chrome on LINUX is told about both flags, and why", () => {
  const advice = webgpuAdvice("missing", { userAgent: UA.chromeLinux });
  // the fact the old message got wrong: Linux was never in Chrome's 113 list
  assert.match(advice.summary, /144/);
  assert.match(advice.summary, /Intel Gen12\+/);
  assert.ok(pastes(advice).includes("chrome://flags/#enable-unsafe-webgpu"));
  assert.ok(pastes(advice).includes("chrome://flags/#enable-vulkan"),
    "Chrome's own troubleshooting page requires #enable-vulkan on Linux");
  assert.ok(pastes(advice).includes("chrome://gpu"));
});

test("🔴 a WINDOWS Chrome is NOT told to enable Vulkan", () => {
  // It has had WebGPU since 113 there, so the cause is local - acceleration
  // off, a blocklisted driver, a policy - and a Linux-only flag would be noise
  // that makes the real step harder to find.
  const advice = webgpuAdvice("missing", { userAgent: UA.chromeWindows });
  assert.ok(!pastes(advice).includes("chrome://flags/#enable-vulkan"));
  assert.ok(pastes(advice).includes("chrome://settings/system"),
    "graphics acceleration is the commonest cause on Windows");
  assert.match(advice.summary, /113/);
  // a two-GPU laptop is a Windows-shaped problem
  assert.ok(pastes(advice).includes("chrome://flags/#force-high-performance-gpu"));
});

test("🔴 an out-of-date Chrome is told to update, and nothing else", () => {
  const advice = webgpuAdvice("missing", { userAgent: UA.chromeOld });
  assert.match(advice.summary, /113/);
  assert.match(advice.summary, /108/);
  assert.ok(!pastes(advice).includes("chrome://flags/#enable-unsafe-webgpu"),
    "a flag cannot add WebGPU to a browser that does not have it");
});

test("🔴 Firefox is not told it has WebGPU where it does not", () => {
  const linux = webgpuAdvice("missing", { userAgent: UA.firefoxLinux });
  assert.match(linux.summary, /Nightly-only|does not enable/);
  assert.ok(pastes(linux).includes("about:config"));
  const windows = webgpuAdvice("missing", { userAgent: UA.firefoxWindows });
  assert.match(windows.summary, /141/,
    "Firefox shipped WebGPU on Windows in 141 and this reader is on 139");
});

test("🔴 the three stages are three different diagnoses", () => {
  const missing = webgpuAdvice("missing", { userAgent: UA.chromeWindows });
  const adapter = webgpuAdvice("no-adapter", { userAgent: UA.chromeWindows });
  const device = webgpuAdvice("device-failed",
    { userAgent: UA.chromeWindows, detail: "maxStorageBufferBindingSize" });
  assert.notEqual(missing.summary, adapter.summary);
  assert.notEqual(adapter.summary, device.summary);
  // a null adapter is NOT "no WebGPU": saying so sends a reader to a flags page
  // for a browser that already has the API
  assert.match(adapter.summary, /has WebGPU/);
  // 🔴 AND A REFUSED DEVICE IS OURS TO FIX. It is the one stage where no
  // browser setting is the answer, so the text must say so and ask for the
  // diagnostics rather than list flags.
  assert.match(device.summary, /our bug/);
  assert.match(device.summary, /maxStorageBufferBindingSize/);
  assert.ok(!pastes(device).includes("chrome://flags/#enable-unsafe-webgpu"));
});

test("🔴 an insecure origin is diagnosed as OURS, before anything else", () => {
  // It hides every other cause - the API is simply not exposed - and it is the
  // only one of these that the site causes rather than the visitor.
  const advice = webgpuAdvice("missing",
    { userAgent: UA.chromeLinux, isSecureContext: false });
  assert.match(advice.summary, /secure/);
  assert.match(advice.summary, /https:\/\/|localhost/);
  // and NOT the Linux flag procedure, which would be a wild goose chase
  assert.ok(!pastes(advice).includes("chrome://flags/#enable-vulkan"));
});

test("🔴 every diagnosis carries a diagnostics line with the stage in it", () => {
  for (const stage of ["missing", "no-adapter", "device-failed"]) {
    const advice = webgpuAdvice(stage, { userAgent: UA.chromeLinux });
    assert.match(advice.diagnostics, new RegExp(`stage ${stage}`));
    assert.match(advice.diagnostics, /Chrome 143 on Linux/);
    assert.match(advice.diagnostics, /ua Mozilla/);
  }
});

test("🔴 a browser nobody recognised still gets somewhere to go", () => {
  const advice = webgpuAdvice("missing", { userAgent: "Mystery/1.0" });
  assert.ok(advice.summary.length > 40);
  assert.ok(advice.steps.length > 0);
  assert.match(advice.diagnostics, /unknown/);
});

test("🔴 no step is a clickable link, because chrome:// cannot be one", () => {
  // Chrome refuses navigation to chrome:// from page content: an anchor would
  // do nothing when clicked, and a reader would read that as a broken page.
  for (const stage of ["missing", "no-adapter", "device-failed"]) {
    for (const ua of Object.values(UA)) {
      const advice = webgpuAdvice(stage, { userAgent: ua });
      for (const step of advice.steps) {
        assert.ok(!("href" in step), "a step must be text to paste, not a link");
        assert.equal(typeof step.do, "string");
        assert.ok(step.do.length > 20, `a step with no instruction: ${step.paste}`);
      }
    }
  }
});

test("🔴 the README's support table is this module's, not a second copy", () => {
  // 🔴 TWO LISTS OF VERSION NUMBERS IS ONE LIST THAT GOES STALE, and the stale
  // one is always the prose. The README prints every row of WEBGPU_BY_DEFAULT.
  const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
  for (const row of WEBGPU_BY_DEFAULT) {
    assert.ok(readme.includes(row.where),
      `the README does not mention "${row.where}"`);
    assert.match(readme, new RegExp(`${row.from}\\b`),
      `the README does not give version ${row.from} for ${row.where}`);
  }
});
