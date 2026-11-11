/**
 * Why this browser has no WebGPU, and what the visitor can do about it.
 *
 * 🔴 THIS EXISTS BECAUSE THE OLD MESSAGE TOLD A CHROME USER THAT WEBGPU "SHIPS
 * IN CURRENT CHROME". Two readers on Linux and Windows, both on Chrome, both
 * with real GPUs, got "This browser has no WebGPU. It ships in current Chrome,
 * Edge, Safari and Firefox." - a sentence that is useless to someone who is
 * already using Chrome, and wrong about Linux, where Chrome did NOT enable
 * WebGPU by default until 144. A visitor cannot act on that, and nor can we:
 * it names no cause, no check and no fix.
 *
 * 🔴 AND THE THREE FAILURE STAGES ARE DIFFERENT PROBLEMS WITH DIFFERENT FIXES,
 * which one sentence cannot cover. `navigator.gpu` absent is the browser not
 * exposing the API at all - a version or a platform question. A null adapter is
 * the API working and refusing to hand over a GPU - graphics acceleration off,
 * or a driver on Chrome's blocklist. A failed `requestDevice` is a limit or a
 * feature this port asked for and the hardware does not have, which is OUR bug
 * and has to be reported as one. Chrome's own troubleshooting page splits them
 * the same way, and the old code collapsed all three into one string.
 *
 * Every version and platform claim below was checked against its source on
 * 2026-11-11 rather than recalled:
 *
 *   - developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips:
 *     "WebGPU requires Chrome 113 or greater on ChromeOS, macOS, Windows and
 *     Chrome 121 or greater on Android" - desktop Linux is NOT in that list.
 *     For a null adapter it names `chrome://settings/system`'s "Use graphics
 *     acceleration when available", `chrome://flags/#enable-unsafe-webgpu`,
 *     `chrome://flags/#ignore-gpu-blocklist`,
 *     `chrome://flags/#force-high-performance-gpu`, `chrome://gpu`, and "For
 *     Linux experimental support, you'll also need to enable the
 *     chrome://flags/#enable-vulkan flag".
 *   - developer.chrome.com/blog/new-in-webgpu-144: Chrome is "carefully rolling
 *     out WebGPU for Linux, starting with support for Intel Gen12+ GPUs but
 *     with a tentative plan to expand it to more devices (AMD, NVIDIA)".
 *   - mozillagfx.wordpress.com, Firefox 141: WebGPU on by default on WINDOWS
 *     only; the gpuweb Implementation-Status wiki has macOS Apple Silicon at
 *     145/147 and Linux as Nightly-only.
 *   - gpuweb Implementation-Status: Safari 26 on macOS 26, iOS 26, iPadOS 26.
 *
 * 🔴 AND A `chrome://` ADDRESS CANNOT BE A LINK. Chrome refuses navigation to
 * `chrome://` from page content, so an `<a href>` here would silently do
 * nothing - which is worse than no link, because the reader concludes the page
 * is broken. Every step below returns the address as `paste`, text for the
 * address bar, and the UI must render it as something copyable rather than as a
 * link.
 *
 * This module is pure: it takes what the page observed and returns what to say.
 * That is what lets test/webgpu-support.test.js check the advice for platforms
 * nobody here can run.
 */

/** What ships WebGPU by default, as checked above. The README prints this. */
export const WEBGPU_BY_DEFAULT = [
  { browser: "Chrome, Edge", where: "Windows, macOS, ChromeOS", from: "113" },
  { browser: "Chrome", where: "Android", from: "121" },
  { browser: "Chrome", where: "Linux, Intel Gen12+ GPUs only", from: "144" },
  { browser: "Firefox", where: "Windows", from: "141" },
  { browser: "Firefox", where: "macOS, Apple silicon", from: "145" },
  { browser: "Safari", where: "macOS 26, iOS 26, iPadOS 26", from: "26" },
];

const CHECK_GPU = {
  do: "Open Chrome's own GPU report and read the WebGPU row and"
    + " “Problems Detected”. It says what Chrome thinks of this"
    + " machine, which is the fastest way to tell a missing flag from a"
    + " blocklisted driver.",
  paste: "chrome://gpu",
};

const ACCELERATION = {
  do: "Turn on “Use graphics acceleration when available”, then"
    + " relaunch. With it off, Chrome renders in software and will not hand a"
    + " page a GPU.",
  paste: "chrome://settings/system",
};

const UNSAFE_WEBGPU = {
  do: "Set “Unsafe WebGPU Support” to Enabled and relaunch.",
  paste: "chrome://flags/#enable-unsafe-webgpu",
};

const VULKAN = {
  do: "Set “Vulkan” to Enabled as well - on Linux, Chrome's WebGPU"
    + " runs on Vulkan - then relaunch.",
  paste: "chrome://flags/#enable-vulkan",
};

const BLOCKLIST = {
  do: "If the GPU report says your driver is blocklisted, this overrides it."
    + " It is a real risk on an old driver: update the driver first if you can.",
  paste: "chrome://flags/#ignore-gpu-blocklist",
};

const HIGH_PERFORMANCE = {
  do: "On a laptop with two GPUs, this stops Chrome picking the integrated one"
    + " and finding it unusable.",
  paste: "chrome://flags/#force-high-performance-gpu",
};

/**
 * The browser and platform, read off a user-agent string.
 *
 * 🔴 UA SNIFFING, AND ONLY FOR ADVICE. `navigator.userAgentData` is
 * Chromium-only, and the question here - "which fix should I show you" - is
 * exactly the question a UA string can answer badly without harm: a wrong guess
 * shows one extra step. No behaviour depends on it.
 */
export function readAgent(userAgent = "") {
  const version = (pattern) => {
    const found = userAgent.match(pattern);
    return found === null ? null : Number.parseInt(found[1], 10);
  };
  const os = /Android/i.test(userAgent) ? "Android"
    : /CrOS/.test(userAgent) ? "ChromeOS"
      : /Windows/i.test(userAgent) ? "Windows"
        : /(iPhone|iPad|iPod)/i.test(userAgent) ? "iOS"
          // 🔴 LINUX IS TESTED LAST OF THE DESKTOPS, because Android's UA
          // contains "Linux" - every Android browser would read as desktop
          // Linux and be told to enable a Vulkan flag that is not its problem.
          : /Mac OS X/i.test(userAgent) ? "macOS"
            : /(Linux|X11)/i.test(userAgent) ? "Linux" : "unknown";
  // 🔴 AND EDGE BEFORE CHROME, because Edge's UA contains "Chrome/" too; the
  // reverse order calls every Edge a Chrome and prints the wrong version.
  const browser = /Edg\//.test(userAgent) ? "Edge"
    : /OPR\//.test(userAgent) ? "Opera"
      : /Firefox\//.test(userAgent) ? "Firefox"
        : /Chrome\//.test(userAgent) ? "Chrome"
          : /Safari\//.test(userAgent) ? "Safari" : "unknown";
  const number = browser === "Firefox" ? version(/Firefox\/(\d+)/)
    : browser === "Edge" ? version(/Edg\/(\d+)/)
      : browser === "Safari" ? version(/Version\/(\d+)/)
        : version(/Chrome\/(\d+)/);
  return { browser, os, version: number };
}

/** True for the browsers the chrome:// steps apply to. */
const isChromium = (browser) =>
  browser === "Chrome" || browser === "Edge" || browser === "Opera";

/**
 * What to tell a visitor whose WebGPU did not start.
 *
 * @param {"missing"|"no-adapter"|"device-failed"} stage where it failed
 * @param {{userAgent?: string, isSecureContext?: boolean, detail?: string}} env
 * @returns {{summary: string, steps: {do: string, paste?: string}[],
 *            diagnostics: string, agent: object}}
 */
export function webgpuAdvice(stage, env = {}) {
  const agent = readAgent(env.userAgent ?? "");
  const { browser, os, version } = agent;
  const secure = env.isSecureContext !== false;
  const diagnostics = [
    `stage ${stage}`,
    `navigator.gpu ${stage === "missing" ? "absent" : "present"}`,
    `secureContext ${secure}`,
    `${browser}${version === null ? "" : ` ${version}`} on ${os}`,
    env.detail === undefined ? null : `detail ${env.detail}`,
    `ua ${env.userAgent ?? "unknown"}`,
  ].filter((part) => part !== null).join(" · ");

  // 🔴 AN INSECURE ORIGIN IS CHECKED FIRST AND SEPARATELY. It hides every other
  // cause - the API is simply not exposed - and it is the one failure that is
  // the SITE's fault rather than the visitor's, so it must not be dressed up as
  // a driver problem. It cannot happen on the published HTTPS site; it happens
  // to anyone serving this tree over plain http:// on a LAN address.
  if (!secure) {
    return {
      summary: "This page is not on a secure origin, and WebGPU is only"
        + " exposed to secure ones - so no browser will offer it here,"
        + " whatever its version. Serve the page over https://, or over"
        + " http://localhost, which counts as secure.",
      steps: [{
        do: "If you must use a LAN address, Chrome can be told to treat one as"
          + " secure: add the origin to this flag's list and relaunch.",
        paste: "chrome://flags/#unsafely-treat-insecure-origin-as-secure",
      }],
      diagnostics,
      agent,
    };
  }

  if (stage === "device-failed") {
    return {
      summary: "This browser has WebGPU and gave us a GPU, but refused the"
        + " device this screen asked for"
        + `${env.detail === undefined ? "" : `: ${env.detail}`}.`
        + " That is a limit or a feature this port requested and your hardware"
        + " does not have, which makes it our bug rather than your setup."
        + " Please send the diagnostics line below.",
      steps: [CHECK_GPU],
      diagnostics,
      agent,
    };
  }

  if (stage === "no-adapter") {
    return {
      summary: "This browser has WebGPU, but it could not give this page a GPU"
        + " adapter. That is almost always graphics acceleration being switched"
        + " off, or a graphics driver on Chrome's blocklist - not a missing"
        + " GPU.",
      steps: isChromium(browser)
        ? [ACCELERATION, CHECK_GPU, BLOCKLIST,
          ...(os === "Windows" ? [HIGH_PERFORMANCE] : [])]
        : [{
          do: "Check that hardware acceleration is on in your browser's"
            + " settings and that your graphics driver is current, then"
            + " relaunch.",
        }],
      diagnostics,
      agent,
    };
  }

  // stage === "missing": the API is not exposed at all.
  if (isChromium(browser) && os === "Linux") {
    return {
      summary: "Chrome on Linux does not enable WebGPU by default below version"
        + " 144, and 144 only turns it on for Intel Gen12+ GPUs - AMD and"
        + " NVIDIA are still being rolled out. This is the usual reason a Linux"
        + " machine with a perfectly good GPU gets no WebGPU."
        + ` You are on ${browser}${version === null ? "" : ` ${version}`}.`
        + " Two flags turn it on, and they work well below 144.",
      steps: [UNSAFE_WEBGPU, VULKAN, CHECK_GPU, ACCELERATION],
      diagnostics,
      agent,
    };
  }
  if (isChromium(browser) && version !== null && version < 113) {
    return {
      summary: `WebGPU needs Chrome or Edge 113 and this is ${version}.`
        + " Updating the browser is the whole fix.",
      steps: [CHECK_GPU],
      diagnostics,
      agent,
    };
  }
  if (isChromium(browser)) {
    return {
      summary: `${browser} has had WebGPU since 113 on ${os === "unknown"
        ? "this platform" : os}, so something on this machine is switching it`
        + " off. The usual causes are graphics acceleration being disabled"
        + " (Chrome then renders in software and exposes no WebGPU), a"
        + " blocklisted graphics driver, or a managed-browser policy.",
      steps: [ACCELERATION, CHECK_GPU, UNSAFE_WEBGPU, BLOCKLIST,
        ...(os === "Windows" ? [HIGH_PERFORMANCE] : [])],
      diagnostics,
      agent,
    };
  }
  if (browser === "Firefox") {
    const where = os === "Windows"
      ? `Firefox has WebGPU by default from 141 on Windows and this is ${
        version === null ? "older" : version}.`
      : os === "macOS"
        ? "Firefox enables WebGPU by default on macOS only from 145, and only"
          + " on Apple silicon."
        : `Firefox does not enable WebGPU by default on ${os} yet - on Linux it`
          + " is Nightly-only.";
    return {
      summary: `${where} Current Chrome is the surest route on this platform;`
        + " in Firefox you can try the switch below, which is experimental.",
      steps: [{
        do: "Open Firefox's configuration editor, search for"
          + " dom.webgpu.enabled and set it to true, then restart. On Linux you"
          + " may also need gfx.webgpu.force-enabled.",
        paste: "about:config",
      }],
      diagnostics,
      agent,
    };
  }
  if (browser === "Safari") {
    return {
      summary: "Safari has WebGPU by default from version 26 (macOS 26, iOS 26"
        + " and iPadOS 26)."
        + `${version === null || version >= 26 ? "" : ` This is Safari ${version}.`}`
        + " On an older Safari it is behind a feature flag.",
      steps: [{
        do: "In Safari, open Settings › Advanced and turn on “Show"
          + " features for web developers”, then Develop › Feature"
          + " Flags and tick WebGPU.",
      }],
      diagnostics,
      agent,
    };
  }
  return {
    summary: "This browser does not expose WebGPU. Current Chrome or Edge is"
      + " the surest route on Windows, macOS and ChromeOS; on Linux, Chrome"
      + " needs two flags (see below) below version 144.",
    steps: [UNSAFE_WEBGPU, VULKAN, CHECK_GPU],
    diagnostics,
    agent,
  };
}
