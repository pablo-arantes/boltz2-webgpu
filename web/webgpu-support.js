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

/**
 * 🔴 WHERE A STEP IS CARRIED OUT, BECAUSE THEY ARE NOT ALL THE ADDRESS BAR.
 * A Linux null adapter needs `vulkaninfo --summary` in a TERMINAL and a
 * relaunch with command-line switches, and a reader who pastes either into the
 * address bar gets a search for it. Every step declares one; the UI labels it.
 */
export const STEP_KINDS = ["address", "terminal", "launch", "none"];

const CHECK_GPU = {
  kind: "address",
  do: "Open Chrome's own GPU report and read the WebGPU row and"
    + " “Problems Detected”. It says what Chrome thinks of this"
    + " machine, which is the fastest way to tell a missing flag from a"
    + " blocklisted driver.",
  paste: "chrome://gpu",
};

const ACCELERATION = {
  kind: "address",
  do: "Turn on “Use graphics acceleration when available”, then"
    + " relaunch. With it off, Chrome renders in software and will not hand a"
    + " page a GPU.",
  paste: "chrome://settings/system",
};

const UNSAFE_WEBGPU = {
  kind: "address",
  do: "Set “Unsafe WebGPU Support” to Enabled and relaunch.",
  paste: "chrome://flags/#enable-unsafe-webgpu",
};

const VULKAN = {
  kind: "address",
  do: "Set “Vulkan” to Enabled as well - on Linux, Chrome's WebGPU"
    + " runs on Vulkan - then relaunch.",
  paste: "chrome://flags/#enable-vulkan",
};

const BLOCKLIST = {
  kind: "address",
  do: "If the GPU report says your driver is blocklisted, this overrides it."
    + " It is a real risk on an old driver: update the driver first if you can.",
  paste: "chrome://flags/#ignore-gpu-blocklist",
};

const HIGH_PERFORMANCE = {
  kind: "address",
  do: "On a laptop with two GPUs, this stops Chrome picking the integrated one"
    + " and finding it unusable.",
  paste: "chrome://flags/#force-high-performance-gpu",
};

/**
 * 🔴 DOES THE MACHINE HAVE VULKAN AT ALL? On Linux this is the first question
 * and the page cannot answer it: Chrome's WebGPU runs on Vulkan there, so a
 * system with no Vulkan driver installed hands out no adapter however many
 * flags are set. It is also the one check that distinguishes "Chrome refused
 * the GPU" from "there is no hardware Vulkan on this box" - if the summary
 * lists only `llvmpipe` or `lavapipe`, that is software rendering and WebGPU
 * will stay refused.
 */
const VULKANINFO = {
  kind: "terminal",
  do: "In a terminal, check the system has a hardware Vulkan driver. On"
    + " Debian or Ubuntu, install it first with"
    + " \u201csudo apt install mesa-vulkan-drivers vulkan-tools\u201d. If the"
    + " summary lists only llvmpipe or lavapipe, that is software rendering and"
    + " no flag will help - the GPU's own driver is what is missing.",
  paste: "vulkaninfo --summary",
};

/**
 * 🔴 THE DOCUMENTED LINUX FALLBACK, IN A THROWAWAY PROFILE. The gpuweb
 * implementation-status wiki gives this exact launch for Linux configurations
 * the default build will not serve, and `--ozone-platform=x11` is in it for a
 * reason: Chromium reports "'--ozone-platform=wayland' is not compatible with
 * Vulkan", which has been seen on machines where `vulkaninfo` finds the GPU
 * perfectly well. The separate profile means a reader can try it without
 * changing the browser they use for everything else.
 *
 * 🔴 AND THE COMMAND-LINE SWITCH IS `--enable-features=Vulkan`, NOT
 * `--enable-vulkan`. The latter is the name of the entry on the flags page and
 * does nothing on a command line.
 */
const LINUX_FALLBACK = {
  kind: "launch",
  do: "If the flags above did not do it, try this launch in a separate,"
    + " throwaway profile - it is the fallback the WebGPU implementation-status"
    + " wiki documents for Linux. X11 rather than Wayland is deliberate:"
    + " Chromium refuses Vulkan under Wayland on some setups and says so in its"
    + " own log.",
  paste: "google-chrome --user-data-dir=/tmp/chrome-webgpu"
    + " --enable-unsafe-webgpu --ozone-platform=x11 --use-angle=vulkan"
    + " --enable-features=Vulkan,VulkanFromANGLE",
};

/**
 * 🔴 FIREFOX'S PREFS, CHECKED AGAINST MOZILLA'S OWN `StaticPrefList.yaml` ON
 * 2026-11-13 - because the first version of this advice named
 * `gfx.webgpu.force-enabled`, WHICH DOES NOT EXIST. It appears in third-party
 * guides; `grep force-enabled` over that file finds twenty-one prefs and none
 * of them is this one. Telling a reader to create a pref that does nothing is
 * worse than saying nothing, because they will believe they have tried it.
 *
 * What the file actually defines:
 *
 *   dom.webgpu.enabled         `true` on `XP_WIN`, or `XP_MACOSX &&
 *                              MOZ_AARCH64`; otherwise `@IS_NIGHTLY_BUILD@` -
 *                              so on LINUX it is on in Nightly and off in
 *                              Release and Beta, and on an Intel Mac it is off.
 *   gfx.webgpu.ignore-blocklist  `bool`, default `false`. The real blocklist
 *                              override, and Chrome's `#ignore-gpu-blocklist`
 *                              by another name.
 *   dom.webgpu.wgpu-backend    names the wgpu backends, "vulkan" among them,
 *                              which is why the Vulkan-driver check applies to
 *                              Firefox on Linux as much as to Chrome.
 */
const FIREFOX_PREF = {
  kind: "address",
  do: "Open Firefox's configuration editor, accept the warning, search for"
    + " dom.webgpu.enabled and set it to true, then restart. This is the only"
    + " pref that turns the API on.",
  paste: "about:config",
};

const FIREFOX_SUPPORT = {
  kind: "address",
  do: "Firefox's own graphics report - the equivalent of Chrome's GPU page."
    + " Read the Graphics section for the WebGPU rows and any blocklist"
    + " decision, and send that section if you report this.",
  paste: "about:support",
};

const FIREFOX_BLOCKLIST = {
  kind: "address",
  do: "If the graphics report says WebGPU is blocklisted, this is the override:"
    + " in about:config set gfx.webgpu.ignore-blocklist to true and restart. As"
    + " with Chrome's, it bypasses a decision rather than fixing it - update the"
    + " driver first if you can.",
  paste: "about:config",
};

/**
 * 🔴 AND A LINUX ESR IS THE ONE CASE WHERE "JUST FLIP THE PREF" IS BAD ADVICE.
 * Mozilla's Andy Leiserson, on bugzilla 2063383: "ESR 153 does not officially
 * support WebGPU on Linux", and they "do not plan to backport Linux-specific
 * WebGPU security fixes to ESR153". The pref can still be flipped, which is
 * exactly why this has to be said: a reader following generic instructions
 * would enable a feature whose Linux security fixes are not coming.
 */
const ESR_CAUTION = {
  kind: "none",
  do: "If this is an ESR build, do not force it on. Mozilla state that ESR 153"
    + " does not support WebGPU on Linux and that Linux-specific WebGPU"
    + " security fixes are not being backported to it (bugzilla 2063383), so"
    + " the pref would turn on code that is not being patched for you. Use a"
    + " current Firefox, or Chrome, instead.",
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
        kind: "address",
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
    // 🔴 AND THE LIKELIEST CAUSE IS NOT THE SAME ON LINUX. This sentence named
    // graphics acceleration and the blocklist on every platform, and a reader
    // on Linux read it, checked both, and was no further forward: there,
    // WebGPU runs on Vulkan, and the usual reason for a null adapter is that
    // there is no Vulkan backend behind the API - either the flag is off or the
    // machine has no hardware Vulkan driver. Naming the wrong cause first
    // sends someone to the wrong place even when the right step is further
    // down the list.
    const why = browser === "Firefox" && os === "Linux"
      ? "On Linux, Firefox's WebGPU runs on Vulkan, so the usual causes are a"
        + " blocklist decision or no hardware Vulkan driver on the machine."
        + " Its graphics report says which."
      : browser === "Firefox"
        ? "Firefox's graphics report says whether it blocklisted WebGPU on this"
          + " hardware, which is the usual reason."
        : os === "Linux"
      ? "On Linux that is usually Vulkan: Chrome's WebGPU runs on it, and"
        + " turning WebGPU on without a Vulkan backend behind it gives exactly"
        + " this - the API present, no adapter to hand out. Either the Vulkan"
        + " flag is off, or the machine has no hardware Vulkan driver"
        + " installed. The check below answers which, and the flags above are"
        + " the fix for the first."
      : "That is almost always graphics acceleration being switched off, or a"
        + " graphics driver on Chrome's blocklist - not a missing GPU.";
    return {
      summary: "This browser has WebGPU, but it could not give this page a GPU"
        + ` adapter. ${why}`,
      // 🔴 THE LINUX STEPS WERE FILED UNDER THE WRONG STAGE, and a reader hit
      // exactly this. Chrome's own troubleshooting page lists
      // `chrome://flags/#enable-vulkan` under "When GPU adapter is null" - not
      // under a missing `navigator.gpu` - because `#enable-unsafe-webgpu` alone
      // exposes the API while leaving no backend to hand out an adapter. This
      // branch offered acceleration, the GPU report and the blocklist, so the
      // one flag that fixes the commonest Linux case was in the other branch.
      steps: isChromium(browser)
        ? [...(os === "Linux" ? [UNSAFE_WEBGPU, VULKAN] : []),
          ACCELERATION, CHECK_GPU, BLOCKLIST,
          ...(os === "Windows" ? [HIGH_PERFORMANCE] : []),
          ...(os === "Linux" ? [VULKANINFO, LINUX_FALLBACK] : [])]
        : browser === "Firefox"
          // 🔴 FIREFOX GOT ONE GENERIC SENTENCE HERE, which is no better than
          // the Chrome branch was before it was fixed. Its graphics report and
          // its blocklist override are different controls with different names,
          // and on Linux its wgpu backend is Vulkan too - so the driver check
          // is the same question.
          ? [FIREFOX_SUPPORT, FIREFOX_BLOCKLIST,
            ...(os === "Linux" ? [VULKANINFO, ESR_CAUTION] : [])]
          : [{
            kind: "none",
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
        + " in Firefox the pref below turns the API on, which is experimental"
        + " rather than supported.",
      steps: [FIREFOX_PREF, FIREFOX_SUPPORT,
        ...(os === "Linux" ? [ESR_CAUTION] : [])],
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
        // a menu path, with nothing to paste - see STEP_KINDS
        kind: "none",
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
