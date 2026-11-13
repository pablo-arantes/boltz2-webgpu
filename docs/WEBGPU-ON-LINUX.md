# Turning WebGPU on, on Linux

This page screens molecules on your own GPU through **WebGPU**. On Linux that is
usually off until you turn it on, which is why the page may tell you it got no
GPU. Everything below is a browser setting — nothing here installs or changes
anything to do with this page.

Every version number and preference name was checked against its primary source;
the sources are listed at the end.

## First: which of the two problems do you have?

They look similar and have **different fixes**, so start here. The page tells
you which one it hit, but you can also check for yourself — open the browser
console on this page (F12 → Console) and paste:

```js
console.log("secure context:", window.isSecureContext,
            "· WebGPU API:", !!navigator.gpu,
            "· adapter:", navigator.gpu ? await navigator.gpu.requestAdapter() : "n/a");
```

| What you see | What it means | Go to |
| --- | --- | --- |
| `WebGPU API: false` | The browser is not offering WebGPU at all | **Problem A** |
| `WebGPU API: true` but `adapter: null` | The browser has WebGPU and cannot find a GPU to give it | **Problem B** |
| `secure context: false` | You are on `http://` — WebGPU is only offered to secure origins. Use the `https://` address, or `http://localhost` | — |
| an adapter object | WebGPU works; if the page still fails, that is our bug — please send the diagnostics line the page prints | — |

---

# Chrome (and Edge, and other Chromium browsers)

## Problem A — no WebGPU at all

**Chrome enables WebGPU by default on Linux only from version 144, and only for
Intel Gen12+ GPUs.** NVIDIA on Wayland followed in 147. On anything older, or
on AMD, you turn it on with two flags. Check your version at
`chrome://version`.

1. Paste into the address bar, set to **Enabled**:
   ```
   chrome://flags/#enable-unsafe-webgpu
   ```
2. Paste into the address bar, set to **Enabled**:
   ```
   chrome://flags/#enable-vulkan
   ```
   🔴 **Both of them.** Chrome's WebGPU runs on Vulkan on Linux. The first flag
   exposes the API; without the second there is no backend behind it, and you
   land on Problem B instead — the API present, no adapter.
3. **Relaunch** Chrome (the button the flags page offers is enough).
4. Reload this page.

## Problem B — WebGPU is there, but no GPU adapter

Work down this list; the first two fix most cases.

1. **Is Vulkan on?** `chrome://flags/#enable-vulkan` → Enabled → relaunch. This
   is the commonest cause, and it is why Problem A's step 2 matters.
2. **Is hardware acceleration on?**
   ```
   chrome://settings/system
   ```
   Turn on **"Use graphics acceleration when available"**, then relaunch. With
   it off, Chrome renders in software and will not hand a page a GPU.
3. **What does Chrome itself say?**
   ```
   chrome://gpu
   ```
   Read the **WebGPU** row and the **Problems Detected** section. This is the
   fastest way to tell a missing flag from a driver Chrome has blocklisted.
4. **Does the machine have a hardware Vulkan driver at all?** In a **terminal**:
   ```bash
   sudo apt install mesa-vulkan-drivers vulkan-tools   # Debian / Ubuntu
   vulkaninfo --summary
   ```
   🔴 **If the summary lists only `llvmpipe` or `lavapipe`, that is software
   rendering** — no browser flag can fix a missing GPU driver. Install your
   GPU's own Vulkan driver first (`mesa-vulkan-drivers` covers Intel and AMD
   through Mesa; NVIDIA's comes with its proprietary driver package).
5. **Is the driver blocklisted?** If `chrome://gpu` says so:
   ```
   chrome://flags/#ignore-gpu-blocklist
   ```
   → Enabled → relaunch. This overrides a decision rather than fixing it, and
   that decision is sometimes right — update the driver first if you can.
6. **The documented fallback**, in a throwaway profile so it cannot disturb the
   browser you use for everything else:
   ```bash
   google-chrome --user-data-dir=/tmp/chrome-webgpu --enable-unsafe-webgpu \
     --ozone-platform=x11 --use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE
   ```
   🔴 **`--ozone-platform=x11` is deliberate.** Chromium reports
   `'--ozone-platform=wayland' is not compatible with Vulkan` on some setups,
   including ones where `vulkaninfo` finds the GPU perfectly well. If you are on
   a Wayland session, this is worth trying before concluding the hardware cannot
   do it.
   🔴 **And the command-line switch is `--enable-features=Vulkan`.**
   `--enable-vulkan` is the *name of the entry on the flags page* and does
   nothing on a command line.

---

# Firefox

🔴 **On Linux, Firefox has WebGPU off by default in Release and Beta — Nightly
only.** That is not a guess about rollout: `dom.webgpu.enabled` in Mozilla's own
preference list is

```
#if defined(XP_WIN) || defined(XP_MACOSX) && defined(MOZ_AARCH64)
  value: true
#else
  value: @IS_NIGHTLY_BUILD@
```

— true on Windows and on Apple-silicon macOS, and Nightly-only everywhere else,
which includes Linux and Intel Macs.

**If this is an ESR build, stop here.** Mozilla state that ESR 153 does not
support WebGPU on Linux and that Linux-specific WebGPU security fixes are *not*
being backported to it. The preference can still be flipped, which is exactly
why this needs saying: you would be enabling code that is not being patched for
you. Use a current Firefox, or Chrome.

## Problem A — no WebGPU at all

1. Address bar: `about:config`, accept the warning.
2. Search for **`dom.webgpu.enabled`** and set it to **`true`**.
3. **Restart** Firefox and reload this page.

That is the only preference that turns the API on. There is **no**
`gfx.webgpu.force-enabled` — it appears in several third-party guides and does
not exist in Firefox.

## Problem B — WebGPU is there, but no GPU adapter

1. **What does Firefox say?** Address bar `about:support`, then the **Graphics**
   section. This is Firefox's equivalent of `chrome://gpu`, and it is the part
   to quote if you report the problem.
2. **Is it blocklisted?** If the graphics report says so: `about:config` →
   **`gfx.webgpu.ignore-blocklist`** → `true` → restart.
3. **Does the machine have a hardware Vulkan driver?** Firefox's WebGPU uses
   wgpu, whose backend list includes `vulkan`, so this is the same question as
   for Chrome:
   ```bash
   sudo apt install mesa-vulkan-drivers vulkan-tools   # Debian / Ubuntu
   vulkaninfo --summary
   ```
   Only `llvmpipe` or `lavapipe` means software rendering, and no preference
   fixes that.

---

## Honest summary

**Current Chrome is the surest route on Linux.** Firefox on Linux is
experimental even with the preference set, and Chrome's own Linux support only
began rolling out in 144 and still does not cover every GPU. If a machine has no
hardware Vulkan driver, neither browser can run this page — and the screen needs
WebGPU, with no fallback: the affinity module has no WebGL path.

If none of this works, the page prints a **diagnostics line** under its error.
Copy that, plus `chrome://gpu` → Problems Detected (or `about:support` →
Graphics), and open an issue — those two together say which of these cases you
are in.

## Sources

Checked on 2026-11-13.

- Chrome's version and platform requirements, the flag names, the Linux
  `#enable-vulkan` requirement under a null adapter, and the secure-context
  rule: [WebGPU troubleshooting tips](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips),
  developer.chrome.com.
- Chrome's Linux rollout starting at Intel Gen12+:
  [What's New in WebGPU (Chrome 144)](https://developer.chrome.com/blog/new-in-webgpu-144).
- The fallback launch line and the Wayland/Vulkan incompatibility: the
  [gpuweb Implementation Status wiki](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)
  and Chromium issue 556278702.
- Firefox's preference values and names, quoted above:
  `modules/libpref/init/StaticPrefList.yaml` in
  [mozilla-firefox/firefox](https://github.com/mozilla-firefox/firefox) —
  `dom.webgpu.enabled`, `gfx.webgpu.ignore-blocklist`,
  `dom.webgpu.wgpu-backend`. `grep force-enabled` over that file returns
  twenty-one preferences, none of them `gfx.webgpu.force-enabled`.
- Firefox ESR on Linux:
  [bugzilla 2063383](https://bugzilla.mozilla.org/show_bug.cgi?id=2063383),
  filed by Mozilla's Andy Leiserson.
