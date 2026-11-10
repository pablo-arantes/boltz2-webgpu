/**
 * Two token-embedding weights, decoded off the shipped bundle.
 *
 * 🔴 WHY THIS IS A TOOL AND NOT A SCRIPT. The bundle's tensors are int5 in
 * shards, and the only decoder that is known-correct is the store's own -
 * `check-bundle-device-decode.js` exists because laying out a shard by hand
 * gets it wrong. Served over HTTP by gpu-chrome.mjs the store works normally;
 * a hand-rolled `fetch` stub does not, because the store streams its shards.
 */
import { openAf3Store } from "../../shared/af3/weights/weights.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const names = option(args, "tensors",
    "diffuser/boltz2_res_type_encoding/weights,"
    + "diffuser/boltz2_msa_profile_encoding/weights").split(",");
  const store = await openAf3Store(manifest);
  const out = {};
  for (const name of names) {
    const values = await store.tensor(name);
    out[name] = { shape: store.shape(name), values: Array.from(values) };
  }
  return out;
}
