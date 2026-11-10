/**
 * Everything a screen needs on the device, loaded once.
 *
 * 🔴 TWO BUNDLES, FETCHED SEPARATELY, AND THE SECOND ONE LAZILY. The structure
 * bundle is 365 MiB and the affinity bundle is 6.5 MiB; they are separate
 * checkpoints (boltz2_conf.ckpt and boltz2_aff.ckpt) sharing 5,019 tensors byte
 * for byte, so the affinity bundle holds only the 520 that are its own. Both go
 * through `openStore`, which means the pinned remote, the CacheStorage shard
 * cache and the download dial with no new code - see MODEL_BUNDLES.
 *
 * 🔴 THE AFFINITY TENSORS ARE READ EAGERLY INTO MEMORY, ON PURPOSE. 520
 * tensors decode to ~33 MiB of f32, and `affinityWeights` takes a SYNCHRONOUS
 * reader because the module's weight adapters are getters that the kernels pull
 * from mid-dispatch. Awaiting inside those would turn every weight access into
 * a promise; 33 MiB resident for the life of the page is the cheaper side of
 * that trade, and it is read once rather than once per ligand.
 */
import { openStore } from "./model.js";
import { trunkDepths, trunkWeights, confidenceWeights } from "../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../shared/af3/weights/diffusion-weights.js";

/** The structure model: trunk, diffusion, confidence, target-feat. */
export async function loadStructureWeights(onProgress) {
  const store = await openStore(onProgress, "boltz2");
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks,
                              { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
  };
  // 🔴 NOT OPTIONAL HERE, where a fold tool may skip it: the affinity pass
  // picks which diffusion sample the module is given BY ipTM, so without a
  // confidence head there is no selection to make and silently taking sample 0
  // would be a different algorithm under the same name.
  try {
    weights.confidence = await confidenceWeights(store);
  } catch (error) {
    throw new Error("the confidence head would not load, and the affinity pass"
      + " selects its sample by ipTM: " + String(error?.message ?? error));
  }
  return { store, weights, depths };
}

/**
 * The affinity checkpoint's 520 tensors, as the synchronous reader
 * `affinityWeights` wants.
 *
 * @returns {Promise<{read: (name: string) => Float64Array|null, tensors: number,
 *                    bytes: number}>}
 */
export async function loadAffinityWeights(onProgress) {
  const store = await openStore(onProgress, "boltz2-affinity");
  const names = Object.keys(store.manifest?.tensors ?? {});
  if (names.length === 0) {
    throw new Error("the affinity bundle's manifest has no tensors");
  }
  const values = new Map();
  let bytes = 0;
  for (const name of names) {
    // 🔴 Float64Array, NOT the store's Float32Array. The weight adapters in
    // shared/af3/weights/affinity-weights.js re-cut tensors (transposes, the
    // interleaved triangle pack) in double precision so the re-layout cannot
    // itself lose a bit; the kernels round to f32 when they upload.
    const tensor = await store.tensor(name);
    values.set(name, Float64Array.from(tensor));
    bytes += tensor.length * 4;
  }
  return {
    read: (name) => values.get(name) ?? null,
    tensors: values.size,
    bytes,
  };
}
