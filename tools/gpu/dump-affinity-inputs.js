/**
 * This port's own `s_inputs` and `z` entering the affinity module, written out
 * for comparison against upstream's.
 *
 *   node tools/gpu-chrome.mjs tools/gpu/dump-affinity-inputs.js \
 *     --sequence=<one-letter> --ligands=<smiles> --out-prefix=ours
 *
 * 🔴 THIS IS THE OTHER HALF OF THE G8 LOCALISATION. G8 measured a 0.343
 * log-unit gap end to end - 6x upstream's own seed-to-seed spread - with the
 * affinity module's arithmetic already excluded at 1e-5..1e-7 against
 * upstream's dumps and the pose measured not to be the cause. What is left is
 * the two inputs the module is conditioned on, and neither had ever been
 * compared to upstream elementwise. `tools/oracle/dump_boltz2_affinity_inputs.py`
 * captures upstream's; this captures ours, from the same two-pass path the page
 * runs.
 *
 * 🔴 IT PRINTS THE TOKEN COUNT AND THE CROP SIZE BECAUSE THE COMPARISON IS ONLY
 * VALID WHERE THE CROP IS THE IDENTITY. The trunk reads features rather than
 * coordinates, so `s_inputs` and `z` are pose-independent GIVEN the same token
 * set - and the crop's token set depends on the pose. On a target small enough
 * that the crop keeps everything (43 of 43 on the G8 job) both sides featurise
 * the same tokens in the same order and an elementwise comparison means
 * something. On a target the crop cuts, it does not, and the numbers here say
 * which case it is rather than leaving it to be assumed.
 */
import { openAf3Store, trunkDepths, trunkWeights, confidenceWeights }
  from "../../shared/af3/weights/weights.js";
import { diffusionWeights, atomReference, targetFeatureWeights }
  from "../../shared/af3/weights/diffusion-weights.js";
import { makeFoldStructure, makeFoldAffinityPass, affinitySInputs }
  from "../../web/affinity-passes.js";
import { affinityCrop } from "../../shared/af3/affinity/crop.js";
import { AFFINITY_TOKEN_S, AFFINITY_TOKEN_Z }
  from "../../shared/af3/weights/affinity-weights.js";
import { crossPairMaskF32 } from "../../webgpu/af3/affinity/affinity.js";
import { parseSmiles } from "../../shared/chem/smiles.js";
import { perAtomConditioning } from "../../cpu/af3/diffusion/atom-conditioning.js";
import { Af3AtomEncoderGpu } from "../../webgpu/af3/diffusion/atom-encoder.js";
import { affinityProfile } from "../../shared/af3/affinity/features.js";

const option = (args, name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit === undefined ? fallback : hit.slice(name.length + 3);
};

/**
 * 🔴 CHUNKED, BECAUSE SPREADING A TYPED ARRAY INTO String.fromCharCode BLOWS
 * THE STACK. `z` here is 43 x 43 x 128 floats - 946 kB, so 946,000 arguments -
 * and the spread form throws RangeError on exactly the tensor this dump exists
 * for while working fine on every small one beside it.
 */
function base64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset,
                               values.byteLength);
  let text = "";
  const step = 0x8000;
  for (let at = 0; at < bytes.length; at += step) {
    text += String.fromCharCode(...bytes.subarray(at, at + step));
  }
  return btoa(text);
}

/** A few numbers that identify a tensor without printing it. */
function summarise(values) {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSquares = 0;
  let nonFinite = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) { nonFinite += 1; continue; }
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    sumSquares += value * value;
  }
  const n = values.length - nonFinite;
  return { count: values.length, nonFinite, min, max,
    mean: sum / n, rms: Math.sqrt(sumSquares / n) };
}

export async function main(device, args = []) {
  const manifest = option(args, "model", "/model-boltz2-int5/manifest.json");
  const affinityBase = option(args, "affinity", "/model-boltz2-affinity-f32");
  const sequence = option(args, "sequence",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ");
  const smiles = option(args, "ligands", "CCCc1ccccc1O");
  const seed = Number(option(args, "seed", "0"));
  const structRecycles = Number(option(args, "struct-recycles", "3"));
  const structSteps = Number(option(args, "struct-steps", "200"));
  const affinityRecycles = Number(option(args, "affinity-recycles", "5"));
  const affinitySteps = Number(option(args, "affinity-steps", "200"));
  const affinitySamples = Number(option(args, "affinity-samples", "5"));
  // 🔴 THE EXPERIMENT THAT PRICES "REPRODUCE RDKit's CANONICAL RANKING".
  // The remaining G8 gap is the ligand's atom NAMES - 256 of the atom
  // encoder's 388 input columns, and not one of ten agrees with upstream's.
  // Matching them means reproducing `CanonicalRankAtoms`, which is days of
  // work with a silent failure mode, and NOBODY HAS MEASURED WHAT IT WOULD
  // BUY. Substituting upstream's own names for this one ligand answers that
  // for the cost of one run: if `a`'s ligand half falls to the quantisation
  // floor and the binary probability lands in upstream's range, the work is
  // worth doing; if it does not, it is not.
  const overrideNames = option(args, "ligand-names", "");

  const store = await openAf3Store(manifest);
  const depths = trunkDepths(store);
  const weights = {
    trunk: await trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks,
                              { allowPrefix: true }),
    diffusion: await diffusionWeights(store),
    atomReference: await atomReference(store),
    targetFeat: await targetFeatureWeights(store),
    confidence: await confidenceWeights(store),
  };

  const foldStructure = makeFoldStructure(device, weights,
    { sequence, recycles: structRecycles, steps: structSteps, seed });
  // 🔴 THE TRUNK'S SEAMS, BECAUSE `z` IS 6.6x THE int5 FLOOR. Measured:
  // quantising upstream's own trunk to this bundle's 5-bit scheme moves its `z`
  // by 3.04e-2 where this port is 2.00e-1 away, and `s_inputs` is AT the floor
  // - so the disagreement is real and enters between the embedder's pair and
  // the module. The LAST write of each name wins, which is the last recycle:
  // upstream's hooks keep the same one.
  const seams = {};
  const foldAffinity = makeFoldAffinityPass(device, weights, {
    seed,
    onSeam: (name, value) => { seams[name] = value; },
  });

  const structure = await foldStructure({ smiles, id: "lig1" });
  const crop = affinityCrop(structure.cropTokens);
  const pass2 = await foldAffinity({ structure, crop,
    recyclingSteps: affinityRecycles, samplingSteps: affinitySteps,
    diffusionSamples: affinitySamples });

  // the sample the module is actually given: upstream takes the best ipTM
  let best = 0;
  for (let index = 1; index < pass2.samples.length; index += 1) {
    if ((pass2.samples[index].iptm ?? -Infinity)
        > (pass2.samples[best].iptm ?? -Infinity)) best = index;
  }
  const sample = pass2.samples[best];
  const n = pass2.feats.tokens;
  const mask = pass2.feats.isLigand ?? new Uint8Array(n);
  const molType = pass2.feats.molType ?? Int32Array.from(mask, (v) => (v ? 3 : 0));
  const padMask = new Uint8Array(n).fill(1);

  // 🔴 THE PRE-MASK IS PART OF `z`, NOT A DETAIL. upstream hands the module
  // `z * cross_pair_mask`, zeroing every pair that is not ligand-receptor or
  // ligand-ligand, and that is what must be compared - comparing the raw trunk
  // pair against upstream's masked one would report a difference in every
  // receptor-receptor entry that neither model reads.
  // 🔴 THE ARGUMENT ORDER IS (molType, affinityMask, padMask, n) - I wrote it
  // as (n, mask, molType, padMask) first, which is four arguments of the right
  // types in the wrong places: `n` read as molType made every token a
  // non-receptor, and the mask came back all zeros rather than throwing. The
  // `excludeDiagonal` default is false, which is the PAIRFORMER's mask and is
  // what boltz2.py applies to `z` - the heads use the other one.
  const crossMask = crossPairMaskF32(molType, mask, padMask, n, false);
  const zRaw = Float32Array.from(sample.z);
  const zAffinity = new Float32Array(zRaw.length);
  const channels = AFFINITY_TOKEN_Z;
  for (let pair = 0; pair < n * n; pair += 1) {
    const keep = crossMask[pair];
    for (let c = 0; c < channels; c += 1) {
      zAffinity[pair * channels + c] = zRaw[pair * channels + c] * keep;
    }
  }
  const sInputs = Float32Array.from(
    await affinitySInputs(device, pass2.feats, weights.targetFeat));

  // 🔴 THE ATOM-ENCODER TERM ON ITS OWN, because `s_inputs` is a SUM and it
  // disagrees with upstream at relRMS 9.0e-1. upstream's InputEmbedder is
  //
  //     s = a + res_type_encoding(..) + msa_profile_encoding(..) + 4 conditionings
  //
  // and `a` - the atom attention encoder's token activation - is the dominant
  // term. Running the same encoder `buildTargetFeat` runs, with the same
  // inputs, rather than reimplementing it: the point is to compare this port's
  // encoder with upstream's, not this tool's arithmetic with either.
  const batch = pass2.feats;
  const tf = weights.targetFeat;
  let namesApplied = null;
  if (overrideNames !== "") {
    const wanted = overrideNames.split(",");
    const slots = [];
    for (let token = 0; token < batch.tokens; token += 1) {
      if (!batch.isLigand?.[token]) continue;
      for (let slot = 0; slot < batch.dense; slot += 1) {
        const at = token * batch.dense + slot;
        if (batch.refMask[at] > 0) slots.push(at);
      }
    }
    if (slots.length !== wanted.length) {
      throw new Error(`--ligand-names has ${wanted.length} names for`
        + ` ${slots.length} live ligand atoms`);
    }
    // 🔴 THE SAME ENCODING THE FEATURISER USES: four characters, ASCII minus
    // 32, zero-padded. Writing raw ASCII here would shift every character's
    // one-hot by 32 columns and look like a different name again.
    slots.forEach((at, index) => {
      const name = wanted[index].padEnd(4, "\0");
      for (let c = 0; c < 4; c += 1) {
        const code = name.charCodeAt(c);
        batch.refAtomNameChars[at * 4 + c] = code === 0 ? 0 : code - 32;
      }
    });
    namesApplied = wanted;
  }
  const { profile, deletionMean } = affinityProfile(batch);
  const conditioning = perAtomConditioning({
    positions: batch.refPos, mask: batch.refMask,
    element: batch.refElement, charge: batch.refCharge,
    atomNameChars: batch.refAtomNameChars,
  }, batch.tokens, batch.dense, tf.reference, tf.dialect);
  const atoms = batch.tokens * batch.dense;
  const encoded = await new Af3AtomEncoderGpu(device).run({
    shape: batch.shape, dialect: tf.dialect,
    conditioning, atomMask: batch.refMask,
    refPos: batch.refPos, refSpaceUid: batch.refSpaceUid,
    tokenAtomsToQueries: batch.tokenAtomsToQueries,
    queriesToKeys: batch.queriesToKeys,
    queriesToTokenAtoms: batch.queriesToTokenAtoms,
    tokensToQueries: batch.tokensToQueries,
    tokensToKeys: batch.tokensToKeys,
    tokenAtomsAct: new Float32Array(atoms * 3),
    trunkSingleCond: null, trunkPairCond: null,
  }, tf.encoder, { hostStatics: false });
  const atomTerm = Float32Array.from(
    encoded.tokenAct.subarray(0, batch.tokens * AFFINITY_TOKEN_S));
  // everything s_inputs has that the atom term does not: the restype one-hot,
  // the profile, the deletion mean and boltz2's four conditionings, summed.
  const restTerms = new Float32Array(atomTerm.length);
  for (let i = 0; i < atomTerm.length; i += 1) {
    restTerms[i] = sInputs[i] - atomTerm[i];
  }

  const tensors = {
    s_inputs: { shape: [n, AFFINITY_TOKEN_S], values: sInputs },
    a_atom_encoder: { shape: [n, AFFINITY_TOKEN_S], values: atomTerm },
    s_minus_a: { shape: [n, AFFINITY_TOKEN_S], values: restTerms },
    profile_affinity: { shape: [n, profile.length / n], values: profile },
    deletion_mean_affinity: { shape: [n], values: deletionMean },
    z: { shape: [n, n, channels], values: zAffinity },
    z_unmasked: { shape: [n, n, channels], values: zRaw },
    cross_pair_mask: { shape: [n, n], values: crossMask },
    positions: { shape: [sample.positions.length / 3, 3],
      values: Float32Array.from(sample.positions) },
    mol_type: { shape: [n], values: Float32Array.from(molType) },
    affinity_token_mask: { shape: [n], values: Float32Array.from(mask) },
  };

  const report = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    const expected = tensor.shape.reduce((a, b) => a * b, 1);
    report[name] = { shape: tensor.shape, ...summarise(tensor.values) };
    if (tensor.values.length !== expected) {
      report[name].SHAPE_MISMATCH =
        `${tensor.values.length} values for shape ${tensor.shape.join("x")}`;
    }
  }

  return {
    tokens: n,
    ligandNamesOverridden: namesApplied,
    croppedProteinTokens: crop.kept.filter((t) => !structure.batch.isLigand?.[t]).length,
    uncroppedTokens: structure.batch.tokens,
    // 🔴 THE CONDITION THE WHOLE COMPARISON RESTS ON, stated as a number.
    cropIsIdentity: crop.kept.length === structure.batch.tokens,
    bestSample: best,
    iptm: pass2.samples.map((s) => s.iptm),
    settings: { seed, structRecycles, structSteps, affinityRecycles,
      affinitySteps, affinitySamples },
    tensors: report,
    // base64 so a JSON-only harness can read the floats back out
    data: Object.fromEntries(Object.entries(tensors).map(([name, tensor]) => [
      name, base64(tensor.values),
    ])),
    // the trunk's own seams, same encoding, with their lengths beside them so
    // a reader can shape them without knowing this pass's token count
    seamShapes: Object.fromEntries(Object.entries(seams)
      .map(([name, value]) => [name, value.length])),
    seams: Object.fromEntries(Object.entries(seams)
      .map(([name, value]) => [name, base64(value)])),
    // 🔴 THE TWO BOND PLANES AS NUMBERS, NOT FLOATS, because `z_init`'s
    // residual is 10.7x the int5 floor on exactly the 20 bonded pairs and
    // these are the only summands supported there. Emitted as the sparse list
    // of non-zero entries so a reader can diff them against upstream's
    // `token_bonds`/`type_bonds` pair by pair without shaping anything.
    bondPlanes: (() => {
      const plane = (name) => {
        const values = batch[name];
        if (values === undefined) return null;
        const out = [];
        for (let i = 0; i < n; i += 1) {
          for (let k = 0; k < n; k += 1) {
            const v = values[i * n + k];
            if (v) out.push([i, k, v]);
          }
        }
        return out;
      };
      return {
        present: ["bondMatrix", "bondOrderMatrix"]
          .filter((name) => batch[name] !== undefined),
        bondMatrix: plane("bondMatrix"),
        bondOrderMatrix: plane("bondOrderMatrix"),
      };
    })(),
  };
}
