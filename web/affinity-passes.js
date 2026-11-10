/**
 * The two folds the screening loop injects, wired to this port's own stacks.
 *
 * 🔴 `targetFeat` IS BOLTZ'S `s_inputs`, AND THAT IS WHAT MAKES THE
 * `affinity=True` SWITCH ONE LINE RATHER THAN A SECOND EMBEDDER. Read off the
 * code rather than by analogy: `buildTargetFeat` calls
 *
 *     targetFeatures({ aatype, profile, deletionMean, atomFeatures,
 *                      sum: weights.encoder.targetFeatSum, ... })
 *
 * at **384 channels**, which is Boltz's `token_s`, summing an atom-encoder
 * output, a restype one-hot, the MSA profile and the deletion mean - plus, for
 * boltz2 alone, the six extra summands `targetFeatSum` carries, which are the
 * method, modified, cyclic and mol-type conditionings its InputEmbedder adds.
 * That is term for term what `InputEmbedder.forward` builds.
 *
 * upstream's `affinity=True` changes exactly two of those inputs - `profile`
 * and `deletion_mean` become their depth-1 variants - so the affinity
 * `s_inputs` is `buildTargetFeat` over a batch with
 * shared/af3/affinity/features.js's profile substituted, and nothing else moves.
 *
 * 🔴 WHAT THAT CHAIN DOES AND DOES NOT ESTABLISH. docs/ records `target_feat`
 * at 1e-8 against the reference's own batch across every AF3-lineage model, so
 * the MACHINERY is verified and the substitution is a documented input changed
 * in a documented way, gated by tools/gpu/check-affinity-sinputs.js. What is
 * NOT established is that upstream's `input_embedder(feats, affinity=True)`
 * output equals ours elementwise - that needs an oracle dump of upstream's
 * embedder on a fully featurised input, which means standing up its whole
 * featuriser (Tokenized + MSA + Structure), and it is named as open rather
 * than assumed.
 *
 * 🔴 AND PASS 2 RE-FEATURISES; IT DOES NOT REUSE PASS 1'S BATCH. The crop
 * renumbers tokens, so the mask, the profile and every gather are rebuilt
 * against the cropped order. `reusable` from pass 1 is deliberately NOT passed
 * to pass 2: it skips the trunk for an IDENTICAL batch, and pass 2's is a
 * different one - a pocket at five recycles rather than a whole target at
 * three.
 */
import { foldBatch, buildTargetFeat } from "../webgpu/af3/fold.js";
import { af3BatchFromA3m } from "../shared/af3/featurise/batch.js";
import { dialectFor, featuriserDialect } from "../shared/af3/dialect.js";
import { affinityProfile, affinityTokenMask }
  from "../shared/af3/affinity/features.js";
import { centreCoords } from "../shared/af3/affinity/centre-atoms.js";
import { featuriseCrop } from "../shared/af3/affinity/featurise-crop.js";
import { affinityForwardGpu, crossPairMaskF32 }
  from "../webgpu/af3/affinity/affinity.js";
import { smilesComponent, ligandName } from "../shared/chem/component.js";
import { AffinityPairStackGpu } from "../webgpu/af3/affinity/pair-stack.js";
import { affinityWeights, AFFINITY_MEMBERS, AFFINITY_TOKEN_S,
  AFFINITY_TOKEN_Z } from "../shared/af3/weights/affinity-weights.js";
import { AFFINITY_PASS } from "./affinity-run.js";

/**
 * `s_inputs` for the affinity module: the structure pass's own machinery over
 * the depth-1 profile.
 *
 * Exported on its own so tools/gpu/check-affinity-sinputs.js can hold it to
 * the structure pass's and show the substitution reaching the output.
 */
export async function affinitySInputs(device, batch, targetFeat) {
  // 🔴 THE ARGUMENT IS `weights.targetFeat`, NOT `weights`. `buildTargetFeat`
  // reads `.reference`, `.encoder` and `.dialect` off what it is handed, and
  // the whole bundle carries none of those at its top level - the first thing
  // that touches it is `weights.channels` inside `perAtomConditioning`, which
  // dies as "cannot read properties of undefined" naming neither the argument
  // nor this function. tools/gpu/check-affinity-sinputs.js passes
  // `targetFeatureWeights(store)` directly, which is the same object.
  if (targetFeat?.reference === undefined) {
    throw new Error("affinitySInputs wants the targetFeat weight set"
      + " (weights.targetFeat), not the whole bundle");
  }
  const { profile, deletionMean } = affinityProfile(batch);
  // A shallow copy: every other field is shared, which is the point - only the
  // two upstream names change.
  return buildTargetFeat({ ...batch, profile, deletionMean }, targetFeat, device);
}

/** Pass 1: the ordinary fold of protein + ligand. */
export function makeFoldStructure(device, weights, options = {}) {
  return async function foldStructure({ smiles, id, signal }) {
    const dialect = featuriserDialect(weights.trunk.dialect);
    // 🔴 A LIGAND REACHES THE FEATURISER AS A COMPONENT, NOT A SMILES. It is
    // the one object `parseCcdComponent` and `smilesComponent` both produce -
    // `{code, atoms, bonds}` - and shared/chem/component.js's header calls
    // that "the whole integration... everything downstream cannot tell which
    // one it got". Handing it `{code, smiles}` dies inside the featuriser's
    // token count, where the error names neither.
    // 🔴 AND THE NAMING RULE IS THE DIALECT'S, NOT THIS CALLER'S. Boltz-2
    // names a SMILES ligand's atoms from a canonical rank where AF3 numbers
    // each element as it is met, and the names ARE a model input
    // (`ref_atom_name_chars`). Read from the flag rather than hard-coded here
    // even though this path only ever runs boltz2, because the convention
    // belongs with the other eight in dialect.js and a second copy of it is
    // this repository's standing complaint.
    const component = await smilesComponent(smiles, {
      code: ligandName(0),
      ...(weights.trunk.dialect?.smilesCanonicalAtomNames
        ? { atomNames: "canonical" } : {}),
    });
    const { batch } = af3BatchFromA3m(options.sequence, options.alignment ?? null, {
      ...dialect,
      ligands: [component],
      signal,
    });
    const fold = await foldBatch(device, batch, weights, {
      recycles: options.recycles ?? 3,
      steps: options.steps ?? 25,
      seed: options.seed ?? 0,
      signal,
    });
    return {
      id, batch, pdb: fold.pdb, positions: fold.positions,
      meanPlddt: fold.meanPlddt, ptm: fold.ptm, iptm: fold.iptm,
      geometry: fold.geometry,
      // what the crop ranks on: one CENTRE atom a token, which is a DIFFERENT
      // gather from the pseudo-beta the affinity module's distances use.
      cropTokens: cropTokensFrom(batch, fold),
    };
  };
}

/**
 * The crop's view of pass 1's prediction.
 *
 * 🔴 CENTRE ATOMS, NOT PSEUDO-BETA, AND THE DIFFERENCE IS NOT A CORNER CASE.
 * upstream ranks the crop on `center_coords` (`center_idx`) and the affinity
 * module's own distances use `disto_idx`. Counted off its tables, the two
 * differ on **28 of 31 residue types** and agree only on GLY, N and DN - CA
 * against CB for every other amino acid. An earlier version of this function
 * had no centre gather to reach for and ranked on pseudo-beta, with a comment
 * saying so; `shared/af3/affinity/centre-atoms.js` is the fix, and
 * test/affinity-centre-atoms.test.js asserts the difference rather than just
 * the value.
 */
export function cropTokensFrom(batch, fold) {
  const tokens = batch.tokens;
  const centre = centreCoords(batch, fold.positions);
  return {
    centreCoords: centre.coords,
    // how the centres were resolved, so a caller can assert the split rather
    // than trust it: a protein target should be all `fromName`, and every
    // fallback should be a ligand or atomised token
    centreFromName: centre.fromName,
    centreFromPseudoBeta: centre.fromPseudoBeta,
    asymId: batch.asymId,
    resIdx: batch.residueOfToken ?? Int32Array.from(batch.asymId, (_, i) => i),
    molType: batch.molType ?? Int32Array.from(batch.isLigand ?? [], (v) => (v ? 3 : 0)),
    atomNum: batch.atomsPerToken
      ?? Int32Array.from({ length: tokens }, () => 1),
    affinityMask: batch.isLigand ?? new Uint8Array(tokens),
    resolvedMask: new Uint8Array(tokens).fill(1),
  };
}

/**
 * Pass 2: the pocket, at upstream's own settings, plus the affinity `s_inputs`.
 *
 * 🔴 FIVE RECYCLES, 200 STEPS, FIVE SAMPLES - and those are not ours to
 * default. `predict_affinity_args` in upstream's main.py sets them, and the
 * confidence head runs because its ipTM is what picks the sample the affinity
 * module is given.
 */
export function makeFoldAffinityPass(device, weights, options = {}) {
  return async function foldAffinityPass({ structure, crop, signal,
    recyclingSteps, samplingSteps, diffusionSamples }) {
    const dialect = featuriserDialect(weights.trunk.dialect);
    // 🔴 THE TOKEN TABLE IS CROPPED, NOT A SEQUENCE RE-FEATURISED. A crop keeps
    // a residue neighbourhood per chain, so what it leaves is generally not
    // contiguous and joining it into a string would imply a peptide bond
    // across every gap. `featuriseCrop` subsets the batch and re-derives the
    // gathers through `atomGathers`, and `npm run test:boltz2-featurise` holds
    // the result to upstream's own featuriser on two cropped cases - gaps in
    // `residue_index` included.
    const croppedBatch = featuriseCrop(structure.batch, crop.kept, dialect);
    const cropped = { batch: croppedBatch,
      // the binder's asym id survives the crop unchanged; the MASK is what
      // moves, and affinityTokenMask rebuilds it against the cropped order
      binderChain: options.binderChain
        ?? binderChainOf(structure.batch, crop.kept) };
    const samples = [];
    // 🔴 ONE TRUNK FOR THE WHOLE PASS, N DIFFUSION TRAJECTORIES. upstream runs
    // the trunk once and draws its samples from that single conditioning; this
    // called `foldBatch` once per sample on an IDENTICAL batch and threaded
    // nothing between them, so at upstream's 5 recycles / 5 samples the pass-2
    // trunk ran 25 times instead of 5 - and retained five identical copies of a
    // 33 MB pair tensor. `reusable` is foldBatch's own mechanism for exactly
    // this: it skips the trunk when the batch is the same object, which here it
    // is by construction.
    //
    // 🔴 AND IT IS ONLY SOUND BECAUSE THIS TRUNK IS DETERMINISTIC. boltz2's
    // dialect has `recycleMcDropout: null` (the 0.4 in dialect.js is
    // protenix2's), so re-running it would give the same pair every time. A
    // dialect WITH recycle dropout must not share one trunk across samples,
    // which is why this reads the flag rather than assuming.
    const dialectFlags = weights.trunk.dialect ?? {};
    const shareTrunk = (dialectFlags.recycleMcDropout ?? null) === null;
    let reusable;
    for (let sample = 0; sample < (diffusionSamples ?? AFFINITY_PASS.diffusionSamples);
      sample += 1) {
      const fold = await foldBatch(device, cropped.batch, weights, {
        recycles: recyclingSteps ?? AFFINITY_PASS.recyclingSteps,
        steps: samplingSteps ?? AFFINITY_PASS.samplingSteps,
        seed: (options.seed ?? 0) + sample,
        signal,
        // 🔴 NO TEMPLATE STAGE IN THIS PASS, BECAUSE UPSTREAM'S AFFINITY
        // CHECKPOINT HAS NO TEMPLATE MODULE. `boltz2_aff.ckpt` carries every
        // other trunk tensor byte-identically to `boltz2_conf.ckpt` (relRMS
        // exactly 0 over 5019 shared tensors) and no `template_module` at all,
        // so `use_templates` is false for the whole affinity run and the
        // `z = z + template_module(z, ...)` line never executes. Running it
        // here with four empty slots adds a term upstream does not have, and
        // trunk.js's own note says an empty stage is "still a quarter of what
        // enters the MSA stack" - not a no-op.
        noTemplate: options.noTemplate ?? true,
        // 🔴 FORWARDED SO THE TRUNK'S SEAMS CAN BE COMPARED TO UPSTREAM'S
        // STAGES. `readSeam` is gated on this being defined, so without it the
        // trunk reads nothing back and costs nothing - and with it this pass's
        // `z_init`/`z_after_msa`/`trunk_out_pair` line up with
        // `dump_boltz2_affinity_inputs.py`'s `z-stages.npz`.
        ...(options.onSeam === undefined ? {} : { onSeam: options.onSeam }),
        // ...and NOT pass 1's `reusable`: a different batch, see the header.
        // 🔴 THE KEY IS `reuse`, NOT `reusable`. foldBatch RETURNS `reusable`
        // and ACCEPTS `reuse` (fold.js:680 reads `options.reuse?.trunk`), so
        // passing it back under the name it arrived with is a silent no-op -
        // the trunk re-runs and nothing complains. Written that way first and
        // caught by grepping for the consumer rather than the producer.
        ...(shareTrunk && reusable !== undefined ? { reuse: reusable } : {}),
      });
      if (shareTrunk) reusable = fold.reusable ?? reusable;
      samples.push({ iptm: fold.iptm, positions: fold.positions,
        pseudoBeta: fold.pseudoBeta, pdb: fold.pdb,
        // the pair the affinity module reads, from THIS pass's trunk
        z: fold.trunk?.pair });
    }
    const sInputs = await affinitySInputs(device, cropped.batch, weights.targetFeat);
    // 🔴 THE MASK TRAVELS **ON** THE FEATS, because that is what the module
    // reads. Returned beside them it was computed and thrown away.
    cropped.batch.affinityMask = affinityTokenMask(cropped.batch.asymId,
                                                   cropped.binderChain);
    return {
      samples, sInputs, feats: cropped.batch,
      affinityMask: cropped.batch.affinityMask,
      z: samples[0].z,
    };
  };
}

/**
 * Which asym id the binder is, read off the batch rather than assumed.
 *
 * 🔴 NOT "THE LAST CHAIN" OR "CHAIN B". upstream names it in the job
 * (`properties.affinity.binder`), and this derives it from the kept tokens'
 * own ligand flags so a job with the chains in another order cannot silently
 * score the protein. More than one candidate is a refusal: Boltz-2 predicts
 * affinity for ONE ligand at a time.
 */
export function binderChainOf(batch, kept) {
  const chains = new Set();
  for (const token of kept) {
    if (batch.isLigand?.[token]) chains.add(batch.asymId[token]);
  }
  if (chains.size === 0) {
    throw new Error("no ligand token survived the crop: there is no binder to"
      + " predict an affinity for");
  }
  if (chains.size > 1) {
    throw new Error(`the crop kept ${chains.size} ligand chains`
      + ` (${[...chains].join(", ")}) and Boltz-2 scores ONE binder at a time;`
      + " name it explicitly with binderChain");
  }
  return [...chains][0];
}

/**
 * `z * cross_pair_mask`, as upstream hands it over.
 *
 * 🔴 THE PAIRFORMER'S MASK, NOT THE HEADS' - `excludeDiagonal` false. The two
 * differ on the diagonal and boltz2.py applies the pairformer's to `z`.
 */
function preMaskedPair(z, n, mask, molType, padMask) {
  const keep = crossPairMaskF32(molType, mask, padMask, n, false);
  const channels = AFFINITY_TOKEN_Z;
  const out = new Float32Array(n * n * channels);
  for (let pair = 0; pair < n * n; pair += 1) {
    const factor = keep[pair];
    if (factor === 0) continue;
    const base = pair * channels;
    for (let c = 0; c < channels; c += 1) out[base + c] = z[base + c] * factor;
  }
  return out;
}

/** One ensemble member on the device. */
export function makeAffinityModule(device, read, options = {}) {
  const stack = new AffinityPairStackGpu(device);
  const cache = new Map();
  return async function affinityModule({ member, sInputs, z, positions, feats }) {
    if (!cache.has(member)) {
      const entry = AFFINITY_MEMBERS.find((m) => m.scope === member);
      if (entry === undefined) throw new Error(`unknown affinity member ${member}`);
      cache.set(member, affinityWeights(read, entry.scope, entry.layers));
    }
    const n = feats.tokens;
    // 🔴 THE DERIVED MASK, NOT `isLigand`. `makeFoldAffinityPass` already
    // builds `affinityTokenMask(asymId, binderChain)` - the binder CHAIN's
    // tokens - and returned it for nothing while this recomputed a mask from
    // `isLigand`. They coincide for a single SMILES ligand chain and diverge
    // the moment a job has two ligand chains and one binder, which is exactly
    // the case `binderChainOf` exists to name. Using the derived one also
    // brings its empty-mask refusal into the path.
    const mask = feats.affinityMask ?? feats.isLigand ?? new Uint8Array(n);
    // 🔴 HOISTED, because the pre-mask needs them and an object literal cannot
    // read its own later properties. Written inline first, which threw
    // `molType is not defined` on the first real ligand - caught by running it
    // rather than by reading it.
    const molType = feats.molType ?? Int32Array.from(mask, (v) => (v ? 3 : 0));
    const padMask = new Uint8Array(n).fill(1);
    const out = await affinityForwardGpu(device, {
      tokens: n, tokenS: AFFINITY_TOKEN_S, tokenZ: AFFINITY_TOKEN_Z,
      sInputs: Float32Array.from(sInputs),
      // 🔴 THE PRE-MASK, WHICH THIS COMMENT CLAIMED AND THE LINE DID NOT DO.
      // upstream hands the module `z * cross_pair_mask` (boltz2.py:620),
      // zeroing every pair that is not ligand-receptor or ligand-ligand -
      // before the module's own LayerNorm and projections, which otherwise
      // normalise and mix receptor-receptor pairs that upstream has already
      // removed. tools/gpu/dump-affinity-inputs.js did mask it, so the dumps
      // and the page disagreed about the module's input.
      zAffinity: preMaskedPair(z, n, mask, molType, padMask),
      positions: Float32Array.from(positions),
      molType,
      affinityMask: mask,
      padMask,
    }, cache.get(member), { stack, precision: options.precision ?? "f32" });
    return { predValue: out.predValue, probabilityBinary: out.probabilityBinary,
      logitsBinary: out.logitsBinary, timings: out.timings };
  };
}
