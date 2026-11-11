/**
 * How big a target this machine can co-fold, and what to do when it is bigger.
 *
 * 🔴 THE PAGE HAD NO UPPER LIMIT AT ALL. `sequence.length < 4` was the only
 * check, so a reader pasted a 2,244-residue target, pressed Run, and the screen
 * failed somewhere in the middle - after the weights had downloaded and the
 * first ligand had started. Nothing told them the length was the problem, and
 * nothing could, because nothing in the page knew what a length costs.
 *
 * 🔴 AND THE CEILING IS THE DEVICE'S, NOT A NUMBER WE PICKED. The pair
 * representation is `tokens^2 x 128 channels x 4 bytes`, and a WebGPU binding
 * cannot exceed `maxStorageBufferBindingSize` - so the hard ceiling is
 * `sqrt(limit / (128 * 4))` residues and it moves with the hardware. Measured
 * through Chrome 157 on an M3 Pro, the adapter reports 4 GiB - 4 and the
 * ceiling is 2,896; where a driver reports 2 GiB it is 2,047, and a
 * 2,000-residue target is then on the edge of refusal rather than inside it.
 * A hardcoded number would be wrong on both machines.
 *
 * 🔴 AND THE WARNING THRESHOLD IS "PAST WHAT WAS MEASURED", NOT A CLIFF SOMEONE
 * IMAGINED. docs/MEASURED.md times one ligand at 99, 198 and 297 residues
 * (7.4 s, 20.9 s, 32.2 s on an M3 Pro at 1 recycle / 12 steps). 297 is the
 * largest length anyone here has timed, so `MEASURED_TO` is 297 and the page
 * says "beyond what has been measured" rather than inventing a performance
 * promise. The other number in the advice is exact arithmetic: several
 * pair-shaped tensors have to be live at once for the triangle operations, so
 * the usable ceiling is well below the binding one and depends on free memory.
 *
 * 🔴 AND THE ADVICE IS THE SAME ADVICE WHATEVER THE MACHINE: crop to the domain
 * that holds the site. That is not a workaround. `shared/af3/affinity/crop.js`
 * carries upstream's own `AffinityCropper` defaults - at most 256 tokens, at
 * most 200 of them protein, within 10 A of the ligand - so the affinity head
 * never reads a large target anyway. A full-length target changes the structure
 * the score is read off, and nothing else.
 *
 * Pure, so test/target-size.test.js can check the thresholds and the arithmetic
 * for devices this machine does not have.
 */

/** Channels in the pair representation, and bytes per f32 element. */
export const PAIR_CHANNELS = 128;
const BYTES = 4;

/** The largest target anyone has timed here. docs/MEASURED.md has the table. */
export const MEASURED_TO = 297;

/** `AffinityCropper`'s protein budget - shared/af3/affinity/crop.js. */
export const AFFINITY_PROTEIN_TOKENS = 200;

/**
 * The conservative binding limit, for before a device has been asked.
 *
 * 🔴 2 GiB AND NOT 4, because the page draws this note while the weights are
 * still downloading and must not promise a ceiling the visitor's hardware does
 * not have. 2 GiB is what a driver that reports less than this machine commonly
 * reports; the number sharpens once `getDevice` has answered.
 */
export const ASSUMED_BINDING_BYTES = 2 * 1024 ** 3;

/** Bytes in one f32 pair tensor for a target of `residues` tokens. */
export function pairBytes(residues) {
  return residues * residues * PAIR_CHANNELS * BYTES;
}

/** The largest target one pair binding can hold on a device. */
export function residueCeiling(bindingBytes) {
  return Math.floor(Math.sqrt(bindingBytes / (PAIR_CHANNELS * BYTES)));
}

const gib = (bytes) => `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
const mib = (bytes) => `${Math.round(bytes / 1024 ** 2)} MiB`;
const size = (bytes) => (bytes >= 1024 ** 3 ? gib(bytes) : mib(bytes));

/** The sentence about cropping, which is the same on every path that needs it. */
export const CROP_ADVICE =
  "Crop to the domain that contains the binding site. That costs you nothing in"
  + ` the score: Boltz-2's affinity module reads at most ${AFFINITY_PROTEIN_TOKENS}`
  + " protein residues within 10 Å of the ligand, so a longer target only"
  + " changes the structure the score is read off. Keep the whole domain rather"
  + " than a shell - the crop still has to fold on its own - and keep both"
  + " domains when the site sits between them.";

/**
 * What to say about a target of this length on this device.
 *
 * @param {number} residues
 * @param {{bindingBytes?: number, measured?: boolean}} [device]
 *   `bindingBytes` is the device's `maxStorageBufferBindingSize`; `measured`
 *   says whether it came from a real adapter or is the conservative assumption
 * @returns {{level: "empty"|"ok"|"note"|"warn"|"refuse", residues: number,
 *            ceiling: number, pair: number, note: string, message: string}}
 *   `note` is one line for the sequence box; `message` is the paragraph a
 *   refusal needs. `level` "refuse" means the run must not start.
 */
export function targetSizeAdvice(residues, device = {}) {
  const bindingBytes = device.bindingBytes ?? ASSUMED_BINDING_BYTES;
  const ceiling = residueCeiling(bindingBytes);
  const pair = pairBytes(residues);
  const common = { residues, ceiling, pair };

  if (residues === 0) {
    return { ...common, level: "empty", note: "one protein chain, amino acids",
      message: "" };
  }

  if (residues > ceiling) {
    const message = `${residues} residues is past what this GPU can co-fold.`
      + ` One pair tensor alone would be ${size(pair)}, and a WebGPU binding on`
      + ` this device stops at ${size(bindingBytes)} - a hard ceiling of`
      + ` ${ceiling} residues, which several tensors have to share. ${CROP_ADVICE}`;
    return { ...common, level: "refuse",
      note: `${residues} residues — too long for this GPU (ceiling`
        + ` ${ceiling}); crop to the domain around the site`,
      message };
  }

  if (residues > MEASURED_TO) {
    // 🔴 NOT A TIME ESTIMATE. The measured curve runs 99-297 residues and
    // nothing here has any business extrapolating it four times past its last
    // point; what IS exact is the pair tensor's size and the fact that several
    // are live at once.
    const message = `${residues} residues is beyond what has been measured here`
      + ` (${MEASURED_TO} is the largest timed: 32 s a ligand on an M3 Pro at 1`
      + " recycle and 12 steps). One pair tensor is"
      + ` ${size(pair)} and the trunk holds several at once, so expect minutes`
      + ` a ligand and a real chance of running out of memory before the`
      + ` ${ceiling}-residue ceiling. ${CROP_ADVICE}`;
    return { ...common, level: "warn",
      note: `${residues} residues — past the ${MEASURED_TO} measured here;`
        + " expect minutes a ligand, and consider cropping to the domain",
      message };
  }

  if (residues > AFFINITY_PROTEIN_TOKENS) {
    return { ...common, level: "note",
      note: `${residues} residues — the affinity head reads at most`
        + ` ${AFFINITY_PROTEIN_TOKENS} of them, round the ligand`,
      message: CROP_ADVICE };
  }

  return { ...common, level: "ok", note: `${residues} residues`, message: "" };
}
