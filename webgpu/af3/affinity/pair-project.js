/**
 * LayerNorm over a pair row, then one bias-free projection - on the device.
 *
 * 🔴 THIS EXISTS BECAUSE THE MEASUREMENT SAID SO, NOT BECAUSE IT LOOKED SLOW.
 * The affinity module's non-pairformer work was kept on the host and justified
 * with an O(tokens^2) against O(tokens^3) argument. Profiled at 256 tokens -
 * the crop's own ceiling - that argument was backwards: the host glue was
 * 16.5 s against the pairformer's 1.8. Moving the conditioner's two transitions
 * onto `Af3TransitionGpu` took it to 6.9 s, and what is left is two plain pair
 * projections:
 *
 *     conditionerInit  LayerNorm(256) + LinearNoBias(256 -> 128)   3718 ms
 *     z_linear         LayerNorm(128) + LinearNoBias(128 -> 128)    954 ms
 *
 * The O() reasoning ignored the channel constant: one pair projection is
 * `tokens^2 * C_in * C_out` multiply-adds, which at 256 tokens and C = 128 is
 * 1.07e9 - and 2.1e9 for the 256-wide one. Four point seven seconds a member,
 * about ten a ligand across the two, on the critical path of every ligand in a
 * screen.
 *
 * 🔴 THE NORM IS FUSED WITH THE PROJECTION BECAUSE BOTH CALLERS WANT IT THAT
 * WAY, and because splitting them would write the normalised pair back to
 * global memory and read it again - `tokens^2 * C_in` floats of traffic for
 * nothing. `z_linear` is `z_linear(z_norm(z))` and the conditioner's init is
 * `LinearNoBias(LayerNorm(cat(z, distogram)))`; neither has an intermediate
 * anyone reads.
 *
 * 🔴 AND THERE IS NO BIAS, BECAUSE BOLTZ HAS NONE HERE. Both are `LinearNoBias`
 * upstream. A bias slot left for symmetry would be a weight nobody writes, and
 * `weight-sources.js` records what an unwritten weight reaching a consumer
 * costs: present in the source, present in the offsets, absent from the buffer,
 * and every oracle seam matching the previous run to the last digit.
 *
 * The LayerNorm is torch's: mean and BIASED variance over the row, `eps` added
 * inside the square root's argument, then scale and offset.
 */
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { pipelineCacheForDevice } from "../../runtime/pipeline-cache.js";

/** WebGPU's own ceiling on one dispatch dimension. */
const GRID_WIDTH = 65535;
/** One workgroup a pair row; 64 lanes cooperate on the row's channels. */
const LANES = 64;

/**
 * 🔴 THE ROW IS STAGED IN WORKGROUP MEMORY ONCE. Every output channel reads
 * every input channel, so without staging each of the C_out lanes would re-read
 * all C_in values from global memory - C_in * C_out loads a row where staging
 * makes it C_in. The bound is the 256-channel caller, which is 1 KiB of f32.
 */
const MAX_STAGED_CHANNELS = 512;

export function createPairProjectShader(inChannels, outChannels, epsilon) {
  if (inChannels > MAX_STAGED_CHANNELS) {
    throw new Error(`pair-project: ${inChannels} input channels exceeds the`
      + ` ${MAX_STAGED_CHANNELS} this kernel stages in workgroup memory`);
  }
  return `
const IN: u32 = ${inChannels}u;
const OUT: u32 = ${outChannels}u;
const LANES: u32 = ${LANES}u;
const EPS: f32 = ${epsilon};
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;

@group(0) @binding(0) var<storage, read> pairIn: array<f32>;
// scale[IN], offset[IN], then the projection as (out, in) - torch's layout, so
// a row of the weight is contiguous and lane-friendly.
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> pairOut: array<f32>;

var<workgroup> staged: array<f32, ${inChannels}u>;
var<workgroup> partial: array<f32, ${LANES}u>;

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
  let row: u32 = group.x + group.y * GRID_WIDTH;
  let lane: u32 = local.x;
  let base: u32 = row * IN;

  // --- the row's mean --------------------------------------------------------
  var sum: f32 = 0.0;
  var c: u32 = lane;
  loop {
    if (c >= IN) { break; }
    let v = pairIn[base + c];
    staged[c] = v;
    sum = sum + v;
    c = c + LANES;
  }
  partial[lane] = sum;
  workgroupBarrier();
  // 🔴 A FIXED-TRIP REDUCTION, NOT A LOOP OVER A WORKGROUP VALUE. WGSL's
  // uniformity analysis refuses a barrier inside control flow that depends on
  // a value read back out of workgroup memory, which is what
  // docs/ records costing the SMILES conformer kernel its early exit.
  var stride: u32 = LANES >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mean: f32 = partial[0] / f32(IN);
  workgroupBarrier();

  // --- and its BIASED variance, as torch computes it -------------------------
  var squares: f32 = 0.0;
  c = lane;
  loop {
    if (c >= IN) { break; }
    let d = staged[c] - mean;
    squares = squares + d * d;
    c = c + LANES;
  }
  partial[lane] = squares;
  workgroupBarrier();
  stride = LANES >> 1u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inverse: f32 = 1.0 / sqrt(partial[0] / f32(IN) + EPS);
  workgroupBarrier();

  // --- normalise in place, then project -------------------------------------
  c = lane;
  loop {
    if (c >= IN) { break; }
    staged[c] = (staged[c] - mean) * inverse * weights[c] + weights[IN + c];
    c = c + LANES;
  }
  workgroupBarrier();

  let weightBase: u32 = IN * 2u;
  var o: u32 = lane;
  loop {
    if (o >= OUT) { break; }
    var total: f32 = 0.0;
    let wrow: u32 = weightBase + o * IN;
    for (var k: u32 = 0u; k < IN; k = k + 1u) {
      total = total + staged[k] * weights[wrow + k];
    }
    pairOut[row * OUT + o] = total;
    o = o + LANES;
  }
}
`;
}

export class AffinityPairProjectGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {Float32Array} input rows * inChannels
   * @param {{rows: number, inChannels: number, outChannels: number}} shape
   * @param {{normWeight: ArrayLike<number>, normBias: ArrayLike<number>,
   *          projection: ArrayLike<number>}} weights
   *   `projection` is (out, in) - torch's own layout, unchanged
   * @param {{epsilon?: number}} [options]
   */
  async run(input, shape, weights, options = {}) {
    const { rows, inChannels, outChannels } = shape;
    const epsilon = options.epsilon ?? 1e-5;
    if (input.length !== rows * inChannels) {
      throw new Error(`pair-project input has ${input.length} elements;`
        + ` expected ${rows * inChannels}`);
    }
    for (const [name, wanted] of [["normWeight", inChannels],
      ["normBias", inChannels], ["projection", inChannels * outChannels]]) {
      const value = weights[name];
      // 🔴 null AND undefined: the loader writes null, and that null reaching
      // a consumer is what weight-sources.js records killing boltz2.
      if (value === null || value === undefined) {
        throw new Error(`pair-project: ${name} is absent`);
      }
      if (value.length !== wanted) {
        throw new Error(`pair-project: ${name} has ${value.length} elements;`
          + ` expected ${wanted}`);
      }
    }

    const packed = new Float32Array(inChannels * 2 + inChannels * outChannels);
    packed.set(weights.normWeight, 0);
    packed.set(weights.normBias, inChannels);
    packed.set(weights.projection, inChannels * 2);

    const source = createPairProjectShader(inChannels, outChannels, epsilon);
    // 🔴 THE KEY NAMES EVERY CONSTANT THE SHADER EMBEDS. CLAUDE.md records two
    // collisions from keys that did not - and the cache indexes by source as
    // well, so a miss here is refused and named rather than served.
    const pipeline = await this.pipelines.get(
      `boltz2-affinity-pair-project:${inChannels}:${outChannels}:${epsilon}:${LANES}`,
      source);

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const bytes = rows * outChannels * 4;
      const inputBuffer = keep(this.allocator.upload(
        "affinity-pair-project.input", input, storage));
      const weightBuffer = keep(this.allocator.upload(
        "affinity-pair-project.weights", packed, storage));
      const outputBuffer = keep(this.allocator.allocate(
        "affinity-pair-project.output", bytes, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "affinity-pair-project.readback", bytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder(
        { label: "affinity-pair-project" });
      const pass = encoder.beginComputePass({ label: "affinity-pair-project" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [inputBuffer, weightBuffer, outputBuffer].map((a, binding) => ({
          binding, resource: { buffer: a.buffer },
        })),
      }));
      // one workgroup a row, folded into two dimensions past WebGPU's ceiling
      pass.dispatchWorkgroups(Math.min(rows, GRID_WIDTH),
                              Math.ceil(rows / GRID_WIDTH));
      pass.end();
      encoder.copyBufferToBuffer(outputBuffer.buffer, 0, readback.buffer, 0, bytes);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) {
        throw new Error(`WebGPU validation failed: ${error.message}`);
      }
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot() };
    } finally {
      for (let i = allocations.length - 1; i >= 0; i -= 1) allocations[i].release();
    }
  }
}
