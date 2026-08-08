//go:build js

package webgpu

// Q4KMatvecWGSL is a from-scratch WGSL reimplementation of this project's
// own scalar Q4_K matvec kernel (simd.go's DequantRowQ4KInto/
// getScaleMinK4), authored directly from the GGUF Q4_K block layout (a
// public file-format fact, not copyrightable expression): 144 bytes/256
// elements = f16 d (2B) + f16 dmin (2B) + 12B of packed 6-bit scale/min
// pairs for 8 sub-blocks of 32 + 128B of nibble-packed 4-bit quants (low
// nibble = even sub-block, high nibble = odd sub-block within each 64-wide
// byte-pair span). Dequant per element: value = d*scale - dmin*min, using
// that element's sub-block's 6-bit scale/min (getScaleMinK4's packing
// trick fits 8x6-bit scales + 8x6-bit mins into 12 bytes).
//
// One workgroup computes one output row: each of 64 threads accumulates a
// partial dot product over the row's blocks (blockIdx = localId, localId+64,
// ...), then a standard workgroup-shared-memory tree reduction sums the 64
// partials into the final value, written by thread 0. Bindings: 0=weights
// (this dispatch's row-chunk's raw Q4_K bytes, array<u32>), 1=x (activation
// vector, length cols), 2=out (output vector, write out[rowOffset+row]),
// 3=params (array<u32>: [cols, rowOffset]).
const Q4KMatvecWGSL = wgslCommon + `
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

var<workgroup> partial: array<f32, 64>;

const Q4K_BLOCK_BYTES: u32 = 144u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  let localId = lid.x;
  let cols = params[0];
  let rowOffset = params[1];
  let numBlocks = cols / 256u;
  let rowByteOffset = row * numBlocks * Q4K_BLOCK_BYTES;

  var acc: f32 = 0.0;
  var blockIdx = localId;
  loop {
    if (blockIdx >= numBlocks) { break; }
    let base = rowByteOffset + blockIdx * Q4K_BLOCK_BYTES;
    let d = f16_to_f32(read_u16(base));
    let dmin = f16_to_f32(read_u16(base + 2u));
    let scalesBase = base + 4u;
    let qBase = base + 16u;
    let yoff = blockIdx * 256u;

    var jj: u32 = 0u;
    loop {
      if (jj >= 4u) { break; }
      let is0 = jj * 2u;
      let sm1 = get_scale_min_k4(is0, scalesBase);
      let sm2 = get_scale_min_k4(is0 + 1u, scalesBase);
      let d1 = d * f32(sm1.x);
      let d2 = d * f32(sm2.x);
      let min1 = dmin * f32(sm1.y);
      let min2 = dmin * f32(sm2.y);
      let qOff = qBase + jj * 32u;
      let xOff = yoff + jj * 64u;

      var l: u32 = 0u;
      loop {
        if (l >= 32u) { break; }
        let qByte = read_u8(qOff + l);
        let lo = qByte & 0x0fu;
        let hi = qByte >> 4u;
        let vLo = d1 * f32(lo) - min1;
        let vHi = d2 * f32(hi) - min2;
        acc = acc + vLo * x[xOff + l];
        acc = acc + vHi * x[xOff + 32u + l];
        l = l + 1u;
      }
      jj = jj + 1u;
    }
    blockIdx = blockIdx + 64u;
  }

  partial[localId] = acc;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (localId < stride) {
      partial[localId] = partial[localId] + partial[localId + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (localId == 0u) {
    out[rowOffset + row] = partial[0];
  }
}
`
