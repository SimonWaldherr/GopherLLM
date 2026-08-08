//go:build js

package webgpu

// Q6KMatvecWGSL is a from-scratch WGSL reimplementation of this project's
// own scalar Q6_K matvec kernel (simd.go's DequantRowQ6KInto), authored
// directly from the GGUF Q6_K block layout: 210 bytes/256 elements = 128B
// low nibbles (ql) + 64B high 2-bit planes (qh) + 16B signed int8
// sub-block scales (sc) + f16 d (2B, at the very end of the block). 256
// values split into two 128-wide halves; within each half, for l in
// [0,32) four 6-bit quant values are reassembled per l from ql/qh (each qh
// byte supplies the top 2 bits for 4 different 6-bit values spread 32
// apart; each ql byte supplies two 4-bit low halves). Each reassembled
// 6-bit value (range [0,63]) has 32 subtracted (signed range [-32,31]),
// multiplied by a signed int8 scale for its sub-block, then by the shared
// f16 d: value = d * sc[k] * (q6bits - 32).
//
// Same one-workgroup-per-row, 64-thread partial-sum + workgroup tree
// reduction structure as Q4KMatvecWGSL. Bindings identical: 0=weights,
// 1=x, 2=out, 3=params ([cols, rowOffset]).
const Q6KMatvecWGSL = wgslCommon + `
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

var<workgroup> partial6: array<f32, 64>;

const Q6K_BLOCK_BYTES: u32 = 210u;

fn read_i8(byteOffset: u32) -> i32 {
  let v = read_u8(byteOffset);
  return select(i32(v), i32(v) - 256, v >= 128u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  let localId = lid.x;
  let cols = params[0];
  let rowOffset = params[1];
  let numBlocks = cols / 256u;
  let rowByteOffset = row * numBlocks * Q6K_BLOCK_BYTES;

  var acc: f32 = 0.0;
  var blockIdx = localId;
  loop {
    if (blockIdx >= numBlocks) { break; }
    let base = rowByteOffset + blockIdx * Q6K_BLOCK_BYTES;
    let d = f16_to_f32(read_u16(base + 208u));
    let yoff = blockIdx * 256u;

    var n: u32 = 0u;
    loop {
      if (n >= 2u) { break; }
      let qlBase = base + n * 64u;
      let qhBase = base + 128u + n * 32u;
      let scBase = base + 192u + n * 8u;
      let xOff = yoff + n * 128u;

      var l: u32 = 0u;
      loop {
        if (l >= 32u) { break; }
        let is0 = l / 16u;
        let qlL = read_u8(qlBase + l);
        let qlL32 = read_u8(qlBase + l + 32u);
        let qhL = read_u8(qhBase + l);

        let q1 = i32((qlL & 0x0fu) | ((qhL & 0x03u) << 4u)) - 32;
        let q2 = i32((qlL32 & 0x0fu) | (((qhL >> 2u) & 0x03u) << 4u)) - 32;
        let q3 = i32((qlL >> 4u) | (((qhL >> 4u) & 0x03u) << 4u)) - 32;
        let q4 = i32((qlL32 >> 4u) | (((qhL >> 6u) & 0x03u) << 4u)) - 32;

        let sc0 = f32(read_i8(scBase + is0));
        let sc2 = f32(read_i8(scBase + is0 + 2u));
        let sc4 = f32(read_i8(scBase + is0 + 4u));
        let sc6 = f32(read_i8(scBase + is0 + 6u));

        acc = acc + (d * sc0 * f32(q1)) * x[xOff + l];
        acc = acc + (d * sc2 * f32(q2)) * x[xOff + 32u + l];
        acc = acc + (d * sc4 * f32(q3)) * x[xOff + 64u + l];
        acc = acc + (d * sc6 * f32(q4)) * x[xOff + 96u + l];
        l = l + 1u;
      }
      n = n + 1u;
    }
    blockIdx = blockIdx + 64u;
  }

  partial6[localId] = acc;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (localId < stride) {
      partial6[localId] = partial6[localId] + partial6[localId + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (localId == 0u) {
    out[rowOffset + row] = partial6[0];
  }
}
`
