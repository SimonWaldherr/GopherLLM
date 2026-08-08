//go:build js

package webgpu

// wgslCommon holds WGSL helper functions shared by every dequantizing
// matvec kernel: a byte-from-u32-word reader (WGSL has no u8 storage
// buffer element type, so packed bytes must be read as u32 words and
// unpacked with shifts/masks) and an IEEE-754 half-to-single-precision
// float decoder (hand-written from the format's own bit layout -- sign,
// 5-bit exponent, 10-bit mantissa, with the standard subnormal
// renormalization loop and Inf/NaN passthrough -- not from any external
// shader source; this is the textbook IEEE 754 conversion algorithm, the
// same one this project's own scalar Go F16ToF32 implements).
//
// weights, x, and out are declared by each kernel file (their exact
// binding numbers differ per kernel), so this file only provides functions
// that reference a `weights: array<u32>` binding by name -- every kernel
// must name its raw-bytes storage binding `weights` for these to resolve.
const wgslCommon = `
fn read_u8(byteOffset: u32) -> u32 {
  let wordIdx = byteOffset >> 2u;
  let shift = (byteOffset & 3u) * 8u;
  return (weights[wordIdx] >> shift) & 0xFFu;
}

fn read_u16(byteOffset: u32) -> u32 {
  return read_u8(byteOffset) | (read_u8(byteOffset + 1u) << 8u);
}

fn f16_to_f32(h: u32) -> f32 {
  let sign: u32 = (h & 0x8000u) << 16u;
  let exp: u32 = (h >> 10u) & 0x1fu;
  var mant: u32 = h & 0x3ffu;
  if (exp == 0u) {
    if (mant == 0u) {
      return bitcast<f32>(sign);
    }
    var e: u32 = 0u;
    for (var i: u32 = 0u; i < 10u; i = i + 1u) {
      if ((mant & 0x400u) != 0u) { break; }
      mant = mant << 1u;
      e = e + 1u;
    }
    mant = mant & 0x3ffu;
    return bitcast<f32>(sign | ((127u - 15u + 1u - e) << 23u) | (mant << 13u));
  }
  if (exp == 31u) {
    return bitcast<f32>(sign | (0xffu << 23u) | (mant << 13u));
  }
  return bitcast<f32>(sign | ((exp + 127u - 15u) << 23u) | (mant << 13u));
}

fn get_scale_min_k4(j: u32, scalesBase: u32) -> vec2<u32> {
  if (j < 4u) {
    let sc = read_u8(scalesBase + j) & 63u;
    let mn = read_u8(scalesBase + j + 4u) & 63u;
    return vec2<u32>(sc, mn);
  }
  let qj4 = read_u8(scalesBase + j + 4u);
  let qjm4 = read_u8(scalesBase + j - 4u);
  let qj = read_u8(scalesBase + j);
  let sc = (qj4 & 0x0fu) | ((qjm4 >> 6u) << 4u);
  let mn = (qj4 >> 4u) | ((qj >> 6u) << 4u);
  return vec2<u32>(sc, mn);
}
`
