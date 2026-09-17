//go:build arm64

package gopherllm

import "math"

// Decode the entire row in one call. The block kernel remains the fallback
// and the reference for the exact floating-point accumulation order.
var q6kDecodeAsmOK = hasDotProd && validateQ6KDecodeAsm()

//go:noescape
func q6kRowDecodeAsm(row *byte, q8 *int8, scales, sums, lut *float32, blocks int) float32

func q6kRowDecode(row []byte, q8 []int8, scales, sums []float32, blocks int) float32 {
	if blocks <= 0 {
		return 0
	}
	// Division avoids overflowing when validating an invalid block count.
	if blocks > len(row)/210 || blocks > len(q8)/256 || blocks > len(scales) || blocks > len(sums)/16 {
		panic("Q6_K decode: truncated input")
	}
	return q6kRowDecodeAsm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], blocks)
}

func validateQ6KDecodeAsm() bool {
	const blocks = 3
	row := q8kSelfCheckBytes(blocks*210, 37, 11)
	// f16LUT is filled in init(), after this self-check. Supply a local table
	// and exercise negative, zero and positive scales independently of it.
	lut := []float32{0.25, -0.125, 0}
	var q8 [blocks * 256]int8
	var sums [blocks * 16]float32
	scales := [blocks]float32{0.01, 0.03, 0.02}
	for i := range q8 {
		q8[i] = int8(i*17 + i/256*13)
	}
	for i := range sums {
		sums[i] = float32(i%23-11) * 0.125
	}
	var want float32
	for b := range blocks {
		block := row[b*210 : (b+1)*210]
		block[208], block[209] = byte(b), 0
		// Independent scalar unpack; do not depend on another SDOT kernel
		// passing its self-check to detect an instruction/layout error here.
		var dots [16]int32
		for half := range 2 {
			for l := range 32 {
				lo := block[half*64+l]
				lo2 := block[half*64+l+32]
				hi := block[128+half*32+l]
				quants := [4]byte{lo&15 | (hi&3)<<4, lo2&15 | ((hi>>2)&3)<<4,
					lo>>4 | ((hi>>4)&3)<<4, lo2>>4 | (hi>>6)<<4}
				for g, q := range quants {
					dots[half*8+g*2+l/16] += int32(q) * int32(q8[b*256+half*128+g*32+l])
				}
			}
		}
		var dot int32
		var offset float32
		for i := range 16 {
			s := int32(int8(block[192+i]))
			dot += s * dots[i]
			offset += float32(s) * sums[b*16+i]
		}
		want += lut[b] * (scales[b]*float32(dot) - offset)
	}
	got := q6kRowDecodeAsm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], blocks)
	return q8kSelfCheck("Q6_K full row", []int32{int32(math.Float32bits(got))}, []int32{int32(math.Float32bits(want))})
}
