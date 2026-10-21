package gopherllm

import "encoding/binary"

// QuantizeRowQ5_0 encodes signed 32-element blocks. Mixed K-quant presets
// use this fallback when a matrix row is not divisible by 256.
func QuantizeRowQ5_0(row []float32, cols int) []byte {
	out := make([]byte, cols/32*22)
	for b := 0; b < cols/32; b++ {
		x := row[b*32 : (b+1)*32]
		var amax, extreme float32
		for _, v := range x {
			if a := abs32(v); a > amax {
				amax, extreme = a, v
			}
		}
		d, inv := extreme/-16, float32(0)
		if d != 0 {
			inv = 1 / d
		}
		block := out[b*22 : (b+1)*22]
		putF16(block, d)
		var high uint32
		for i := 0; i < 16; i++ {
			lo := uint32(clampInt(int(roundHalfAwayFromZero(x[i]*inv))+16, 0, 31))
			hi := uint32(clampInt(int(roundHalfAwayFromZero(x[i+16]*inv))+16, 0, 31))
			block[6+i] = byte(lo&15) | byte(hi&15)<<4
			high |= (lo>>4)<<i | (hi>>4)<<(i+16)
		}
		binary.LittleEndian.PutUint32(block[2:6], high)
	}
	return out
}

// QuantizeRowQ5_1 encodes affine 32-element blocks for the Q5_K fallback.
func QuantizeRowQ5_1(row []float32, cols int) []byte {
	out := make([]byte, cols/32*24)
	for b := 0; b < cols/32; b++ {
		x := row[b*32 : (b+1)*32]
		minimum, maximum := x[0], x[0]
		for _, v := range x[1:] {
			minimum, maximum = min(minimum, v), max(maximum, v)
		}
		d, inv := (maximum-minimum)/31, float32(0)
		if d != 0 {
			inv = 1 / d
		}
		block := out[b*24 : (b+1)*24]
		putF16(block, d)
		putF16(block[2:], minimum)
		var high uint32
		for i := 0; i < 16; i++ {
			lo := uint32(clampInt(int(roundHalfAwayFromZero((x[i]-minimum)*inv)), 0, 31))
			hi := uint32(clampInt(int(roundHalfAwayFromZero((x[i+16]-minimum)*inv)), 0, 31))
			block[8+i] = byte(lo&15) | byte(hi&15)<<4
			high |= (lo>>4)<<i | (hi>>4)<<(i+16)
		}
		binary.LittleEndian.PutUint32(block[4:8], high)
	}
	return out
}

func quantizeRowF16(row []float32, cols int) []byte {
	out := make([]byte, 2*cols)
	for i, v := range row[:cols] {
		putF16(out[2*i:], v)
	}
	return out
}
