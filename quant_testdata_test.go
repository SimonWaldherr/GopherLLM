package gopherllm

import "math/rand"

// Synthetic weight rows for the quantized-kernel tests, one builder per GGML
// block format.
//
// These live in an untagged file on purpose. They started out next to the amd64
// kernel tests that first needed them, which meant every new per-architecture
// kernel test had to relocate one of them — Q5_K, then Q8_0, then Q4_0/Q4_1/MXFP4
// — because the arm64 tests could not see an amd64-only helper. Nothing about
// generating a row is architecture-specific, so they all belong here where any
// test can reach them.
//
// Every builder writes small positive f16 scales rather than random bytes in the
// scale positions. Uniformly random scale bytes produce NaN and Inf block scales,
// which makes a differential comparison vacuously equal under an exact check and
// vacuously unequal under NaN != NaN.

func randomVec(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

// randomQ4KRow: 144-byte superblocks — f16 d, f16 dmin, 12 packed scale bytes,
// 128 packed nibbles.
func randomQ4KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*144)
	for b := 0; b < cols/256; b++ {
		block := row[b*144 : (b+1)*144]
		// 0x2c00 ~ 0.0625, 0x1c00 ~ 0.0039.
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		block[2], block[3] = byte(rng.Intn(256)), 0x1c
		for i := 4; i < 144; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// randomQ5KRow: 176-byte superblocks — Q4_K's header plus a 32-byte fifth-bit
// plane before the nibbles.
func randomQ5KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*176)
	for b := 0; b < cols/256; b++ {
		block := row[b*176 : (b+1)*176]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		block[2], block[3] = byte(rng.Intn(256)), 0x1c
		for i := 4; i < 176; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// randomQ6KRow: 210-byte superblocks — 128 low-nibble bytes, 64 high-bit bytes,
// 16 signed scales, then the f16 d at the END rather than the start.
func randomQ6KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*210)
	for b := 0; b < cols/256; b++ {
		block := row[b*210 : (b+1)*210]
		for i := 0; i < 208; i++ {
			block[i] = byte(rng.Intn(256))
		}
		block[208], block[209] = byte(rng.Intn(256)), 0x2c
	}
	return row
}

// randomQ8_0Row: 34-byte blocks — f16 scale then 32 int8 weights.
func randomQ8_0Row(rng *rand.Rand, cols int) []byte {
	blocks := cols / 32
	row := make([]byte, blocks*34)
	for b := 0; b < blocks; b++ {
		block := row[b*34 : (b+1)*34]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		for i := 2; i < 34; i++ {
			block[i] = byte(int8(rng.Intn(255) - 127)) // [-127, 127]
		}
	}
	return row
}

// randomQ4_0Row: 18-byte blocks — f16 scale then 16 packed nibble bytes.
func randomQ4_0Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*18)
	for b := 0; b < cols/32; b++ {
		block := row[b*18 : (b+1)*18]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		for i := 2; i < 18; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// randomQ4_1Row: 20-byte blocks — f16 scale, f16 minimum, 16 nibble bytes.
func randomQ4_1Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*20)
	for b := 0; b < cols/32; b++ {
		block := row[b*20 : (b+1)*20]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		block[2], block[3] = byte(rng.Intn(256)), 0x1c
		for i := 4; i < 20; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// randomMXFP4Row: 17-byte blocks — 16 nibble bytes then one raw E8M0 exponent.
func randomMXFP4Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*17)
	for b := 0; b < cols/32; b++ {
		block := row[b*17 : (b+1)*17]
		for i := 0; i < 16; i++ {
			block[i] = byte(rng.Intn(256))
		}
		// Exponents around 1.0 (127 +/- 8) keep the sums well-conditioned.
		block[16] = byte(119 + rng.Intn(17))
	}
	return row
}
