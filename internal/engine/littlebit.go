package engine

import (
	"encoding/binary"
	"math"
	"sync"
)

// LittleBitWeight stores an affine projection with one or two binary
// factorizations. Weight carries the input and output dimensions.
type LittleBitWeight struct {
	Main     LittleBitFactors
	Residual *LittleBitFactors
	Bias     LittleBitScale
	Metal    *MetalLittleBitWeight
}

// LittleBitFactors borrows the GGUF tensor bytes for one pair of binary
// matrices and their scale vectors. U is [out, Rank] and V is [Rank, in].
// The binary rows are padded to 32 bits, with bit zero encoding +1.
type LittleBitFactors struct {
	Rank           int
	U, V           []byte
	U1, U2, V1, V2 LittleBitScale
}

// LittleBitScale holds a validated GGUF F16, BF16 or F32 vector in its
// original little-endian representation.
type LittleBitScale struct {
	Raw  []byte
	Type GGMLType
}

func (s LittleBitScale) at(i int) float32 {
	switch s.Type {
	case GGMLTypeF32:
		return math.Float32frombits(binary.LittleEndian.Uint32(s.Raw[4*i:]))
	case GGMLTypeF16:
		return F16ToF32(binary.LittleEndian.Uint16(s.Raw[2*i:]))
	case GGMLTypeBF16:
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(s.Raw[2*i:])) << 16)
	default:
		panic("invalid LittleBit scale type")
	}
}

// littleBitPackedRowBytes expects a strictly positive column count.
func littleBitPackedRowBytes(cols int) int {
	return ((cols-1)/32 + 1) * 4
}

func littleBitDotScalar(packed []byte, x []float32) float32 {
	var sum float32
	for i, value := range x {
		if packed[i>>3]&(1<<uint(i&7)) != 0 {
			sum -= value
		} else {
			sum += value
		}
	}
	return sum
}

type littleBitMatvecTask struct {
	scaledX, latent   []float32
	input, output     []float32
	packed            []byte
	rowScales, rankU2 LittleBitScale
	bias              LittleBitScale
	rowBytes          int
	optimized, add    bool
}

var littleBitMatvecPool = sync.Pool{New: func() any { return new(littleBitMatvecTask) }}

func (t *littleBitMatvecTask) runRows(start, end int) {
	row := start
	// Share the activation loads across four rows at a time: a single
	// decode-time token has no sibling tokens to fuse against (unlike the
	// batch path's littleBitDot4), so fuse across output rows instead.
	if t.optimized && hasFastLittleBitDotRows4 {
		for ; row+4 <= end; row += 4 {
			p0 := t.packed[row*t.rowBytes : (row+1)*t.rowBytes]
			p1 := t.packed[(row+1)*t.rowBytes : (row+2)*t.rowBytes]
			p2 := t.packed[(row+2)*t.rowBytes : (row+3)*t.rowBytes]
			p3 := t.packed[(row+3)*t.rowBytes : (row+4)*t.rowBytes]
			a, b, c, d := littleBitDotRows4(p0, p1, p2, p3, t.input)
			t.storeRow(row, a)
			t.storeRow(row+1, b)
			t.storeRow(row+2, c)
			t.storeRow(row+3, d)
		}
	}
	for ; row < end; row++ {
		packed := t.packed[row*t.rowBytes : (row+1)*t.rowBytes]
		var dot float32
		if t.optimized {
			dot = littleBitDot(packed, t.input)
		} else {
			dot = littleBitDotScalar(packed, t.input)
		}
		t.storeRow(row, dot)
	}
}

func (t *littleBitMatvecTask) storeRow(row int, dot float32) {
	scale := t.rowScales.at(row)
	if len(t.rankU2.Raw) != 0 {
		scale *= t.rankU2.at(row)
	}
	result := dot * scale
	if t.add {
		result = t.output[row] + result
	}
	if len(t.bias.Raw) != 0 {
		result += t.bias.at(row)
	}
	t.output[row] = result
}

func (t *littleBitMatvecTask) project(input, output []float32, packed []byte, scales, rankU2, bias LittleBitScale, add bool) {
	t.input, t.output, t.packed = input, output, packed
	t.rowScales, t.rankU2, t.bias, t.add = scales, rankU2, bias, add
	t.rowBytes = littleBitPackedRowBytes(len(input))
	// Use the coarse row scheduler only when the amount of binary dot
	// work can amortize waking the workers. Division keeps the comparison
	// in range even for very wide inputs.
	if t.optimized && len(output) >= 64 && len(input) >= 65536/len(output) {
		parallelRowsBatchedTask(len(output), t)
	} else {
		t.runRows(0, len(output))
	}
}

func (t *littleBitMatvecTask) factors(f *LittleBitFactors, x, out []float32, bias LittleBitScale, add bool) {
	ensureLenNoClear(&t.latent, f.Rank)
	for i, value := range x {
		t.scaledX[i] = value * f.V2.at(i)
	}
	t.project(t.scaledX, t.latent, f.V, f.V1, f.U2, LittleBitScale{}, false)
	t.project(t.latent, out, f.U, f.U1, LittleBitScale{}, bias, add)
}

func (w *LittleBitWeight) matvecInto(x []float32, rows, cols int, out *[]float32, optimized bool) {
	if len(x) != cols {
		panic("LittleBit projection input width mismatch")
	}
	ensureLenNoClear(out, rows)
	t := littleBitMatvecPool.Get().(*littleBitMatvecTask)
	ensureLenNoClear(&t.scaledX, cols)
	t.optimized = optimized
	if w.Residual == nil {
		t.factors(&w.Main, x, *out, w.Bias, false)
	} else {
		t.factors(&w.Main, x, *out, LittleBitScale{}, false)
		t.factors(w.Residual, x, *out, w.Bias, true)
	}
	// Only the activation scratch survives between calls. In particular the
	// task must never retain a model's mmap-backed factor or scale slices.
	*t = littleBitMatvecTask{scaledX: t.scaledX[:0], latent: t.latent[:0]}
	littleBitMatvecPool.Put(t)
}

var littleBitArgmaxPool = sync.Pool{New: func() any { return new([]float32) }}
