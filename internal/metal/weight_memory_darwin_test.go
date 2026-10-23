//go:build darwin && cgo && metal

package metal

import (
	"encoding/binary"
	"math"
	"strings"
	"syscall"
	"testing"
)

type memoryTestFormat struct {
	name       string
	blockSize  int
	blockBytes int
	prepare    func([]byte, int, int, bool) *Weight
	matvec     func(*Weight, []float32, []float32) bool
}

var memoryTestFormats = []memoryTestFormat{
	{"Q4_K", 256, 144, PrepareQ4K, MatvecQ4K},
	{"Q5_K", 256, 176, PrepareQ5K, MatvecQ5K},
	{"Q6_K", 256, 210, PrepareQ6K, MatvecQ6K},
	{"Q8_0", 32, 34, PrepareQ8_0, MatvecQ8_0},
}

func TestQuantizedWeightDataBounds(t *testing.T) {
	for _, f := range memoryTestFormats {
		t.Run(f.name, func(t *testing.T) {
			const rows, cols = 3, 256
			matrixBytes := rows * (cols / f.blockSize) * f.blockBytes
			data := make([]byte, matrixBytes+128)
			got := quantizedWeightData(data, rows, cols, f.blockSize, f.blockBytes)
			if len(got) != matrixBytes || &got[0] != &data[0] {
				t.Fatal("matrix view must exclude trailing bytes without copying")
			}
			for _, shape := range [][2]int{
				{0, cols}, {-1, cols}, {rows, 0}, {rows, cols + 1},
				{math.MaxInt32 + 1, cols}, {rows, math.MaxInt32 + 1},
				{math.MaxInt32, cols}, {rows, math.MaxInt32 - (math.MaxInt32 % f.blockSize)},
			} {
				if got := f.prepare(data, shape[0], shape[1], true); got != nil {
					Release(got)
					t.Fatalf("accepted invalid shape %v", shape)
				}
			}
			if got := f.prepare(data[:matrixBytes-1], rows, cols, true); got != nil {
				Release(got)
				t.Fatal("accepted truncated matrix")
			}
		})
	}
}

func TestMetalBorrowedWeightsAliasMapping(t *testing.T) {
	if !Available() {
		t.Skip(LastError())
	}
	for _, f := range memoryTestFormats {
		t.Run(f.name, func(t *testing.T) {
			const rows, cols, offset = 8, 256, 32
			matrixBytes := rows * (cols / f.blockSize) * f.blockBytes
			page := syscall.Getpagesize()
			mapping, err := syscall.Mmap(-1, 0, (offset+matrixBytes+page-1)/page*page,
				syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
			if err != nil {
				t.Fatal(err)
			}
			defer syscall.Munmap(mapping)
			data := mapping[offset : offset+matrixBytes]
			borrowed := f.prepare(data, rows, cols, true)
			owned := f.prepare(data, rows, cols, false)
			if borrowed == nil || owned == nil {
				Release(borrowed)
				Release(owned)
				t.Fatalf("prepare matrix: %s", LastError())
			}
			defer Release(borrowed)
			defer Release(owned)
			x := make([]float32, cols)
			for i := range x {
				x[i] = 1
			}
			before := make([]float32, rows)
			if !f.matvec(borrowed, x, before) {
				t.Fatalf("initial matvec: %s", LastError())
			}
			for _, value := range before {
				if value != 0 {
					t.Fatalf("initial zero matrix yielded %v", before)
				}
			}
			fillMemoryTestOnes(data, f)
			alias, copy := make([]float32, rows), make([]float32, rows)
			if !f.matvec(borrowed, x, alias) || !f.matvec(owned, x, copy) {
				t.Fatalf("matvec after mapping update: %s", LastError())
			}
			for i := range alias {
				if alias[i] != cols || copy[i] != 0 {
					t.Fatalf("row %d: borrowed=%g owned=%g, want %d/0", i, alias[i], copy[i], cols)
				}
			}
		})
	}
}

func fillMemoryTestOnes(data []byte, f memoryTestFormat) {
	for len(data) >= f.blockBytes {
		block := data[:f.blockBytes]
		switch f.name {
		case "Q4_K", "Q5_K":
			binary.LittleEndian.PutUint16(block, 0x3c00)
			for i := 4; i < 16; i++ {
				block[i] = 1
			}
			start := 16
			if f.name == "Q5_K" {
				start = 48
			}
			for i := start; i < len(block); i++ {
				block[i] = 0x11
			}
		case "Q6_K":
			for i := 0; i < 128; i++ {
				block[i] = 0x11
			}
			for i := 128; i < 192; i++ {
				block[i] = 0xaa
			}
			for i := 192; i < 208; i++ {
				block[i] = 1
			}
			binary.LittleEndian.PutUint16(block[208:], 0x3c00)
		case "Q8_0":
			binary.LittleEndian.PutUint16(block, 0x3c00)
			for i := 2; i < len(block); i++ {
				block[i] = 1
			}
		}
		data = data[f.blockBytes:]
	}
}

func TestMetalOversizedWeightFailsWithoutReadingMapping(t *testing.T) {
	if !Available() {
		t.Skip(LastError())
	}
	limit := MaxBufferLength()
	if limit == 0 {
		t.Fatal("available device returned no buffer limit")
	}
	for _, f := range memoryTestFormats {
		t.Run(f.name, func(t *testing.T) {
			const cols = 65536
			rowBytes := (cols / f.blockSize) * f.blockBytes
			rows := int(limit/uint64(rowBytes)) + 1
			if rows > math.MaxInt32 || rows > math.MaxInt/rowBytes {
				t.Skip("device buffer limit exceeds representable test matrix")
			}
			// Reserve address space only. An accidental copy would read the
			// protected mapping, while the valid rejection consumes no RAM.
			mapping, err := syscall.Mmap(-1, 0, rows*rowBytes, syscall.PROT_NONE,
				syscall.MAP_ANON|syscall.MAP_PRIVATE)
			if err != nil {
				t.Skipf("reserve virtual address space: %v", err)
			}
			defer syscall.Munmap(mapping)
			for _, noCopy := range []bool{true, false} {
				w := f.prepare(mapping, rows, cols, noCopy)
				if w != nil {
					Release(w)
					t.Fatalf("oversized matrix accepted (noCopy=%t)", noCopy)
				}
				if !strings.Contains(LastError(), "maxBufferLength") {
					t.Fatalf("unexpected rejection: %s", LastError())
				}
			}
		})
	}
}
