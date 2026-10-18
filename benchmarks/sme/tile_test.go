package sme

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"strings"
	"testing"
)

func reference(a, b []int8, rows, cols, depth int) []int32 {
	out := make([]int32, rows*cols)
	for row := range rows {
		for col := range cols {
			for k := range depth {
				out[row*cols+col] += int32(a[row*depth+k]) * int32(b[col*depth+k])
			}
		}
	}
	return out
}

func testRows(rows, depth int, seed int64) []int8 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]int8, rows*depth)
	for i := range x {
		x[i] = int8(rng.Intn(256) - 128)
	}
	return x
}

func packForTest(t testing.TB, src []int8, rows, depth, lanes int) []int8 {
	t.Helper()
	size, err := packedSize(rows, depth, lanes)
	if err != nil {
		t.Fatal(err)
	}
	guarded := make([]int8, size+2)
	for i := range guarded {
		guarded[i] = 91
	}
	packed := guarded[1 : size+1]
	original := slices.Clone(src)
	if err := packRows(packed, src, rows, depth, lanes); err != nil {
		t.Fatal(err)
	}
	if guarded[0] != 91 || guarded[size+1] != 91 || !slices.Equal(src, original) {
		t.Fatal("packing modified its guards or input")
	}
	for group := range (depth + 3) / 4 {
		for row := range lanes {
			for k := range 4 {
				var want int8
				if row < rows && group*4+k < depth {
					want = src[row*depth+group*4+k]
				}
				if got := packed[(group*lanes+row)*4+k]; got != want {
					t.Fatalf("group=%d row=%d k=%d: got %d, want %d", group, row, k, got, want)
				}
			}
		}
	}
	return packed
}

func TestPackingAndReference(t *testing.T) {
	for _, lanes := range []int{4, 8, 16, 32, 64} {
		for _, shape := range [][2]int{{1, 1}, {min(3, lanes), min(5, lanes)}, {lanes, lanes}} {
			for _, depth := range []int{1, 3, 4, 5, 31, 256, 257} {
				t.Run(fmt.Sprintf("lanes%d/m%d/n%d/k%d", lanes, shape[0], shape[1], depth), func(t *testing.T) {
					rows, cols := shape[0], shape[1]
					a, b := testRows(rows, depth, 18), testRows(cols, depth, 19)
					pa := packForTest(t, a, rows, depth, lanes)
					pb := packForTest(t, b, cols, depth, lanes)
					out := make([]int32, rows*cols+2)
					for i := range out {
						out[i] = 0x345678
					}
					if err := packedReference(pa, pb, out[1:len(out)-1], rows, cols, depth, lanes); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(out[1:len(out)-1], reference(a, b, rows, cols, depth)) || out[0] != 0x345678 || out[len(out)-1] != 0x345678 {
						t.Fatal("packed result or output guards differ")
					}
				})
			}
		}
	}
}

func TestPackingReuseClearsPadding(t *testing.T) {
	const lanes, depth = 16, 5
	size, _ := packedSize(lanes, depth, lanes)
	packed := make([]int8, size)
	if err := packRows(packed, testRows(lanes, depth, 20), lanes, depth, lanes); err != nil {
		t.Fatal(err)
	}
	if err := packRows(packed, []int8{-128, -1, 0, 127, 13}, 1, depth, lanes); err != nil {
		t.Fatal(err)
	}
	want := packForTest(t, []int8{-128, -1, 0, 127, 13}, 1, depth, lanes)
	if !slices.Equal(packed, want) {
		t.Fatal("old rows or depth tail survived reuse")
	}
}

func TestTileBounds(t *testing.T) {
	for _, dims := range [][3]int{{0, 1, 4}, {-1, 1, 4}, {5, 1, 4}, {1, 0, 4}, {1, -1, 4}, {1, maxDepth + 1, 4}, {1, 1, 0}, {1, 1, 3}, {1, 1, 65}, {1, 1, int(^uint(0) >> 1)}} {
		if _, err := packedSize(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("accepted dimensions %v", dims)
		}
	}
	for _, lengths := range [][2]int{{15, 4}, {16, 3}} {
		dst := make([]int8, lengths[0])
		for i := range dst {
			dst[i] = 91
		}
		if err := packRows(dst, make([]int8, lengths[1]), 1, 4, 4); err == nil || !slices.Equal(dst, slices.Repeat([]int8{91}, len(dst))) {
			t.Fatal("short packing buffer accepted or modified")
		}
	}
	for _, lengths := range [][3]int{{15, 16, 1}, {16, 15, 1}, {16, 16, 0}} {
		if err := packedReference(make([]int8, lengths[0]), make([]int8, lengths[1]), make([]int32, lengths[2]), 1, 1, 4, 4); err == nil {
			t.Fatal("short tile buffer accepted")
		}
	}
	if err := (tileKernel{}).run(nil, nil, nil, 1, 1, 4); err == nil {
		t.Fatal("zero-value kernel accepted")
	}
}

func TestFeatureGate(t *testing.T) {
	const sme = "hw.optional.arm.FEAT_SME"
	const i8 = "hw.optional.arm.SME_I8I32"
	const sme2 = "hw.optional.arm.FEAT_SME2"
	for _, tc := range []struct {
		name     string
		values   map[string]uint32
		wantOK   bool
		wantSME2 bool
	}{
		{"missing", nil, false, false},
		{"disabled", map[string]uint32{sme: 0, i8: 1}, false, false},
		{"no integer arithmetic", map[string]uint32{sme: 1, i8: 0}, false, false},
		{"SME", map[string]uint32{sme: 1, i8: 1}, true, false},
		{"SME2", map[string]uint32{sme: 1, i8: 1, sme2: 1}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotSME2, err := featureGate(func(name string) (uint32, error) {
				if value, ok := tc.values[name]; ok {
					return value, nil
				}
				return 0, errors.New("missing or denied")
			})
			if (err == nil) != tc.wantOK || gotSME2 != tc.wantSME2 {
				t.Fatalf("got SME2=%t err=%v", gotSME2, err)
			}
		})
	}
	calls := 0
	_, _ = featureGate(func(string) (uint32, error) {
		calls++
		return 0, errors.New("denied")
	})
	if calls != 1 {
		t.Fatal("queried additional features after failed SME gate")
	}
}

func TestSMEOptIn(t *testing.T) {
	t.Setenv("GOPHERLLM_SME_EXPERIMENT", "")
	if _, err := newSMEKernel(); err == nil || !strings.Contains(err.Error(), "opt in") {
		t.Fatalf("missing opt-in: %v", err)
	}
	t.Setenv("GOPHERLLM_SME_EXPERIMENT", "1")
	t.Setenv("GOPHERLLM_DISABLE_SIMD", "1")
	if _, err := newSMEKernel(); err == nil || !strings.Contains(err.Error(), "DISABLE_SIMD") {
		t.Fatalf("SIMD disable ignored: %v", err)
	}
}

func TestSMETileBitExact(t *testing.T) {
	kernel, err := newSMEKernel()
	if err != nil {
		t.Skip(err)
	}
	t.Logf("streaming int32 lanes=%d, SME2=%t", kernel.lanes, kernel.sme2)
	for _, shape := range [][2]int{{1, 1}, {min(3, kernel.lanes), min(5, kernel.lanes)}, {kernel.lanes, kernel.lanes}} {
		for _, depth := range []int{1, 3, 4, 5, 31, 256, 3072, 9216} {
			rows, cols := shape[0], shape[1]
			a, b := testRows(rows, depth, 21), testRows(cols, depth, 22)
			pa := packForTest(t, a, rows, depth, kernel.lanes)
			pb := packForTest(t, b, cols, depth, kernel.lanes)
			originalA, originalB := slices.Clone(pa), slices.Clone(pb)
			out := make([]int32, rows*cols+2)
			for i := range out {
				out[i] = 0x345678
			}
			if err := kernel.run(pa, pb, out[1:len(out)-1], rows, cols, depth); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(out[1:len(out)-1], reference(a, b, rows, cols, depth)) || out[0] != 0x345678 || out[len(out)-1] != 0x345678 {
				t.Fatalf("SME result/guards differ: m=%d n=%d k=%d", rows, cols, depth)
			}
			if !slices.Equal(pa, originalA) || !slices.Equal(pb, originalB) {
				t.Fatal("SME modified packed inputs")
			}
		}
	}
	// A different per-thread VL must reject already packed buffers before
	// writing output. This branch executes SME only after feature gating.
	changed := kernel
	changed.lanes = 4
	if kernel.lanes == 4 {
		changed.lanes = 8
	}
	pa := packForTest(t, []int8{1, 2, 3, 4}, 1, 4, changed.lanes)
	out := []int32{91}
	if err := changed.run(pa, pa, out, 1, 1, 4); err == nil || out[0] != 91 {
		t.Fatal("VL mismatch accepted or modified output")
	}
}

func BenchmarkSMEInt8Tile(b *testing.B) {
	kernel, err := newSMEKernel()
	if err != nil {
		b.Skip(err)
	}
	b.Logf("streaming int32 lanes=%d, SME2=%t", kernel.lanes, kernel.sme2)
	for _, rows := range []int{4, kernel.lanes} {
		if rows == 4 && kernel.lanes == 4 {
			continue
		}
		for _, depth := range []int{256, 3072, 9216} {
			cols := kernel.lanes
			a, bb := testRows(rows, depth, 23), testRows(cols, depth, 24)
			pa := packForTest(b, a, rows, depth, kernel.lanes)
			pb := packForTest(b, bb, cols, depth, kernel.lanes)
			out := make([]int32, rows*cols)
			if err := kernel.run(pa, pb, out, rows, cols, depth); err != nil {
				b.Fatal(err)
			}
			if !slices.Equal(out, reference(a, bb, rows, cols, depth)) {
				b.Fatal("SME parity check failed before measurement")
			}
			for _, mode := range []string{"scalar_packed", "sme_packed", "sme_with_packing"} {
				b.Run(fmt.Sprintf("m%d/n%d/k%d/%s", rows, cols, depth, mode), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64((rows + cols) * depth))
					for b.Loop() {
						if mode == "scalar_packed" {
							if err := packedReference(pa, pb, out, rows, cols, depth, kernel.lanes); err != nil {
								b.Fatal(err)
							}
							continue
						}
						if mode == "sme_with_packing" {
							if err := packRows(pa, a, rows, depth, kernel.lanes); err != nil {
								b.Fatal(err)
							}
							if err := packRows(pb, bb, cols, depth, kernel.lanes); err != nil {
								b.Fatal(err)
							}
						}
						if err := kernel.run(pa, pb, out, rows, cols, depth); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
