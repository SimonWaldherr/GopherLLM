//go:build !amd64

package main

// The int8-activation Q4_K path is amd64-only (it relies on VPMADDUBSW). On
// other targets useQ8Activations is a compile-time false, so these stubs are
// referenced by the (dead-code-eliminated) branches in the matvec entry points
// but never executed.

const useQ8Activations = false

func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, release func()) {
	panic("acquireQ8 is amd64-only")
}

func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	panic("dotQ4KRowsQ8 is amd64-only")
}
