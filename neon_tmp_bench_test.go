package main

import "testing"

func BenchmarkDotQ4KNEON_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 144)
	x := benchFloatSlice(4096)
	scratch := []float32{}
	xs := fillQ4KXSums(x, 4096, &scratch)
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = dotQ4KF32NEONWithXSums(row, x, xs, 4096)
	}
}

func BenchmarkDotQ6KNEON_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 210)
	x := benchFloatSlice(4096)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, 4096, &scratch)
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = dotQ6KF32NEONWithXSums(row, x, xs, 4096)
	}
}
