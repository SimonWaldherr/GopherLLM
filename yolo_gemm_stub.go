//go:build !darwin || !cgo

package gopherllm

func yoloGEMM(m, n, k int, a, b, c []float32) { yoloGEMMPortable(m, n, k, a, b, c) }
