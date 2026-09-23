//go:build !darwin || !cgo

package yolo

func yoloGEMM(m, n, k int, a, b, c []float32) { yoloGEMMPortable(m, n, k, a, b, c) }
