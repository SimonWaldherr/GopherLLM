//go:build !darwin || !arm64 || !cgo

package gopherllm

const hasApplePrefillBLAS = false

func matvecBatchBLAS(_ Weight, _, _ [][]float32, _ bool) bool { return false }
