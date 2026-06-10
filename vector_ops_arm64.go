//go:build arm64

package main

func axpyF32(out []float32, alpha float32, x []float32)
func scaleF32(out []float32, alpha float32)
func scaleAddF32(out []float32, alpha float32, x []float32)
func mulScaleF32(x []float32, weight []float32, scale float32, out []float32)
