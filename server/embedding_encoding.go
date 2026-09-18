package server

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
)

func encodeEmbedding(v []float32, format string) any {
	if format != "base64" {
		return v
	}
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}

func validateEmbeddingVector(v []float32, dimension int) error {
	if dimension <= 0 || len(v) != dimension {
		return fmt.Errorf("embedding dimension mismatch")
	}
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("non-finite embedding")
		}
	}
	return nil
}
