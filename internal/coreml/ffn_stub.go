//go:build !darwin || !cgo || !coreml

package coreml

func OpenFFN(path string, units ComputeUnits) (*FFN, error) {
	return nil, ErrUnavailable
}
