//go:build !darwin || !cgo || !coreml

package coreml

func Open(path string, c Config) (*Model, error) {
	return nil, ErrUnavailable
}

func Inspect(path string, c Config) (Plan, error) {
	return Plan{}, ErrUnavailable
}
