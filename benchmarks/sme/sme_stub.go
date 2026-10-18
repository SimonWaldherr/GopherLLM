//go:build !gopherllm_sme_experiment || !darwin || !arm64 || !cgo || ios

package sme

import "errors"

func initSME() (tileKernel, error) {
	return tileKernel{}, errors.New("SME experiment requires macOS ARM64, cgo and -tags=gopherllm_sme_experiment")
}

func runSME(_, _ []int8, _ []int32, _, _, _, _ int) error {
	return errors.New("SME experiment was not compiled")
}
