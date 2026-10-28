//go:build coreml

package engine

import (
	"errors"
	"fmt"
	"io"
	"time"

	nativecoreml "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

// Keep at most one candidate alive for this layer. When both modes are valid,
// CPU must win both sides of a CPU/ANE/CPU comparison by 10%; otherwise retain
// the default CPU/ANE permission. Each load, including a final reopen,
// validates shape, source binding and quality.
func tuneCoreMLFFN(load func(nativecoreml.ComputeUnits) (*coreMLFFNLayer, error),
	measure func(*coreMLFFNLayer) (time.Duration, error), logw io.Writer) (*coreMLFFNLayer, error) {
	trial := func(units nativecoreml.ComputeUnits, name string) (*coreMLFFNLayer, time.Duration, error) {
		layer, err := load(units)
		if err != nil {
			fmt.Fprintf(logw, "  %s rejected: %v\n", name, err)
			return nil, 0, err
		}
		latency, err := measure(layer)
		if err == nil && latency <= 0 {
			err = errors.New("calibration clock did not advance")
		}
		if err != nil {
			_ = layer.model.Close()
			fmt.Fprintf(logw, "  %s failed: %v\n", name, err)
			return nil, 0, err
		}
		fmt.Fprintf(logw, "  %s warm median: %s\n", name, latency.Round(time.Microsecond))
		return layer, latency, nil
	}
	cpu, firstCPU, cpuErr := trial(nativecoreml.CPUOnly, "CPU")
	if cpu != nil {
		_ = cpu.model.Close()
	}
	ane, baseline, aneErr := trial(nativecoreml.CPUAndNeuralEngine, "CPU/Neural Engine")
	if cpuErr != nil && aneErr != nil {
		return nil, errors.Join(cpuErr, aneErr)
	}
	threshold := baseline - baseline/10
	if aneErr == nil && (cpuErr != nil || firstCPU > threshold) {
		fmt.Fprintln(logw, "  selected CPU/Neural Engine")
		return ane, nil
	}
	if ane != nil {
		_ = ane.model.Close()
	}
	cpu, repeatedCPU, repeatErr := trial(nativecoreml.CPUOnly, "CPU recheck")
	if repeatErr == nil && (aneErr != nil || repeatedCPU <= threshold) {
		fmt.Fprintln(logw, "  selected CPU")
		return cpu, nil
	}
	if cpu != nil {
		_ = cpu.model.Close()
	}
	if aneErr != nil {
		return nil, errors.Join(aneErr, repeatErr)
	}
	// A transient CPU advantage or a failed recheck must not change the
	// default device permission. Revalidate rather than retain both models.
	ane, err := load(nativecoreml.CPUAndNeuralEngine)
	if err == nil {
		fmt.Fprintln(logw, "  selected CPU/Neural Engine after CPU recheck")
	}
	return ane, err
}

func measureCoreMLFFN(model coreMLFFNPredictor, input, output []float32) (time.Duration, error) {
	for range 2 {
		if err := model.Predict(input, output); err != nil {
			return 0, err
		}
	}
	var samples [9]float64
	for i := range samples {
		start := time.Now()
		if err := model.Predict(input, output); err != nil {
			return 0, err
		}
		samples[i] = float64(time.Since(start))
	}
	return time.Duration(medianFloat(samples[:])), nil
}
