package engine

import (
	"errors"
	"math"
)

// Scheduling follows Diffusers FlowMatchEulerDiscreteScheduler. Turbo stores
// its distilled sigmas in model_index.json; they are not a uniform 8-step grid.
type qi21SchedulerConfig struct {
	ClassName     string   `json:"_class_name"`
	TrainSteps    int      `json:"num_train_timesteps"`
	BaseSequence  int      `json:"base_image_seq_len"`
	MaxSequence   int      `json:"max_image_seq_len"`
	BaseShift     float64  `json:"base_shift"`
	MaxShift      float64  `json:"max_shift"`
	Shift         float64  `json:"shift"`
	ShiftTerminal *float64 `json:"shift_terminal"`
	Dynamic       bool     `json:"use_dynamic_shifting"`
	ShiftType     string   `json:"time_shift_type"`
	Invert        bool     `json:"invert_sigmas"`
	Stochastic    bool     `json:"stochastic_sampling"`
	Karras        bool     `json:"use_karras_sigmas"`
	Exponential   bool     `json:"use_exponential_sigmas"`
	Beta          bool     `json:"use_beta_sigmas"`
}

func qi21Schedule(c qi21SchedulerConfig, steps, imageTokens int, savedSigmas []float32) ([]float32, error) {
	if c.ClassName != "FlowMatchEulerDiscreteScheduler" || c.TrainSteps != 1000 || c.Invert || c.Stochastic || c.Karras || c.Exponential || c.Beta || imageTokens < 1 || c.Shift <= 0 || math.IsInf(c.Shift, 0) || math.IsNaN(c.Shift) {
		return nil, errors.New("qwen image: unsupported flow scheduler")
	}
	if len(savedSigmas) != 0 {
		steps = len(savedSigmas)
	}
	if steps < 1 || steps > 1000 {
		return nil, errors.New("qwen image: steps must be between 1 and 1000")
	}
	sigmas := make([]float32, steps+1)
	mu := float64(0)
	if c.Dynamic {
		if c.ShiftType != "exponential" || c.BaseSequence <= 0 || c.MaxSequence <= c.BaseSequence || math.IsNaN(c.BaseShift) || math.IsInf(c.BaseShift, 0) || math.IsNaN(c.MaxShift) || math.IsInf(c.MaxShift, 0) {
			return nil, errors.New("qwen image: invalid dynamic shift")
		}
		mu = c.BaseShift + float64(imageTokens-c.BaseSequence)*(c.MaxShift-c.BaseShift)/float64(c.MaxSequence-c.BaseSequence)
	}
	previous := float32(2)
	for i := 0; i < steps; i++ {
		value := float32(1 - float64(i)/float64(steps))
		if len(savedSigmas) != 0 {
			value = savedSigmas[i]
		}
		if !finite32(value) || value <= 0 || value > 1 || value >= previous {
			return nil, errors.New("qwen image: sigmas must be finite, positive and strictly decreasing")
		}
		previous = value
		if c.Dynamic {
			a := float32(math.Exp(mu))
			value = a / (a + (1/value - 1))
		} else {
			shift := float32(c.Shift)
			value = shift * value / (1 + (shift-1)*value)
		}
		if !finite32(value) {
			return nil, errors.New("qwen image: non-finite shifted sigma")
		}
		sigmas[i] = value
	}
	if c.ShiftTerminal != nil && *c.ShiftTerminal != 0 {
		terminal := *c.ShiftTerminal
		if math.IsNaN(terminal) || terminal <= 0 || terminal >= 1 || sigmas[steps-1] >= 1 {
			return nil, errors.New("qwen image: invalid terminal shift for this schedule")
		}
		factor := (1 - sigmas[steps-1]) / (1 - float32(terminal))
		for i := 0; i < steps; i++ {
			sigmas[i] = 1 - (1-sigmas[i])/factor
		}
	}
	for i := 0; i < steps; i++ {
		if !finite32(sigmas[i]) || sigmas[i] <= 0 || sigmas[i] > 1 || (i > 0 && sigmas[i] >= sigmas[i-1]) {
			return nil, errors.New("qwen image: shifted sigma schedule is not strictly decreasing")
		}
	}
	return sigmas, nil
}

func qi21EulerStep(latents, prediction []float32, sigma, next float32) error {
	if len(latents) != len(prediction) || !finite32(sigma) || !finite32(next) || sigma <= 0 || sigma > 1 || next < 0 || next >= sigma {
		return errors.New("qwen image: invalid Euler step")
	}
	dt := next - sigma
	for i, v := range prediction {
		if !finite32(v) || !finite32(latents[i]) {
			return errors.New("qwen image: non-finite Euler input")
		}
		if !finite32(latents[i] + dt*v) {
			return errors.New("qwen image: non-finite Euler output")
		}
	}
	for i, v := range prediction {
		latents[i] += dt * v
	}
	return nil
}
