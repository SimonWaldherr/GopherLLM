package gopherllm

import "math"

// voxtralSlaneyMelFilterbank computes a Slaney-normalized mel filterbank,
// matching librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,
// norm="slaney") and matching stt.frontend.mel_norm=="slaney" in every
// known-good Voxtral GGUF's metadata. The official (non-GGUF) Voxtral
// release ships no precomputed filterbank tensor at all (see
// LoadVoxtralRealtimeModelFromSafetensors), unlike both known GGUF
// conventions, which bake one in for convenience -- this reconstructs the
// identical values from the standard formula instead.
//
// Returns row-major [nMels][nFFT/2+1], matching the layout
// VoxtralRealtimeWeights.MelFilterbank (Weight.MatvecInto's convention:
// Cols must equal the caller's power-spectrum input length) needs directly,
// with no further transpose. Verified byte-close (see
// TestVoxtralSlaneyMelFilterbankMatchesShippedFilterbank) against a working
// GGUF's own precomputed filterbank tensor for n_fft=400, nMels=128,
// sampleRate=16000, fMin=0, fMax=8000 -- the exact Voxtral frontend config.
func voxtralSlaneyMelFilterbank(sampleRate, nFFT, nMels int, fMin, fMax float64) []float32 {
	nFreq := nFFT/2 + 1
	fftFreqs := make([]float64, nFreq)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(sampleRate) / float64(nFFT)
	}

	// Slaney's hz<->mel scale: linear below 1 kHz, logarithmic above --
	// distinct from the simpler HTK formula (mel = 2595*log10(1+f/700))
	// GGUF's htk field would flag if this checkpoint used it (it doesn't).
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	const minLogMel = minLogHz / fSp // 15.0
	logStep := math.Log(6.4) / 27.0
	hzToMel := func(f float64) float64 {
		if f < minLogHz {
			return f / fSp
		}
		return minLogMel + math.Log(f/minLogHz)/logStep
	}
	melToHz := func(m float64) float64 {
		if m < minLogMel {
			return fSp * m
		}
		return minLogHz * math.Exp(logStep*(m-minLogMel))
	}

	melMin, melMax := hzToMel(fMin), hzToMel(fMax)
	melPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
	}
	edges := make([]float64, nMels+2)
	for i, m := range melPoints {
		edges[i] = melToHz(m)
	}

	weights := make([]float32, nMels*nFreq)
	for m := 0; m < nMels; m++ {
		lower, center, upper := edges[m], edges[m+1], edges[m+2]
		// Slaney-style area normalization: each filter is scaled so a
		// constant-power input contributes equal energy regardless of the
		// (non-uniform) width of its triangle in Hz.
		enorm := 2.0 / (upper - lower)
		for k, f := range fftFreqs {
			var w float64
			if f > lower && f < upper {
				if f <= center {
					w = (f - lower) / (center - lower)
				} else {
					w = (upper - f) / (upper - center)
				}
				if w < 0 {
					w = 0
				}
			}
			weights[m*nFreq+k] = float32(w * enorm)
		}
	}
	return weights
}
