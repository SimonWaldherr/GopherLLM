package gopherllm

import (
	"context"
	"sync"
)

// These immutable bases cover the shipped Voxtral and Parakeet frontends.
// Keep the cache bounded: unusual FFT sizes remain request-local.
var (
	speechDFT400 = sync.OnceValue(func() voxtralDFTTables { return buildVoxtralDFTTables(400) })
	speechDFT512 = sync.OnceValue(func() voxtralDFTTables { return buildVoxtralDFTTables(512) })
)

func cachedSpeechDFT(nfft int) voxtralDFTTables {
	switch nfft {
	case 400:
		return speechDFT400()
	case 512:
		return speechDFT512()
	default:
		return buildVoxtralDFTTables(nfft)
	}
}

// Dispatch complete frames, not three small matvecs per frame. Within a frame
// the original SIMD dot products and float32 power calculation are unchanged.
// Padding and model-specific log/normalization stay with the callers.
type speechMelTask struct {
	ctx                     context.Context
	dft                     voxtralDFTTables
	filterbank              Weight
	samples, window, mel    []float32
	frames, hop, windowLeft int
	preEmphasis             float32
	rowMajor                bool
	workers, mels           int
	scratch                 []float32
}

func (t *speechMelTask) run() error {
	// A streaming session keeps the scratch capacity, but not views into the
	// previous audio chunk or its output, after all workers have returned.
	defer func() { *t = speechMelTask{scratch: t.scratch} }()
	if err := t.ctx.Err(); err != nil {
		return err
	}
	t.mels = len(t.mel) / t.frames
	t.workers = 1
	if t.dft.nfft >= 128 && t.filterbank.F32 != nil {
		t.workers = min(t.frames, numThreads())
	}
	ensureLenNoClear(&t.scratch, t.workers*(t.dft.nfft+t.dft.nFreq+t.mels))
	parallelChunksTask(t.workers, t)
	return t.ctx.Err()
}

func (t *speechMelTask) runRows(start, end int) {
	width := t.dft.nfft
	stride := width + t.dft.nFreq + t.mels
	for worker := start; worker < end; worker++ {
		storage := t.scratch[worker*stride : (worker+1)*stride]
		frame := storage[:width]
		power := storage[width : width+t.dft.nFreq]
		mel := storage[width+t.dft.nFreq:]
		clear(frame)
		for pos := worker; pos < t.frames; pos += t.workers {
			if t.ctx.Err() != nil {
				return
			}
			for i, w := range t.window {
				frame[t.windowLeft+i] = t.samples[pos*t.hop+i] * w
			}
			if t.preEmphasis != 0 {
				// Preemphasis also writes past a short window's last sample.
				// Restore its zero padding before reusing this worker's frame.
				clear(frame[:t.windowLeft])
				clear(frame[t.windowLeft+len(t.window):])
				prev := float32(0)
				for i, value := range frame {
					frame[i] = value - t.preEmphasis*prev
					prev = value
				}
			}
			for k := range power {
				re := DotF32(t.dft.cos.F32[k*width:(k+1)*width], frame)
				im := DotF32(t.dft.sin.F32[k*width:(k+1)*width], frame)
				power[k] = re*re + im*im
			}
			if t.filterbank.F32 != nil {
				projection := matvecF32Task{data: t.filterbank.F32, x: power, out: mel, cols: len(power)}
				projection.runRows(0, t.mels)
			} else {
				// Non-F32 filterbanks stay on the caller, allowing their
				// existing matvec path to use the worker pool without nesting.
				mel = speechMelMatvec(t.filterbank, power, mel)
			}
			for c, value := range mel {
				if t.rowMajor {
					t.mel[pos*t.mels+c] = value
				} else {
					t.mel[c*t.frames+pos] = value
				}
			}
		}
	}
}

// Keep MatvecInto's escaping slice header off the common F32 worker path.
func speechMelMatvec(filterbank Weight, power, out []float32) []float32 {
	filterbank.MatvecInto(power, &out)
	return out
}
