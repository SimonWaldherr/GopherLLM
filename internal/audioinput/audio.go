// Package audioinput decodes audio for the transcription commands.
package audioinput

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"

	"github.com/SimonWaldherr/GopherLLM/internal/wav"
)

// Load16kMono returns mono float32 PCM samples at 16 kHz using FFmpeg when
// available. The native fallback accepts PCM16/float32 WAV files and resamples
// with linear interpolation, without an anti-alias filter.
func Load16kMono(path string) ([]float32, error) {
	if ffmpegPath, err := exec.LookPath("ffmpeg"); err == nil {
		samples, err := decodeWithFFmpeg(ffmpegPath, path)
		if err == nil {
			return samples, nil
		}
		fmt.Fprintf(os.Stderr, "warning: ffmpeg decode failed (%v), falling back to the native WAV reader\n", err)
	}
	return decodeWAVFallback(path)
}

func decodeWithFFmpeg(ffmpegPath, path string) ([]float32, error) {
	cmd := exec.Command(ffmpegPath, "-v", "error", "-i", path, "-f", "f32le", "-ar", "16000", "-ac", "1", "-")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw)%4 != 0 {
		raw = raw[:len(raw)-len(raw)%4]
	}
	samples := make([]float32, len(raw)/4)
	for i := range samples {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples, nil
}

// decodeWAVFallback parses a canonical (non-extensible) PCM WAV file: RIFF
// container, one "fmt " chunk (PCM=1 or IEEE float=3), one "data" chunk.
func decodeWAVFallback(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening audio file: %w", err)
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("reading WAV size: %w", err)
	}
	info, err := wav.Inspect(f, st.Size())
	if err != nil {
		return nil, err
	}
	if info.SampleRate == 0 || info.Channels == 0 || info.BitsPerSample == 0 {
		return nil, fmt.Errorf("WAV file missing fmt/data chunks")
	}
	if !info.PCM() {
		return nil, fmt.Errorf("unsupported WAV format (audioFormat=%d bitsPerSample=%d); install ffmpeg for broader format support", info.Encoding, info.BitsPerSample)
	}
	mono, err := info.ReadMono(f)
	if err != nil {
		return nil, err
	}

	sampleRate := info.SampleRate
	if sampleRate == 16000 {
		return mono, nil
	}
	fmt.Fprintf(os.Stderr, "warning: resampling %dHz -> 16000Hz with a naive linear interpolator (install ffmpeg for a proper resample)\n", sampleRate)
	return linearResample(mono, int(sampleRate), 16000), nil
}

func linearResample(in []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate || len(in) == 0 {
		return in
	}
	ratio := float64(fromRate) / float64(toRate)
	outLen := int(float64(len(in)) / ratio)
	out := make([]float32, outLen)
	for i := range out {
		srcPos := float64(i) * ratio
		i0 := int(srcPos)
		frac := float32(srcPos - float64(i0))
		if i0+1 < len(in) {
			out[i] = in[i0]*(1-frac) + in[i0+1]*frac
		} else if i0 < len(in) {
			out[i] = in[i0]
		}
	}
	return out
}
