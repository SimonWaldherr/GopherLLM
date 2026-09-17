package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
)

// loadAudio16kMono returns mono float32 PCM samples at 16kHz -- what
// gopherllm.EncodeAudioVoxtralRealtime expects. It prefers shelling out to
// ffmpeg (correct resampling/decoding for any input format ffmpeg supports);
// when ffmpeg isn't on PATH, it falls back to a minimal native WAV reader
// (PCM16/PCM32F only) with a naive linear-interpolation resampler, which is
// adequate for a first "does this work at all" test but not production
// quality.
func loadAudio16kMono(path string) ([]float32, error) {
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

	var riffHeader [12]byte
	if _, err := f.Read(riffHeader[:]); err != nil {
		return nil, fmt.Errorf("reading WAV header: %w", err)
	}
	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV file (and ffmpeg is unavailable to decode other formats): %s", path)
	}

	var sampleRate uint32
	var channels, bitsPerSample uint16
	var audioFormat uint16
	var pcmData []byte

	for {
		var chunkHeader [8]byte
		n, err := f.Read(chunkHeader[:])
		if n < 8 || err != nil {
			break
		}
		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])
		body := make([]byte, chunkSize)
		if _, err := f.Read(body); err != nil {
			return nil, fmt.Errorf("reading WAV chunk %q: %w", chunkID, err)
		}
		if chunkSize%2 == 1 {
			var pad [1]byte
			f.Read(pad[:])
		}
		switch chunkID {
		case "fmt ":
			if len(body) < 16 {
				return nil, fmt.Errorf("WAV fmt chunk too short")
			}
			audioFormat = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			sampleRate = binary.LittleEndian.Uint32(body[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
		case "data":
			pcmData = body
		}
	}
	if sampleRate == 0 || channels == 0 || bitsPerSample == 0 || pcmData == nil {
		return nil, fmt.Errorf("WAV file missing fmt/data chunks")
	}

	var mono []float32
	switch {
	case audioFormat == 1 && bitsPerSample == 16: // PCM16
		frames := len(pcmData) / (2 * int(channels))
		mono = make([]float32, frames)
		for i := range frames {
			var sum int32
			for c := range int(channels) {
				off := (i*int(channels) + c) * 2
				sum += int32(int16(binary.LittleEndian.Uint16(pcmData[off:])))
			}
			mono[i] = float32(sum) / float32(channels) / 32768
		}
	case audioFormat == 3 && bitsPerSample == 32: // IEEE float32
		frames := len(pcmData) / (4 * int(channels))
		mono = make([]float32, frames)
		for i := range frames {
			var sum float32
			for c := range int(channels) {
				off := (i*int(channels) + c) * 4
				sum += math.Float32frombits(binary.LittleEndian.Uint32(pcmData[off:]))
			}
			mono[i] = sum / float32(channels)
		}
	default:
		return nil, fmt.Errorf("unsupported WAV format (audioFormat=%d bitsPerSample=%d); install ffmpeg for broader format support", audioFormat, bitsPerSample)
	}

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
