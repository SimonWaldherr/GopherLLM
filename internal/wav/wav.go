// Package wav reads canonical PCM16 and IEEE float32 WAV containers.
package wav

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

type Format struct {
	Encoding, Channels, BitsPerSample, BlockAlign uint16
	SampleRate                                    uint32
}

// File describes one RIFF/WAVE container. Inspect only reads chunk headers and
// the 16-byte format record; neither PCM nor unknown metadata is copied.
type File struct {
	Format
	DataOffset, DataSize int64
}

// Inspect accepts trailing bytes outside the declared RIFF container. Callers
// with stricter transport framing must check the container length separately.
// Duplicate fmt/data chunks are ambiguous and are rejected.
func Inspect(r io.ReaderAt, size int64) (File, error) {
	var header [16]byte
	if _, err := r.ReadAt(header[:12], 0); err != nil {
		return File{}, fmt.Errorf("reading WAV header: %w", err)
	}
	if string(header[:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return File{}, fmt.Errorf("not a WAV file")
	}
	end := int64(binary.LittleEndian.Uint32(header[4:])) + 8
	if end < 12 {
		return File{}, fmt.Errorf("WAV RIFF size is smaller than the WAVE header")
	}
	if end > size {
		return File{}, fmt.Errorf("reading WAV container: %w", io.ErrUnexpectedEOF)
	}
	var file File
	var haveFormat, haveData bool
	for offset := int64(12); offset < end; {
		if end-offset < 8 {
			return File{}, fmt.Errorf("reading WAV chunk header: %w", io.ErrUnexpectedEOF)
		}
		if _, err := r.ReadAt(header[:8], offset); err != nil {
			return File{}, fmt.Errorf("reading WAV chunk header: %w", err)
		}
		length := int64(binary.LittleEndian.Uint32(header[4:8]))
		start := offset + 8
		if length > end-start {
			return File{}, fmt.Errorf("reading WAV chunk %q: %w", header[:4], io.ErrUnexpectedEOF)
		}
		next := start + length + length%2
		if next > end {
			return File{}, fmt.Errorf("missing WAV chunk padding: %w", io.ErrUnexpectedEOF)
		}
		switch string(header[:4]) {
		case "fmt ":
			if haveFormat || length < 16 {
				return File{}, fmt.Errorf("invalid WAV fmt chunk")
			}
			if _, err := r.ReadAt(header[:], start); err != nil {
				return File{}, fmt.Errorf("reading WAV fmt chunk: %w", err)
			}
			file.Format = Format{
				Encoding: binary.LittleEndian.Uint16(header[:]), Channels: binary.LittleEndian.Uint16(header[2:]),
				SampleRate: binary.LittleEndian.Uint32(header[4:]), BlockAlign: binary.LittleEndian.Uint16(header[12:]),
				BitsPerSample: binary.LittleEndian.Uint16(header[14:]),
			}
			haveFormat = true
		case "data":
			if haveData {
				return File{}, fmt.Errorf("multiple WAV data chunks are unsupported")
			}
			file.DataOffset, file.DataSize = start, length
			haveData = true
		}
		offset = next
	}
	if !haveFormat || !haveData {
		return File{}, fmt.Errorf("WAV file missing fmt/data chunks")
	}
	return file, nil
}

// PCM reports whether the record describes supported integer or float samples.
// Channel/rate restrictions and block alignment are caller-specific policies.
func (f Format) PCM() bool {
	return f.Encoding == 1 && f.BitsPerSample == 16 || f.Encoding == 3 && f.BitsPerSample == 32
}

// DecodeMono converts complete interleaved frames to mono. Float samples must
// be finite. It preserves PCM16 scaling and averages channels in input order.
func DecodeMono(data []byte, format Format) ([]float32, error) {
	frames, _, err := monoShape(format, int64(len(data)))
	if err != nil {
		return nil, err
	}
	mono := make([]float32, frames)
	if err := decodeMonoInto(mono, data, format); err != nil {
		return nil, err
	}
	return mono, nil
}

// ReadMono decodes the inspected data chunk without retaining a second copy of
// the recording. Scratch holds at most 32 KiB, or one frame if it is larger.
func (f File) ReadMono(r io.ReaderAt) ([]float32, error) {
	if f.DataOffset < 0 || f.DataSize < 0 || f.DataOffset > math.MaxInt64-f.DataSize {
		return nil, fmt.Errorf("invalid WAV data range")
	}
	frames, frameBytes, err := monoShape(f.Format, f.DataSize)
	if err != nil {
		return nil, err
	}
	mono := make([]float32, frames)
	if frames == 0 {
		return mono, nil
	}
	tileFrames := min(frames, max(1, (32<<10)/frameBytes))
	data := make([]byte, tileFrames*frameBytes)
	for start := 0; start < frames; start += tileFrames {
		count := min(tileFrames, frames-start)
		tile := data[:count*frameBytes]
		if _, err := r.ReadAt(tile, f.DataOffset+int64(start)*int64(frameBytes)); err != nil {
			return nil, fmt.Errorf("reading WAV data chunk: %w", err)
		}
		if err := decodeMonoInto(mono[start:start+count], tile, f.Format); err != nil {
			return nil, err
		}
	}
	return mono, nil
}

func monoShape(format Format, dataSize int64) (frames, frameBytes int, err error) {
	if !format.PCM() || format.Channels == 0 {
		return 0, 0, fmt.Errorf("unsupported WAV format (audioFormat=%d bitsPerSample=%d channels=%d)", format.Encoding, format.BitsPerSample, format.Channels)
	}
	frameBytes = int(format.Channels) * int(format.BitsPerSample/8)
	if dataSize < 0 || dataSize%int64(frameBytes) != 0 {
		return 0, 0, fmt.Errorf("WAV data ends in a partial sample frame")
	}
	count := dataSize / int64(frameBytes)
	if uint64(count) > uint64(^uint(0)>>1)/4 {
		return 0, 0, fmt.Errorf("decoded WAV exceeds addressable memory")
	}
	return int(count), frameBytes, nil
}

// Callers validate the format, complete frames and output length before use.
func decodeMonoInto(mono []float32, data []byte, format Format) error {
	channels := int(format.Channels)
	frameBytes := channels * int(format.BitsPerSample/8)
	for i := range mono {
		frame := data[i*frameBytes : (i+1)*frameBytes]
		if format.Encoding == 1 {
			if channels == 1 {
				mono[i] = float32(int16(binary.LittleEndian.Uint16(frame))) / 32768
				continue
			}
			var sum int32
			for c := range channels {
				sum += int32(int16(binary.LittleEndian.Uint16(frame[c*2:])))
			}
			mono[i] = float32(sum) / float32(channels) / 32768
		} else {
			if channels == 1 {
				// No channel mixing or division; retain the original sign bit.
				mono[i] = math.Float32frombits(binary.LittleEndian.Uint32(frame))
				continue
			}
			var sum float32
			for c := range channels {
				sum += math.Float32frombits(binary.LittleEndian.Uint32(frame[c*4:]))
			}
			mono[i] = sum / float32(channels)
		}
	}
	if format.Encoding == 3 && !numeric.AllFiniteF32(mono) {
		return fmt.Errorf("WAV contains non-finite samples")
	}
	return nil
}
