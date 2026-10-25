package gopherllm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// WriteWAV writes mono 16-bit PCM with a RIFF/WAVE header. Samples outside
// [-1,1] are clipped; non-finite samples are rejected before writing anything.
// Conversion uses a small buffer rather than copying the entire recording.
func (a SpeechAudio) WriteWAV(w io.Writer) error {
	if a.SampleRate <= 0 || uint64(a.SampleRate) > (1<<32-1)/2 {
		return fmt.Errorf("WAV: invalid sample rate %d", a.SampleRate)
	}
	if uint64(len(a.Samples)) > (1<<32-1-36)/2 {
		return fmt.Errorf("WAV: recording exceeds the RIFF size limit")
	}
	if !numeric.AllFiniteF32(a.Samples) {
		return fmt.Errorf("WAV: non-finite sample")
	}
	var header [44]byte
	copy(header[:], "RIFF")
	dataSize := uint32(len(a.Samples)) * 2
	binary.LittleEndian.PutUint32(header[4:], 36+dataSize)
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], uint32(a.SampleRate))
	binary.LittleEndian.PutUint32(header[28:], uint32(a.SampleRate)*2)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], dataSize)
	write := func(data []byte) error {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n != len(data) {
			return io.ErrShortWrite
		}
		return nil
	}
	if err := write(header[:]); err != nil {
		return err
	}
	var buffer [4096]byte
	for start := 0; start < len(a.Samples); start += len(buffer) / 2 {
		count := min(len(buffer)/2, len(a.Samples)-start)
		for i, value := range a.Samples[start : start+count] {
			value = max(-1, min(1, value))
			pcm := int16(math.Round(float64(value) * 32767))
			if value == -1 {
				pcm = -32768
			}
			binary.LittleEndian.PutUint16(buffer[2*i:], uint16(pcm))
		}
		if err := write(buffer[:count*2]); err != nil {
			return err
		}
	}
	return nil
}
