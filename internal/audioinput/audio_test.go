package audioinput

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestLoad16kMonoWAV(t *testing.T) {
	// Exercise the native path independently of an installed FFmpeg.
	t.Setenv("PATH", t.TempDir())
	floatPCM := make([]byte, 16)
	for i, v := range []float32{.5, -.25, -1, 1} {
		binary.LittleEndian.PutUint32(floatPCM[4*i:], math.Float32bits(v))
	}
	for _, tc := range []struct {
		name         string
		format, bits uint16
		pcm          []byte
		want         []float32
	}{
		{"PCM16 stereo", 1, 16, []byte{0, 64, 0, 192, 0, 128, 255, 127}, []float32{0, -1.0 / 65536}},
		{"float32 stereo", 3, 32, floatPCM, []float32{.125, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "audio.wav")
			if err := os.WriteFile(path, testWAV(tc.format, tc.bits, 2, 16000, tc.pcm), 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := Load16kMono(path)
			if err != nil || !slices.Equal(got, tc.want) {
				t.Fatalf("Load16kMono = %v, %v; want %v", got, err, tc.want)
			}
		})
	}
}

func TestLoad16kMonoResamplesWAV(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	path := filepath.Join(t.TempDir(), "audio.wav")
	if err := os.WriteFile(path, testWAV(1, 16, 1, 8000, []byte{0, 0, 0, 64, 0, 0}), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := Load16kMono(path)
	want := []float32{0, .25, .5, .25, 0, 0}
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("Load16kMono = %v, %v; want %v", got, err, want)
	}
	if got := linearResample(want, 16000, 8000); !slices.Equal(got, []float32{0, .5, 0}) {
		t.Fatalf("downsample = %v", got)
	}
}

func TestWAVErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"not WAV", []byte("not a WAV file"), "not a WAV file"},
		{"missing chunks", []byte("RIFF\x04\x00\x00\x00WAVE"), "missing fmt/data chunks"},
		{"unsupported PCM8", testWAV(1, 8, 1, 16000, []byte{128}), "unsupported WAV format"},
		{"zero sample rate", testWAV(1, 16, 1, 0, []byte{0, 0}), "missing fmt/data chunks"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "audio.wav")
			if err := os.WriteFile(path, tc.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := decodeWAVFallback(path); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("decodeWAVFallback = %v; want %q", err, tc.want)
			}
		})
	}
}

func TestWAVSkipsUnknownChunks(t *testing.T) {
	data := testWAV(1, 16, 1, 16000, []byte{0, 64})
	// The three-byte metadata chunk has one byte of RIFF alignment padding.
	metadata := []byte("JUNK\x03\x00\x00\x00abc\x00")
	data = append(append(append([]byte(nil), data[:12]...), metadata...), data[12:]...)
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	// Bytes outside the declared RIFF container must not become extra chunks.
	data = append(data, []byte("data\xff\xff\xff\xff")...)
	path := filepath.Join(t.TempDir(), "audio.wav")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := decodeWAVFallback(path)
	if err != nil || !slices.Equal(got, []float32{.5}) {
		t.Fatalf("decodeWAVFallback = %v, %v", got, err)
	}
}

func TestWAVRejectsTruncation(t *testing.T) {
	valid := testWAV(1, 16, 1, 16000, []byte{0, 64})
	appendChunk := func(chunk []byte) []byte {
		data := append(slices.Clone(valid), chunk...)
		binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
		return data
	}
	oversizedData := slices.Clone(valid)
	binary.LittleEndian.PutUint32(oversizedData[40:], 4)
	for _, tc := range []struct {
		name string
		data []byte
		want string
	}{
		{"RIFF header", valid[:10], "reading WAV header"},
		{"container", valid[:len(valid)-1], "unexpected EOF"},
		{"chunk header", appendChunk([]byte("JUNK")), "chunk header"},
		{"data exceeds container", oversizedData, "unexpected EOF"},
		{"unknown exceeds container", appendChunk([]byte("JUNK\xff\xff\xff\xff")), "unexpected EOF"},
		{"missing odd padding", appendChunk([]byte("JUNK\x03\x00\x00\x00abc")), "unexpected EOF"},
		{"partial PCM16 sample", testWAV(1, 16, 1, 16000, []byte{0}), "partial sample frame"},
		{"partial stereo frame", testWAV(1, 16, 2, 16000, []byte{0, 0}), "partial sample frame"},
		{"partial float32 sample", testWAV(3, 32, 1, 16000, []byte{0, 0, 0}), "partial sample frame"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "audio.wav")
			if err := os.WriteFile(path, tc.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := decodeWAVFallback(path); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("decodeWAVFallback = %v; want %q", err, tc.want)
			}
		})
	}
}

func TestWAVRejectsDuplicateChunksAndNonFiniteSamples(t *testing.T) {
	valid := testWAV(1, 16, 1, 16000, []byte{0, 0})
	for _, chunk := range [][]byte{valid[12:36], valid[36:]} {
		data := append(slices.Clone(valid), chunk...)
		binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
		path := filepath.Join(t.TempDir(), "duplicate.wav")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := decodeWAVFallback(path); err == nil {
			t.Fatalf("accepted duplicate %q chunk", chunk[:4])
		}
	}
	data := testWAV(3, 32, 1, 16000, []byte{0, 0, 128, 127}) // +Inf
	path := filepath.Join(t.TempDir(), "infinite.wav")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeWAVFallback(path); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite PCM: %v", err)
	}
}

func testWAV(format, bits, channels uint16, rate uint32, pcm []byte) []byte {
	data := make([]byte, 44+len(pcm)+len(pcm)%2)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], format)
	binary.LittleEndian.PutUint16(data[22:], channels)
	binary.LittleEndian.PutUint32(data[24:], rate)
	frameBytes := channels * bits / 8
	binary.LittleEndian.PutUint32(data[28:], rate*uint32(frameBytes))
	binary.LittleEndian.PutUint16(data[32:], frameBytes)
	binary.LittleEndian.PutUint16(data[34:], bits)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], uint32(len(pcm)))
	copy(data[44:], pcm)
	return data
}
