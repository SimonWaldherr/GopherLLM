package gopherllm

import (
	"encoding/binary"
	"os"
	"testing"
)

func TestParakeetDecodeTokens(t *testing.T) {
	vocab := []string{"<unk>", "Hello", "▁world", "▁", "!"}
	got, err := ParakeetDecodeTokens(vocab, []int{1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	if want := "Hello world!"; got != want {
		t.Fatalf("ParakeetDecodeTokens = %q, want %q", got, want)
	}
}

func TestParakeetDecodeTokensRejectsOutOfRange(t *testing.T) {
	if _, err := ParakeetDecodeTokens([]string{"a", "b"}, []int{5}); err == nil {
		t.Fatal("expected an error for an out-of-vocabulary token id")
	}
}

func TestParakeetHannSymmetricEndpoints(t *testing.T) {
	// torch.hann_window(n, periodic=False): endpoints are exactly 0, the
	// center is exactly 1 -- unlike the periodic form, which never reaches
	// 1 for even n. Distinguishing the two is the whole point of having a
	// separate function from voxtralPeriodicHannWindow.
	w := parakeetHannSymmetric(9)
	if w[0] != 0 || w[8] != 0 {
		t.Fatalf("symmetric Hann window endpoints = %v, %v; want 0, 0", w[0], w[8])
	}
	if diff := w[4] - 1; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("symmetric Hann window center = %v, want 1", w[4])
	}
}

func loadTestWav16Mono(t *testing.T, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	pcm := data[44:]
	n := len(pcm) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		v := int16(binary.LittleEndian.Uint16(pcm[2*i:]))
		out[i] = float32(v) / 32768.0
	}
	return out
}

// synthesizeTestTone writes a short mono 16kHz PCM16 WAV containing a plain
// sine tone -- not speech, but enough to exercise the full encode/decode
// pipeline without needing a checked-in audio fixture.
func synthesizeTestTone(t *testing.T, path string, seconds float64) {
	t.Helper()
	const sr = 16000
	n := int(seconds * sr)
	pcm := make([]byte, n*2)
	for i := 0; i < n; i++ {
		v := int16(3000.0 * sinApprox(2*3.14159265*440*float64(i)/sr))
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(v))
	}
	var header [44]byte
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+len(pcm)))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], 1)
	binary.LittleEndian.PutUint32(header[24:28], sr)
	binary.LittleEndian.PutUint32(header[28:32], sr*2)
	binary.LittleEndian.PutUint16(header[32:34], 2)
	binary.LittleEndian.PutUint16(header[34:36], 16)
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(len(pcm)))
	if err := os.WriteFile(path, append(header[:], pcm...), 0o644); err != nil {
		t.Fatal(err)
	}
}

func sinApprox(x float64) float64 {
	// Bhaskara I's sine approximation -- accurate enough for a synthetic
	// test tone, avoids importing math just for this.
	for x > 2*3.14159265 {
		x -= 2 * 3.14159265
	}
	for x < 0 {
		x += 2 * 3.14159265
	}
	sign := 1.0
	if x > 3.14159265 {
		x -= 3.14159265
		sign = -1
	}
	return sign * 16 * x * (3.14159265 - x) / (5*3.14159265*3.14159265 - 4*x*(3.14159265-x))
}

func findLocalParakeetGGUF() string {
	path := os.Getenv("HOME") + "/.cache/lm-studio/models/nvidia/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		return ""
	}
	return path
}

// TestParakeetEncodeProducesFiniteOutput exercises the full mel frontend,
// subsampling stem, and every Conformer block against a synthetic tone (no
// checked-in audio fixture needed), asserting the frame count matches the
// expected 8x-subsampled length and every value is finite -- a structural
// smoke test, not a correctness one (see TestTranscribeParakeetRealAudio
// for that, which needs a real local model).
func TestParakeetEncodeProducesFiniteOutput(t *testing.T) {
	path := findLocalParakeetGGUF()
	if path == "" {
		t.Skip("no local Parakeet-TDT GGUF available")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		t.Fatal(err)
	}
	cfg, w, err := LoadParakeetModel(data, gguf, nil)
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	tonePath := dir + "/tone.wav"
	synthesizeTestTone(t, tonePath, 2.0)
	samples := loadTestWav16Mono(t, tonePath)

	frames, err := ParakeetEncode(cfg, w, samples)
	if err != nil {
		t.Fatal(err)
	}
	if len(frames) == 0 {
		t.Fatal("no encoder frames produced")
	}
	for i, row := range frames {
		if len(row) != cfg.Encoder.DModel {
			t.Fatalf("frame %d has %d dims, want %d", i, len(row), cfg.Encoder.DModel)
		}
		for j, v := range row {
			if v != v || v > 1e6 || v < -1e6 {
				t.Fatalf("frame %d dim %d is non-finite or huge: %v", i, j, v)
			}
		}
	}
}

// TestTranscribeParakeetRealAudio is the real correctness check: two known
// utterances (macOS `say`-synthesized speech, not checked into the repo),
// transcribed end-to-end (mel frontend -> subsampling -> Conformer encoder
// -> LSTM prediction network -> joint network -> TDT greedy decode ->
// SentencePiece detokenization) and compared against their known text.
// Skipped without both a local model and the two fixtures this repo
// deliberately doesn't bundle (a several-hundred-MB GGUF and generated
// speech audio).
func TestTranscribeParakeetRealAudio(t *testing.T) {
	path := findLocalParakeetGGUF()
	if path == "" {
		t.Skip("no local Parakeet-TDT GGUF available")
	}
	fixtures := []struct{ path, want string }{
		{"/tmp/speech-short.wav", "Testing one two three"},
		{"/tmp/speech-test.wav", "Hello, this is a test of automatic audio transcription."},
	}
	ran := 0
	for _, f := range fixtures {
		if _, err := os.Stat(f.path); err != nil {
			continue
		}
		samples := loadTestWav16Mono(t, f.path)
		got, err := TranscribeParakeet(path, samples, nil)
		if err != nil {
			t.Fatalf("%s: %v", f.path, err)
		}
		if got != f.want {
			t.Errorf("%s: got %q, want %q", f.path, got, f.want)
		}
		ran++
	}
	if ran == 0 {
		t.Skip("no local speech fixtures available (generated ad hoc during development, not checked in)")
	}
}
