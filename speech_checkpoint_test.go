package gopherllm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"runtime"
	"testing"
	"time"
)

// Optional local comparison harness. PCM is headerless mono float32 little
// endian at 16 kHz; neither model nor audio is downloaded by the test.
func TestSpeechTranscriptionCheckpoint(t *testing.T) {
	path, audio := os.Getenv("GOPHERLLM_STT_MODEL"), os.Getenv("GOPHERLLM_STT_PCM")
	if path == "" || audio == "" {
		t.Skip("set GOPHERLLM_STT_MODEL (Parakeet) and GOPHERLLM_STT_PCM (16 kHz f32le)")
	}
	setPerformanceThreads(t, 8)
	data, err := os.ReadFile(audio)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 || len(data)%4 != 0 {
		t.Fatal("invalid f32le PCM size")
	}
	samples := make([]float32, len(data)/4)
	for i := range samples {
		samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(samples[i])) || math.IsInf(float64(samples[i]), 0) {
			t.Fatal("non-finite audio")
		}
	}
	m, err := OpenParakeet(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var first string
	for run := range 3 {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		start := time.Now()
		frames, err := ParakeetEncode(m.cfg, m.weights, samples)
		if err != nil {
			t.Fatal(err)
		}
		encoded := time.Now()
		tokens := ParakeetGreedyDecodeTDT(m.cfg, m.weights, frames)
		text, err := ParakeetDecodeTokens(m.weights.Vocab, tokens)
		if err != nil {
			t.Fatal(err)
		}
		finished := time.Now()
		runtime.ReadMemStats(&after)
		h := sha256.New()
		var bits [4]byte
		for _, row := range frames {
			for _, value := range row {
				binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
				h.Write(bits[:])
			}
		}
		digest := hex.EncodeToString(h.Sum(nil))
		if run == 0 {
			first = digest
		} else if digest != first {
			t.Fatal("encoder output changed across repeated requests")
		}
		if want := os.Getenv("GOPHERLLM_STT_EXPECT_TEXT"); want != "" && text != want {
			t.Fatalf("transcript %q, want %q", text, want)
		}
		record, err := json.Marshal(map[string]any{"run": run, "audio_seconds": float64(len(samples)) / 16000,
			"encode_ms":       float64(encoded.Sub(start)) / float64(time.Millisecond),
			"decode_ms":       float64(finished.Sub(encoded)) / float64(time.Millisecond),
			"total_ms":        float64(finished.Sub(start)) / float64(time.Millisecond),
			"allocated_bytes": after.TotalAlloc - before.TotalAlloc, "frames_sha256": digest, "tokens": tokens, "text": text})
		if err != nil {
			t.Fatal(err)
		}
		t.Log(string(record))
	}
}
