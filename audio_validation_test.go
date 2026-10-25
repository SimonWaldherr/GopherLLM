package gopherllm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestAudioValidationBeforeInference(t *testing.T) {
	ctx := context.Background()
	for _, input := range []struct {
		name    string
		samples []float32
		want    string
	}{
		{"empty", nil, "audio is empty"},
		{"NaN", []float32{float32(math.NaN())}, "audio contains non-finite samples"},
		{"positive infinity", []float32{float32(math.Inf(1))}, "audio contains non-finite samples"},
		{"negative infinity", []float32{float32(math.Inf(-1))}, "audio contains non-finite samples"},
	} {
		t.Run(input.name, func(t *testing.T) {
			for name, transcribe := range map[string]func() (string, error){
				"voxtral GGUF": func() (string, error) {
					return TranscribeVoxtralRealtime(ctx, "missing.gguf", input.samples, 0, nil)
				},
				"voxtral safetensors": func() (string, error) {
					return TranscribeVoxtralRealtimeFromSafetensors(ctx, "missing", input.samples, 0, nil)
				},
				"voxtral loaded": func() (string, error) {
					return decodeVoxtralRealtimeOffline(ctx, VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, nil, 0, 0, input.samples, 0, false, nil)
				},
				"parakeet GGUF": func() (string, error) {
					return TranscribeParakeet("missing.gguf", input.samples, nil)
				},
				"parakeet loaded": func() (string, error) {
					return new(ParakeetModel).TranscribeOffline(ctx, input.samples, nil)
				},
			} {
				if _, err := transcribe(); err == nil || err.Error() != input.want {
					t.Fatalf("%s: got %v, want %q before loading or encoding", name, err, input.want)
				}
			}
		})
	}
	// Empty streaming chunks are valid, unlike empty offline recordings.
	if err := validateAudioSamples(nil); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := validateVoxtralOfflineInput(canceled, nil, -1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation precedence: %v", err)
	}
	if err := validateVoxtralOfflineInput(ctx, []float32{float32(math.NaN())}, -1); err == nil || !strings.Contains(err.Error(), "max extra steps") {
		t.Fatalf("option validation precedence: %v", err)
	}
	if _, err := new(ParakeetModel).TranscribeOffline(canceled, nil, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("Parakeet cancellation precedence: %v", err)
	}
}

func TestAudioValidationRejectsVectorTailBeforeMutation(t *testing.T) {
	for _, length := range []int{16, 17, 31, 32, 33, 3200} {
		pcm := make([]float32, length)
		pcm[length-1] = float32(math.NaN())
		session := new(VoxtralRealtimeSession)
		if _, err := session.Push(context.Background(), pcm, true); err == nil {
			t.Fatalf("stream accepted NaN at index %d", length-1)
		}
		if session.audio != nil || session.decoder != nil || session.finished {
			t.Fatal("invalid input mutated streaming state")
		}
		var out bytes.Buffer
		if err := (SpeechAudio{Samples: pcm, SampleRate: 16000}).WriteWAV(&out); err == nil || out.Len() != 0 {
			t.Fatalf("WAV emitted data before validating tail: %v, %d bytes", err, out.Len())
		}
	}
}

func BenchmarkAudioSampleValidation(b *testing.B) {
	for _, size := range []int{16000, 30 * 16000} {
		for _, check := range []struct {
			name string
			fn   func([]float32) error
		}{
			{"scalar", func(samples []float32) error {
				for _, sample := range samples {
					if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
						return errors.New("audio contains non-finite samples")
					}
				}
				return nil
			}},
			{"shared", validateAudioSamples},
		} {
			b.Run(fmt.Sprintf("%d/%s", size, check.name), func(b *testing.B) {
				samples := make([]float32, size)
				b.SetBytes(int64(size * 4))
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if err := check.fn(samples); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
