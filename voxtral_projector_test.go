package gopherllm

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestVoxtralProjectorQuantizedMatchesPerGroupReference(t *testing.T) {
	const dim, hidden, ds = 128, 256, 2
	cfg := VoxtralRealtimeConfig{Encoder: VoxtralRealtimeEncoderConfig{DModel: dim}, Decoder: VoxtralRealtimeDecoderConfig{HiddenSize: hidden}, Projector: VoxtralRealtimeProjectorConfig{InputDim: dim * ds, DownsampleFactor: ds}}
	weight := func(seed int) Weight {
		data := make([]byte, hidden*(hidden/32)*34)
		for b := 0; b < len(data)/34; b++ {
			binary.LittleEndian.PutUint16(data[b*34:], 0x2000) // exactly 1/128
			for j := 0; j < 32; j++ {
				data[b*34+2+j] = byte(int8((b*7+j*3+seed)%19 - 9))
			}
		}
		return Weight{Type: GGMLTypeQ8_0, Raw: data, Rows: hidden, Cols: hidden}
	}
	w := VoxtralRealtimeWeights{Projector: VoxtralRealtimeProjectorWeights{Linear1: weight(1), Linear2: weight(3)}}
	encoder := make([][]float32, 11) // a remainder frame must be dropped
	for i := range encoder {
		encoder[i] = make([]float32, dim)
		for j := range encoder[i] {
			encoder[i][j] = float32((i*17+j*7)%23-11) / 11
		}
	}
	for _, q8 := range []bool{false, true} {
		withQ8Activations(q8, func() {
			want := referenceVoxtralProjector(cfg, w, encoder)
			got, err := projectAudioVoxtralRealtime(cfg, w, encoder)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				for j, v := range want[i] {
					if d := math.Abs(float64(got[i][j] - v)); d > 1e-5*(1+math.Abs(float64(v))) {
						t.Fatalf("q8=%v [%d][%d]: got %g, want %g", q8, i, j, got[i][j], v)
					}
				}
			}
		})
	}
}

func referenceVoxtralProjector(cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights, encoder [][]float32) [][]float32 {
	ds, dim := cfg.Projector.DownsampleFactor, cfg.Encoder.DModel
	result := make([][]float32, len(encoder)/ds)
	concat := make([]float32, cfg.Projector.InputDim)
	var hidden []float32
	for g := range result {
		for j := 0; j < ds; j++ {
			copy(concat[j*dim:(j+1)*dim], encoder[g*ds+j])
		}
		w.Projector.Linear1.MatvecInto(concat, &hidden)
		for i, v := range hidden {
			hidden[i] = geluTanh(v)
		}
		w.Projector.Linear2.MatvecInto(hidden, &result[g])
	}
	return result
}

func TestVoxtralProjectorMatchesPerGroupReference(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	for _, n := range []int{2, 5, 18} {
		encoder := make([][]float32, n)
		for i := range encoder {
			encoder[i] = make([]float32, cfg.Encoder.DModel)
			for j := range encoder[i] {
				encoder[i][j] = float32((i*17+j*7)%23-11) / 11
			}
		}
		want := referenceVoxtralProjector(cfg, w, encoder)
		got, err := projectAudioVoxtralRealtime(cfg, w, encoder)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			for j, v := range want[i] {
				if d := math.Abs(float64(got[i][j] - v)); d > 1e-6 {
					t.Fatalf("n=%d [%d][%d]: got %g, want %g", n, i, j, got[i][j], v)
				}
			}
		}
	}
}

func BenchmarkVoxtralProjector(b *testing.B) {
	const dim, hidden, ds, groups = 1280, 3072, 4, 24
	cfg := VoxtralRealtimeConfig{Encoder: VoxtralRealtimeEncoderConfig{DModel: dim}, Decoder: VoxtralRealtimeDecoderConfig{HiddenSize: hidden}, Projector: VoxtralRealtimeProjectorConfig{InputDim: dim * ds, DownsampleFactor: ds}}
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(i%19-9) / 200
		}
		return v
	}
	w := VoxtralRealtimeWeights{Projector: VoxtralRealtimeProjectorWeights{Linear1: Weight{F32: fill(hidden * dim * ds)}, Linear2: Weight{F32: fill(hidden * hidden)}}}
	xs := make([][]float32, groups*ds)
	for i := range xs {
		xs[i] = fill(dim)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := projectAudioVoxtralRealtime(cfg, w, xs); err != nil {
			b.Fatal(err)
		}
	}
}
