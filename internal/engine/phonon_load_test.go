package engine

import (
	"encoding/binary"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/phonon"
)

func TestPhononTensorMappingAndShape(t *testing.T) {
	for _, tc := range []struct {
		parakeet, native string
		shape            []int
	}{
		{"encoder.pre_encode.conv.0.weight", "encoder.subsampling.layers.0.weight", []int{256, 1, 3, 3}},
		{"encoder.layers.23.self_attn.linear_pos.weight", "encoder.layers.23.self_attn.relative_k_proj.weight", []int{1024, 1024}},
		{"encoder.layers.1.self_attn.pos_bias_v", "encoder.layers.1.self_attn.bias_v", []int{8, 128}},
		{"encoder.layers.2.conv.batch_norm.running_var", "encoder.layers.2.conv.norm.running_var", []int{1024}},
		{"decoder.prediction.dec_rnn.lstm.ih_l0.weight", "decoder.lstm.weight_ih_l0", []int{2560, 640}},
		{"decoder.prediction.dec_rnn.lstm.hh_l1.bias", "decoder.lstm.bias_hh_l1", []int{2560}},
		{"joint.pred.weight", "decoder.decoder_projector.weight", []int{640, 640}},
		{"joint.joint_net.2.weight", "joint.head.weight", []int{8198, 640}},
	} {
		t.Run(tc.native, func(t *testing.T) {
			model := &phonon.Model{Tensors: map[string]phonon.Tensor{
				tc.native: {Shape: tc.shape},
			}}
			if _, err := phononTensor(model, tc.parakeet, tc.shape); err != nil {
				t.Fatal(err)
			}
			// Allow neither an accidentally flattened weight nor a missing
			// tensor to flow through to the FastConformer graph.
			if _, err := phononTensor(model, tc.parakeet, append([]int{1}, tc.shape...)); err == nil ||
				!strings.Contains(err.Error(), "shape") {
				t.Fatalf("expected shape error, got %v", err)
			}
			delete(model.Tensors, tc.native)
			if _, err := phononTensor(model, tc.parakeet, tc.shape); err == nil ||
				!strings.Contains(err.Error(), tc.native) {
				t.Fatalf("expected missing native-tensor error, got %v", err)
			}
		})
	}
}

func TestPhononPositionTable(t *testing.T) {
	got := phononPositionTable(4, 2)
	if !slices.Equal(got[4:8], []float32{0, 1, 0, 1}) {
		t.Fatalf("zero relative position has wrong values: %v", got[4:8])
	}
	if math.Abs(float64(got[0])-math.Sin(1)) > 1e-7 ||
		math.Abs(float64(got[2])-math.Sin(0.01)) > 1e-7 ||
		got[8] != -got[0] || got[9] != got[1] {
		t.Fatalf("positive/negative relative positions have wrong order or frequencies: %v", got)
	}
}

func TestParakeetF16ProjectionBatchMatchesF32(t *testing.T) {
	raw, expanded := make([]byte, 12), make([]float32, 6)
	for i, bits := range []uint16{0x3c00, 0x3400, 0xbe00, 0x1, 0, 0x4000} {
		binary.LittleEndian.PutUint16(raw[2*i:], bits)
		expanded[i] = F16ToF32(bits)
	}
	x := [][]float32{{1, -2, 3}, {-0.5, 2.5, 0}}
	got, want := audioMatrixRows(2, 2), audioMatrixRows(2, 2)
	parakeetMatvecBatch(Weight{Raw: raw, Type: GGMLTypeF16, Rows: 2, Cols: 3}, x, got)
	blasMatvecBatch(Weight{F32: expanded}, x, want)
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("frame %d = %v, want %v", i, got[i], want[i])
		}
	}
}
