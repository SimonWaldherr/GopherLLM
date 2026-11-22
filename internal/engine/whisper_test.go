package engine

import (
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestWhisperRejectsUnsupportedCheckpoints(t *testing.T) {
	header := []uint32{0x67676d6c, 51865, 1500, 512, 8, 6, 448, 512, 8, 6, 80, 1}
	for _, tc := range []struct {
		name string
		edit func([]byte) []byte
		want string
	}{
		{"wrong magic", func(b []byte) []byte { b[0] = 0; return b }, "legacy GGML"},
		{"short header", func(b []byte) []byte { return b[:16] }, "truncated"},
		{"quantized", func(b []byte) []byte { b[44] = 2; return b }, "precision"},
		{"wrong heads", func(b []byte) []byte { b[16] = 0; return b }, "dimensions"},
		{"missing filters", func(b []byte) []byte { return b }, "filterbank"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := binary.LittleEndian.AppendUint32(nil, header[0])
			for _, v := range header[1:] {
				data = binary.LittleEndian.AppendUint32(data, v)
			}
			if m, err := parseWhisper(context.Background(), tc.edit(data)); m != nil || err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("model=%v, error=%v, want %q", m, err, tc.want)
			}
		})
	}
}

func TestWhisperAttentionUsesAllSuppliedKeys(t *testing.T) {
	q := audioMatrixRows(1, 4)
	copy(q[0], []float32{1, 1, 0, 1})
	key := []float32{1, 0, 1, 0, 0, 2, 0, 1}
	value := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	got := whisperAttend(context.Background(), q, key, value, 2)[0]
	p := float32(1 / (1 + math.Exp(-1/math.Sqrt2)))
	want := []float32{1 + 4*p, 2 + 4*p, 3 + 4*p, 4 + 4*p}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 1e-6 {
			t.Fatalf("attention[%d]=%g, want %g", i, v, want[i])
		}
	}
}

func TestWhisperLanguageTokens(t *testing.T) {
	m := WhisperModel{vocab: 51865, start: 50258, languages: 99}
	for lang, want := range map[string]int{"auto": -1, "de": 50261, "en": 50259, "su": 50357} {
		id, err := m.whisperLanguage(lang)
		if id != want || err != nil {
			t.Fatalf("%s maps to %d (%v), want %d", lang, id, err, want)
		}
	}
	if _, err := m.whisperLanguage("yue"); err == nil {
		t.Fatal("accepted v3-only language in a v2 vocabulary")
	}
}
