package main

import (
	"encoding/binary"
	"math"
	"slices"
	"testing"
)

func appendProtoBytes(dst []byte, field uint64, contents []byte) []byte {
	dst = binary.AppendUvarint(dst, field<<3|2)
	dst = binary.AppendUvarint(dst, uint64(len(contents)))
	return append(dst, contents...)
}

func TestSentencePieceVocabulary(t *testing.T) {
	var model []byte
	for i, term := range []string{"<unk>", "▁", "▁a"} {
		piece := appendProtoBytes(nil, 1, []byte(term))
		piece = append(piece, 2<<3|5)
		piece = binary.LittleEndian.AppendUint32(piece, math.Float32bits(float32(-i)))
		if i == 0 {
			piece = append(piece, 3<<3, 2)
		}
		model = appendProtoBytes(model, 1, piece)
	}
	model = appendProtoBytes(model, 2, []byte{3 << 3, 2})
	got, err := loadSentencePiece(model)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.pieces, []string{"<unk>", "▁", "▁a"}) ||
		!slices.Equal(got.scores, []float32{0, -1, -2}) ||
		!slices.Equal(got.types, []uint32{2, 1, 1}) {
		t.Fatalf("unexpected tokenizer vocabulary %+v", got)
	}
	if _, err := loadSentencePiece(model[:len(model)-1]); err == nil {
		t.Fatal("truncated SentencePiece model accepted")
	}
}
