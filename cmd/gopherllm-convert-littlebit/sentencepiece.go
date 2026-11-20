package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Visit the protobuf fields needed by the vocabulary importer. Scalar bytes
// stay borrowed until each piece is converted to its GGUF string.
func protoFields(data []byte, visit func(uint64, uint64, []byte) error) error {
	for len(data) > 0 {
		tag, n := binary.Uvarint(data)
		if n <= 0 || tag>>3 == 0 {
			return fmt.Errorf("invalid SentencePiece protobuf tag")
		}
		data = data[n:]
		wire := tag & 7
		value := uint64(0)
		size := 0
		switch wire {
		case 0:
			value, n = binary.Uvarint(data)
			if n <= 0 {
				return fmt.Errorf("invalid SentencePiece varint")
			}
			size = n
		case 1:
			size = 8
		case 2:
			value, n = binary.Uvarint(data)
			if n <= 0 || value > uint64(len(data)-n) {
				return fmt.Errorf("invalid SentencePiece field length")
			}
			size, data = int(value), data[n:]
		case 5:
			size = 4
		default:
			return fmt.Errorf("unsupported SentencePiece wire type %d", wire)
		}
		if size > len(data) {
			return fmt.Errorf("truncated SentencePiece field")
		}
		if err := visit(tag>>3, value, data[:size]); err != nil {
			return err
		}
		data = data[size:]
	}
	return nil
}

func loadSentencePiece(raw []byte) (tokenTable, error) {
	// ModelProto repeats SentencePiece entries in field 1; its TrainerSpec
	// (field 2) identifies BPE as model_type=2.
	var t tokenTable
	var bpe bool
	err := protoFields(raw, func(field, _ uint64, payload []byte) error {
		switch field {
		case 1:
			piece, score, kind := "", float32(0), uint32(1)
			err := protoFields(payload, func(field, value uint64, data []byte) error {
				switch field {
				case 1:
					piece = string(data)
				case 2:
					if len(data) != 4 {
						return fmt.Errorf("invalid SentencePiece score")
					}
					score = math.Float32frombits(binary.LittleEndian.Uint32(data))
				case 3:
					if value < 1 || value > 6 {
						return fmt.Errorf("invalid SentencePiece token type %d", value)
					}
					kind = uint32(value)
				}
				return nil
			})
			if err != nil {
				return err
			}
			if piece == "" {
				return fmt.Errorf("empty SentencePiece vocabulary entry")
			}
			t.pieces = append(t.pieces, piece)
			t.scores = append(t.scores, score)
			t.types = append(t.types, kind)
		case 2:
			return protoFields(payload, func(trainerField, value uint64, _ []byte) error {
				if trainerField == 3 {
					bpe = value == 2
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return tokenTable{}, err
	}
	if !bpe || len(t.pieces) == 0 {
		return tokenTable{}, fmt.Errorf("expected a SentencePiece BPE model")
	}
	return t, nil
}
