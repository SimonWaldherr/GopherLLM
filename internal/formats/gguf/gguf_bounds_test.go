package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func boundsTestHeader(tensors, metadata uint64, tail []byte) []byte {
	data := []byte{'G', 'G', 'U', 'F'}
	data = binary.LittleEndian.AppendUint32(data, 3)
	data = binary.LittleEndian.AppendUint64(data, tensors)
	data = binary.LittleEndian.AppendUint64(data, metadata)
	data = append(data, tail...)
	return append(data, make([]byte, 64)...)
}

func boundsTestMetadata(typ uint32, value []byte) []byte {
	data := binary.LittleEndian.AppendUint64(nil, 1)
	data = append(data, 'x')
	data = binary.LittleEndian.AppendUint32(data, typ)
	return append(data, value...)
}

func TestParseRejectsUnboundedAllocations(t *testing.T) {
	badRank := binary.LittleEndian.AppendUint64(nil, 1)
	badRank = append(badRank, 'w')
	badRank = binary.LittleEndian.AppendUint32(badRank, math.MaxUint32)
	cases := map[string][]byte{
		"tensor_count":   boundsTestHeader(math.MaxUint64, 0, nil),
		"metadata_count": boundsTestHeader(0, math.MaxUint64, nil),
		"tensor_rank":    boundsTestHeader(1, 0, badRank),
		"key_length":     boundsTestHeader(0, 1, binary.LittleEndian.AppendUint64(nil, math.MaxUint64)),
		"string_length":  boundsTestHeader(0, 1, boundsTestMetadata(8, binary.LittleEndian.AppendUint64(nil, uint64(^uint(0)>>1)))),
	}
	for _, typ := range []uint32{0, 6, 7, 8, 10} {
		value := binary.LittleEndian.AppendUint32(nil, typ)
		value = binary.LittleEndian.AppendUint64(value, math.MaxUint64)
		cases["array_count_"+string(rune('a'+typ))] = boundsTestHeader(0, 1, boundsTestMetadata(9, value))
	}
	var nested []byte
	for range 17 {
		nested = binary.LittleEndian.AppendUint32(nested, 9)
		nested = binary.LittleEndian.AppendUint64(nested, 1)
	}
	nested = binary.LittleEndian.AppendUint32(nested, 7)
	nested = binary.LittleEndian.AppendUint64(nested, 1)
	nested = append(nested, 1)
	cases["nested_arrays"] = boundsTestHeader(0, 1, boundsTestMetadata(9, nested))
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseQuiet(data); err == nil {
				t.Fatal("accepted malformed GGUF header")
			}
		})
	}
}

func TestCursorNeedRejectsAdditionOverflow(t *testing.T) {
	c := cursor{data: make([]byte, 16), pos: 8}
	if err := c.need(int(^uint(0) >> 1)); err == nil {
		t.Fatal("accepted a size that overflows pos + n")
	}
	if err := c.need(8); err != nil {
		t.Fatal(err)
	}
}
