package gopherllm

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"
)

func TestWriteCompressedTensorMatchesWholeTensorEncoding(t *testing.T) {
	const cols, rows = 256, 15
	values := smallWeights(cols*rows, 19)
	for _, source := range []GGMLType{GGMLTypeF32, GGMLTypeF16, GGMLTypeBF16, GGMLTypeF64, GGMLTypeQ8_0, GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(source.String(), func(t *testing.T) {
			var raw []byte
			switch source {
			case GGMLTypeF32:
				raw = f32Bytes(values)
			case GGMLTypeF16, GGMLTypeBF16:
				raw = make([]byte, 2*len(values))
				for i, v := range values {
					if source == GGMLTypeF16 {
						putF16(raw[i*2:], v)
					} else {
						binary.LittleEndian.PutUint16(raw[i*2:], uint16(math.Float32bits(v)>>16))
					}
				}
			case GGMLTypeF64:
				raw = make([]byte, 8*len(values))
				for i, v := range values {
					binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(float64(v)))
				}
			default:
				for row := 0; row < rows; row++ {
					raw = append(raw, quantizeRowFor(source)(values[row*cols:(row+1)*cols], cols)...)
				}
			}
			decoded, ok := tensorToF32(source, raw, len(values))
			if !ok {
				t.Fatalf("reference decoder does not support %s", source)
			}
			for _, target := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
				var got, want bytes.Buffer
				for row := 0; row < rows; row++ {
					want.Write(quantizeRowFor(target)(decoded[row*cols:(row+1)*cols], cols))
				}
				tensor := TensorInfo{Name: "blk.0.ffn_down_exps.weight", Dims: []uint64{cols, 3, 5}, DType: source}
				if err := writeCompressedTensor(context.Background(), &got, tensor, raw, target); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Bytes(), want.Bytes()) {
					t.Fatalf("%s -> %s differs from original whole-tensor encoding", source, target)
				}
			}
		})
	}
}

type compressTrackingWriter struct {
	writes int
	maxLen int
	bytes  int
	cancel context.CancelFunc
	err    error
	short  bool
}

func (w *compressTrackingWriter) Write(p []byte) (int, error) {
	w.writes++
	w.maxLen = max(w.maxLen, len(p))
	w.bytes += len(p)
	if w.cancel != nil {
		w.cancel()
	}
	if w.err != nil {
		return 0, w.err
	}
	if w.short {
		return len(p) - 1, nil
	}
	return len(p), nil
}

func TestWriteCompressedTensorChunksAndCancellation(t *testing.T) {
	const cols = 256
	rowBytes, _ := GGMLTypeQ8_0.DataSize(cols)
	chunkRows := compressChunkRows(1<<20, cols, rowBytes)
	rows := chunkRows + 3
	raw := make([]byte, rows*cols*2) // zero F16 values need no initialization
	tensor := TensorInfo{Name: "experts", Dims: []uint64{cols, uint64(rows)}, DType: GGMLTypeF16}
	w := &compressTrackingWriter{}
	if err := writeCompressedTensor(context.Background(), w, tensor, raw, GGMLTypeQ8_0); err != nil {
		t.Fatal(err)
	}
	if w.writes != 2 || w.bytes != rows*rowBytes || w.maxLen != chunkRows*rowBytes {
		t.Fatalf("unexpected chunk writes: %+v", w)
	}
	if chunkRows*(cols*4+rowBytes) > compressScratchBytes {
		t.Fatal("combined float/output scratch exceeds budget")
	}
	if n := compressChunkRows(5, compressScratchBytes, 1); n != 1 {
		t.Fatalf("oversized row chunk size = %d, want 1", n)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w = &compressTrackingWriter{cancel: cancel}
	if err := writeCompressedTensor(ctx, w, tensor, raw, GGMLTypeQ8_0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation error = %v", err)
	}
	if w.writes != 1 {
		t.Fatalf("wrote %d chunks after cancellation, want 1", w.writes)
	}
}

func TestWriteCompressedTensorErrors(t *testing.T) {
	tensor := TensorInfo{Name: "weights", Dims: []uint64{256, 2}, DType: GGMLTypeF32}
	raw := make([]byte, 256*2*4)
	sentinel := errors.New("disk full")
	for _, tc := range []struct {
		name string
		dst  *compressTrackingWriter
		want error
	}{
		{"writer", &compressTrackingWriter{err: sentinel}, sentinel},
		{"short_writer", &compressTrackingWriter{short: true}, io.ErrShortWrite},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := writeCompressedTensor(context.Background(), tc.dst, tensor, raw, GGMLTypeQ4_K); !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
	for _, dims := range [][]uint64{nil, {256}, {0, 2}, {257, 2}, {256, ^uint64(0)}} {
		invalid := tensor
		invalid.Dims = dims
		if err := writeCompressedTensor(context.Background(), io.Discard, invalid, raw, GGMLTypeQ4_K); err == nil {
			t.Fatalf("accepted invalid dimensions %v", dims)
		}
	}
	if err := writeCompressedTensor(context.Background(), io.Discard, tensor, raw[:len(raw)-1], GGMLTypeQ4_K); err == nil {
		t.Fatal("accepted truncated source")
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		binary.LittleEndian.PutUint32(raw, math.Float32bits(value))
		w := &compressTrackingWriter{}
		if err := writeCompressedTensor(context.Background(), w, tensor, raw, GGMLTypeQ4_K); err == nil || w.writes != 0 {
			t.Fatalf("non-finite input error = %v, writes = %d", err, w.writes)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w := &compressTrackingWriter{}
	if err := writeCompressedTensor(ctx, w, tensor, raw, GGMLTypeQ4_K); !errors.Is(err, context.Canceled) || w.writes != 0 {
		t.Fatalf("already canceled error = %v, writes = %d", err, w.writes)
	}
}
