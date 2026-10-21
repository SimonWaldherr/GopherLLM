package gguf

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestGGUFWriterTensorStreamRoundTrip(t *testing.T) {
	planned := []PlannedTensor{
		{Name: "experts", Dims: []uint64{32, 3, 2}, DType: GGMLTypeQ8_0},
		{Name: "tail", Dims: []uint64{3}, DType: GGMLTypeF32},
	}
	first := bytes.Repeat([]byte{37}, 6*34)
	last := bytes.Repeat([]byte{83}, 12)
	var buf bytes.Buffer
	w, err := NewGGUFWriter(&buf, nil, planned, 64)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteTensorStream(func(dst io.Writer) error {
		for i := 0; i < len(first); i += 17 {
			if _, err := dst.Write(first[i:min(i+17, len(first))]); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteTensor(last); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	g, err := ParseQuiet(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range [][]byte{first, last} {
		start := g.DataOffset + int(g.Tensors[i].Offset)
		if !bytes.Equal(buf.Bytes()[start:start+len(want)], want) {
			t.Fatalf("tensor %d did not preserve streamed bytes", i)
		}
		if start%64 != 0 {
			t.Fatalf("tensor %d begins at unaligned offset %d", i, start)
		}
	}
}

func TestGGUFWriterTensorStreamFailurePoisonsWriter(t *testing.T) {
	sentinel := errors.New("encoding stopped")
	for _, tc := range []struct {
		name  string
		write func(io.Writer) error
		cause error
	}{
		{"short", func(w io.Writer) error { _, err := w.Write([]byte{1}); return err }, io.ErrUnexpectedEOF},
		{"overrun", func(w io.Writer) error { _, err := w.Write(make([]byte, 5)); return err }, nil},
		{"ignored_overrun", func(w io.Writer) error { _, _ = w.Write(make([]byte, 5)); return nil }, nil},
		{"callback", func(w io.Writer) error { _, _ = w.Write(make([]byte, 4)); return sentinel }, sentinel},
		{"nil", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			w, err := NewGGUFWriter(&buf, nil, []PlannedTensor{{Name: "one", Dims: []uint64{1}, DType: GGMLTypeF32}}, 32)
			if err != nil {
				t.Fatal(err)
			}
			err = w.WriteTensorStream(tc.write)
			if err == nil || tc.cause != nil && !errors.Is(err, tc.cause) {
				t.Fatalf("stream error %v, want cause %v", err, tc.cause)
			}
			if got := w.WriteTensor(make([]byte, 4)); got != err {
				t.Fatalf("write after failure = %v, want original error %v", got, err)
			}
			if got := w.WriteTensorStream(func(io.Writer) error { t.Fatal("callback ran after failure"); return nil }); got != err {
				t.Fatalf("stream after failure = %v, want original error %v", got, err)
			}
			if got := w.Close(); got != err {
				t.Fatalf("close after failure = %v, want original error %v", got, err)
			}
		})
	}
}

type shortTensorWriter struct{}

func (shortTensorWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestTensorStreamWriterRejectsShortWrite(t *testing.T) {
	w := tensorStreamWriter{w: shortTensorWriter{}, remaining: 4}
	if _, err := w.Write(make([]byte, 4)); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write error = %v", err)
	}
	if _, err := w.Write([]byte{0}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("subsequent write error = %v", err)
	}
}

type failingTensorWriter struct{ err error }

func (w failingTensorWriter) Write([]byte) (int, error) { return 0, w.err }

func TestGGUFWriterTensorStreamUnderlyingError(t *testing.T) {
	sentinel := errors.New("output unavailable")
	w, err := NewGGUFWriter(failingTensorWriter{sentinel}, nil, []PlannedTensor{{Name: "large", Dims: []uint64{1 << 20}, DType: GGMLTypeF32}}, 32)
	if err != nil {
		t.Fatal(err)
	}
	err = w.WriteTensorStream(func(dst io.Writer) error {
		_, err := dst.Write(make([]byte, 2<<20))
		return err
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("stream error = %v, want underlying error", err)
	}
	if got := w.Close(); got != err {
		t.Fatalf("close after failed write = %v, want %v", got, err)
	}
}
