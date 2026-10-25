package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"testing"
)

func TestDecodeMono(t *testing.T) {
	floatData := func(values ...float32) []byte {
		data := make([]byte, 4*len(values))
		for i, value := range values {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
		}
		return data
	}
	for _, tc := range []struct {
		name   string
		format Format
		data   []byte
		want   []float32
	}{
		{"PCM16 mono", Format{Encoding: 1, Channels: 1, BitsPerSample: 16}, []byte{0, 128, 255, 127}, []float32{-1, 32767.0 / 32768}},
		{"PCM16 stereo", Format{Encoding: 1, Channels: 2, BitsPerSample: 16}, []byte{0, 128, 255, 127}, []float32{-1.0 / 65536}},
		{"float mono", Format{Encoding: 3, Channels: 1, BitsPerSample: 32}, floatData(-.5, 1), []float32{-.5, 1}},
		{"float stereo", Format{Encoding: 3, Channels: 2, BitsPerSample: 32}, floatData(-.5, 1), []float32{.25}},
		{"signed zero", Format{Encoding: 3, Channels: 1, BitsPerSample: 32}, floatData(math.Float32frombits(1 << 31)), []float32{math.Float32frombits(1 << 31)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := slices.Clone(tc.data)
			got, err := DecodeMono(tc.data, tc.format)
			if err != nil || !slices.Equal(got, tc.want) || len(got) > 0 && math.Float32bits(got[0]) != math.Float32bits(tc.want[0]) {
				t.Fatalf("DecodeMono = %v, %v; want %v", got, err, tc.want)
			}
			if !bytes.Equal(tc.data, original) {
				t.Fatal("changed the caller's PCM bytes")
			}
		})
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := DecodeMono(floatData(value), Format{Encoding: 3, Channels: 1, BitsPerSample: 32}); err == nil {
			t.Fatalf("accepted non-finite sample %g", value)
		}
	}
	for _, format := range []Format{{}, {Encoding: 1, Channels: 1, BitsPerSample: 8}, {Encoding: 1, Channels: 2, BitsPerSample: 16}} {
		if _, err := DecodeMono([]byte{0, 0}, format); err == nil {
			t.Fatalf("accepted invalid or incomplete PCM: %+v", format)
		}
	}
}

type countedReader struct {
	*bytes.Reader
	bytesRead int
}

type boundedReader struct {
	*bytes.Reader
	maxRead int
	limit   int
}

func (r *boundedReader) ReadAt(p []byte, offset int64) (int, error) {
	if len(p) > r.limit {
		return 0, fmt.Errorf("read of %d exceeds %d-byte scratch limit", len(p), r.limit)
	}
	r.maxRead = max(r.maxRead, len(p))
	return r.Reader.ReadAt(p, offset)
}

func TestReadMonoMatchesBufferDecode(t *testing.T) {
	for _, channels := range []uint16{1, 2, 3, 65535} {
		for _, format := range []Format{
			{Encoding: 1, BitsPerSample: 16, Channels: channels},
			{Encoding: 3, BitsPerSample: 32, Channels: channels},
		} {
			width := int(format.BitsPerSample / 8)
			frameBytes := int(channels) * width
			frames := 2*max(1, (32<<10)/frameBytes) + 1
			raw := make([]byte, frames*frameBytes)
			for i := 0; i < len(raw)/width; i++ {
				if width == 2 {
					binary.LittleEndian.PutUint16(raw[i*2:], uint16(i*127))
				} else {
					v := float32(i%17-8) / 8
					if i%17 == 0 {
						v = math.Float32frombits(0x80000000)
					}
					binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
				}
			}
			want, err := DecodeMono(raw, format)
			if err != nil {
				t.Fatal(err)
			}
			file := File{Format: format, DataOffset: 7, DataSize: int64(len(raw))}
			r := &boundedReader{Reader: bytes.NewReader(append(make([]byte, 7), raw...)), limit: max(32<<10, frameBytes)}
			got, err := file.ReadMono(r)
			if err != nil || len(got) != len(want) {
				t.Fatalf("%+v: %v, frames=%d", format, err, len(got))
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%+v: sample %d differs across block boundary", format, i)
				}
			}
			if r.maxRead >= len(raw) {
				t.Fatal("read the whole recording into scratch")
			}
		}
	}
}

func TestReadMonoErrors(t *testing.T) {
	format := Format{Encoding: 3, BitsPerSample: 32, Channels: 1}
	file := File{Format: format, DataOffset: 0, DataSize: 32772}
	raw := make([]byte, file.DataSize)
	if samples, err := file.ReadMono(bytes.NewReader(raw[:len(raw)-1])); samples != nil || !errors.Is(err, io.EOF) {
		t.Fatalf("truncated final block: samples=%d err=%v", len(samples), err)
	}
	binary.LittleEndian.PutUint32(raw[32768:], 0x7fc00000)
	if samples, err := file.ReadMono(bytes.NewReader(raw)); samples != nil || err == nil {
		t.Fatal("accepted a non-finite sample in the final block")
	}
	for _, bad := range []File{
		{Format: format, DataOffset: -1},
		{Format: format, DataSize: -4},
		{Format: format, DataSize: 3},
		{Format: format, DataOffset: math.MaxInt64, DataSize: 4},
		{DataSize: 4},
	} {
		if _, err := bad.ReadMono(bytes.NewReader(nil)); err == nil {
			t.Fatalf("accepted invalid layout: %+v", bad)
		}
	}
	if samples, err := (File{Format: format}).ReadMono(nil); err != nil || len(samples) != 0 {
		t.Fatalf("empty data chunk: %v, %v", samples, err)
	}
}

func BenchmarkReadMono(b *testing.B) {
	for _, format := range []Format{
		{Encoding: 1, BitsPerSample: 16, Channels: 1},
		{Encoding: 3, BitsPerSample: 32, Channels: 1},
	} {
		data := make([]byte, 30*16000*int(format.BitsPerSample/8))
		reader := bytes.NewReader(data)
		file := File{Format: format, DataSize: int64(len(data))}
		for _, mode := range []string{"whole_chunk", "bounded"} {
			b.Run(fmt.Sprintf("%dbit/%s", format.BitsPerSample, mode), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(len(data)))
				for b.Loop() {
					var samples []float32
					var err error
					if mode == "whole_chunk" {
						raw := make([]byte, len(data))
						if _, err = reader.ReadAt(raw, 0); err == nil {
							samples, err = DecodeMono(raw, format)
						}
					} else {
						samples, err = file.ReadMono(reader)
					}
					if err != nil || len(samples) != 30*16000 {
						b.Fatalf("decoded %d samples: %v", len(samples), err)
					}
				}
			})
		}
	}
}

func (r *countedReader) ReadAt(p []byte, offset int64) (int, error) {
	r.bytesRead += len(p)
	return r.Reader.ReadAt(p, offset)
}

func TestInspectReadsOnlyHeaders(t *testing.T) {
	format := make([]byte, 16)
	binary.LittleEndian.PutUint16(format, 1)
	binary.LittleEndian.PutUint16(format[2:], 1)
	binary.LittleEndian.PutUint32(format[4:], 16000)
	binary.LittleEndian.PutUint16(format[12:], 2)
	binary.LittleEndian.PutUint16(format[14:], 16)
	var raw bytes.Buffer
	raw.WriteString("RIFF\x00\x00\x00\x00WAVE")
	for _, chunk := range []struct {
		name string
		data []byte
	}{{"JUNK", make([]byte, 1<<20)}, {"data", []byte{0, 0}}, {"fmt ", format}} {
		raw.WriteString(chunk.name)
		_ = binary.Write(&raw, binary.LittleEndian, uint32(len(chunk.data)))
		raw.Write(chunk.data)
	}
	data := raw.Bytes()
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	r := &countedReader{Reader: bytes.NewReader(data)}
	info, err := Inspect(r, int64(len(data)))
	if err != nil || info.SampleRate != 16000 || info.DataSize != 2 || info.DataOffset != 12+8+(1<<20)+8 {
		t.Fatalf("Inspect = %+v, %v", info, err)
	}
	if r.bytesRead != 12+3*8+16 {
		t.Fatalf("read %d bytes, expected only container/chunk headers and fmt", r.bytesRead)
	}
}
