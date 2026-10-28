package engine

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"
)

func TestSpeechAudioWAV(t *testing.T) {
	a := SpeechAudio{SampleRate: 16000, Samples: make([]float32, 2053)}
	copy(a.Samples, []float32{-2, -1, -.5, 0, .5, 1, 2})
	var out bytes.Buffer
	if err := a.WriteWAV(&out); err != nil {
		t.Fatal(err)
	}
	data := out.Bytes()
	if len(data) != 44+2*len(a.Samples) || string(data[:4]) != "RIFF" || string(data[8:16]) != "WAVEfmt " || string(data[36:40]) != "data" {
		t.Fatal("invalid WAV container")
	}
	if binary.LittleEndian.Uint32(data[4:]) != uint32(len(data)-8) || binary.LittleEndian.Uint32(data[40:]) != uint32(len(a.Samples)*2) || binary.LittleEndian.Uint32(data[24:]) != 16000 || binary.LittleEndian.Uint32(data[28:]) != 32000 {
		t.Fatal("invalid WAV lengths or rates")
	}
	for i, want := range []int16{-32768, -32768, -16384, 0, 16384, 32767, 32767} {
		if got := int16(binary.LittleEndian.Uint16(data[44+2*i:])); got != want {
			t.Fatalf("sample %d: %d, want %d", i, got, want)
		}
	}
}

type shortWAVWriter struct{}

func (shortWAVWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }

func TestSpeechAudioWAVErrors(t *testing.T) {
	for _, a := range []SpeechAudio{{SampleRate: 0}, {SampleRate: 16000, Samples: []float32{float32(math.NaN())}}, {SampleRate: 16000, Samples: []float32{float32(math.Inf(1))}}} {
		var out bytes.Buffer
		if err := a.WriteWAV(&out); err == nil || out.Len() != 0 {
			t.Fatal("invalid audio wrote a partial WAV")
		}
	}
	if err := (SpeechAudio{SampleRate: 16000}).WriteWAV(shortWAVWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short writer: %v", err)
	}
}
