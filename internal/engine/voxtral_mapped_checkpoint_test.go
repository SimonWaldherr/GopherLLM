//go:build darwin && cgo && metal

package engine

import (
	"context"
	"io"
	"os"
	"testing"
	"time"
)

// An incomplete layer must remain reachable by the caller's error cleanup.
// Otherwise Metal can outlive the file mapping even though loading failed.
func TestVoxtralMappedCheckpointPartialLoad(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VOXTRAL_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VOXTRAL_MODEL")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapping.Close()
	gguf, err := ParseGGUF(mapping.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	names := detectVoxtralTensorNames(indexTensors(gguf))
	for _, decoder := range []bool{false, true} {
		missing := names.encBlock(0).K
		if decoder {
			missing = names.decBlock(0).K
		}
		t.Run(missing, func(t *testing.T) {
			broken := *gguf
			broken.Tensors = nil
			for _, tensor := range gguf.Tensors {
				if tensor.Name != missing {
					broken.Tensors = append(broken.Tensors, tensor)
				}
			}
			_, weights, err := loadVoxtralRealtimeModel(mapping.Bytes(), &broken, true, io.Discard, true)
			defer releaseVoxtralRealtimeWeights(&weights)
			if err == nil {
				t.Fatal("missing tensor accepted")
			}
			layers := weights.Encoder.Layers
			var q *Weight
			if decoder {
				if len(weights.Decoder.Layers) == 0 {
					t.Fatal("partial decoder layer lost")
				}
				q = &weights.Decoder.Layers[0].Q
			} else {
				if len(layers) == 0 {
					t.Fatal("partial encoder layer lost")
				}
				q = &layers[0].Q
			}
			if q.Metal == nil && len(q.F32) == 0 && len(q.Raw) == 0 {
				t.Fatal("fixture did not load a projection before failure")
			}
			if decoder && q.Metal == nil {
				t.Fatal("decoder fixture did not create a Metal projection")
			}
			releaseVoxtralRealtimeWeights(&weights)
			if q.Metal != nil {
				t.Fatal("partial projection was not released")
			}
		})
	}
}

func TestVoxtralMappedCheckpointOwnership(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VOXTRAL_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VOXTRAL_MODEL")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	m, err := OpenVoxtral(ctx, path, WithVoxtralMetal(true))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	source := m.source
	if source.mmap == nil || !source.mmap.IsMapped() {
		t.Fatal("model is not mapped")
	}
	if !weightBytesInMapping(source.weights.Decoder.TokenEmbd.Raw, source.mmap.Bytes()) {
		t.Fatal("embedding was copied out of mapping")
	}
	s, err := m.NewSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.fast == nil {
		t.Fatal("Metal decoder inactive")
	}
	gpu := s.fast.(*voxtralMetalDecoder)
	if gpu.borrowedWeightBytes == 0 || gpu.copiedWeightBytes != 0 {
		t.Fatalf("decoder weights: borrowed=%d copied=%d", gpu.borrowedWeightBytes, gpu.copiedWeightBytes)
	}
	if gpu.capacity != 256 {
		t.Fatalf("decoder eagerly reserved %d KV positions", gpu.capacity)
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if m.source == nil {
		t.Fatal("active session lost its mapping")
	}
	if _, err = s.Push(ctx, make([]float32, 3200), false); err != nil {
		t.Fatal(err)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
	if m.source != nil {
		t.Fatal("mapping retained after final session close")
	}
}
