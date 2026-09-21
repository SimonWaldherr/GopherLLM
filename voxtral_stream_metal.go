//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"math"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

type voxtralMetalDecoder struct {
	decoder                         *metalbackend.Decoder
	owned                           []*metalbackend.Weight
	refs                            []*metalbackend.Weight
	physical, capacity, window      int
	interleaved                     bool
	inv, sin, cos, residual, normed []float32
}

func newVoxtralFastDecoder(cfg VoxtralRealtimeConfig, w VoxtralRealtimeWeights, state *VoxtralRealtimeDecoderState) voxtralFastDecoder {
	dc := cfg.Decoder
	if !MetalAvailable() || dc.HeadDim != 128 || dc.SlidingWindow <= 1 || dc.SlidingWindow > 8192 {
		return nil
	}
	s := &voxtralMetalDecoder{capacity: dc.SlidingWindow + 256, window: dc.SlidingWindow, interleaved: cfg.RopeInterleaved, inv: standardRopeInvFreqSlice(128, dc.RopeTheta), residual: make([]float32, dc.HiddenSize), normed: make([]float32, dc.HiddenSize)}
	s.decoder = metalbackend.NewDecoder(dc.HiddenSize, dc.IntermediateSize, dc.NHeads, dc.NKVHeads, len(w.Decoder.Layers), s.capacity, dc.Epsilon, float32(1/math.Sqrt(128)), w.Decoder.OutputNorm)
	if s.decoder == nil {
		return nil
	}
	fail := func() voxtralFastDecoder { s.Close(); return nil }
	bind := func(m Weight) (*metalbackend.Weight, uint32) {
		var p *metalbackend.Weight
		var q uint32
		switch m.Type {
		case GGMLTypeQ4_K:
			q = 4
			if m.Metal != nil {
				p = m.Metal.q4
			}
		case GGMLTypeQ6_K:
			q = 6
			if m.Metal != nil {
				p = m.Metal.q6
			}
		case GGMLTypeQ8_0:
			q = 8
			if m.Metal != nil {
				p = m.Metal.q8
			}
		default:
			return nil, 0
		}
		if p == nil {
			switch q {
			case 4:
				p = metalbackend.PrepareQ4K(m.Raw, m.Rows, m.Cols, false)
			case 6:
				p = metalbackend.PrepareQ6K(m.Raw, m.Rows, m.Cols, false)
			case 8:
				p = metalbackend.PrepareQ8_0(m.Raw, m.Rows, m.Cols, false)
			}
			if p != nil {
				s.owned = append(s.owned, p)
			}
		}
		s.refs = append(s.refs, p)
		return p, q
	}
	for i, l := range w.Decoder.Layers {
		matrices := [7]Weight{l.Q, l.K, l.V, l.O, l.FFNGate, l.FFNUp, l.FFNDown}
		var refs [7]*metalbackend.Weight
		var quant [7]uint32
		for j, m := range matrices {
			refs[j], quant[j] = bind(m)
			if refs[j] == nil {
				return fail()
			}
		}
		// Ada is constant. Fold its gate into FFN RMSNorm weights for the fused kernel.
		ffn := make([]float32, len(l.FFNNorm))
		for j, v := range l.FFNNorm {
			ffn[j] = v * (1 + state.AdaScale[i][j])
		}
		if !s.decoder.BindLayer(i, refs, quant, l.AttnNorm, ffn, nil, nil, dc.SlidingWindow-1) {
			return fail()
		}
	}
	output, q := bind(w.Decoder.TokenEmbd)
	if output == nil || !s.decoder.BindOutput(output, q) {
		return fail()
	}
	return s
}

func (s *voxtralMetalDecoder) Step(input []float32, pos int, generate bool) (int, error) {
	if s.physical == s.capacity {
		drop := s.physical - s.window + 1
		if !s.decoder.ShiftCache(s.physical, drop) {
			return 0, fmt.Errorf("cannot compact Voxtral GPU cache")
		}
		s.physical -= drop
	}
	_, pairs := prepareRopeScratch(pos, 128, 128, s.inv, 1, &s.sin, &s.cos)
	var token uint32
	var next *uint32
	if generate {
		next = &token
	}
	if !s.decoder.Step(input, s.sin, s.cos, pairs, s.physical, s.interleaved, 1, s.residual, s.normed, nil, nil, 1, next) {
		return 0, fmt.Errorf("Voxtral GPU decoder failed")
	}
	s.physical++
	return int(token), nil
}
func (s *voxtralMetalDecoder) Close() {
	if s == nil {
		return
	}
	s.decoder.Close()
	for _, w := range s.owned {
		metalbackend.Release(w)
	}
	s.owned = nil
	s.refs = nil
}
