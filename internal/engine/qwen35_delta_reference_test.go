package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// Compare the complete production-size recurrence against the implementation
// preceding the fused kernels. Exercise a nonzero layer offset, tiled key
// groups, nonempty convolution history and state, and changing projected inputs.
func TestQwen35DeltaNetProductionGeometryExactReference(t *testing.T) {
	cfg := Config{SSMInner: 6144, SSMState: 128, SSMHeads: 48, SSMGroups: 16, SSMConv: 4, RMSNormEps: 1e-6}
	const layer = 1
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	w := Qwen35DeltaNetWeights{
		ConvKernel: Weight{F32: randomExpertInput(channels*cfg.SSMConv, 43910), Rows: channels, Cols: cfg.SSMConv},
		A:          randomExpertInput(cfg.SSMHeads, 43911),
		DTBias:     randomExpertInput(cfg.SSMHeads, 43912),
		Norm:       randomExpertInput(cfg.SSMState, 43913),
	}
	for i, value := range w.A {
		w.A[i] = -.2 - float32(math.Abs(float64(value)))
	}
	for i, value := range w.Norm {
		w.Norm[i] = .5 + float32(math.Abs(float64(value)))
	}
	initial := newQwen35Cache(cfg, 2)
	copy(initial.Conv, randomExpertInput(len(initial.Conv), 43914))
	copy(initial.State, randomExpertInput(len(initial.State), 43915))
	checkBits := func(t *testing.T, step int, name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("step=%d %s length=%d, want %d", step, name, len(got), len(want))
		}
		for i, value := range got {
			if math.Float32bits(value) != math.Float32bits(want[i]) {
				t.Fatalf("step=%d %s[%d]=%g (%08x), want %g (%08x)", step, name, i,
					value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
			}
		}
	}
	for _, threads := range []int{1, 5, 8} {
		t.Run(fmt.Sprintf("threads=%d", threads), func(t *testing.T) {
			setPerformanceThreads(t, threads)
			got, want := initial.snapshot(), initial.snapshot()
			gotBuf, wantBuf := &DecodeBuffer{}, &DecodeBuffer{}
			for step := range 6 {
				qkv := randomExpertInput(channels, int64(43920+step*4))
				gate := randomExpertInput(cfg.SSMInner, int64(43921+step*4))
				alpha := randomExpertInput(cfg.SSMHeads, int64(43922+step*4))
				beta := randomExpertInput(cfg.SSMHeads, int64(43923+step*4))
				qwen35DeltaNetOriginalStep(cfg, w, want, qkv, gate, alpha, beta, layer, wantBuf)
				qwen35DeltaNetStep(cfg, w, got, qkv, gate, alpha, beta, layer, gotBuf)
				checkBits(t, step, "output", gotBuf.MambaY, wantBuf.MambaY)
				checkBits(t, step, "convolution", got.Conv, want.Conv)
				checkBits(t, step, "state", got.State, want.State)
			}
			// The active layer must change, while the preceding cache slot must
			// retain its original, nonzero contents.
			convOffset, stateOffset := got.convOffset(layer, 0), got.stateOffset(layer, 0)
			checkBits(t, 6, "untouched convolution layer", got.Conv[:convOffset], initial.Conv[:convOffset])
			checkBits(t, 6, "untouched state layer", got.State[:stateOffset], initial.State[:stateOffset])
			if slices.Equal(got.Conv[convOffset:], initial.Conv[convOffset:]) ||
				slices.Equal(got.State[stateOffset:], initial.State[stateOffset:]) {
				t.Fatal("active layer convolution or recurrent state did not advance")
			}
		})
	}
}

// Frozen from qwen35DeltaNetStep before the fused DeltaNet/SIMD changes.
// Keep the original scalar activations and per-row state kernels independent
// from the optimized helpers so future changes cannot move both sides together.
func qwen35DeltaNetOriginalStep(cfg Config, w Qwen35DeltaNetWeights, state *Qwen35Cache, qkv, gate, alphaInput, betaInput []float32, layer int, buf *DecodeBuffer) {
	dInner, dState, nHeads, nGroups := cfg.SSMInner, cfg.SSMState, cfg.SSMHeads, cfg.SSMGroups
	headDim := dInner / nHeads
	channels := dInner + 2*nGroups*dState
	if len(qkv) < channels {
		panic("Qwen hybrid Gated DeltaNet qkv projection has an invalid shape")
	}
	ensureLenNoClear(&buf.MambaConv, channels)
	copy(buf.MambaConv, qkv[:channels])

	// Causal depthwise convolution, then SiLU over the Q|K|V channels.
	ensureLenNoClear(&buf.MambaKernel, cfg.SSMConv)
	// F32/F16/BF16 convolution weights are intentionally materialized by the
	// loader, including in out-of-core mode. Direct row views avoid a copy and
	// method dispatch for every channel; quantized/future kernels retain the
	// general RowInto fallback.
	f32Kernel := w.ConvKernel.F32
	for ch := range channels {
		kernel := buf.MambaKernel[:cfg.SSMConv]
		if len(f32Kernel) >= (ch+1)*cfg.SSMConv {
			kernel = f32Kernel[ch*cfg.SSMConv : (ch+1)*cfg.SSMConv]
		} else {
			w.ConvKernel.RowInto(ch, cfg.SSMConv, &buf.MambaKernel)
		}
		off := state.convOffset(layer, ch)
		v := float32(0)
		for k := 0; k < state.ConvLen; k++ {
			v += kernel[k] * state.Conv[off+k]
		}
		v += kernel[state.ConvLen] * buf.MambaConv[ch]
		if state.ConvLen > 0 {
			copy(state.Conv[off:off+state.ConvLen-1], state.Conv[off+1:off+state.ConvLen])
			state.Conv[off+state.ConvLen-1] = buf.MambaConv[ch]
		}
		buf.MambaConv[ch] = v * fastSigmoidF32(v)
	}

	ensureLenNoClear(&buf.MambaX, dInner)         // V, one vector per value head.
	ensureLenNoClear(&buf.MambaB, nGroups*dState) // Q, one vector per key group.
	ensureLenNoClear(&buf.MambaC, nGroups*dState) // K, one vector per key group.
	ensureLenNoClear(&buf.MambaY, dInner)
	keyDim := nGroups * dState
	copy(buf.MambaB, buf.MambaConv[:keyDim])
	copy(buf.MambaC, buf.MambaConv[keyDim:2*keyDim])
	copy(buf.MambaX, buf.MambaConv[2*keyDim:channels])

	// GGUF conversion keeps Q/K in key-head order and tiles the V-head stream
	// to match its broadcast layout. Normalize each shared Q/K once; Q carries
	// the 1/sqrt(d_k) DeltaNet attention scale after L2 normalization.
	qScale := float32(1 / math.Sqrt(float64(dState)))
	for group := range nGroups {
		qwen35L2NormalizeInPlace(buf.MambaB[group*dState:(group+1)*dState], cfg.RMSNormEps, qScale)
		qwen35L2NormalizeInPlace(buf.MambaC[group*dState:(group+1)*dState], cfg.RMSNormEps, 1)
	}

	// Each DeltaNet head owns a disjoint state matrix and output range. On the
	// production 27B geometry a head updates a 128x128 matrix, so splitting
	// heads across the persistent worker pool is substantially coarser than an
	// ordinary small-row matvec and exposes useful CPU parallelism. Give every
	// head its own recall span; the former one-head scratch was necessarily
	// serial.
	ensureLenNoClear(&buf.MambaRecall, nHeads*headDim)
	updateHeads := func(start, end int) {
		for h := start; h < end; h++ {
			alpha := nemotronSoftplus(alphaInput[h] + w.DTBias[h])
			decay := float32(math.Exp(float64(alpha * w.A[h])))
			beta := fastSigmoidF32(betaInput[h])
			group := h % nGroups
			q := buf.MambaB[group*dState : group*dState+dState]
			k := buf.MambaC[group*dState : group*dState+dState]
			xv := buf.MambaX[h*headDim : h*headDim+headDim]
			recall := buf.MambaRecall[h*headDim : (h+1)*headDim]

			off := state.stateOffset(layer, h)
			for d := range headDim {
				row := state.State[off+d*headDim : off+d*headDim+headDim]
				ScaleF32(row, decay)
				recall[d] = DotF32(row, k)
			}
			for d := range headDim {
				row := state.State[off+d*headDim : off+d*headDim+headDim]
				delta := beta * (xv[d] - recall[d])
				AxpyF32(row, delta, k)
				buf.MambaY[h*headDim+d] = DotF32(row, q)
			}
		}
	}
	if nHeads >= 8 && headDim >= 64 {
		parallelChunks(nHeads, updateHeads)
	} else {
		updateHeads(0, nHeads)
	}

	// Qwen's Gated RMSNorm normalizes the recurrent output first, then applies
	// SiLU(z). Applying the gate before RMSNorm would mostly cancel its scale
	// and materially changes the reference graph.
	for h := range nHeads {
		part := buf.MambaY[h*headDim : h*headDim+headDim]
		ss := DotF32(part, part)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+cfg.RMSNormEps)))
		mulScaleF32(part, w.Norm[:headDim], scale, part)
	}
	for i, g := range gate[:dInner] {
		buf.MambaY[i] *= g * fastSigmoidF32(g)
	}
}
