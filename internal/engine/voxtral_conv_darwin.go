//go:build darwin && cgo

package engine

import "context"

// Offline audio has enough frames to amortize a small im2col tile. Keep both
// tiles bounded independently of clip duration; the input is read directly,
// so no channel-major padded copy of the complete recording is necessary.
func voxtralCausalConvBatch(ctx context.Context, x []float32, length int, conv VoxtralRealtimeEncoderConv, padLeft, outLen int, out []float32) (bool, error) {
	// Small/custom encoder geometries keep the existing SIMD reduction order.
	// The measured Voxtral stems have at least 128 input/output channels.
	if conv.In < 128 || conv.Out < 128 || outLen < 8 || outLen*conv.Out*conv.In*conv.Kernel < 1<<18 {
		return false, nil
	}
	const tileFrames = 32
	width := conv.In * conv.Kernel
	tile := min(tileFrames, outLen)
	patch := make([]float32, width*tile)
	projected := out
	if outLen > tile {
		projected = make([]float32, conv.Out*tile)
	}
	activation := voxtralConvActivation{bias: conv.Bias, out: out, outLen: outLen}
	for start := 0; start < outLen; start += tile {
		if err := ctx.Err(); err != nil {
			return true, err
		}
		frames := min(tile, outLen-start)
		patch = patch[:width*frames]
		projected = projected[:conv.Out*frames]
		for c := range conv.In {
			input := x[c*length : (c+1)*length]
			for k := range conv.Kernel {
				row := patch[(c*conv.Kernel+k)*frames : (c*conv.Kernel+k+1)*frames]
				first := start*conv.Stride - padLeft + k
				if conv.Stride == 1 {
					copyPaddedWindow(row, input, first)
					continue
				}
				for t := range frames {
					at := first + t*conv.Stride
					if at >= 0 && at < length {
						row[t] = input[at]
					} else {
						row[t] = 0
					}
				}
			}
		}
		if !blasGemmNN(conv.Out, frames, width, conv.Weight, patch, projected) {
			return false, nil
		}
		activation.projected, activation.frames, activation.start = projected, frames, start
		if len(projected) >= 1<<14 {
			parallelRowsBatchedTask(conv.Out, &activation)
		} else {
			activation.runRows(0, conv.Out)
		}
	}
	return true, ctx.Err()
}

// A tile is fully projected before workers write disjoint output channels.
// Reusing the task across tiles avoids one escaping callback per dispatch.
type voxtralConvActivation struct {
	projected, bias, out  []float32
	frames, outLen, start int
}

func (t *voxtralConvActivation) runRows(start, end int) {
	for o := start; o < end; o++ {
		src := t.projected[o*t.frames : (o+1)*t.frames]
		dst := t.out[o*t.outLen+t.start : o*t.outLen+t.start+t.frames]
		for i, v := range src {
			if len(t.bias) > o {
				v += t.bias[o]
			}
			dst[i] = geluTanh(v)
		}
	}
}
