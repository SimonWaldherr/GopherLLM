package gopherllm

import (
	"context"
	"fmt"
)

const ttsConvTileFrames = 128

// One synthesis reuses these bounded tiles across its convolutions. They are
// local to the call, so finishing a long utterance does not retain its buffers.
type ttsConvScratch struct {
	patch, projected, packedWeight []float32
	gemm                           denseGemmScratch
}

func ttsConvInputValid(x ttsMatrix, w ttsWeight, bias []float32, transpose bool) (in, out, kernel int, err error) {
	if x.channels <= 0 || x.length <= 0 || x.channels > len(x.data)/x.length || x.channels*x.length != len(x.data) || len(w.shape) != 3 {
		return 0, 0, 0, fmt.Errorf("tts convolution: invalid input or weight rank")
	}
	a, b, k := w.shape[0], w.shape[1], w.shape[2]
	if a <= 0 || b <= 0 || k <= 0 || a > len(w.data)/b || a*b > len(w.data)/k || a*b*k != len(w.data) {
		return 0, 0, 0, fmt.Errorf("tts convolution: invalid weight dimensions")
	}
	in, out = b, a
	if transpose {
		in, out = a, b
	}
	if bias != nil && len(bias) != out {
		return 0, 0, 0, fmt.Errorf("tts convolution: bias width %d, want %d", len(bias), out)
	}
	return in, out, k, nil
}

// ttsConv1D uses PyTorch's OIK layout and unit stride. Only a fixed-size
// im2col tile is materialized, keeping scratch independent of audio duration.
func ttsConv1D(ctx context.Context, x ttsMatrix, weight ttsWeight, bias []float32, dilation, padding, groups int) (ttsMatrix, error) {
	var dst []float32
	var scratch ttsConvScratch
	return ttsConv1DInto(ctx, x, weight, bias, dilation, padding, groups, &dst, &scratch)
}

func ttsConv1DInto(ctx context.Context, x ttsMatrix, weight ttsWeight, bias []float32, dilation, padding, groups int, dst *[]float32, scratch *ttsConvScratch) (ttsMatrix, error) {
	if err := ctx.Err(); err != nil {
		return ttsMatrix{}, err
	}
	in, out, kernel, err := ttsConvInputValid(x, weight, bias, false)
	if err != nil {
		return ttsMatrix{}, err
	}
	if dilation <= 0 || padding < 0 || groups <= 0 || x.channels%groups != 0 || out%groups != 0 || in != x.channels/groups {
		return ttsMatrix{}, fmt.Errorf("tts convolution: invalid dilation, padding or groups")
	}
	maxInt := int(^uint(0) >> 1)
	if kernel-1 > (maxInt-1)/dilation || padding > (maxInt-x.length)/2 {
		return ttsMatrix{}, fmt.Errorf("tts convolution: output size overflow")
	}
	length := x.length + 2*padding - dilation*(kernel-1)
	if length <= 0 || out > maxInt/length {
		return ttsMatrix{}, fmt.Errorf("tts convolution: invalid output size")
	}
	ensureLenNoClear(dst, out*length)
	if batchSlicesOverlap(*dst, x.data) || batchSlicesOverlap(*dst, weight.data) || batchSlicesOverlap(*dst, bias) {
		return ttsMatrix{}, fmt.Errorf("tts convolution: output overlaps input or weights")
	}
	y := ttsMatrix{data: *dst, channels: out, length: length}
	if groups == x.channels && in == 1 {
		// Depthwise duration-predictor convolutions are too narrow for GEMM.
		parallelRows(out, func(start, end int) {
			for oc := start; oc < end; oc++ {
				if ctx.Err() != nil {
					return
				}
				ic := oc / (out / groups)
				for pos := range length {
					var sum float32
					for k := range kernel {
						at := pos - padding + k*dilation
						if at >= 0 && at < x.length {
							sum += weight.data[oc*kernel+k] * x.data[ic*x.length+at]
						}
					}
					if bias != nil {
						sum += bias[oc]
					}
					y.data[oc*length+pos] = sum
				}
			}
		})
		return y, ctx.Err()
	}
	rows, cols := out/groups, in*kernel
	tile := min(ttsConvTileFrames, length)
	ensureLenNoClear(&scratch.patch, cols*tile)
	ensureLenNoClear(&scratch.projected, rows*tile)
	patch, projected := scratch.patch, scratch.projected
	for g := range groups {
		weights := weight.data[g*rows*cols : (g+1)*rows*cols]
		for start := 0; start < length; start += tile {
			if err := ctx.Err(); err != nil {
				return ttsMatrix{}, err
			}
			n := min(tile, length-start)
			p := patch[:cols*n]
			for ic := range in {
				input := x.data[(g*in+ic)*x.length : (g*in+ic+1)*x.length]
				for k := range kernel {
					row := p[(ic*kernel+k)*n : (ic*kernel+k+1)*n]
					copyPaddedWindow(row, input, start-padding+k*dilation)
				}
			}
			if err := scratch.gemm.multiply(ctx, rows, n, cols, weights, p, projected[:rows*n]); err != nil {
				return ttsMatrix{}, err
			}
			for oc := range rows {
				var offset float32
				if bias != nil {
					offset = bias[g*rows+oc]
				}
				for j := range n {
					y.data[(g*rows+oc)*length+start+j] = projected[oc*n+j] + offset
				}
			}
		}
	}
	return y, nil
}

// ttsConvTranspose1D uses IOK weights. It projects one kernel offset at a time
// and scatters bounded output tiles, without creating a zero-inserted signal.
func ttsConvTranspose1D(ctx context.Context, x ttsMatrix, weight ttsWeight, bias []float32, stride, padding int) (ttsMatrix, error) {
	var dst []float32
	var scratch ttsConvScratch
	return ttsConvTranspose1DInto(ctx, x, weight, bias, stride, padding, &dst, &scratch)
}

func ttsConvTranspose1DInto(ctx context.Context, x ttsMatrix, weight ttsWeight, bias []float32, stride, padding int, dst *[]float32, scratch *ttsConvScratch) (ttsMatrix, error) {
	if err := ctx.Err(); err != nil {
		return ttsMatrix{}, err
	}
	in, out, kernel, err := ttsConvInputValid(x, weight, bias, true)
	if err != nil {
		return ttsMatrix{}, err
	}
	if in != x.channels || stride <= 0 || padding < 0 {
		return ttsMatrix{}, fmt.Errorf("tts transposed convolution: invalid channels, stride or padding")
	}
	maxInt := int(^uint(0) >> 1)
	if x.length-1 > (maxInt-kernel)/stride {
		return ttsMatrix{}, fmt.Errorf("tts transposed convolution: output size overflow")
	}
	full := (x.length-1)*stride + kernel
	if padding > (full-1)/2 {
		return ttsMatrix{}, fmt.Errorf("tts transposed convolution: invalid output size")
	}
	length := full - 2*padding
	if out > maxInt/length {
		return ttsMatrix{}, fmt.Errorf("tts transposed convolution: output size overflow")
	}
	ensureLenNoClear(dst, out*length)
	if batchSlicesOverlap(*dst, x.data) || batchSlicesOverlap(*dst, weight.data) || batchSlicesOverlap(*dst, bias) {
		return ttsMatrix{}, fmt.Errorf("tts transposed convolution: output overlaps input or weights")
	}
	clear(*dst)
	y := ttsMatrix{data: *dst, channels: out, length: length}
	tile := min(ttsConvTileFrames, x.length)
	ensureLenNoClear(&scratch.packedWeight, out*in)
	ensureLenNoClear(&scratch.patch, in*tile)
	ensureLenNoClear(&scratch.projected, out*tile)
	packedWeight, input, projected := scratch.packedWeight, scratch.patch, scratch.projected
	for k := range kernel {
		if err := ctx.Err(); err != nil {
			return ttsMatrix{}, err
		}
		for oc := range out {
			for ic := range in {
				packedWeight[oc*in+ic] = weight.data[(ic*out+oc)*kernel+k]
			}
		}
		for start := 0; start < x.length; start += tile {
			if err := ctx.Err(); err != nil {
				return ttsMatrix{}, err
			}
			n := min(tile, x.length-start)
			for ic := range in {
				copy(input[ic*n:(ic+1)*n], x.data[ic*x.length+start:ic*x.length+start+n])
			}
			if err := scratch.gemm.multiply(ctx, out, n, in, packedWeight, input[:in*n], projected[:out*n]); err != nil {
				return ttsMatrix{}, err
			}
			for oc := range out {
				for j := range n {
					at := (start+j)*stride - padding + k
					if at >= 0 && at < length {
						y.data[oc*length+at] += projected[oc*n+j]
					}
				}
			}
		}
	}
	if bias != nil {
		for oc := range out {
			for j := range length {
				y.data[oc*length+j] += bias[oc]
			}
		}
	}
	return y, ctx.Err()
}
