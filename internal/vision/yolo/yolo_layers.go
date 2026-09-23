package yolo

import (
	"fmt"
	"math"
)

// This file holds the building blocks of the native YOLO executor in
// yolo_model.go: CHW tensors, convolution (im2col + GEMM, plus a direct
// depthwise path), and the Ultralytics modules the supported generations are
// assembled from (Bottleneck, C2f/C3k2, C3k, SPPF, C2PSA). Every module is
// immutable after loading; forward passes allocate their own activations.

// yoloTensor is a CHW activation map.
type yoloTensor struct {
	c, h, w int
	data    []float32
}

func (t yoloTensor) plane() int { return t.h * t.w }

// channels returns the view of channels [from, to) without copying; CHW
// layout keeps each channel range contiguous.
func (t yoloTensor) channels(from, to int) yoloTensor {
	return yoloTensor{c: to - from, h: t.h, w: t.w, data: t.data[from*t.plane() : to*t.plane()]}
}

func yoloConcat(parts ...yoloTensor) (yoloTensor, error) {
	out := yoloTensor{h: parts[0].h, w: parts[0].w}
	for _, p := range parts {
		if p.h != out.h || p.w != out.w {
			return yoloTensor{}, fmt.Errorf("concatenating %dx%d with %dx%d feature maps", out.w, out.h, p.w, p.h)
		}
		out.c += p.c
	}
	out.data = make([]float32, 0, out.c*out.plane())
	for _, p := range parts {
		out.data = append(out.data, p.data...)
	}
	return out, nil
}

func yoloAddInPlace(dst, src []float32) {
	for i, v := range src {
		dst[i] += v
	}
}

// yoloConv is one Conv2d with BatchNorm folded in, optionally followed by
// SiLU. Dense weights are [out][in*k*k], matching the im2col row order; a
// depthwise convolution (one filter per channel, in == out) stores
// [out][k*k].
type yoloConv struct {
	w                  []float32
	b                  []float32
	in, out, k, stride int
	depthwise, act     bool
}

func (cv *yoloConv) forward(x yoloTensor) (yoloTensor, error) {
	if x.c != cv.in {
		return yoloTensor{}, fmt.Errorf("convolution expects %d input channels, got %d", cv.in, x.c)
	}
	pad := cv.k / 2
	oh := (x.h+2*pad-cv.k)/cv.stride + 1
	ow := (x.w+2*pad-cv.k)/cv.stride + 1
	n := oh * ow
	out := yoloTensor{c: cv.out, h: oh, w: ow, data: make([]float32, cv.out*n)}
	if cv.depthwise {
		yoloDepthwise(x, cv, pad, out)
	} else {
		cols := x.data
		if cv.k != 1 || cv.stride != 1 {
			cols = yoloIm2col(x, cv.k, cv.stride, pad, oh, ow)
		}
		yoloGEMM(cv.out, n, cv.in*cv.k*cv.k, cv.w, cols, out.data)
	}
	parallelChunks(cv.out, func(start, end int) {
		for o := start; o < end; o++ {
			row, bias := out.data[o*n:(o+1)*n], cv.b[o]
			if cv.act {
				for i, v := range row {
					v += bias
					row[i] = v * yoloFastSigmoid(v)
				}
			} else {
				for i := range row {
					row[i] += bias
				}
			}
		}
	})
	return out, nil
}

// yoloDepthwise writes a per-channel k×k convolution (without bias) into
// out. It is too little arithmetic per output to benefit from im2col.
func yoloDepthwise(x yoloTensor, cv *yoloConv, pad int, out yoloTensor) {
	k, s := cv.k, cv.stride
	parallelChunks(x.c, func(start, end int) {
		for c := start; c < end; c++ {
			src, dst, w := x.data[c*x.plane():(c+1)*x.plane()], out.data[c*out.plane():(c+1)*out.plane()], cv.w[c*k*k:(c+1)*k*k]
			for oy := range out.h {
				for ox := range out.w {
					var sum float32
					for ky := range k {
						iy := oy*s - pad + ky
						if iy < 0 || iy >= x.h {
							continue
						}
						for kx := range k {
							if ix := ox*s - pad + kx; ix >= 0 && ix < x.w {
								sum += src[iy*x.w+ix] * w[ky*k+kx]
							}
						}
					}
					dst[oy*out.w+ox] = sum
				}
			}
		}
	})
}

// yoloIm2col lays out every k×k receptive field as one column of a
// [in*k*k][oh*ow] matrix, zero-padded at the borders, so the convolution
// becomes a single weights × columns product.
func yoloIm2col(x yoloTensor, k, stride, pad, oh, ow int) []float32 {
	n := oh * ow
	cols := make([]float32, x.c*k*k*n)
	parallelChunks(x.c, func(start, end int) {
		for c := start; c < end; c++ {
			src := x.data[c*x.plane() : (c+1)*x.plane()]
			for ky := range k {
				for kx := range k {
					dst := cols[((c*k+ky)*k+kx)*n:][:n]
					for oy := range oh {
						iy := oy*stride - pad + ky
						row := dst[oy*ow : (oy+1)*ow]
						if iy < 0 || iy >= x.h {
							continue // already zero
						}
						srow := src[iy*x.w : (iy+1)*x.w]
						for ox := range ow {
							if ix := ox*stride - pad + kx; ix >= 0 && ix < x.w {
								row[ox] = srow[ix]
							}
						}
					}
				}
			}
		}
	})
	return cols
}

// yoloGEMMPortable is the pure-Go c = a·b used when Accelerate is not
// available. It walks b in column tiles that stay cache-resident and
// updates four output rows together per pass over each tile row.
func yoloGEMMPortable(m, n, k int, a, b, c []float32) {
	const tile = 512
	clear(c[:m*n])
	tiles := (n + tile - 1) / tile
	parallelChunks(tiles, func(start, end int) {
		for t := start; t < end; t++ {
			j0 := t * tile
			w := min(n, j0+tile) - j0
			i := 0
			for ; i+4 <= m; i += 4 {
				c0, c1, c2, c3 := c[i*n+j0:i*n+j0+w], c[(i+1)*n+j0:(i+1)*n+j0+w], c[(i+2)*n+j0:(i+2)*n+j0+w], c[(i+3)*n+j0:(i+3)*n+j0+w]
				a0, a1, a2, a3 := a[i*k:], a[(i+1)*k:], a[(i+2)*k:], a[(i+3)*k:]
				for p := range k {
					row := b[p*n+j0 : p*n+j0+w]
					v0, v1, v2, v3 := a0[p], a1[p], a2[p], a3[p]
					for j, x := range row {
						c0[j] += v0 * x
						c1[j] += v1 * x
						c2[j] += v2 * x
						c3[j] += v3 * x
					}
				}
			}
			for ; i < m; i++ {
				row := c[i*n+j0 : i*n+j0+w]
				for p := range k {
					yoloAxpy(row, a[i*k+p], b[p*n+j0:p*n+j0+w])
				}
			}
		}
	})
}

// yoloBlock is one entry of a C2f/C3k2 or C3k inner module list.
type yoloBlock interface {
	forward(yoloTensor) (yoloTensor, error)
}

// yoloChain runs convolutions back to back.
type yoloChain []yoloConv

func (ch yoloChain) forward(x yoloTensor) (yoloTensor, error) {
	var err error
	for i := range ch {
		if x, err = ch[i].forward(x); err != nil {
			return x, err
		}
	}
	return x, nil
}

type yoloBottleneck struct {
	cv1, cv2 yoloConv
	residual bool
}

func (b *yoloBottleneck) forward(x yoloTensor) (yoloTensor, error) {
	y, err := yoloChain{b.cv1, b.cv2}.forward(x)
	if err == nil && b.residual {
		yoloAddInPlace(y.data, x.data)
	}
	return y, err
}

// yoloC3k is YOLO11's C3k: two parallel 1x1 projections, one through a
// bottleneck stack, concatenated and fused by cv3.
type yoloC3k struct {
	cv1, cv2, cv3 yoloConv
	m             []yoloBottleneck
}

func (c *yoloC3k) forward(x yoloTensor) (yoloTensor, error) {
	a, err := c.cv1.forward(x)
	for i := 0; err == nil && i < len(c.m); i++ {
		a, err = c.m[i].forward(a)
	}
	if err != nil {
		return a, err
	}
	b, err := c.cv2.forward(x)
	if err != nil {
		return b, err
	}
	cat, err := yoloConcat(a, b)
	if err != nil {
		return cat, err
	}
	return c.cv3.forward(cat)
}

// yoloC2f is YOLOv8's C2f and, with C3k or bottleneck inner blocks, YOLO11's
// C3k2 (which subclasses C2f and changes only what m holds).
type yoloC2f struct {
	cv1, cv2 yoloConv
	m        []yoloBlock
}

func (c *yoloC2f) forward(x yoloTensor) (yoloTensor, error) {
	y, err := c.cv1.forward(x)
	if err != nil {
		return y, err
	}
	half := y.c / 2
	parts := []yoloTensor{y.channels(0, half), y.channels(half, y.c)}
	for _, m := range c.m {
		next, err := m.forward(parts[len(parts)-1])
		if err != nil {
			return next, err
		}
		parts = append(parts, next)
	}
	cat, err := yoloConcat(parts...)
	if err != nil {
		return cat, err
	}
	return c.cv2.forward(cat)
}

type yoloSPPF struct {
	cv1, cv2 yoloConv
	k        int
}

func (s *yoloSPPF) forward(x yoloTensor) (yoloTensor, error) {
	y, err := s.cv1.forward(x)
	if err != nil {
		return y, err
	}
	p1 := yoloMaxPool(y, s.k)
	p2 := yoloMaxPool(p1, s.k)
	p3 := yoloMaxPool(p2, s.k)
	cat, err := yoloConcat(y, p1, p2, p3)
	if err != nil {
		return cat, err
	}
	return s.cv2.forward(cat)
}

// yoloMaxPool is a stride-1, "same"-padded k×k max pool. Max is separable,
// so it runs as a horizontal then a vertical 1-D pass per channel.
func yoloMaxPool(x yoloTensor, k int) yoloTensor {
	out := yoloTensor{c: x.c, h: x.h, w: x.w, data: make([]float32, len(x.data))}
	r := k / 2
	parallelChunks(x.c, func(start, end int) {
		tmp := make([]float32, x.plane())
		for c := start; c < end; c++ {
			src, dst := x.data[c*x.plane():(c+1)*x.plane()], out.data[c*x.plane():(c+1)*x.plane()]
			for y := range x.h {
				row := src[y*x.w : (y+1)*x.w]
				for xx := range x.w {
					m := row[xx]
					for j := max(0, xx-r); j <= min(x.w-1, xx+r); j++ {
						m = max(m, row[j])
					}
					tmp[y*x.w+xx] = m
				}
			}
			for y := range x.h {
				for xx := range x.w {
					m := tmp[y*x.w+xx]
					for j := max(0, y-r); j <= min(x.h-1, y+r); j++ {
						m = max(m, tmp[j*x.w+xx])
					}
					dst[y*x.w+xx] = m
				}
			}
		}
	})
	return out
}

func yoloUpsample2x(x yoloTensor) yoloTensor {
	out := yoloTensor{c: x.c, h: 2 * x.h, w: 2 * x.w}
	out.data = make([]float32, out.c*out.plane())
	parallelChunks(x.c, func(start, end int) {
		for c := start; c < end; c++ {
			src, dst := x.data[c*x.plane():], out.data[c*out.plane():]
			for y := range out.h {
				srow, drow := src[(y/2)*x.w:], dst[y*out.w:(y+1)*out.w]
				for xx := range drow {
					drow[xx] = srow[xx/2]
				}
			}
		}
	})
	return out
}

// yoloAttention is the spatial multi-head self-attention inside YOLO11's
// PSABlock. qkv packs, per head, keyDim query channels, keyDim key channels
// and headDim value channels; pe is a depthwise 3x3 positional term added
// to the attended values before proj.
type yoloAttention struct {
	qkv, proj, pe          yoloConv
	heads, keyDim, headDim int
}

func (a *yoloAttention) forward(x yoloTensor) (yoloTensor, error) {
	qkv, err := a.qkv.forward(x)
	if err != nil {
		return qkv, err
	}
	n := x.plane()
	per := 2*a.keyDim + a.headDim
	scale := float32(1 / math.Sqrt(float64(a.keyDim)))
	values := yoloTensor{c: a.heads * a.headDim, h: x.h, w: x.w, data: make([]float32, a.heads*a.headDim*n)}
	attended := yoloTensor{c: values.c, h: x.h, w: x.w, data: make([]float32, len(values.data))}
	for h := range a.heads {
		copy(values.data[h*a.headDim*n:(h+1)*a.headDim*n], qkv.data[(h*per+2*a.keyDim)*n:(h+1)*per*n])
	}
	// One (head, query) row per work item: scores over all n keys, softmax,
	// then the value-weighted sum. At YOLO11's P5 resolution n is 400 for a
	// 640 input, so the n×n score matrix never needs to exist in full.
	parallelChunks(a.heads*n, func(start, end int) {
		scores := make([]float32, n)
		for item := start; item < end; item++ {
			h, i := item/n, item%n
			q := qkv.data[h*per*n:]
			k := qkv.data[(h*per+a.keyDim)*n:]
			v := values.data[h*a.headDim*n:]
			largest := float32(math.Inf(-1))
			for j := range n {
				var s float32
				for d := range a.keyDim {
					s += q[d*n+i] * k[d*n+j]
				}
				s *= scale
				scores[j] = s
				largest = max(largest, s)
			}
			var sum float32
			for j, s := range scores {
				scores[j] = yoloFastExp(s - largest)
				sum += scores[j]
			}
			inv := 1 / sum
			out := attended.data[h*a.headDim*n:]
			for d := range a.headDim {
				row := v[d*n : (d+1)*n]
				var acc float32
				for j, p := range scores {
					acc += row[j] * p
				}
				out[d*n+i] = acc * inv
			}
		}
	})
	pe, err := a.pe.forward(values)
	if err != nil {
		return pe, err
	}
	yoloAddInPlace(attended.data, pe.data)
	return a.proj.forward(attended)
}

type yoloPSABlock struct {
	attn yoloAttention
	ffn  yoloChain
}

func (b *yoloPSABlock) forward(x yoloTensor) (yoloTensor, error) {
	a, err := b.attn.forward(x)
	if err != nil {
		return a, err
	}
	yoloAddInPlace(a.data, x.data)
	f, err := b.ffn.forward(a)
	if err != nil {
		return f, err
	}
	yoloAddInPlace(f.data, a.data)
	return f, nil
}

// yoloC2PSA splits cv1's output in half, runs PSA blocks over the second
// half only, and fuses both halves again with cv2.
type yoloC2PSA struct {
	cv1, cv2 yoloConv
	m        []yoloPSABlock
}

func (c *yoloC2PSA) forward(x yoloTensor) (yoloTensor, error) {
	y, err := c.cv1.forward(x)
	if err != nil {
		return y, err
	}
	half := y.c / 2
	b := y.channels(half, y.c)
	for i := range c.m {
		if b, err = c.m[i].forward(b); err != nil {
			return b, err
		}
	}
	cat, err := yoloConcat(y.channels(0, half), b)
	if err != nil {
		return cat, err
	}
	return c.cv2.forward(cat)
}
