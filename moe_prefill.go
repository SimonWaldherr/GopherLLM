package gopherllm

import (
	"context"
	"os"
	"sync"
)

const moePrefillScratchBytes = 16 << 20

// Group only the active (token, expert) pairs. Expert banks remain borrowed
// from the mmap; no dequantized or GPU copy of the bank is retained.
type moePrefillScratch struct {
	gate, up, hidden, down, scales, sums []float32
	q8                                   []int8
	gateRows, upRows, downRows           [][]float32
	choices, inverse, entries, offsets   []int
	probabilities                        []float32
	groups                               []moePrefillGroup
}

type moePrefillGroup struct{ expert, first, count int }

func moePrefillReady(w *SparseMoEWeights, p int) bool {
	if w == nil || !useQ8Activations.Load() || p < 1 || p > 32 ||
		w.ExpertUsed < 1 || w.ExpertUsed > 32 || w.Gate.Experts < w.ExpertUsed || w.Gate.Experts > 4096 ||
		w.Gate.Experts != w.Up.Experts || w.Gate.Experts != w.Down.Experts ||
		w.Gate.Input <= 0 || w.Gate.Input > moePrefillScratchBytes/4 || w.Gate.Input%256 != 0 ||
		w.Gate.Output <= 0 || w.Gate.Output > moePrefillScratchBytes/4 || w.Gate.Output%256 != 0 ||
		w.Up.Input != w.Gate.Input || w.Up.Output != w.Gate.Output ||
		w.Down.Input != w.Gate.Output || w.Down.Output != w.Gate.Input ||
		w.Gate.Weight.Type != w.Up.Weight.Type {
		return false
	}
	// These are the formats whose serial gate/up path also uses Q8 inputs.
	if w.Gate.Weight.Type != GGMLTypeQ4_K && w.Gate.Weight.Type != GGMLTypeQ6_K {
		return false
	}
	switch w.Down.Weight.Type {
	case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4:
	default:
		return false
	}
	n, dim, hidden := p*w.ExpertUsed, w.Gate.Input, w.Gate.Output
	// Include all retained slabs and row/index metadata in the admission bound.
	width := max(dim, hidden)
	perEntry := int64(4)*(3*int64(hidden)+int64(dim)+int64(width/16)+int64(width/256)+1) + int64(width) + 128
	if int64(n)*perEntry+int64(w.Gate.Experts+1)*8 > moePrefillScratchBytes {
		return false
	}
	for _, ew := range []ExpertWeight{w.Gate, w.Up, w.Down} {
		if ew.Weight.F32 != nil || ew.Weight.GPU != nil || ew.Weight.Metal != nil || ew.Weight.Prepared != nil {
			return false
		}
		// The last plane covers all preceding planes, including fused views.
		if _, ok := expertPlaneWeight(ew, ew.Experts-1); !ok {
			return false
		}
	}
	return true
}

func (s *moePrefillScratch) bytes() int {
	return 4*(cap(s.gate)+cap(s.up)+cap(s.hidden)+cap(s.down)+cap(s.scales)+cap(s.sums)+cap(s.probabilities)) +
		cap(s.q8) + 24*(cap(s.gateRows)+cap(s.upRows)+cap(s.downRows)+cap(s.groups)) +
		8*(cap(s.choices)+cap(s.inverse)+cap(s.entries)+cap(s.offsets))
}

func (s *moePrefillScratch) prepare(w *SparseMoEWeights, p int) {
	n, dim, hidden := p*w.ExpertUsed, w.Gate.Input, w.Gate.Output
	width := max(dim, hidden)
	// Allocate exact capacities. When heterogeneous layers trade dimensions,
	// discard their combined maxima rather than exceeding the scratch budget.
	retained := 4*(max(cap(s.gate), n*hidden)+max(cap(s.up), n*hidden)+max(cap(s.hidden), n*hidden)+
		max(cap(s.down), n*dim)+max(cap(s.scales), n*width/256)+max(cap(s.sums), n*width/16)+max(cap(s.probabilities), n)) +
		max(cap(s.q8), n*width) + 24*(max(cap(s.gateRows), n)+max(cap(s.upRows), n)+max(cap(s.downRows), n)+max(cap(s.groups), n)) +
		8*(max(cap(s.choices), n)+max(cap(s.inverse), n)+max(cap(s.entries), n)+max(cap(s.offsets), w.Gate.Experts+1))
	if retained > moePrefillScratchBytes {
		*s = moePrefillScratch{}
	}
	reuseBatchViews(&s.gate, &s.gateRows, n, hidden)
	reuseBatchViews(&s.up, &s.upRows, n, hidden)
	reuseBatchViews(&s.down, &s.downRows, n, dim)
	// A shorter chunk in a wider layer can replace a flat slab while reusing
	// the row-view allocation. Its spare views must not retain the old slab.
	clear(s.gateRows[n:cap(s.gateRows)])
	clear(s.upRows[n:cap(s.upRows)])
	clear(s.downRows[n:cap(s.downRows)])
	ensureLenNoClear(&s.hidden, n*hidden)
	ensureLenNoClear(&s.q8, n*width)
	ensureLenNoClear(&s.scales, n*width/256)
	ensureLenNoClear(&s.sums, n*width/16)
	ensureLenNoClear(&s.choices, n)
	ensureLenNoClear(&s.inverse, n)
	ensureLenNoClear(&s.entries, n)
	ensureLenNoClear(&s.probabilities, n)
	ensureLenNoClear(&s.offsets, w.Gate.Experts+1)
	ensureLenNoClear(&s.groups, n)
	s.groups = s.groups[:0]
}

func moePrefillSumWidth(typ GGMLType, cols int) int {
	switch typ {
	case GGMLTypeQ4_K, GGMLTypeQ5_K:
		return cols / 32
	case GGMLTypeQ6_K:
		return cols / 16
	default:
		return 0
	}
}

func moePrefillQuantize(s *moePrefillScratch, index int, typ GGMLType, x []float32) {
	cols := len(x)
	quantizeQ8Into(x, s.q8[index*cols:(index+1)*cols], s.scales[index*(cols/256):(index+1)*(cols/256)])
	width := moePrefillSumWidth(typ, cols)
	sums := s.sums[index*width : (index+1)*width : (index+1)*width]
	switch typ {
	case GGMLTypeQ4_K, GGMLTypeQ5_K:
		fillQ4KXSums(x, cols, &sums)
	case GGMLTypeQ6_K:
		fillQ6KXSums16(x, cols, &sums)
		ScaleF32(sums, 32)
	}
}

type moePrefillTask struct {
	a, b       ExpertWeight
	groups     []moePrefillGroup
	x          *moePrefillScratch
	aOut, bOut [][]float32
}

func (t *moePrefillTask) runRows(start, end int) {
	rows := t.a.Output
	segments := 1
	if t.b.Output > 0 {
		segments = 2
	}
	for _, group := range t.groups {
		if (group.first+group.count)*segments*rows <= start {
			continue
		}
		if group.first*segments*rows >= end {
			break
		}
		for segment := 0; segment < segments; segment++ {
			base := (group.first*segments + segment*group.count) * rows
			// One virtual row represents one token dot. Weight ranges by the
			// group's token count so a popular expert cannot pin one worker.
			lo := (max(0, start-base) + group.count - 1) / group.count
			hi := (min(rows*group.count, end-base) + group.count - 1) / group.count
			if lo >= hi {
				continue
			}
			w, outs := t.a, t.aOut
			if segment == 1 {
				w, outs = t.b, t.bOut
			}
			plane, _ := expertPlaneWeight(w, group.expert)
			cols, blocks := w.Input, w.Input/256
			sw := moePrefillSumWidth(w.Weight.Type, cols)
			q8 := t.x.q8[group.first*cols : (group.first+group.count)*cols]
			sc := t.x.scales[group.first*blocks : (group.first+group.count)*blocks]
			sums := t.x.sums[group.first*sw : (group.first+group.count)*sw]
			dst := outs[group.first : group.first+group.count]
			if batchQ4KRows4(plane, dst, q8, sc, sums, lo, hi) ||
				batchQ6KRows4(plane, dst, q8, sc, sums, lo, hi) ||
				batchQ8_0Rows4(plane, dst, q8, sc, lo, hi) {
				continue
			}
			rb, _ := plane.Type.DataSize(cols)
			// Keep a small row tile hot even when an expert has fewer than four
			// matching tokens. Arithmetic is the serial projection's row kernel.
			for tile := lo; tile < hi; tile += 16 {
				last := min(tile+16, hi)
				for i := range group.count {
					qx, sx, ox := q8[i*cols:], sc[i*blocks:], sums[i*sw:]
					switch plane.Type {
					case GGMLTypeQ4_K:
						dotQ4KRowsQ8(plane.Raw, qx, sx, ox, cols, rb, tile, last, dst[i])
					case GGMLTypeQ5_K:
						dotQ5KRowsQ8(plane.Raw, qx, sx, ox, cols, rb, tile, last, dst[i])
					case GGMLTypeQ6_K:
						dotQ6KRowsQ8(plane.Raw, qx, sx, ox, cols, rb, tile, last, dst[i])
					case GGMLTypeQ8_0:
						dotQ8_0RowsQ8(plane.Raw, qx, sx, cols, rb, tile, last, dst[i])
					case GGMLTypeMXFP4:
						dotMXFP4RowsQ8(plane.Raw, qx, sx, cols, rb, tile, last, dst[i])
					}
				}
			}
		}
	}
}

var moePrefillTaskPool = sync.Pool{New: func() any { return new(moePrefillTask) }}

func sparseMoEPrefill(ctx context.Context, w *SparseMoEWeights, xs, outs [][]float32, buf *DecodeBuffer) error {
	s := &buf.moePrefill
	p, k, h, dim := len(xs), w.ExpertUsed, w.Gate.Output, w.Gate.Input
	s.prepare(w, p)
	clear(s.offsets)
	for i, x := range xs {
		if err := ctx.Err(); err != nil {
			return err
		}
		selected, probabilities := sparseMoERoute(w, x, buf)
		for rank, choice := range selected {
			entry := i*k + rank
			s.choices[entry], s.probabilities[entry] = choice.Index, probabilities[rank]
			s.offsets[choice.Index+1]++
		}
	}
	for expert := 0; expert < w.Gate.Experts; expert++ {
		count := s.offsets[expert+1]
		s.offsets[expert+1] += s.offsets[expert]
		if count > 0 {
			s.groups = append(s.groups, moePrefillGroup{expert, s.offsets[expert], count})
		}
	}
	// Stable counting sort preserves token order within each expert.
	for entry, expert := range s.choices {
		index := s.offsets[expert]
		s.entries[index], s.inverse[entry] = entry, index
		s.offsets[expert]++
	}
	for token, x := range xs {
		// Quantize once per token, then copy its small activation row for K
		// experts. Each matrix plane is streamed only for its matching group.
		first := s.inverse[token*k]
		moePrefillQuantize(s, first, w.Gate.Weight.Type, x)
		sw := moePrefillSumWidth(w.Gate.Weight.Type, dim)
		for rank := 1; rank < k; rank++ {
			index := s.inverse[token*k+rank]
			copy(s.q8[index*dim:(index+1)*dim], s.q8[first*dim:(first+1)*dim])
			copy(s.scales[index*(dim/256):(index+1)*(dim/256)], s.scales[first*(dim/256):(first+1)*(dim/256)])
			copy(s.sums[index*sw:(index+1)*sw], s.sums[first*sw:(first+1)*sw])
		}
	}
	task := moePrefillTaskPool.Get().(*moePrefillTask)
	defer func() { *task = moePrefillTask{}; moePrefillTaskPool.Put(task) }()
	task.a, task.b, task.groups, task.x, task.aOut, task.bOut = w.Gate, w.Up, s.groups, s, s.gateRows, s.upRows
	parallelRowsBatchedTask(p*k*2*h, task)
	if err := ctx.Err(); err != nil {
		return err
	}
	if !sparseMoEActivateBatch(w, s.gate, s.up, s.hidden) {
		for index, entry := range s.entries {
			expert := s.choices[entry]
			gate, up, hidden := s.gateRows[index], s.upRows[index], s.hidden[index*h:(index+1)*h]
			w.GateBias.addTo(expert, gate)
			w.UpBias.addTo(expert, up)
			if w.OAIActivation {
				for j := range h {
					g := min(gate[j], float32(7))
					hidden[j] = g * nemotronSigmoid(1.702*g) * (clamp(up[j], -7, 7) + 1)
				}
			} else {
				siluMulF32(gate, up, hidden)
			}
		}
	}
	// Hidden rows are block-aligned and contiguous in expert order. Prepare
	// their independent Q8 scales and offset sums in one pass over the slab.
	moePrefillQuantize(s, 0, w.Down.Weight.Type, s.hidden)
	task.a, task.b, task.aOut, task.bOut = w.Down, ExpertWeight{}, s.downRows, nil
	parallelRowsBatchedTask(p*k*dim, task)
	if err := ctx.Err(); err != nil {
		return err
	}
	originalProj := buf.Proj
	defer func() { buf.Proj = originalProj }()
	for token, out := range outs {
		clear(out)
		for rank := 0; rank < k; rank++ {
			entry := token*k + rank
			result := s.downRows[s.inverse[entry]]
			w.DownBias.addTo(s.choices[entry], result)
			AxpyF32(out, s.probabilities[entry]*w.Scale, result)
		}
		// Retain the established shared expert CPU/Metal reduction exactly.
		buf.Proj = out
		sparseMoESharedForward(w, xs[token], buf)
	}
	return ctx.Err()
}

// Full-model paging measurements have not established a repeatable latency
// win. Keep the scheduling experiment explicit until that changes.
func moePrefillEnabled() bool {
	return os.Getenv("GOPHERLLM_MOE_PREFILL") == "1" && os.Getenv("GOPHERLLM_NO_MOE_PREFILL") == ""
}
