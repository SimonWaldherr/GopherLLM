package gopherllm

import (
	"math"
	"unicode/utf8"
)

// Both BPE families merge the best adjacent symbol pair until none is left:
// SentencePiece scores a pair by the merged token's vocabulary score (highest
// wins), GPT-2 by its merge rank (lowest wins). Rescanning every pair after
// each merge is O(n^2) and dominates prompt latency once a symbol run gets
// long — a 4k-character SentencePiece prompt, or one unspaced CJK "word" under
// the GPT-2 pretokenizers, which emit \p{L}+ runs of unbounded length.
//
// A merge only invalidates the pairs touching the two symbols it consumes, so
// the scan is replaced by a doubly linked symbol list plus a candidate heap:
// each merge pops one candidate and pushes at most two. Ordering is (priority,
// then leftmost symbol), reproducing the scan's first-best-wins tie-break
// exactly, so the emitted token sequence is unchanged.

// bpeSymbol is one entry of the merge list. Symbols are only ever extended on
// the right, so a symbol's text length doubles as a version stamp: a candidate
// whose endpoints still have their pushed lengths still describes the pair it
// was built from.
type bpeSymbol struct {
	text string
	id   uint32 // SentencePiece token id; unused by the GPT-2 loop
	prev int32
	next int32
	dead bool
}

// bpeCandidate is a mergeable pair queued for consideration.
type bpeCandidate struct {
	prio     float64 // lower merges first
	left     int32
	right    int32
	id       uint32 // SentencePiece id of the merged token
	leftLen  int32
	rightLen int32
}

type bpeHeap []bpeCandidate

func (h bpeHeap) less(i, j int) bool {
	if h[i].prio != h[j].prio {
		return h[i].prio < h[j].prio
	}
	return h[i].left < h[j].left
}

func (h *bpeHeap) push(c bpeCandidate) {
	*h = append(*h, c)
	s := *h
	for i := len(s) - 1; i > 0; {
		p := (i - 1) / 2
		if !s.less(i, p) {
			break
		}
		s[i], s[p] = s[p], s[i]
		i = p
	}
}

func (h *bpeHeap) pop() bpeCandidate {
	s := *h
	top := s[0]
	n := len(s) - 1
	s[0] = s[n]
	s = s[:n]
	*h = s
	for i := 0; ; {
		l := 2*i + 1
		if l >= n {
			break
		}
		m := l
		if r := l + 1; r < n && s.less(r, l) {
			m = r
		}
		if !s.less(m, i) {
			break
		}
		s[i], s[m] = s[m], s[i]
		i = m
	}
	return top
}

// bpeScanMaxSymbols is where the heap starts paying for itself. The GPT-2
// pretokenizers hand the merge loop one short word at a time, and for a
// handful of symbols the scan's tight loop beats building and sifting a heap.
// Both paths merge in the same order, so this is purely a speed knob.
//
// Kept a variable, like useGroupedGQAAttention, so the differential tests can
// force either path in one process and prove they agree.
var bpeScanMaxSymbols = 32

// mergeBPESymbols runs the merge loop. rate reports the priority of merging
// the pair (left, right) plus the merged token's id, or ok=false when the pair
// is not in the vocabulary. joined builds the merged symbol's text.
func mergeBPESymbols(syms []bpeSymbol, heap *bpeHeap,
	rate func(left, right *bpeSymbol) (prio float64, id uint32, ok bool),
	joined func(left, right *bpeSymbol, id uint32) string) {
	if len(syms) <= bpeScanMaxSymbols {
		mergeBPESymbolsScan(syms, rate, joined)
		return
	}
	// Sizing and resetting here rather than at each call site: a leftover
	// candidate from the previous word would index this word's symbols, and
	// short runs that take the scan never pay for a heap at all.
	if cap(*heap) < len(syms) {
		*heap = make(bpeHeap, 0, len(syms)+8)
	} else {
		*heap = (*heap)[:0]
	}
	consider := func(l, r int32) {
		if l < 0 || r < 0 {
			return
		}
		prio, id, ok := rate(&syms[l], &syms[r])
		// A non-finite priority never won the scan's strict comparison, so
		// it must not win here either.
		if !ok || math.IsNaN(prio) || math.IsInf(prio, 1) {
			return
		}
		heap.push(bpeCandidate{
			prio: prio, left: l, right: r, id: id,
			leftLen: int32(len(syms[l].text)), rightLen: int32(len(syms[r].text)),
		})
	}
	for i := range syms {
		consider(int32(i), syms[i].next)
	}
	for len(*heap) > 0 {
		c := heap.pop()
		l, r := &syms[c.left], &syms[c.right]
		if l.dead || r.dead || l.next != c.right ||
			int32(len(l.text)) != c.leftLen || int32(len(r.text)) != c.rightLen {
			continue
		}
		l.text = joined(l, r, c.id)
		l.id = c.id
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = c.left
		}
		r.dead = true
		consider(l.prev, c.left)
		consider(c.left, l.next)
	}
}

// mergeBPESymbolsScan is the linear-scan merge, and the semantics both paths
// are held to. It walks the live list left to right so the strict comparison
// keeps the leftmost of two equally good pairs.
func mergeBPESymbolsScan(syms []bpeSymbol,
	rate func(left, right *bpeSymbol) (prio float64, id uint32, ok bool),
	joined func(left, right *bpeSymbol, id uint32) string) {
	for {
		bestPrio := math.Inf(1)
		bestLeft, bestRight := int32(-1), int32(-1)
		var bestID uint32
		for l := int32(0); l >= 0; l = syms[l].next {
			r := syms[l].next
			if r < 0 {
				break
			}
			prio, id, ok := rate(&syms[l], &syms[r])
			if !ok || math.IsNaN(prio) || math.IsInf(prio, 1) {
				continue
			}
			if prio < bestPrio {
				bestPrio, bestLeft, bestRight, bestID = prio, l, r, id
			}
		}
		if bestLeft < 0 {
			return
		}
		l, r := &syms[bestLeft], &syms[bestRight]
		l.text = joined(l, r, bestID)
		l.id = bestID
		l.next = r.next
		if r.next >= 0 {
			syms[r.next].prev = bestLeft
		}
		r.dead = true
	}
}

// newBPESymbols links n symbols into one list; the caller fills in the text
// and ids through the returned slice.
func newBPESymbols(n int) []bpeSymbol {
	syms := make([]bpeSymbol, n)
	for i := range syms {
		syms[i] = bpeSymbol{prev: int32(i - 1), next: int32(i + 1)}
	}
	if n > 0 {
		syms[n-1].next = -1
	}
	return syms
}

// bpeSymbolsFromRunes relinks buf as one symbol per rune of s, reusing its
// capacity. Every symbol's text is a substring of s, so seeding a word costs
// no allocation and only one pass over it.
func bpeSymbolsFromRunes(buf []bpeSymbol, s string) []bpeSymbol {
	buf = buf[:0]
	for i := 0; i < len(s); {
		_, size := utf8.DecodeRuneInString(s[i:])
		n := int32(len(buf))
		buf = append(buf, bpeSymbol{text: s[i : i+size], prev: n - 1, next: n + 1})
		i += size
	}
	if n := len(buf); n > 0 {
		buf[n-1].next = -1
	}
	return buf
}
