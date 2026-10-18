package gopherllm

import (
	"math/rand"
	"slices"
	"testing"
)

func TestPromptLookupDraftLongestAndLatest(t *testing.T) {
	tests := []struct {
		name              string
		prompt, generated []uint32
		max               int
		want              []uint32
	}{
		{
			name: "longest before younger shorter match",
			prompt: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 90, 91,
				5, 6, 7, 8, 70, 71, 1, 2, 3, 4, 5, 6, 7},
			generated: []uint32{8}, max: 2, want: []uint32{90, 91},
		},
		{
			name: "latest equally long match",
			prompt: []uint32{1, 2, 3, 4, 5, 6, 7, 8, 90, 91,
				1, 2, 3, 4, 5, 6, 7, 8, 70, 71, 1, 2, 3, 4, 5, 6, 7},
			generated: []uint32{8}, max: 2, want: []uint32{70, 71},
		},
		{
			name:      "suffix across prompt boundary",
			prompt:    []uint32{1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 99, 1, 2, 3, 4},
			generated: []uint32{5, 6, 7, 8}, max: 2, want: []uint32{90, 91},
		},
		{
			name:      "matched source across prompt boundary",
			prompt:    []uint32{99, 1, 2, 3, 4},
			generated: []uint32{5, 6, 7, 8, 90, 91, 1, 2, 3, 4, 5, 6, 7, 8},
			max:       2, want: []uint32{90, 91},
		},
		{
			name:      "generated history only",
			generated: []uint32{1, 2, 3, 4, 90, 91, 99, 1, 2, 3, 4},
			max:       2, want: []uint32{90, 91},
		},
		{
			name:      "overlapping match has only known continuation",
			generated: []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1},
			max:       8, want: []uint32{1},
		},
		{
			name: "full continuation before younger overlapping short match",
			generated: []uint32{1, 2, 1, 2, 1, 2, 1, 2,
				90, 91, 92, 93, 94, 95, 96, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
			max: 7, want: []uint32{90, 91, 92, 93, 94, 95, 96},
		},
		{
			name: "repeated token supplies full seven-token window",
			generated: []uint32{0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			max: 7, want: []uint32{0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "no self match",
			generated: []uint32{1, 2, 3, 4, 5, 6, 7, 8}, max: 8,
		},
		{
			name:      "no three token lookup",
			generated: []uint32{1, 2, 3, 90, 1, 2, 3}, max: 8,
		},
		{
			name:   "no seed",
			prompt: []uint32{1, 2, 3, 4, 90, 1, 2, 3, 4}, max: 8,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := promptLookupDraft(tc.prompt, tc.generated, tc.max, nil)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPromptLookupDraftHistoryBound(t *testing.T) {
	prompt := append([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 90, 91}, make([]uint32, promptLookupHistoryTokens)...)
	generated := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	if got := promptLookupDraft(prompt, generated, 8, nil); len(got) != 0 {
		t.Fatalf("matched outside history bound: %v", got)
	}
	// The same boundary applies after generated tokens fill the whole window.
	generated = append(append(slices.Clone(prompt), generated...), 99)
	generated = append(generated, 1, 2, 3, 4, 5, 6, 7, 8)
	want := promptLookupDraft(nil, generated[len(generated)-promptLookupHistoryTokens:], 8, nil)
	got := promptLookupDraft([]uint32{1, 2, 3, 4, 5, 6, 7, 8, 77}, generated, 8, nil)
	if !slices.Equal(got, want) {
		t.Fatalf("window with old prompt = %v, bounded window = %v", got, want)
	}
}

func TestPromptLookupDraftScratchAndInputs(t *testing.T) {
	prompt := []uint32{1, 2, 3, 4, 90, 91, 99}
	generated := []uint32{1, 2, 3, 4}
	wantPrompt, wantGenerated := slices.Clone(prompt), slices.Clone(generated)
	scratch := make([]uint32, 0, 8)
	got := promptLookupDraft(prompt, generated, 2, scratch)
	if !slices.Equal(got, []uint32{90, 91}) || &got[0] != &scratch[:cap(scratch)][0] {
		t.Fatalf("scratch was not reused for %v", got)
	}
	for _, maxDraft := range []int{-1, 0, 1, 2} {
		got = promptLookupDraft(prompt, generated, maxDraft, scratch)
		if len(got) > max(0, maxDraft) {
			t.Fatalf("max=%d returned %v", maxDraft, got)
		}
	}
	for _, aliased := range [][]uint32{prompt[:0], generated[:0], prompt[2:2]} {
		got = promptLookupDraft(prompt, generated, 2, aliased)
		if !slices.Equal(got, []uint32{90, 91}) {
			t.Fatalf("aliasing changed proposal: %v", got)
		}
		if !slices.Equal(prompt, wantPrompt) || !slices.Equal(generated, wantGenerated) {
			t.Fatal("lookup mutated input history")
		}
	}
	allocs := testing.AllocsPerRun(10, func() {
		got = promptLookupDraft(prompt, generated, 2, scratch)
	})
	if allocs != 0 {
		t.Fatalf("owned scratch lookup allocates: %g", allocs)
	}
	repeated := make([]uint32, 32)
	allocs = testing.AllocsPerRun(10, func() {
		got = promptLookupDraft(nil, repeated, 7, scratch)
	})
	if allocs != 0 || len(got) != 7 {
		t.Fatalf("full repeated-token window allocates or is truncated: allocs=%g tokens=%v", allocs, got)
	}
}

func TestPromptLookupDraftMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := range 300 {
		history := make([]uint32, 8+rng.Intn(80))
		for i := range history {
			history[i] = uint32(rng.Intn(4))
		}
		// Deliberately repeat passages, including overlap and boundary cases.
		length := 4 + rng.Intn(5)
		start := rng.Intn(len(history) - length + 1)
		history = append(history, history[start:start+length]...)
		split := rng.Intn(len(history)) // Keep at least the current seed generated.
		limit := 1 + rng.Intn(8)
		var bestStart, bestGram, bestCount = -1, -1, -1
		for start := 0; start < len(history); start++ {
			for gram := 4; gram <= 8; gram++ {
				if start+gram >= len(history) || gram > len(history) ||
					!slices.Equal(history[start:start+gram], history[len(history)-gram:]) {
					continue
				}
				count := min(limit, len(history)-start-gram)
				full, bestFull := count == limit, bestCount == limit
				if gram > bestGram || (gram == bestGram && (full && !bestFull || full == bestFull && start > bestStart)) {
					bestStart, bestGram, bestCount = start, gram, count
				}
			}
		}
		var want []uint32
		if bestStart >= 0 {
			end := bestStart + bestGram
			want = history[end:min(len(history), end+limit)]
		}
		got := promptLookupDraft(history[:split], history[split:], limit, nil)
		if !slices.Equal(got, want) {
			t.Fatalf("trial=%d split=%d limit=%d history=%v: got %v want %v", trial, split, limit, history, got, want)
		}
	}
}
