package gopherllm

import (
	"context"
	"strings"
	"testing"
)

// TestRunnerEmbedBatchMatchesEmbedPerItem pins EmbedBatch's contract: it must
// be observably identical to calling Embed once per item, since it is an
// amortized-setup rewrite of the same per-text computation, not a new
// algorithm.
func TestRunnerEmbedBatchMatchesEmbedPerItem(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	texts := []string{"hello world", "a second sentence to embed", "third"}
	var want []EmbeddingResult
	for _, text := range texts {
		res, err := r.Embed(text)
		if err != nil {
			t.Fatalf("Embed(%q): %v", text, err)
		}
		want = append(want, res)
	}

	got, err := r.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("EmbedBatch returned %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].TokenCount != want[i].TokenCount {
			t.Fatalf("item %d: TokenCount = %d, want %d", i, got[i].TokenCount, want[i].TokenCount)
		}
		if len(got[i].Embedding) != len(want[i].Embedding) {
			t.Fatalf("item %d: embedding dim = %d, want %d", i, len(got[i].Embedding), len(want[i].Embedding))
		}
		for j := range want[i].Embedding {
			if got[i].Embedding[j] != want[i].Embedding[j] {
				t.Fatalf("item %d: embedding[%d] = %v, want %v", i, j, got[i].Embedding[j], want[i].Embedding[j])
			}
		}
	}
}

// TestRunnerEmbedBatchEmptyInputIsANoop covers the boundary explicitly named
// in EmbedBatch's contract.
func TestRunnerEmbedBatchEmptyInputIsANoop(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	got, err := r.EmbedBatch(context.Background(), nil)
	if err != nil || got != nil {
		t.Fatalf("got = %v, err = %v, want nil, nil", got, err)
	}
}

// TestRunnerEmbedBatchHonorsCancellation covers the property EmbedBatch adds
// over a bare loop of Embed calls: ctx is checked BETWEEN texts, so a large
// batch can actually be interrupted mid-ingest.
func TestRunnerEmbedBatchHonorsCancellation(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	texts := []string{"a", "b", "c"}
	got, err := r.EmbedBatch(ctx, texts)
	if err == nil {
		t.Fatal("expected a cancellation error")
	}
	if got != nil {
		t.Fatalf("a cancelled batch must not return a partial result, got %v", got)
	}
}

// TestRunnerEmbedBatchWrapsItemErrorWithIndex covers that a failing item is
// identifiable in a batch, not just reported as an opaque failure.
func TestRunnerEmbedBatchWrapsItemErrorWithIndex(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	// Force the second item over the context window, the same failure mode
	// TestEmbedRejectsInvalidContextLength exercises directly against Embed.
	r.config.MaxSeqLen = 4

	_, err = r.EmbedBatch(context.Background(), []string{"ok", "this text has enough tokens to exceed the tiny context window configured above"})
	if err == nil {
		t.Fatal("expected an error for the over-length item")
	}
	if !strings.Contains(err.Error(), "item 1") {
		t.Fatalf("error = %q, want it to name item 1", err.Error())
	}
}
