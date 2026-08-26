package gopherllm

import (
	"fmt"
	"math"
)

type EmbeddingResult struct {
	Embedding  []float32
	TokenCount int
}

// Embed produces a text embedding by mean-pooling the final-layer hidden
// states over all input tokens and L2-normalizing the result (so dot product
// equals cosine similarity). Dimension is the model's hidden size — note this
// uses the generation model's hidden states, not a dedicated embedding head.
func (r *Runner) Embed(text string) (EmbeddingResult, error) {
	if err := r.acquireModelLease(); err != nil {
		return EmbeddingResult{}, err
	}
	defer r.releaseModelLease()
	r.genLock.Lock()
	defer r.genLock.Unlock()
	if r.kind == loadedBERT {
		return r.embedBERT(text)
	}
	// Embeddings reuse the same scratch KV workspace but have unrelated token
	// positions, so they must never overwrite a live chat-prefix cache.
	r.clearPrefixCache()
	tokens := r.tok.Encode(text)
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: input tokenised to zero tokens")
	}
	// generationWorkspace allocates its KV cache from MaxSeqLen. A malformed
	// or manually constructed decoder with a zero context length used to get
	// as far as a zero-sized cache and panic inside the forward pass.
	if r.config.MaxSeqLen <= 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: model has an invalid context length (%d)", r.config.MaxSeqLen)
	}
	if len(tokens) > r.config.MaxSeqLen {
		return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+1)
	cache, buf := r.generationWorkspace(cacheLen)
	sum := make([]float32, r.config.Dim)
	if r.canBatchPrefill() {
		// Mean pooling needs every final hidden state, not just the last
		// prompt logit. The batched path streams each projection weight once
		// per chunk and accumulates those states before its scratch is reused.
		weights, _ := r.batchPrefillWeights()
		chunk := prefillChunkSize(r.config)
		for start := 0; start < len(tokens); start += chunk {
			end := min(start+chunk, len(tokens))
			forwardBatchPoolInto(r.config, weights, cache, buf, tokens[start:end], start, sum)
		}
	} else {
		for pos, tok := range tokens {
			h := r.forwardHiddenToken(cache, buf, tok, pos)
			for i, v := range h {
				sum[i] += v
			}
		}
	}
	meanPoolInPlace(sum, len(tokens))
	l2NormalizeInPlace(sum)
	return EmbeddingResult{Embedding: sum, TokenCount: len(tokens)}, nil
}

func meanPoolInPlace(values []float32, count int) {
	if count == 0 {
		return
	}
	scale := float32(1) / float32(count)
	for i := range values {
		values[i] *= scale
	}
}

func l2NormalizeInPlace(values []float32) {
	var ss float32
	for _, v := range values {
		ss += v * v
	}
	norm := float32(math.Sqrt(float64(ss)))
	if norm > 1e-8 {
		for i := range values {
			values[i] /= norm
		}
	}
}

func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("cosine_similarity: dimension mismatch (%d vs %d)", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("cosine_similarity: empty vectors")
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if denom <= 1e-12 {
		return 0, fmt.Errorf("cosine_similarity: zero-norm vector encountered")
	}
	return dot / denom, nil
}
