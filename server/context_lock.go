package server

import (
	"context"
	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"sync"
	"time"
)

func readLockContext(ctx context.Context, mu *sync.RWMutex) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if mu.TryRLock() {
		return nil
	}
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if mu.TryRLock() {
				return nil
			}
		}
	}
}
func (s *runnerState) withRunnerContext(ctx context.Context, fn func(*gopherllm.Runner)) error {
	if err := readLockContext(ctx, &s.mu); err != nil {
		return err
	}
	defer s.mu.RUnlock()
	fn(s.r)
	return nil
}
func withEmbeddingRunnerContext(ctx context.Context, chat *runnerState, embedder *embeddingState, fn func(*gopherllm.Runner)) error {
	if err := readLockContext(ctx, &embedder.mu); err != nil {
		return err
	}
	if embedder.r != nil {
		defer embedder.mu.RUnlock()
		fn(embedder.r)
		return nil
	}
	embedder.mu.RUnlock()
	return chat.withRunnerContext(ctx, fn)
}
