package engine

import (
	"context"
	"errors"
	"testing"
)

func requireQwen35SerialWorkspace(t *testing.T, r *Runner) {
	t.Helper()
	s := &r.workspaceBuf.speculative
	if r.workspaceCache.Qwen35MTP != nil || r.prefixCache.qwen35MTP != nil ||
		s.checkpoint != nil || s.qwenReplay.layers != nil || s.hidden != nil ||
		s.outputLogits != nil || s.outputRows != nil || s.boundaryX != nil || s.boundaryXN != nil {
		t.Fatal("serial request retained draft KV or speculative backing storage")
	}
}

func TestQwen35SerialRequestsReleaseSpeculativeWorkspace(t *testing.T) {
	r, baseline := speculativeFixedQwenRunner(t, 0), speculativeFixedQwenRunner(t, 0)
	opts := speculativeRuntimeOptions()
	opts.MaxTokens = 8
	messages := []ChatMessage{UserMessage("aaaaaaaa")}
	var previousMTP *Qwen35MTPState
	for _, draft := range []int{0, 3, 0, 3} {
		opts.MTPDraftTokens = 0
		want, err := baseline.GenerateChat(messages, opts)
		if err != nil {
			t.Fatal(err)
		}
		opts.MTPDraftTokens = draft
		got, err := r.GenerateChat(messages, opts)
		if err != nil {
			t.Fatal(err)
		}
		speculativeRequireSameResult(t, got, want)
		speculativeRequireSameVisibleState(t, r, baseline)
		if draft == 0 {
			requireQwen35SerialWorkspace(t, r)
		} else {
			state := r.workspaceCache.Qwen35MTP
			if state == nil || state == previousMTP || r.workspaceBuf.speculative.checkpoint == nil || len(r.workspaceBuf.speculative.qwenReplay.layers) == 0 {
				t.Fatal("MTP did not allocate a fresh draft cache and verification checkpoint")
			}
			if got.PromptCache == nil || got.PromptCache.Hit {
				t.Fatal("MTP reused a target prefix without its released draft rows")
			}
			speculativeRequireConsistentBoundary(t, r)
			previousMTP = state
		}
		messages = append(messages, AssistantMessage(got.Text), UserMessage("aaaa"))
	}
}

type qwen35CancelAfterDraftRelease struct {
	context.Context
	r      *Runner
	cancel context.CancelFunc
}

func (c *qwen35CancelAfterDraftRelease) Err() error {
	if c.r.workspaceCache != nil && c.r.workspaceCache.Qwen35MTP == nil {
		c.cancel()
	}
	return c.Context.Err()
}

func TestQwen35DraftReleaseCancellationInvalidatesOnlyMTPPrefix(t *testing.T) {
	r := speculativeFixedQwenRunner(t, 0)
	opts := speculativeRuntimeOptions()
	opts.MaxTokens, opts.MTPDraftTokens = 8, 3
	messages := []ChatMessage{UserMessage("aaaaaaaa")}
	first, err := r.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	oldMTP := r.workspaceCache.Qwen35MTP
	if oldMTP == nil || r.prefixCache.qwen35MTP == nil {
		t.Fatal("fixture did not retain an MTP prefix")
	}
	messages = append(messages, AssistantMessage(first.Text), UserMessage("aaaa"))
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &qwen35CancelAfterDraftRelease{Context: base, r: r, cancel: cancel}
	serial := opts
	serial.MTPDraftTokens = 0
	if _, err := r.GenerateChat(messages, serial.WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Fatalf("serial request did not cancel after draft release: %v", err)
	}
	requireQwen35SerialWorkspace(t, r)
	speculativeRequireConsistentBoundary(t, r)
	got, err := r.GenerateChat(messages, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Hit || r.workspaceCache.Qwen35MTP == oldMTP {
		t.Fatalf("MTP reused released draft state after cancellation: %+v, %v", got.PromptCache, err)
	}
	cold := speculativeFixedQwenRunner(t, 0)
	want, err := cold.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, r, cold)
	speculativeRequireSameCacheRows(t, r.workspaceCache.Qwen35MTP.KV, cold.workspaceCache.Qwen35MTP.KV, len(r.prefixCache.tokens))
}
