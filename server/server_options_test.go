package gopherllm

import "testing"

func f32p(v float32) *float32 { return &v }
func intp(v int) *int         { return &v }

func TestGenerateRequestToMessagesAndOptions(t *testing.T) {
	def := DefaultGenerationOptions()
	req := GenerateRequest{
		Prompt:        "hi",
		MaxTokens:     intp(9),
		Temperature:   f32p(0.3),
		TopP:          f32p(0.8),
		TopK:          intp(5),
		MinP:          f32p(0.1),
		RepeatPenalty: f32p(1.2),
	}
	msgs, opts := req.ToMessagesAndOptions(def)
	if len(msgs) != 1 || msgs[0].Content != "hi" || msgs[0].Role != ChatRoleUser {
		t.Fatalf("messages = %+v", msgs)
	}
	if opts.MaxTokens != 9 || opts.Sampler.Temperature != 0.3 || opts.Sampler.TopP != 0.8 ||
		opts.Sampler.TopK != 5 || opts.Sampler.MinP != 0.1 || opts.Sampler.RepeatPenalty != 1.2 {
		t.Fatalf("opts = %+v", opts.Sampler)
	}
}

func TestOpenAIChatMaxCompletionFallbackAndMinP(t *testing.T) {
	def := DefaultGenerationOptions()
	req := OpenAIChatRequest{
		Messages:            []APIMessage{{Role: "user", Content: "hi"}},
		MaxCompletionTokens: intp(12),
		MinP:                f32p(0.07),
	}
	opts := req.Options(def)
	if opts.MaxTokens != 12 {
		t.Fatalf("MaxTokens = %d, want 12 (from max_completion_tokens)", opts.MaxTokens)
	}
	if opts.Sampler.MinP != 0.07 {
		t.Fatalf("MinP = %v", opts.Sampler.MinP)
	}
	if len(req.ChatMessages()) != 1 {
		t.Fatalf("chat messages = %d", len(req.ChatMessages()))
	}
}

func TestOllamaRequestOptions(t *testing.T) {
	def := DefaultGenerationOptions()
	req := OllamaGenerateRequest{
		Prompt:  "hi",
		System:  "sys",
		Options: OllamaOptions{NumPredict: intp(7), Temperature: f32p(0.5), MinP: f32p(0.2)},
	}
	opts := req.GenerationOptions(def)
	if opts.MaxTokens != 7 || opts.SystemPrompt != "sys" || opts.Sampler.MinP != 0.2 || opts.Sampler.Temperature != 0.5 {
		t.Fatalf("opts = maxtok=%d sys=%q sampler=%+v", opts.MaxTokens, opts.SystemPrompt, opts.Sampler)
	}
}

func TestApplyRequestOptionsOverridesSampler(t *testing.T) {
	def := DefaultGenerationOptions()
	got := applyRequestOptions(def, intp(3), f32p(0.4), f32p(0.7), intp(9), f32p(0.15), f32p(1.3), nil, nil, nil, nil, "")
	if got.MaxTokens != 3 || got.Sampler.Temperature != 0.4 || got.Sampler.TopP != 0.7 ||
		got.Sampler.TopK != 9 || got.Sampler.MinP != 0.15 || got.Sampler.RepeatPenalty != 1.3 {
		t.Fatalf("got = %+v / %+v", got, got.Sampler)
	}
}

func TestApiMessagesRoleMapping(t *testing.T) {
	msgs := apiMessages([]APIMessage{
		{Role: "system", Content: "s"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "u"},
		{Role: "other", Content: "o"},
	})
	want := []ChatRole{ChatRoleSystem, ChatRoleAssistant, ChatRoleUser, ChatRoleUser}
	if len(msgs) != len(want) {
		t.Fatalf("len = %d", len(msgs))
	}
	for i, w := range want {
		if msgs[i].Role != w {
			t.Fatalf("msg %d role = %v, want %v", i, msgs[i].Role, w)
		}
	}
}
