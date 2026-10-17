package gopherllm

import (
	"os"
	"reflect"
	"testing"
)

// Keep the original tokenizer-call boundaries explicit. In particular the
// generic and hybrid thinking headers, and grouped tool-response fragments,
// intentionally use different boundaries even when they spell similar text.
func requireQwenRenderTokenParity(t testing.TB, r *Runner, hybrid, think, toolsEnabled bool) {
	t.Helper()
	call := ToolCall{ID: "call", Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin","days":2}`}}
	messages := []ChatMessage{
		{Role: ChatRoleSystem, Content: " first instruction "},
		UserMessage(" question \n"),
		{Role: ChatRoleAssistant, Content: " answer ", ToolCalls: []ToolCall{call}},
		{Role: ChatRoleSystem, Content: " second instruction "},
		ToolResultMessage("call", "get_weather", " one \n"),
		ToolResultMessage("call", "get_weather", " two \t"),
		UserMessage(" next "),
	}
	original := append([]ChatMessage(nil), messages...)
	var tools []ToolDefinition
	if toolsEnabled {
		tools = []ToolDefinition{sampleTool()}
	}
	start, _ := r.tok.SpecialID("<|im_start|>")
	end, _ := r.tok.SpecialID("<|im_end|>")
	var want []uint32
	text := func(value string) { want = append(want, r.tok.EncodeWithoutBOS(value)...) }
	turn := func(value string) {
		want = append(want, start)
		text(value)
		want = append(want, end)
		text("\n")
	}
	if hybrid {
		system := "first instruction\n\nsecond instruction"
		if toolsEnabled {
			system = qwen35ToolSystemPrompt(tools) + "\n\n" + system
		}
		turn("system\n" + system)
		turn("user\nquestion")
		turn("assistant\n" + renderQwen35AssistantToolCalls("answer", []ToolCall{call}))
		want = append(want, start)
		text("user")
		for _, result := range []string{"one", "two"} {
			text("\n<tool_response>\n")
			text(result)
			text("\n</tool_response>")
		}
		want = append(want, end)
		text("\n")
		turn("user\nnext")
		want = append(want, start)
		text("assistant\n<think>\n")
	} else {
		for _, value := range []string{"system\nfirst instruction", "user\nquestion", "assistant\nanswer", "system\nsecond instruction", "user\none", "user\ntwo", "user\nnext"} {
			turn(value)
		}
		want = append(want, start)
		text("assistant\n")
		if think {
			id, _ := r.tok.SpecialID("<think>")
			want = append(want, id)
			text("\n")
		}
	}
	render := func() ([]uint32, bool) {
		if hybrid {
			return r.renderQwen35Messages(messages, "unused default", tools)
		}
		return r.renderChatMLMessages(messages, "unused default")
	}
	got, ok := render()
	if !ok || !reflect.DeepEqual(got, want) || !reflect.DeepEqual(messages, original) {
		t.Fatalf("renderer changed token boundaries or caller history: hybrid=%t think=%t tools=%t", hybrid, think, toolsEnabled)
	}
	for i := range got {
		got[i] = ^uint32(0)
	}
	again, ok := render()
	if !ok || !reflect.DeepEqual(again, want) {
		t.Fatal("returned render tokens share mutable storage between calls")
	}
}

func TestQwenRenderSeparatorTokenParity(t *testing.T) {
	for _, hybrid := range []bool{false, true} {
		for _, think := range []bool{false, true} {
			tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>", "=")
			r := &Runner{tok: tok, arch: "qwen35"}
			if think {
				r.gguf = &GGUFFile{Metadata: map[string]MetaValue{"tokenizer.chat_template": {Kind: "str", Value: `enable_thinking = enable_thinking if enable_thinking is defined else True '<|im_start|>assistant\n<think>\n'`}}}
			}
			requireQwenRenderTokenParity(t, r, hybrid, think, false)
			if hybrid {
				requireQwenRenderTokenParity(t, r, hybrid, think, true)
			}
		}
	}
}

func TestQwenRenderEmptySeparatorTokenParity(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>", "=")
	tok.Mode, tok.RawWordPieceVocab = TokenizerWordPiece, true
	if len(tok.EncodeWithoutBOS("\n")) != 0 {
		t.Fatal("fixture needs an empty encoded separator")
	}
	r := &Runner{tok: tok}
	requireQwenRenderTokenParity(t, r, false, false, false)
	requireQwenRenderTokenParity(t, r, true, false, true)
}

func qwenConfiguredRenderRunner(t testing.TB) *Runner {
	t.Helper()
	path := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_BENCH_MODEL to a local Qwen GGUF; only metadata/tokenizer pages are read")
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mapping.Close() })
	gguf, err := ParseGGUFQuiet(mapping.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	arch, _ := gguf.GetString("general.architecture")
	if arch != "qwen2" && arch != "qwen2moe" && arch != "qwen3" && arch != "qwen3moe" && !qwen35Family(arch) {
		t.Skip("configured model is not a Qwen-family model")
	}
	return &Runner{tok: tok, gguf: gguf, arch: arch}
}

func TestQwenRenderConfiguredTokenizerParity(t *testing.T) {
	r := qwenConfiguredRenderRunner(t)
	requireQwenRenderTokenParity(t, r, qwen35Family(r.arch), r.chatMLOpensThinkTagByDefault(), qwen35Family(r.arch))
}

func benchmarkQwenRenderSeparators(b *testing.B, r *Runner, hybrid bool) {
	for _, shape := range []string{"short", "history64", "tool-results32"} {
		if shape == "tool-results32" && !hybrid {
			continue
		}
		b.Run(shape, func(b *testing.B) {
			messages := []ChatMessage{UserMessage("hello")}
			var tools []ToolDefinition
			if shape == "history64" {
				for range 32 {
					messages = append(messages, AssistantMessage("answer"), UserMessage("hello"))
				}
			} else if shape == "tool-results32" {
				tools = []ToolDefinition{sampleTool()}
				for range 32 {
					messages = append(messages, ToolResultMessage("call", "get_weather", "a short result"))
				}
			}
			start, _ := r.tok.SpecialID("<|im_start|>")
			b.ReportAllocs()
			for b.Loop() {
				var tokens []uint32
				var ok bool
				if hybrid {
					tokens, ok = r.renderQwen35Messages(messages, "Answer concisely.", tools)
				} else {
					tokens, ok = r.renderChatMLMessages(messages, "Answer concisely.")
				}
				if !ok || len(tokens) == 0 || tokens[0] != start {
					b.Fatal("invalid rendered Qwen prompt")
				}
			}
		})
	}
}

func BenchmarkQwenRenderSeparators(b *testing.B) {
	for _, hybrid := range []bool{false, true} {
		name := "chatml"
		if hybrid {
			name = "hybrid"
		}
		b.Run(name, func(b *testing.B) {
			benchmarkQwenRenderSeparators(b, &Runner{tok: newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>")}, hybrid)
		})
	}
}

func BenchmarkQwenRenderConfiguredTokenizer(b *testing.B) {
	r := qwenConfiguredRenderRunner(b)
	benchmarkQwenRenderSeparators(b, r, qwen35Family(r.arch))
}
