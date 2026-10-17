package gopherllm

import (
	"fmt"
	"strings"
	"testing"
)

func contextReductionBenchMessages(turns int, compressible bool) []ChatMessage {
	messages := make([]ChatMessage, 0, 1+2*turns)
	messages = append(messages, ChatMessage{Role: ChatRoleSystem, Content: "abc"})
	user, assistant := strings.Repeat("abcdefghij ", 4), strings.Repeat("jihgfedcba ", 4)
	if compressible {
		user = "Please explain the application programming interface, for example its context window."
		assistant = "For an application programming interface, for example a large language model, use retrieval augmented generation."
	}
	for range turns {
		messages = append(messages, UserMessage(user), AssistantMessage(assistant))
	}
	return messages
}

func BenchmarkPrepareChatContextReduction(b *testing.B) {
	for _, shape := range []struct {
		mode        ContextWindowMode
		turns, keep int
	}{
		{ContextWindowRecent, 64, 64},
		{ContextWindowRecent, 256, 8},
		{ContextWindowAutoCompress, 128, 8},
	} {
		b.Run(fmt.Sprintf("%s/turns%d/keep%d", shape.mode, shape.turns, shape.keep), func(b *testing.B) {
			r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() { r.Close() })
			opts := DefaultGenerationOptions()
			opts.ContextWindowMode, opts.MaxTokens, opts.SystemPrompt = shape.mode, 1, ""
			messages := contextReductionBenchMessages(shape.turns, shape.mode == ContextWindowAutoCompress)
			retained := append([]ChatMessage{messages[0]}, messages[1+(shape.turns-shape.keep)*2:]...)
			if shape.mode == ContextWindowAutoCompress {
				// Keep the budget fixture independent of the compression function
				// under test: only historical assistant prose is abbreviated.
				for i := 0; i < len(retained)-2; i++ {
					if retained[i].Role == ChatRoleAssistant {
						retained[i].Content = "For an API, e.g. a LLM, use RAG."
					}
				}
			}
			r.config.MaxSeqLen = contextWindowLimitFor(r, retained, opts)
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := r.PrepareChatContext(messages, opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAutoCompressChatHistory(b *testing.B) {
	for _, compressible := range []bool{false, true} {
		b.Run(fmt.Sprintf("compressible=%t", compressible), func(b *testing.B) {
			messages := contextReductionBenchMessages(256, compressible)
			b.ReportAllocs()
			for b.Loop() {
				autoCompressChatMessages(messages)
			}
		})
	}
}
