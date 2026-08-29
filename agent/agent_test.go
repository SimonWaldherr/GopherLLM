package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

func TestOpenFailsOnDuplicateToolNameWithoutPartialState(t *testing.T) {
	dup := NewTool("dup", "first", func(context.Context, struct{}) (string, error) { return "a", nil })
	dup2 := NewTool("dup", "second", func(context.Context, struct{}) (string, error) { return "b", nil })
	_, err := newAgentWithGenerator(context.Background(), func([]gopherllm.ChatMessage, gopherllm.GenerationOptions, func(string) bool) (gopherllm.GenerationResult, error) {
		t.Fatal("generator must not be called: construction should fail before any Chat happens")
		return gopherllm.GenerationResult{}, nil
	}, []Option{WithTool(dup), WithTool(dup2)})
	if err == nil {
		t.Fatal("expected an error for a duplicate tool name")
	}
	if !strings.Contains(err.Error(), "dup") {
		t.Fatalf("error = %v, want it to name the duplicate", err)
	}
}

// scriptedTurn is one canned model response, keyed by round number (1-based)
// so a test can drive a multi-round tool conversation deterministically.
type scriptedTurn struct {
	toolName string
	args     string
	text     string
}

func scriptedGenerator(t *testing.T, turns []scriptedTurn) gopherllm.ChatGenerator {
	t.Helper()
	round := 0
	return func(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions, onToken func(string) bool) (gopherllm.GenerationResult, error) {
		if round >= len(turns) {
			t.Fatalf("generator called more times (%d) than scripted (%d)", round+1, len(turns))
		}
		turn := turns[round]
		round++
		if turn.toolName == "" {
			return gopherllm.GenerationResult{Text: turn.text, FinishReason: "stop"}, nil
		}
		return gopherllm.GenerationResult{
			ToolCalls:    []gopherllm.ToolCall{{ID: "c1", Type: "function", Function: gopherllm.ToolCallFunction{Name: turn.toolName, Arguments: turn.args}}},
			FinishReason: "tool_calls",
		}, nil
	}
}

func TestAskAnswersFromToolAndFromIndexInOneTurn(t *testing.T) {
	callerTool := NewTool("stock_level", "units in stock", func(ctx context.Context, args struct {
		SKU string `json:"sku"`
	}) (string, error) {
		return "42 units of " + args.SKU, nil
	})

	gen := scriptedGenerator(t, []scriptedTurn{
		{toolName: SearchDocumentsToolName, args: `{"query":"return window"}`},
		{toolName: "stock_level", args: `{"sku":"GX-1180"}`},
		{text: "Final synthesized answer."},
	})

	a, err := newAgentWithGenerator(context.Background(), gen, []Option{
		WithTool(callerTool),
		WithDocuments(), // no-op; index populated directly below
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.index.Add(context.Background(), rag.Doc{ID: "handbook", Title: "Handbook", Text: "The return window is 30 days from purchase."}); err != nil {
		t.Fatal(err)
	}

	ans, err := a.Ask(context.Background(), "What is the return window and stock for GX-1180?")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Text != "Final synthesized answer." {
		t.Fatalf("Text = %q", ans.Text)
	}
	if !ans.Retrieved {
		t.Fatal("Retrieved = false, want true (search_documents was called)")
	}
	if len(ans.Sources) != 1 || ans.Sources[0].DocID != "handbook" {
		t.Fatalf("Sources = %+v, want one hit from the handbook", ans.Sources)
	}
}

func TestAskSetsRetrievedFalseWhenModelNeverSearches(t *testing.T) {
	gen := scriptedGenerator(t, []scriptedTurn{{text: "No documents needed for this."}})
	a, err := newAgentWithGenerator(context.Background(), gen, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.index.Add(context.Background(), rag.Doc{ID: "d1", Text: "irrelevant content"}); err != nil {
		t.Fatal(err)
	}
	ans, err := a.Ask(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Retrieved {
		t.Fatal("Retrieved = true, want false: the model never called search_documents")
	}
	if ans.Sources != nil {
		t.Fatalf("Sources = %+v, want nil", ans.Sources)
	}
}

func TestAskSetsRetrievedTrueOnEmptyResult(t *testing.T) {
	gen := scriptedGenerator(t, []scriptedTurn{
		{toolName: SearchDocumentsToolName, args: `{"query":"nonexistent"}`},
		{text: "I could not find anything relevant."},
	})
	a, err := newAgentWithGenerator(context.Background(), gen, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Index has content, but nothing matching the scripted query, so the
	// search returns zero hits — this must still count as "looked".
	if err := a.index.Add(context.Background(), rag.Doc{ID: "d1", Text: "totally unrelated words"}); err != nil {
		t.Fatal(err)
	}
	ans, err := a.Ask(context.Background(), "find the nonexistent thing")
	if err != nil {
		t.Fatal(err)
	}
	if !ans.Retrieved {
		t.Fatal("Retrieved = false, want true: search_documents WAS called, even though it found nothing")
	}
	if ans.Sources != nil {
		t.Fatalf("Sources = %+v, want nil (nothing was found)", ans.Sources)
	}
}

func TestSourcesAreDedupedKeepingBestScore(t *testing.T) {
	sources := []Source{
		{DocID: "d1", ChunkIndex: 0, Score: 0.5},
		{DocID: "d2", ChunkIndex: 0, Score: 0.9},
		{DocID: "d1", ChunkIndex: 0, Score: 0.8}, // duplicate of the first, higher score
	}
	got := dedupeSources(sources)
	if len(got) != 2 {
		t.Fatalf("got %d sources, want 2: %+v", len(got), got)
	}
	if got[0].DocID != "d2" || got[0].Score != 0.9 {
		t.Fatalf("got[0] = %+v, want d2 first (highest score)", got[0])
	}
	if got[1].DocID != "d1" || got[1].Score != 0.8 {
		t.Fatalf("got[1] = %+v, want the higher-scoring d1 occurrence (0.8, not 0.5)", got[1])
	}
}

func TestToolSetRejectsReservedNames(t *testing.T) {
	for _, name := range []string{gopherllm.LoadSkillToolName, SearchDocumentsToolName} {
		tool := NewTool(name, "x", func(context.Context, struct{}) (string, error) { return "", nil })
		if _, err := NewToolSet(tool); err == nil {
			t.Fatalf("expected NewToolSet to reject the reserved name %q", name)
		}
	}
}

func TestToolSetRejectsDuplicateWithinOneAddCall(t *testing.T) {
	a := NewTool("x", "a", func(context.Context, struct{}) (string, error) { return "", nil })
	b := NewTool("x", "b", func(context.Context, struct{}) (string, error) { return "", nil })
	if _, err := NewToolSet(a, b); err == nil {
		t.Fatal("expected an error for two tools named \"x\" in the same call")
	}
}

func TestSearchDocumentsToolAppliesGuessLexicalModeWhenUnset(t *testing.T) {
	ix := rag.New(rag.Options{})
	// "GX-1180" is an identifier-shaped query, so GuessLexicalMode picks
	// LexicalAll: a chunk must contain every term ("gx" AND "1180") to
	// match. The "partial" doc has only "gx" and must therefore be excluded
	// — this is the behavior difference that proves the heuristic actually
	// ran, versus the default LexicalAny which would include both.
	if err := ix.Add(context.Background(),
		rag.Doc{ID: "exact", Text: "Part number GX-1180 is in stock."},
		rag.Doc{ID: "partial", Text: "This mentions GX only, not the full code."},
	); err != nil {
		t.Fatal(err)
	}
	tool := SearchDocumentsTool(ix, rag.Query{})
	call := gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"GX-1180"}`}}
	result, err := tool.Execute(context.Background(), call)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "GX-1180 is in stock") {
		t.Fatalf("result = %q, want the exact match", result)
	}
	if strings.Contains(result, "not the full code") {
		t.Fatalf("result = %q, want the partial (\"gx\" only) match excluded by LexicalAll", result)
	}
}

func TestSearchDocumentsToolExecuteReturnsDeterministicSchema(t *testing.T) {
	ix := rag.New(rag.Options{})
	tool := SearchDocumentsTool(ix, rag.Query{})
	var schema map[string]any
	if err := json.Unmarshal(tool.Definition.Function.Parameters, &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Fatalf("schema = %v", schema)
	}
}

func TestAddToolRejectsAfterConstruction(t *testing.T) {
	a, err := newAgentWithGenerator(context.Background(), scriptedGenerator(t, nil), nil)
	if err != nil {
		t.Fatal(err)
	}
	first := NewTool("one", "x", func(context.Context, struct{}) (string, error) { return "", nil })
	if err := a.AddTool(first); err != nil {
		t.Fatal(err)
	}
	dup := NewTool("one", "y", func(context.Context, struct{}) (string, error) { return "", nil })
	if err := a.AddTool(dup); err == nil {
		t.Fatal("expected AddTool to reject a duplicate name")
	}
	if got := a.Tools(); len(got) != 1 || got[0] != "one" {
		t.Fatalf("Tools() = %v, want exactly [\"one\"] (the rejected duplicate must not appear)", got)
	}
}
