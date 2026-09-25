package laya_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func fixture(t *testing.T) (*gopherllm.LayaModel, gopherllm.DecisionRequest, map[string]struct {
	Probabilities   []float64
	Confidence, Act float64
	IDs             []uint32
}) {
	t.Helper()
	dir := "../../testdata/laya-tiny"
	m, e := gopherllm.OpenLaya(context.Background(), dir)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { m.Close() })
	b, e := os.ReadFile(filepath.Join(dir, "reference.json"))
	if e != nil {
		t.Fatal(e)
	}
	var ref struct {
		Request  gopherllm.DecisionRequest
		Expected map[string]struct {
			Probabilities   []float64
			Confidence, Act float64
			IDs             []uint32
		}
	}
	if e = json.Unmarshal(b, &ref); e != nil {
		t.Fatal(e)
	}
	return m, ref.Request, ref.Expected
}
func near(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 0.00011 {
		t.Errorf("%s: got %.7g, want %.7g", what, got, want)
	}
}
func TestNativeLayaMatchesPyTorch(t *testing.T) {
	m, req, expected := fixture(t)
	result, e := m.Predict(context.Background(), req)
	if e != nil {
		t.Fatal(e)
	}
	tokens := 0
	for name, ref := range expected {
		a := result.Answers[name]
		tokens += len(ref.IDs)
		near(t, name+" action", a.Action.ActProbability, ref.Act)
		best := float64(0)
		for _, p := range ref.Probabilities {
			best = max(best, p)
		}
		near(t, name+" answer confidence", a.AnswerConfidence, best)
		if a.Type == "noul" {
			near(t, name, *a.Noul, ref.Probabilities[1])
			near(t, name+" confidence", a.Confidence, best)
			continue
		}
		near(t, name+" entropy confidence", a.Confidence, ref.Confidence)
		keys := map[string][]string{"category": {"billing", "tech", "other"}, "rating": {"0", "1", "2"}, "single": {"only"}}[name]
		for i, key := range keys {
			near(t, name+" "+key, a.Probabilities[key], ref.Probabilities[i])
		}
		if a.Type == "score" {
			want := float64(0)
			for i, p := range ref.Probabilities {
				want += float64(i) * p
			}
			near(t, "score", *a.Score, want)
		}
	}
	if result.Usage.InputTokens != tokens || result.Usage.OutputTokens != 0 {
		t.Fatalf("usage %+v, expected %d", result.Usage, tokens)
	}
}
func TestLayaValidationAndLifecycle(t *testing.T) {
	m, req, _ := fixture(t)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := m.Predict(canceled, req); !errors.Is(e, context.Canceled) {
		t.Fatalf("cancel: %v", e)
	}
	for _, bad := range []gopherllm.DecisionRequest{
		{Questions: req.Questions},
		{State: "hi"},
		{State: "hi", Questions: map[string]gopherllm.DecisionQuestion{"x": {Type: "bogus"}}},
		{State: "hi", Questions: map[string]gopherllm.DecisionQuestion{"x": {Type: "choice", Criteria: []string{}}}},
		{State: "hi", Questions: map[string]gopherllm.DecisionQuestion{"x": {Type: "score", Criteria: []any{nil}}}},
		{State: "hi", Questions: req.Questions, MaxLen: 10, HeadMaxLen: 8},
		{State: "hi", Questions: req.Questions, MaxLen: -1},
	} {
		if _, e := m.Predict(context.Background(), bad); !errors.Is(e, gopherllm.ErrInvalidDecision) {
			t.Errorf("bad request returned %v", e)
		}
	}
	empty, e := m.Predict(context.Background(), gopherllm.DecisionRequest{State: "hi", Questions: map[string]gopherllm.DecisionQuestion{}})
	if e != nil || len(empty.Answers) != 0 || empty.Usage.InputTokens != 0 {
		t.Fatalf("empty: %+v, %v", empty, e)
	}
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, e := m.Predict(context.Background(), req); e != nil {
				t.Error(e)
			}
		}()
	}
	wg.Wait()
	if e = m.Close(); e != nil {
		t.Fatal(e)
	}
	if e = m.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = m.Predict(context.Background(), req); e == nil {
		t.Fatal("closed model accepted inference")
	}
}
func TestLayaRealCheckpoint(t *testing.T) {
	dir := os.Getenv("GOPHERLLM_LAYA_MODEL")
	if dir == "" {
		t.Skip("set GOPHERLLM_LAYA_MODEL for downloaded checkpoint smoke test")
	}
	m, e := gopherllm.OpenLaya(context.Background(), dir)
	if e != nil {
		t.Fatal(e)
	}
	defer m.Close()
	req := gopherllm.DecisionRequest{State: "I was charged twice. Please refund the duplicate payment.", Questions: map[string]gopherllm.DecisionQuestion{"department": {Type: "choice", Instructions: "Which department should handle this?", Criteria: json.RawMessage(`{"billing":"invoices, payments, refunds","technical":"bugs, outages, system errors","other":"everything else"}`)}}}
	result, e := m.Predict(context.Background(), req)
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("result: %+v", result.Answers["department"])
	if os.Getenv("GOPHERLLM_LAYA_ENGLISH_REFERENCE") == "1" {
		a := result.Answers["department"]
		if a.Choice == nil || *a.Choice != "billing" {
			t.Fatal(a)
		}
		near(t, "real checkpoint", a.Probabilities["billing"], .9171)
	}
}
func TestLayaRejectsIncompatibleCheckpoint(t *testing.T) {
	original := "../../testdata/laya-tiny"
	tmp := t.TempDir()
	for _, name := range []string{"rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "model.safetensors"} {
		b, e := os.ReadFile(filepath.Join(original, name))
		if e != nil {
			t.Fatal(e)
		}
		path := filepath.Join(tmp, name)
		os.MkdirAll(filepath.Dir(path), 0755)
		if e = os.WriteFile(path, b, 0600); e != nil {
			t.Fatal(e)
		}
	}
	b, _ := os.ReadFile(filepath.Join(tmp, "encoder/config.json"))
	var cfg map[string]any
	json.Unmarshal(b, &cfg)
	cfg["hidden_size"] = 10
	b, _ = json.Marshal(cfg)
	os.WriteFile(filepath.Join(tmp, "encoder/config.json"), b, 0600)
	if m, e := gopherllm.OpenLaya(context.Background(), tmp); e == nil {
		m.Close()
		t.Fatal("incompatible shape accepted")
	}
}
