package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

func layaWorkspaceTestModel(t testing.TB) *LayaModel {
	t.Helper()
	dir := filepath.Join("../../testdata", "laya-tiny")
	if _, benchmark := t.(*testing.B); benchmark {
		if selected := os.Getenv("GOPHERLLM_LAYA_BENCH_MODEL"); selected != "" {
			dir = selected
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "rl_agent_config.json")); err != nil {
		t.Skipf("local Laya checkpoint unavailable: %v", err)
	}
	m, err := OpenLaya(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestLayaPredictRejectsNonFiniteOutputs(t *testing.T) {
	for _, target := range []string{"scorer", "action"} {
		t.Run(target, func(t *testing.T) {
			m := layaWorkspaceTestModel(t)
			bias := m.score2.bias
			if target == "action" {
				bias = m.act2.bias
			}
			bias[0] = float32(math.NaN())
			_, err := m.Predict(context.Background(), DecisionRequest{
				State: "Please refund my payment.",
				Questions: map[string]DecisionQuestion{
					"route": {Type: "choice", Criteria: []string{"billing", "technical"}},
				},
			})
			if err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Fatalf("non-finite %s output was accepted: %v", target, err)
			}
		})
	}
}

func BenchmarkLayaPredictRequests(b *testing.B) {
	for _, shape := range []struct {
		name      string
		repeat    int
		questions int
	}{{"short", 1, 1}, {"long", 30, 1}, {"questions8", 1, 8}} {
		b.Run(shape.name, func(b *testing.B) {
			m := layaWorkspaceTestModel(b)
			req := DecisionRequest{State: strings.Repeat("Please refund my payment. The application does not work. ", shape.repeat), Questions: make(map[string]DecisionQuestion)}
			for i := range shape.questions {
				req.Questions[fmt.Sprintf("result%d", i)] = DecisionQuestion{Type: "choice", Instructions: "Which department should handle this request?", Criteria: []string{"billing", "technical support", "sales"}}
			}
			if _, err := m.Predict(context.Background(), req); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := m.Predict(context.Background(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Isolate scratch allocation and the two real ModernBERT RoPE tables at the
// common 512-token/768-wide shape. Model arithmetic is outside this benchmark.
func BenchmarkLayaWorkspaceStorage(b *testing.B) {
	for _, retained := range []bool{false, true} {
		b.Run(fmt.Sprintf("retained%t", retained), func(b *testing.B) {
			var work layaWorkspace
			prepare := func() {
				work.input.resize(512, 768, false)
				work.norm.resize(512, 768, false)
				work.qkv.resize(512, 3*768, false)
				work.attention.resize(512, 768, false)
				work.projection.resize(512, 768, false)
				work.hidden.resize(512, 2*1152, false)
				work.rotations(512, 64, 10000)
				work.rotations(512, 64, 160000)
			}
			if retained {
				prepare()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if !retained {
					work = layaWorkspace{}
				}
				prepare()
			}
		})
	}
}

func TestLayaPredictWorkspaceMatchesColdRequests(t *testing.T) {
	warm, cold := layaWorkspaceTestModel(t), layaWorkspaceTestModel(t)
	choice := DecisionQuestion{Type: "choice", Instructions: "Route?", Criteria: []string{"billing", "technical", "other"}}
	requests := []DecisionRequest{
		{State: "Please refund my payment.", Questions: map[string]DecisionQuestion{"a": choice}},
		{State: strings.Repeat("The application does not work. ", 30), MaxLen: 128, HeadMaxLen: 48, Questions: map[string]DecisionQuestion{"a": choice}},
		{State: "Bug", MaxLen: 64, HeadMaxLen: 32, Questions: map[string]DecisionQuestion{"a": choice}},
		{State: "Which request is urgent?", Questions: map[string]DecisionQuestion{
			"a": choice,
			"b": {Type: "noul", Instructions: "Refund?"},
			"c": {Type: "score", Instructions: "Urgency?", Criteria: []string{"low", "medium", "high", "critical"}},
		}},
		{State: "Please refund my payment.", Questions: map[string]DecisionQuestion{"a": choice}},
	}
	var firstWarm, firstCold DecisionResult
	for i, req := range requests {
		if work := warm.workspace; work != nil {
			// Old activations must never become inputs. Cached position-only
			// rotations intentionally stay intact while every tensor is poisoned.
			for _, buffer := range []*layaBuffer{&work.input, &work.norm, &work.qkv, &work.attention, &work.projection, &work.hidden} {
				backing := buffer.data[:cap(buffer.data)]
				for j := range backing {
					backing[j] = float32(math.NaN())
				}
			}
			backing := work.scores[:cap(work.scores)]
			for j := range backing {
				backing[j] = float32(math.NaN())
			}
		}
		cold.workspace = nil
		got, err := warm.Predict(context.Background(), req)
		if err != nil {
			t.Fatalf("request %d warm: %v", i, err)
		}
		want, err := cold.Predict(context.Background(), req)
		if err != nil {
			t.Fatalf("request %d cold: %v", i, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("request %d differs after workspace reuse:\n got: %+v\nwant: %+v", i, got, want)
		}
		if warm.workspace == nil || warm.workspace.retainedBytes() > layaWorkspaceRetentionBytes {
			t.Fatal("ordinary request workspace was not retained within budget")
		}
		if i == 0 {
			firstWarm, firstCold = got, want
		}
	}
	if !reflect.DeepEqual(firstWarm, firstCold) {
		t.Fatal("later requests changed a previously returned answer")
	}
	if err := warm.Close(); err != nil || warm.workspace != nil {
		t.Fatalf("Close retained workspace: err=%v", err)
	}
}

type layaCancelDuringForward struct {
	context.Context
	cancel    context.CancelFunc
	remaining atomic.Int64
}

func (c *layaCancelDuringForward) Err() error {
	if c.remaining.Add(-1) <= 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestLayaPredictWorkspaceAfterCanceledForward(t *testing.T) {
	warm, cold := layaWorkspaceTestModel(t), layaWorkspaceTestModel(t)
	req := DecisionRequest{State: strings.Repeat("Please refund my payment. ", 8), Questions: map[string]DecisionQuestion{
		"a": {Type: "choice", Instructions: "Route?", Criteria: []string{"billing", "technical", "other"}},
	}}
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &layaCancelDuringForward{Context: base, cancel: cancel}
	ctx.remaining.Store(10)
	if _, err := warm.Predict(ctx, req); !errors.Is(err, context.Canceled) {
		t.Fatalf("partially evaluated request: %v", err)
	}
	if warm.workspace == nil || warm.workspace.attentionTask != nil {
		t.Fatal("canceled workspace was lost or retained the request context")
	}
	got, err := warm.Predict(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	want, err := cold.Predict(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("canceled activations affected later request:\n got: %+v\nwant: %+v", got, want)
	}
}

func TestLayaWorkspaceRetentionBound(t *testing.T) {
	m := &LayaModel{}
	work := &layaWorkspace{}
	work.input.resize(128, 768, false)
	work.rotations(128, 64, 10000)
	work.rotations(128, 64, 160000)
	work.attentionTask = &layaAttentionTask{ctx: context.Background(), qkv: work.input.rows}
	work.attentionTask.scorePool.Put(new([]float32))
	m.retainWorkspace(work)
	if m.workspace != work || work.attentionTask != nil {
		t.Fatal("ordinary workspace or request-local cleanup differs")
	}
	large := &layaWorkspace{}
	// Capacity, including row headers and cached rotations, is what remains
	// reachable after a request; a shortened visible slice must not hide it.
	large.hidden.resize(1, int(layaWorkspaceRetentionBytes/4), false)
	large.hidden.data = large.hidden.data[:1]
	if large.retainedBytes() <= layaWorkspaceRetentionBytes {
		t.Fatal("retention accounting ignored backing capacity")
	}
	m.retainWorkspace(large)
	if m.workspace != nil {
		t.Fatal("oversized one-off workspace remained resident")
	}
}

func TestLayaBufferResizeDropsReplacedBackingStore(t *testing.T) {
	var b layaBuffer
	b.resize(4, 4, false)
	old := &b.data[0]
	for i := range b.data {
		b.data[i] = float32(i + 1)
	}
	rows := b.resize(1, 32, false)
	if len(rows) != 1 || len(rows[0]) != 32 || &rows[0][0] != &b.data[0] || &b.data[0] == old {
		t.Fatal("resized views did not use the replacement storage")
	}
	for i, row := range b.rows[:cap(b.rows)] {
		if i >= len(rows) && row != nil {
			t.Fatalf("inactive row %d retained old backing storage", i)
		}
	}
	// Growing the row count on the same backing store must rebuild every
	// active view, and zero=true must clear all active elements.
	for i := range b.data {
		b.data[i] = 7
	}
	rows = b.resize(4, 8, true)
	for i, row := range rows {
		if len(row) != 8 || &row[0] != &b.data[i*8] {
			t.Fatalf("row %d was not rebuilt", i)
		}
		for _, value := range row {
			if value != 0 {
				t.Fatal("zeroed resize retained an activation")
			}
		}
	}
}
