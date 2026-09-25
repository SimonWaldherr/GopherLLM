package laya_test

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type decisionFunc func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error)

func (f decisionFunc) Predict(ctx context.Context, r gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
	return f(ctx, r)
}
func fixedDecision(s string) gopherllm.DecisionResult {
	return gopherllm.DecisionResult{Answers: map[string]gopherllm.DecisionAnswer{"result": {Type: "choice", Choice: &s}}, Usage: gopherllm.DecisionUsage{InputTokens: 7}}
}
func csvOptions() gopherllm.DecisionCSVOptions {
	return gopherllm.DecisionCSVOptions{Question: gopherllm.DecisionQuestion{Type: "choice", Instructions: "Classify this record", Criteria: []string{"yes", "no"}}}
}

func TestCSVPreservesRecordsAndAppendsResult(t *testing.T) {
	input := "\ufeff\"id\",text\r\n1,\"hello, world\"\r\n2,\"two\nlines\"\r\n"
	var states []string
	f := decisionFunc(func(_ context.Context, r gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		states = append(states, r.State.(string))
		return fixedDecision("answer, with\nnewline"), nil
	})
	opts := csvOptions()
	opts.InputColumn = "text"
	opts.ResultColumn = "category"
	var out bytes.Buffer
	stats, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader(input), &out, opts)
	if err != nil {
		t.Fatal(err)
	}
	records, err := csv.NewReader(&out).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{{"id", "text", "category"}, {"1", "hello, world", "answer, with\nnewline"}, {"2", "two\nlines", "answer, with\nnewline"}}
	if !reflect.DeepEqual(records, want) || !reflect.DeepEqual(states, []string{"hello, world", "two\nlines"}) {
		t.Fatalf("records=%q states=%q", records, states)
	}
	if stats.Rows != 2 || stats.InputTokens != 14 {
		t.Fatal(stats)
	}
}
func TestCSVWholeRecordAndHeaderlessTSV(t *testing.T) {
	opts := csvOptions()
	var out bytes.Buffer
	f := decisionFunc(func(_ context.Context, r gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		raw := r.State.(json.RawMessage)
		if string(raw) != `{"z":"1","a":"x"}` {
			t.Fatalf("column order lost: %s", raw)
		}
		return fixedDecision("yes"), nil
	})
	if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("z,a\n1,x\n"), &out, opts); err != nil {
		t.Fatal(err)
	}
	opts.NoHeader = true
	opts.Delimiter = "\t"
	opts.InputColumn = "2"
	out.Reset()
	f = func(_ context.Context, r gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		if r.State != "x" {
			t.Fatal(r.State)
		}
		return fixedDecision("yes"), nil
	}
	if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("1\tx\n"), &out, opts); err != nil {
		t.Fatal(err)
	}
	if out.String() != "1\tx\tyes\n" {
		t.Fatal(out.String())
	}
	opts.InputColumn = ""
	out.Reset()
	f = func(_ context.Context, r gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		if !reflect.DeepEqual(r.State, []string{"1", "x"}) {
			t.Fatal(r.State)
		}
		return fixedDecision("yes"), nil
	}
	if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("1\tx\n"), &out, opts); err != nil {
		t.Fatal(err)
	}
}
func TestCSVFailuresAndCancellation(t *testing.T) {
	f := decisionFunc(func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		return fixedDecision("yes"), nil
	})
	for _, input := range []string{"a,a\nx,y\n", "a,\nx,y\n", "a,result\nx,y\n", "a,b\nx\n", "a\n\"unterminated"} {
		var out bytes.Buffer
		if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader(input), &out, csvOptions()); !errors.Is(err, gopherllm.ErrInvalidDecision) {
			t.Errorf("%q: %v", input, err)
		}
	}
	opts := csvOptions()
	opts.InputColumn = "missing"
	if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("a\nx\n"), io.Discard, opts); err == nil {
		t.Fatal("unknown column accepted")
	}
	var out bytes.Buffer
	stats, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("a,b\nx,y\nz\n"), &out, csvOptions())
	if !errors.Is(err, gopherllm.ErrInvalidDecision) || stats.Rows != 1 || out.String() != "a,b,result\nx,y,yes\n" {
		t.Fatalf("partial: %+v %v %q", stats, err, out.String())
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out.Reset()
	if _, err := gopherllm.ClassifyCSV(ctx, f, strings.NewReader("a\nx\n"), &out, csvOptions()); !errors.Is(err, context.Canceled) || out.Len() != 0 {
		t.Fatalf("cancel: %v %q", err, out.String())
	}
}
func TestCSVScalarsAndJSON(t *testing.T) {
	for _, typ := range []string{"score", "noul"} {
		t.Run(typ, func(t *testing.T) {
			f := decisionFunc(func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
				v := .75
				a := gopherllm.DecisionAnswer{Type: typ}
				if typ == "score" {
					a.Score = &v
				} else {
					a.Noul = &v
				}
				return gopherllm.DecisionResult{Answers: map[string]gopherllm.DecisionAnswer{"result": a}}, nil
			})
			opts := csvOptions()
			var out bytes.Buffer
			if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("a\nx\n"), &out, opts); err != nil {
				t.Fatal(err)
			}
			if out.String() != "a,result\nx,0.75\n" {
				t.Fatal(out.String())
			}
			opts.ResultJSON = true
			out.Reset()
			if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("a\nx\n"), &out, opts); err != nil {
				t.Fatal(err)
			}
			records, err := csv.NewReader(&out).ReadAll()
			if err != nil || !json.Valid([]byte(records[1][1])) {
				t.Fatalf("JSON: %v %q", err, records)
			}
		})
	}
}
func TestCSVStreamsBeforeEndOfInput(t *testing.T) {
	in, producer := io.Pipe()
	out, consumer := io.Pipe()
	defer in.Close()
	defer producer.Close()
	defer out.Close()
	defer consumer.Close()
	done := make(chan error, 1)
	f := decisionFunc(func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		return fixedDecision("yes"), nil
	})
	go func() {
		_, err := gopherllm.ClassifyCSV(context.Background(), f, in, consumer, csvOptions())
		consumer.CloseWithError(err)
		done <- err
	}()
	go func() { _, _ = io.WriteString(producer, "text\nfirst\n") }()
	received := make(chan error, 1)
	go func() {
		r := csv.NewReader(out)
		header, e := r.Read()
		if e == nil && len(header) != 2 {
			e = errors.New("missing result header")
		}
		if e == nil {
			row, err := r.Read()
			e = err
			if e == nil && !reflect.DeepEqual(row, []string{"first", "yes"}) {
				e = errors.New("wrong first row")
			}
		}
		received <- e
	}()
	select {
	case err := <-received:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("filter waited for EOF instead of flushing a record")
	}
	producer.Close()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

type failedCSVWriter struct{}

func (failedCSVWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func TestCSVOutputFailure(t *testing.T) {
	calls := 0
	f := decisionFunc(func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		calls++
		return fixedDecision("yes"), nil
	})
	if _, err := gopherllm.ClassifyCSV(context.Background(), f, strings.NewReader("a\nx\n"), failedCSVWriter{}, csvOptions()); !errors.Is(err, io.ErrClosedPipe) || calls != 0 {
		t.Fatalf("write failure: %v calls=%d", err, calls)
	}
}
func TestNativeCSVMatchesSinglePredictions(t *testing.T) {
	m, _, _ := fixture(t)
	opts := csvOptions()
	opts.InputColumn = "text"
	var out bytes.Buffer
	if _, err := m.ClassifyCSV(context.Background(), strings.NewReader("id,text\n1,refund\n2,broken\n"), &out, opts); err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(&out).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range []string{"refund", "broken"} {
		res, err := m.Predict(context.Background(), gopherllm.DecisionRequest{State: text, Questions: map[string]gopherllm.DecisionQuestion{"result": opts.Question}})
		if err != nil {
			t.Fatal(err)
		}
		if rows[i+1][2] != *res.Answers["result"].Choice {
			t.Fatal(rows)
		}
	}
}

// Measures CSV overhead only; inference speed depends on the checkpoint/hardware.
func BenchmarkCSVWholeRecords(b *testing.B) {
	input := "id,text,language,source,priority,status\n" + strings.Repeat("1,please refund my payment,de,email,normal,open\n", 1000)
	result := fixedDecision("yes")
	model := decisionFunc(func(context.Context, gopherllm.DecisionRequest) (gopherllm.DecisionResult, error) {
		return result, nil
	})
	opts := csvOptions()
	b.ReportAllocs()
	b.SetBytes(int64(len(input)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gopherllm.ClassifyCSV(context.Background(), model, strings.NewReader(input), io.Discard, opts); err != nil {
			b.Fatal(err)
		}
	}
}
