package main

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSSEEventFraming(t *testing.T) {
	for _, input := range []string{"data: {\ndata: \"x\":1}\n\n", "data: {\r\ndata: \"x\":1}\r\n\r\n"} {
		s := bufio.NewScanner(strings.NewReader(input))
		s.Split(splitSSEEvents)
		if !s.Scan() || s.Text() != "data: {\ndata: \"x\":1}" || s.Scan() || s.Err() != nil {
			t.Fatalf("bad framing: %q %v", s.Text(), s.Err())
		}
	}
	s := bufio.NewScanner(strings.NewReader("data: [DONE]"))
	s.Split(splitSSEEvents)
	if s.Scan() || s.Err() == nil {
		t.Fatal("unterminated event accepted")
	}
}

func TestEvaluateProtocolAndPredicates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"object\":\"chat.completion.chunk\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"{\\\"count\\\":7}\"},\"finish_reason\":null}]}\n\n")
		fmt.Fprint(w, "data: {\"object\":\"chat.completion.chunk\",\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: {\"object\":\"chat.completion.chunk\",\"model\":\"test\",\"choices\":[],\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	m := evaluate(context.Background(), srv.Client(), runConfig{endpoint: srv.URL, model: "test", timeout: time.Second, maxTokens: 32}, syntheticFixtures()[0])
	if !m.Conformant || !m.OutputValid || m.Semantic == nil || !*m.Semantic || m.TTFTMS == nil || m.DecodeTPS != nil || m.ServerPeakRSSBytes != nil {
		t.Fatalf("%+v", m)
	}
}
func TestEvaluateDoesNotCountHTTPErrorAsPass(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(429) }))
	defer srv.Close()
	m := evaluate(context.Background(), srv.Client(), runConfig{endpoint: srv.URL, model: "test", timeout: time.Second}, syntheticFixtures()[0])
	if m.Conformant || m.OutputValid || m.Error != "http_429" {
		t.Fatalf("%+v", m)
	}
}
