package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// TestWithRequestContextBindsTheRequest pins the wiring that makes client
// disconnects actually stop work. Both ends of the mechanism already existed
// and are tested elsewhere — GenerateChatStreamUntil polls the generation
// context between prefill chunks and decoded tokens, and the agentic loop
// hands the same context to every AgenticTool.Execute — but withRequestContext
// returned its input unchanged, so nothing ever reached them. A closed browser
// tab left the model generating to completion and left any in-flight tool
// fetch running for a caller that was already gone.
//
// This is deliberately a unit test rather than an end-to-end one. Driving a
// cancelled request through the handler cannot prove anything: withLimit
// selects between the semaphore and req.Context().Done(), both of which are
// ready, so an already-cancelled request is turned away at the door about half
// the time and never reaches generation at all.
func TestWithRequestContextBindsTheRequest(t *testing.T) {
	options := gopherllm.DefaultGenerationOptions()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req = req.WithContext(ctx)

	got := withRequestContext(options, req)
	if reflect.DeepEqual(got, options) {
		t.Fatal("withRequestContext returned its input unchanged; the request context is not bound")
	}
	if want := options.WithContext(req.Context()); !reflect.DeepEqual(got, want) {
		t.Fatal("withRequestContext did not bind exactly the request's context")
	}
}

// TestWithRequestContextToleratesNoRequest covers the defensive branch: the
// helper is called from every generation route, and a nil request must not
// panic a server that is otherwise fine.
func TestWithRequestContextToleratesNoRequest(t *testing.T) {
	options := gopherllm.DefaultGenerationOptions()
	if got := withRequestContext(options, nil); !reflect.DeepEqual(got, options) {
		t.Fatal("withRequestContext(nil request) should return the options unchanged")
	}
}

// TestCancelledRequestIsNotServed covers the outer half of the same promise at
// the HTTP boundary: a request whose caller is already gone must not produce a
// complete answer. withLimit turns it away without writing a body; what must
// never happen is a fully generated 200.
func TestCancelledRequestIsNotServed(t *testing.T) {
	handler := NewHandler(nil, HandlerOptions{Defaults: gopherllm.DefaultGenerationOptions()})
	t.Cleanup(func() { _ = handler.Close() })

	const body = `{"model":"tiny","messages":[{"role":"user","content":"hello"}],"max_tokens":16}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	ctx, cancel := context.WithCancel(req.Context())
	cancel()
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req.WithContext(ctx))

	if rec.Code == http.StatusOK && strings.Contains(rec.Body.String(), "choices") {
		t.Fatalf("a cancelled request produced a complete answer: %s", rec.Body.String())
	}
}
