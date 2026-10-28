package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestAssetsWorkOutsideSourceDirectory(t *testing.T) {
	t.Chdir(t.TempDir())
	handler := newHandler(nil)
	for _, path := range []string{"/", "/app.js", "/style.css"} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
			if response.Code != http.StatusOK || response.Body.Len() == 0 {
				t.Fatalf("GET %s: status %d, body %q", path, response.Code, response.Body.String())
			}
		})
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/main.go", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("source file returned status %d", response.Code)
	}
}

func TestRequestValidationBeforeGeneration(t *testing.T) {
	const request = `{"order":{"number":"demo-1"},"details":"Does not turn on"}`
	calls := 0
	handler := newHandler(func(context.Context, string, ...gopherllm.GenOption) (gopherllm.GenerationResult, error) {
		calls++
		return gopherllm.GenerationResult{
			Text: `{"classification":"power","priority":"normal","requires_human_review":true,"recommended_actions":["Check the manual"],"customer_reply":"Please check the manual.","safety_notice":""}`,
		}, nil
	})
	for _, tc := range []struct {
		name, body string
	}{
		{"second object", request + `{}`},
		{"trailing invalid JSON", request + `oops`},
		{"oversized whitespace", request + strings.Repeat(" ", maxRequestBytes)},
		{"unknown field", `{"order":{"number":"demo-1"},"details":"test","unexpected":true}`},
		{"missing description", `{"order":{"number":"demo-1"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/assist", strings.NewReader(tc.body)))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400; body %q", response.Code, response.Body.String())
			}
		})
	}
	if calls != 0 {
		t.Fatalf("invalid requests invoked generation %d times", calls)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/assist", strings.NewReader(request+" \n")))
	if response.Code != http.StatusOK || calls != 1 {
		t.Fatalf("valid request: status %d, generation calls %d", response.Code, calls)
	}
	var result assistResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || result.Classification != "power" {
		t.Fatalf("response: %+v, %v", result, err)
	}
}
