package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestDecisionRoutes(t *testing.T) {
	model, e := gopherllm.OpenLaya(context.Background(), "../testdata/laya-tiny")
	if e != nil {
		t.Fatal(e)
	}
	defer model.Close()
	h := NewHandler(nil, HandlerOptions{DecisionModel: model, DecisionModelID: "tiny", MaxConcurrentRequests: 1})
	defer h.Close()
	body := `{"state":"refund","questions":{"q":{"type":"choice","instructions":"Route?","criteria":["billing","tech"]}}}`
	for _, tc := range []struct {
		name, method, body string
		status             int
	}{
		{"success", "POST", body, 200},
		{"method", "GET", "", 405},
		{"trailing", "POST", body + ` {}`, 400},
		{"missing state", "POST", `{"questions":{}}`, 422},
		{"missing instructions", "POST", `{"state":"hi","questions":{"q":{"type":"noul"}}}`, 400},
		{"bad type", "POST", `{"state":"hi","questions":{"q":{"type":"unknown","instructions":"?"}}}`, 422},
		{"unknown model", "POST", `{"model":"some/path","state":"hi","questions":{}}`, 404},
		{"too large", "POST", `{"state":"` + strings.Repeat("x", 2<<20) + `","questions":{}}`, 413},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(tc.method, "/v1/systemone", strings.NewReader(tc.body))
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != tc.status {
				t.Fatalf("status %d want %d: %s", w.Code, tc.status, w.Body)
			}
			if w.Code == 200 {
				var res gopherllm.DecisionResult
				if e := json.Unmarshal(w.Body.Bytes(), &res); e != nil {
					t.Fatal(e)
				}
				if res.Model != "tiny" || len(res.Answers) != 1 || res.Answers["q"].Choice == nil {
					t.Fatal(res)
				}
			}
		})
	}
	for _, opts := range []HandlerOptions{{}, {DecisionModel: model, DeploymentMode: DeploymentBrowser}, {DecisionModel: model, DeploymentMode: DeploymentMode("wasm")}} {
		handler := NewHandler(nil, opts)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
		handler.Close()
		if w.Code != 404 {
			t.Fatalf("disabled decision route: %d", w.Code)
		}
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"id":"tiny"`) || !strings.Contains(w.Body.String(), `"classification"`) {
		t.Fatalf("model discovery: %s", w.Body)
	}
	// NewHandler does not own a caller-supplied decision model.
	if _, e = model.Predict(context.Background(), gopherllm.DecisionRequest{State: "hi", Questions: map[string]gopherllm.DecisionQuestion{}}); e != nil {
		t.Fatal(e)
	}
}
func TestDecisionDeadlineAndCrossSite(t *testing.T) {
	model, e := gopherllm.OpenLaya(context.Background(), "../testdata/laya-tiny")
	if e != nil {
		t.Fatal(e)
	}
	defer model.Close()
	h := NewHandler(nil, HandlerOptions{DecisionModel: model, RequestTimeout: time.Nanosecond})
	defer h.Close()
	body := `{"state":"hi","questions":{"x":{"type":"noul","instructions":"yes?"}}}`
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	if w.Code == http.StatusOK {
		t.Fatal("expired inference deadline ignored")
	}
	r := httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body))
	r.Header.Set("Sec-Fetch-Site", "cross-site")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 403 {
		t.Fatalf("cross-site status %d", w.Code)
	}
}
