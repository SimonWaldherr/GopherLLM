package server

import (
	"bytes"
	"context"
	"encoding/csv"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func decisionCSVUpload(t *testing.T, data, options string) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	form := multipart.NewWriter(&buf)
	// File first: the endpoint must not depend on browser multipart field order.
	part, err := form.CreateFormFile("file", "input.csv")
	if err != nil {
		t.Fatal(err)
	}
	io.WriteString(part, data)
	if err = form.WriteField("options", options); err != nil {
		t.Fatal(err)
	}
	if err = form.Close(); err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest("POST", "/v1/systemone/csv", &buf)
	r.Header.Set("Content-Type", form.FormDataContentType())
	return r
}

const csvTestOptions = `{"question":{"type":"choice","instructions":"Route this?","criteria":["billing","tech"]},"input_column":"text","result_column":"category"}`

func TestDecisionCSVUpload(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("TMPDIR", tmp)
	model, err := gopherllm.OpenLaya(context.Background(), "../testdata/laya-tiny")
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	h := NewHandler(nil, HandlerOptions{DecisionModel: model})
	defer h.Close()
	for _, tc := range []struct {
		name, data, opts string
		status           int
	}{
		{"valid", "id,text\n1,refund\n2,broken\n", csvTestOptions, 200},
		{"late malformed", "id,text\n1,refund\n2\n", csvTestOptions, 422},
		{"unknown column", "id,body\n1,refund\n", csvTestOptions, 422},
		{"missing instruction", "text\nx\n", `{"question":{"type":"noul"}}`, 400},
		{"missing question", "text\nx\n", `{}`, 422},
		{"unknown option", "text\nx\n", `{"bogus":1}`, 400},
		{"trailing options", "text\nx\n", csvTestOptions + ` {}`, 400},
		{"spilled upload cleanup", "id,text\n1," + strings.Repeat("x", (1<<20)+1024) + "\n", csvTestOptions, 422},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			h.ServeHTTP(w, decisionCSVUpload(t, tc.data, tc.opts))
			if w.Code != tc.status {
				t.Fatalf("status %d want %d: %s", w.Code, tc.status, w.Body)
			}
			if w.Code == 200 {
				if w.Header().Get("X-Processed-Rows") != "2" || w.Header().Get("Content-Disposition") == "" {
					t.Fatal(w.Header())
				}
				records, e := csv.NewReader(w.Body).ReadAll()
				if e != nil || len(records) != 3 || len(records[0]) != 3 || records[0][2] != "category" {
					t.Fatalf("%q %v", records, e)
				}
			} else if strings.HasPrefix(w.Header().Get("Content-Type"), "text/csv") || strings.Contains(w.Body.String(), "id,text,category") {
				t.Fatal("partial CSV leaked as successful response")
			}
			files, e := os.ReadDir(tmp)
			if e != nil || len(files) > 0 {
				t.Fatalf("temporary uploads/results not removed: %v %v", files, e)
			}
		})
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/classify", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `type="file"`) || !strings.Contains(w.Body.String(), "FormData") {
		t.Fatalf("upload page: %d", w.Code)
	}
	for _, p := range []string{"/classify", "/v1/systemone/csv"} {
		w := httptest.NewRecorder()
		method := "POST"
		if p != "/classify" {
			method = "GET"
		}
		h.ServeHTTP(w, httptest.NewRequest(method, p, nil))
		if w.Code != 405 {
			t.Fatalf("method %s: %d", p, w.Code)
		}
	}
}
func TestDecisionCSVDeploymentAndDeadline(t *testing.T) {
	model, e := gopherllm.OpenLaya(context.Background(), "../testdata/laya-tiny")
	if e != nil {
		t.Fatal(e)
	}
	defer model.Close()
	for _, mode := range []DeploymentMode{DeploymentBrowser, "wasm"} {
		h := NewHandler(nil, HandlerOptions{DecisionModel: model, DeploymentMode: mode})
		for _, path := range []string{"/classify", "/v1/systemone/csv"} {
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
			if w.Code != 404 {
				t.Fatalf("browser exposed %s: %d", path, w.Code)
			}
		}
		h.Close()
	}
	var endpoint string
	h := NewHandler(nil, HandlerOptions{DecisionModel: model, RequestTimeout: time.Nanosecond, ObserveRequest: func(o RequestObservation) { endpoint = o.Endpoint }})
	defer h.Close()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, decisionCSVUpload(t, "id,text\n1,refund\n", csvTestOptions))
	if w.Code != 504 || endpoint != "/v1/systemone/csv" {
		t.Fatalf("deadline: %d %s", w.Code, endpoint)
	}
	r := decisionCSVUpload(t, "id,text\n1,refund\n", csvTestOptions)
	r.Header.Set("Sec-Fetch-Site", "cross-site")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 403 {
		t.Fatalf("cross-site: %d", w.Code)
	}
}
func TestDecisionCSVOutputCap(t *testing.T) {
	var b bytes.Buffer
	w := decisionCSVWriter{w: &b, remaining: 3}
	if n, e := w.Write([]byte("ab")); n != 2 || e != nil {
		t.Fatal(n, e)
	}
	if n, e := w.Write([]byte("cd")); n != 0 || e != errDecisionCSVOutputLimit || b.String() != "ab" {
		t.Fatal(n, e, b.String())
	}
}
func TestDecisionCSVRejectsInvalidMultipart(t *testing.T) {
	model, e := gopherllm.OpenLaya(context.Background(), "../testdata/laya-tiny")
	if e != nil {
		t.Fatal(e)
	}
	defer model.Close()
	h := NewHandler(nil, HandlerOptions{DecisionModel: model})
	defer h.Close()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone/csv", strings.NewReader("plain csv")))
	if w.Code != 400 {
		t.Fatal(w.Code)
	}
	// Advertise a body that exceeds the cap without allocating a 64 MiB buffer.
	r := httptest.NewRequest("POST", "/v1/systemone/csv", io.MultiReader(strings.NewReader("--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"large.csv\"\r\n\r\n"), io.LimitReader(repeatedCSVByte{}, maxDecisionCSVUpload+1)))
	r.Header.Set("Content-Type", "multipart/form-data; boundary=boundary")
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 413 {
		t.Fatalf("size cap: %d %s", w.Code, w.Body)
	}
}

type repeatedCSVByte struct{}

func (repeatedCSVByte) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'x'
	}
	return len(p), nil
}
