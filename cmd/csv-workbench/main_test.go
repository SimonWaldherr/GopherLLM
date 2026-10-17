package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	g "github.com/SimonWaldherr/GopherLLM"
)

type fakePredictor struct{ state string }

func (p *fakePredictor) Predict(ctx context.Context, request g.DecisionRequest) (g.DecisionResult, error) {
	p.state = string(request.State.(json.RawMessage))
	choice := "Technik"
	return g.DecisionResult{Answers: map[string]g.DecisionAnswer{"result": {Type: "choice", Choice: &choice, AnswerConfidence: .8, Probabilities: map[string]float64{"Technik": .8, "Abrechnung": .2}}}}, nil
}
func TestParseCSV(t *testing.T) {
	tests := []struct {
		name, input, delimiter string
		noHeader, wantErr      bool
		rows                   int
	}{
		{"BOM quoted multiline", "\ufeff\"id\",text\r\n1,\"Hallo;\nWelt\"\r\n", ";", false, true, 0},
		{"BOM and multiline", "\ufeff\"id\",text\r\n1,\"Hallo,\nWelt\"\r\n", ",", false, false, 1},
		{"TSV no header", "1\tText\n2\tNoch mehr\n", "\t", true, false, 2},
		{"semicolon", "id;text\n1;\"Hallo; Welt\"\n", ";", false, false, 1},
		{"duplicates", "text,text\nx,y\n", ",", false, true, 0},
		{"ragged", "id,text\n1\n", ",", false, true, 0},
		{"empty", "", ",", false, true, 0},
		{"header only", "id,text\n", ",", false, true, 0},
		{"bad delimiter", "id,text\n1,hello\n", "\n", false, true, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := parseCSV(strings.NewReader(tt.input), tt.delimiter, tt.noHeader)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error=%v", err)
			}
			if err == nil && len(data.Rows) != tt.rows {
				t.Fatalf("rows=%d", len(data.Rows))
			}
		})
	}
	input := "text\n" + strings.Repeat("hello\n", maxRows+1)
	if _, err := parseCSV(strings.NewReader(input), ",", false); err == nil {
		t.Fatal("row limit not enforced")
	}
}
func call(a *app, path, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "http://127.0.0.1:8092"+path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	a.handler().ServeHTTP(w, r)
	return w
}
func TestClassify(t *testing.T) {
	p := &fakePredictor{}
	a := &app{predictor: p, gate: make(chan struct{}, 1)}
	w := call(a, "/api/classify", `{"state":"Laptop defekt","question":{"type":"choice","instructions":"Team?","criteria":["Technik","Abrechnung"]}}`)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"choice":"Technik"`) {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if p.state != `"Laptop defekt"` {
		t.Fatal(p.state)
	}
	for _, body := range []string{`{"state":"x","question":{"type":"choice","instructions":"","criteria":["a"]}}`, `{"state":"x","question":{"type":"choice","instructions":"Team?","criteria":["a"]}} {}`} {
		if call(a, "/api/classify", body).Code == 200 {
			t.Fatal("invalid request accepted")
		}
	}
}
func TestExportRoundTrip(t *testing.T) {
	a := &app{}
	w := call(a, "/api/export", `{"headers":["text","category","category_model","category_probability","category_corrected"],"rows":[["Hallo;\nWelt","Abrechnung","Technik","0.8","true"]],"delimiter":";","no_header":false}`)
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	reader := csv.NewReader(w.Body)
	reader.Comma = ';'
	rows, err := reader.ReadAll()
	if err != nil || len(rows) != 2 || rows[1][0] != "Hallo;\nWelt" || rows[1][4] != "true" {
		t.Fatalf("%v %v", rows, err)
	}
	if !strings.Contains(w.Header().Get("Content-Disposition"), "classified.csv") {
		t.Fatal("missing attachment")
	}
	w = call(a, "/api/export", `{"headers":["text"],"rows":[["x","extra"]],"delimiter":","}`)
	if w.Code != 422 {
		t.Fatal(w.Code)
	}
}

func TestImportRejectsInvalidUTF8(t *testing.T) {
	for _, tc := range []struct {
		name, input string
	}{
		{"data", "text\nM\xfcller\n"},
		{"header", "na\xffme\nvalue\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body bytes.Buffer
			form := multipart.NewWriter(&body)
			file, err := form.CreateFormFile("file", "input.csv")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := file.Write([]byte(tc.input)); err != nil {
				t.Fatal(err)
			}
			if err := form.WriteField("delimiter", ","); err != nil {
				t.Fatal(err)
			}
			if err := form.Close(); err != nil {
				t.Fatal(err)
			}
			r := httptest.NewRequest("POST", "http://127.0.0.1:8092/api/import", &body)
			r.Header.Set("Content-Type", form.FormDataContentType())
			w := httptest.NewRecorder()
			(&app{}).handler().ServeHTTP(w, r)
			if w.Code != http.StatusUnprocessableEntity || !strings.Contains(w.Body.String(), "UTF-8") {
				t.Fatalf("non-UTF-8 CSV was accepted or not explained: status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
}

func TestParseCSVPreservesQuotedLeadingBOM(t *testing.T) {
	for _, tc := range []struct {
		name, input, want string
		noHeader          bool
	}{
		{"file BOM", "\ufeff\"first\",x\n", "first", true},
		{"quoted data", "\"\ufefffirst\",x\n", "\ufefffirst", true},
		{"quoted header", "\"\ufefftext\",other\nfirst,second\n", "\ufefftext", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := parseCSV(strings.NewReader(tc.input), ",", tc.noHeader)
			if err != nil {
				t.Fatal(err)
			}
			got := data.Headers[0]
			if tc.noHeader {
				got = data.Rows[0][0]
			}
			if got != tc.want {
				t.Fatalf("first CSV field = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestAssistRejectsDuplicateCategories(t *testing.T) {
	a := &app{gate: make(chan struct{}, 1), generate: func(context.Context, string, string) (string, error) {
		return `{"instructions":"Team?","criteria":["Technik","Technik"]}`, nil
	}}
	w := call(a, "/api/assist", `{"mode":"suggest","task":"Team?"}`)
	if w.Code != http.StatusBadGateway {
		t.Fatalf("unusable suggestion accepted: status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestExportRejectsRuneErrorDelimiter(t *testing.T) {
	w := call(&app{}, "/api/export", `{"headers":["text"],"rows":[["hello"]],"delimiter":"\ufffd"}`)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("invalid writer delimiter accepted: status=%d body=%q", w.Code, w.Body.String())
	}
}
func TestAssistAndProtection(t *testing.T) {
	a := &app{gate: make(chan struct{}, 1)}
	if call(a, "/api/assist", `{"mode":"suggest","task":"Team?"}`).Code != 503 {
		t.Fatal("missing model accepted")
	}
	a.generate = func(context.Context, string, string) (string, error) {
		return "```json\n{\"instructions\":\"Team?\",\"criteria\":[\"Technik\",\"Abrechnung\"]}\n```", nil
	}
	if w := call(a, "/api/assist", `{"mode":"suggest","task":"Team?"}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	a.generate = func(context.Context, string, string) (string, error) {
		return `{"instructions":"Team?","criteria":[]}`, nil
	}
	if call(a, "/api/assist", `{"mode":"suggest","task":"Team?"}`).Code != 502 {
		t.Fatal("invalid suggestion accepted")
	}
	r := httptest.NewRequest("POST", "http://evil.example/api/export", strings.NewReader(`{}`))
	w := httptest.NewRecorder()
	a.handler().ServeHTTP(w, r)
	if w.Code != http.StatusForbidden {
		t.Fatal("foreign Host accepted")
	}
	r = httptest.NewRequest("POST", "http://127.0.0.1:8092/api/export", strings.NewReader(`{}`))
	r.Header.Set("Origin", "https://evil.example")
	w = httptest.NewRecorder()
	a.handler().ServeHTTP(w, r)
	if w.Code != http.StatusForbidden {
		t.Fatal("cross-origin write accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.gate <- struct{}{}
	cancel()
	if a.acquire(ctx) == nil {
		t.Fatal("cancelled acquisition succeeded")
	}
}
