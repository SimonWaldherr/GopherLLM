package main

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func modelApp() *app {
	return &app{gate: make(chan struct{}, 1), options: []modelOption{option("laya", "First", "first", ""), option("laya", "Second", "second", ""), option("ministral", "Assistant", "assistant", "")}}
}
func TestSwitchKeepsOldModelOnFailure(t *testing.T) {
	a := modelApp()
	closed := 0
	p := &fakePredictor{}
	a.load = func(context.Context, modelOption) (loadedModel, error) {
		return loadedModel{predictor: p, close: func() error { closed++; return nil }}, nil
	}
	first, second := a.options[0].ID, a.options[1].ID
	if err := a.switchModel(context.Background(), "laya", first); err != nil {
		t.Fatal(err)
	}
	before := a.snapshot()
	a.load = func(context.Context, modelOption) (loadedModel, error) {
		return loadedModel{}, errors.New("broken checkpoint")
	}
	if err := a.switchModel(context.Background(), "laya", second); err == nil {
		t.Fatal("failed load accepted")
	}
	after := a.snapshot()
	if after.LayaID != before.LayaID || after.LayaVersion != before.LayaVersion || a.predictor != p || closed != 0 {
		t.Fatal("old model changed on failure")
	}
	if err := a.switchModel(context.Background(), "laya", ""); err != nil {
		t.Fatal(err)
	}
	if closed != 1 || a.snapshot().Laya || a.snapshot().LayaVersion != before.LayaVersion+1 {
		t.Fatal("unload did not release model or advance version")
	}
}
func TestSwitchReleasesReplacedModelAndCancelledLoad(t *testing.T) {
	a := modelApp()
	closed := 0
	ctx, cancel := context.WithCancel(context.Background())
	a.load = func(_ context.Context, item modelOption) (loadedModel, error) {
		if item.Source == "second" {
			cancel()
		}
		return loadedModel{predictor: &fakePredictor{}, close: func() error { closed++; return nil }}, nil
	}
	if err := a.switchModel(context.Background(), "laya", a.options[0].ID); err != nil {
		t.Fatal(err)
	}
	if err := a.switchModel(ctx, "laya", a.options[1].ID); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if closed != 1 || a.snapshot().LayaID != a.options[0].ID {
		t.Fatal("cancelled load was not released")
	}
	a.load = func(context.Context, modelOption) (loadedModel, error) {
		return loadedModel{predictor: &fakePredictor{}, close: func() error { closed++; return nil }}, nil
	}
	if err := a.switchModel(context.Background(), "laya", a.options[1].ID); err != nil {
		t.Fatal(err)
	}
	if closed != 2 || a.snapshot().LayaVersion != 2 {
		t.Fatal("replacement did not close old checkpoint")
	}
	a.closeModels()
	if closed != 3 {
		t.Fatal("shutdown did not release current model")
	}
}
func TestModelSelectionAndVersions(t *testing.T) {
	a := modelApp()
	a.load = func(_ context.Context, item modelOption) (loadedModel, error) {
		if item.Kind == "ministral" {
			return loadedModel{generate: func(context.Context, string, string) (string, error) { return "Interpretation", nil }}, nil
		}
		return loadedModel{predictor: &fakePredictor{}}, nil
	}
	for _, body := range []string{`{"kind":"laya","id":"/etc/passwd"}`, `{"kind":"other","id":""}`, `{"kind":"ministral","id":"` + a.options[0].ID + `"}`} {
		if call(a, "/api/models/select", body).Code != 422 {
			t.Fatal("invalid model selection accepted")
		}
	}
	if w := call(a, "/api/classify", `{"state":"x","question":{"instructions":"Yes?","type":"noul"}}`); w.Code != 503 {
		t.Fatal("missing classifier accepted")
	}
	if w := call(a, "/api/models/select", `{"kind":"laya","id":"`+a.options[0].ID+`"}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	request := `{"state":"x","model_version":0,"question":{"instructions":"Yes?","type":"noul"}}`
	if w := call(a, "/api/classify", request); w.Code != 409 {
		t.Fatal("stale classifier accepted", w.Code)
	}
	request = strings.Replace(request, `"model_version":0`, `"model_version":1`, 1)
	if w := call(a, "/api/classify", request); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if w := call(a, "/api/models/select", `{"kind":"ministral","id":"`+a.options[2].ID+`"}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if w := call(a, "/api/assist", `{"mode":"explain","task":"Why?","model_version":0,"laya_version":1}`); w.Code != 409 {
		t.Fatal("stale assistant accepted")
	}
	if w := call(a, "/api/assist", `{"mode":"explain","task":"Why?","model_version":1,"laya_version":0}`); w.Code != 409 {
		t.Fatal("stale Laya explanation accepted")
	}
	if w := call(a, "/api/assist", `{"mode":"explain","task":"Why?","model_version":1,"laya_version":1}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	snapshot := a.snapshot()
	if snapshot.LayaID != a.options[0].ID || snapshot.MinistralID != a.options[2].ID {
		t.Fatal("loaded IDs missing")
	}
}
func TestSelectionWaitsForInferenceGate(t *testing.T) {
	a := modelApp()
	entered := make(chan struct{})
	a.load = func(context.Context, modelOption) (loadedModel, error) {
		close(entered)
		return loadedModel{predictor: &fakePredictor{}}, nil
	}
	a.gate <- struct{}{}
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- call(a, "/api/models/select", `{"kind":"laya","id":"`+a.options[0].ID+`"}`) }()
	select {
	case <-entered:
		t.Fatal("loaded during active inference")
	case <-time.After(30 * time.Millisecond):
	}
	// Status remains responsive while a selection is waiting.
	if a.snapshot().Laya {
		t.Fatal("model changed before gate was released")
	}
	<-a.gate
	select {
	case w := <-done:
		if w.Code != 200 {
			t.Fatal(w.Body.String())
		}
	case <-time.After(time.Second):
		t.Fatal("selection did not resume")
	}
}
func TestCatalogLocalLayaAndStableIDs(t *testing.T) {
	root := t.TempDir()
	checkpoint := filepath.Join(root, "multilingual")
	for _, file := range []string{"rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"} {
		path := filepath.Join(checkpoint, file)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(`{}`), 0600); err != nil {
			t.Fatal(err)
		}
	}
	catalog, err := modelCatalog(root)
	if err != nil {
		t.Fatal(err)
	}
	var local modelOption
	for _, item := range catalog {
		if item.Source == checkpoint {
			local = item
		}
	}
	if local.ID == "" || local.Kind != "laya" {
		t.Fatal("local Laya not found")
	}
	if local.ID != option("laya", "different name", checkpoint, "").ID {
		t.Fatal("ID depends on label")
	}
	if _, err := modelCatalog(filepath.Join(root, "missing")); err == nil {
		t.Fatal("missing model directory silently ignored")
	}
}
