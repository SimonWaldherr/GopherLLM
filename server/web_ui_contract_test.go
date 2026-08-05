package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestChatUICompactInteractionContracts protects the deliberately small
// composer and the browser-side affordances that make its less-visible
// controls discoverable.  This stays at the HTTP asset boundary: it does not
// prescribe layout pixels or implementation details of the renderer.
func TestChatUICompactInteractionContracts(t *testing.T) {
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ChatUI: true}))
	t.Cleanup(srv.Close)

	get := func(path, contentType string) string {
		t.Helper()
		resp, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s status = %d", path, resp.StatusCode)
		}
		if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, contentType) {
			t.Fatalf("GET %s content-type = %q, want %q", path, got, contentType)
		}
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		return string(body)
	}

	html := get("/chat", "text/html")
	if strings.Contains(html, `id="clear"`) {
		t.Fatal("compact composer must not restore the duplicate clear/new-chat control")
	}
	for _, want := range []string{
		`id="composerProToggle"`,
		`aria-controls="composerProPanel"`,
		`aria-expanded="false"`,
		`>Controls <span aria-hidden="true">`,
		`id="composerProPanel"`,
		`id="a11yStatus"`,
		`role="status" aria-live="polite" aria-atomic="true"`,
		`id="modelSearch"`,
		`id="modelShowUnsupported"`,
		`id="activeModelSummary"`,
		`id="modelLibrary"`,
		`id="storageMode"`,
		`id="briefShare"`,
		`.xlsx`,
		`.jsonl`,
		`role="tablist" aria-label="Settings sections"`,
		`data-settings-tab="model"`,
		`data-settings-page="generation"`,
	} {
		if !strings.Contains(html, want) {
			t.Errorf("chat page missing compact-composer/accessibility contract %q", want)
		}
	}

	// Every modal surface is a real modal dialog, rather than a merely visual
	// overlay. The browser code owns focus trapping; the markup owns this
	// semantic declaration.
	for _, want := range []string{
		`role="dialog" aria-modal="true" aria-labelledby="briefingTitle"`,
		`role="dialog" aria-modal="true" aria-labelledby="settingsTitle"`,
		`role="dialog" aria-modal="true" aria-labelledby="batchTitle"`,
		`role="dialog" aria-modal="true" aria-labelledby="agentosTitle"`,
	} {
		if !strings.Contains(html, want) {
			t.Errorf("chat page missing modal-dialog contract %q", want)
		}
	}

	script := get("/script.js", "text/javascript")
	if strings.Contains(script, `$("clear")`) {
		t.Error("script still expects the removed clear control")
	}
	for _, want := range []string{
		"function openDialog(",
		"function dialogFocusables(",
		"function trapDialogFocus(",
		"function renderActivityDisclosure(",
		"function recoverMistralToolCalls(",
		`className = "activity-details"`,
		`button.textContent = "Details"`,
		`button.textContent = "Copy message"`,
		`action.textContent = "Regenerate"`,
		`changeModel.textContent = "Change model"`,
		"function changeModelForMessage(",
		"function filterModelOptions(",
		"function renderModelLibrary(",
		"function renderActiveModelSummary(",
		"function setSettingsTab(",
		`name: "/review"`,
		`name: "/plan"`,
		"function parseGoalSpec(",
		`"/batch/parse?filename="`,
	} {
		if !strings.Contains(script, want) {
			t.Errorf("chat script missing interaction contract %q", want)
		}
	}
}
