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
		`id="deploymentBadge"`,
		`data-deployment-mode="local"`,
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
		`id="modelLoadProgress"`,
		`id="modelLoadStage"`,
		`id="modelLoadBar"`,
		`role="progressbar" aria-label="Model loading progress"`,
		`id="modelLibrary"`,
		`id="modelDownloadForm"`,
		`id="modelDownloadRef"`,
		`id="modelHubSearchForm"`,
		`id="modelHubSearchQuery"`,
		`id="modelHubSearchResults"`,
		`id="modelDownloadVariants"`,
		`id="modelDownloadProgress"`,
		`id="modelDownloadStatus"`,
		`id="workflowSelect"`,
		`id="workflowHelp"`,
		`id="storageMode"`,
		`id="briefShare"`,
		`.xlsx`,
		`.jsonl`,
		`role="tablist" aria-label="Settings sections"`,
		`data-settings-tab="model"`,
		`data-settings-page="generation"`,
		`id="settingsSearch"`,
		`id="settingsSearchEmpty"`,
		`id="engineOverviewTitle"`,
		`Inference engine first`,
		`no separate inference server or web UI is required`,
		`Ollama, llama.cpp, LM Studio, or RustyLLM`,
		`through <code>/remote</code>`,
		`Inference engine · optional workspace UI`,
		`class="settings-disclosure"`,
		`#i-chat`,
		`#i-wrench`,
		`#i-sliders`,
		`#i-layout`,
		`id="liveVisionButton"`,
		`id="liveScreenButton"`,
		`id="cameraCaptureMenu"`,
		`id="screenCaptureMenu"`,
		`id="cameraCaptureSnapshot"`,
		`id="screenCaptureSnapshot"`,
		`id="liveOverlay"`,
		`id="liveOutputText"`,
		`id="liveOutputStatus"`,
		`id="liveActionCondition"`,
		`id="liveActionSound"`,
		`id="liveActionNotify"`,
		`id="liveActionMark"`,
		`id="managedAccessNotice"`,
		`id="adminTokenInput"`,
		`id="adminUnlock"`,
		`data-arm="alert"`,
		`value="384" aria-label="Live frame resolution"`,
		`id="liveContextMode"`,
		`value="change"`,
		`value="timeline"`,
		`id="liveZoneInput"`,
		`id="liveHealthCamera"`,
		`id="liveHealthModel"`,
		`id="liveHealthInference"`,
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
		"function deploymentModeValue()",
		"function adminFetch(",
		"function syncDeploymentControls()",
		"browserOnlyDeployment",
		"adminRequiredDeployment || !available",
		"This managed server uses the model selected by its administrator.",
		`adminFetch("/deployment"`,
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
		"const WORKFLOWS = {",
		"function applyWorkflow(",
		"workflowSelectEl.addEventListener(\"change\"",
		`safety: {`,
		`support: {`,
		"function startModelLoadProgress(",
		"function finishModelLoadProgress(",
		"const modelLoadStages = [",
		"function renderModelDownloadVariants(",
		"function startModelDownload(",
		"function readNDJSON(",
		`"/models/download/variants?ref="`,
		`"/models/search?q="`,
		"function searchModelHub(",
		"function renderModelHubSearchResults(",
		"function cancelModelHubSearch(",
		"function setSettingsTab(",
		"function applySettingsSearch(",
		"function clearSettingsSearch(",
		`name: "/review"`,
		`name: "/plan"`,
		"function parseGoalSpec(",
		`"/batch/parse?filename="`,
		"function waitForLiveVideoFrame(",
		"function buildLiveTimelineCollage(",
		"function restartLiveTimelineSampler(",
		"function liveTimelineFrameLimit(",
		"function setLiveHealth(",
		"sampled one second apart",
		"function startLiveFrameProgress(",
		"function setLiveOutputStatus(",
		"function setLiveCaptureButtonState(",
		"function toggleCaptureMenu(",
		"function playLiveAlertTone(",
		"function parseLiveActions(",
		"function fireLiveAction(",
		"function ensureNotifyPermission(",
		"function fireLiveNotification(",
		"const LIVE_ACTIONS = {",
		"Frame captured — waiting for the model response",
		`"Live screen"`,
		`[ALERT]`,
		`[NOTIFY]`,
		`[MARK]`,
		"gopherllm_skills: useTools ? undefined : false",
		"function modelCapabilities(",
		`model.reasoning ? "Thinking" : "No thinking"`,
		`model.vision ? "Vision" : "No vision"`,
		"maxTokens: liveSettings.maxTokens",
		"maxTokens: Math.min(64",
		"You are describing one current live camera frame.",
	} {
		if !strings.Contains(script, want) {
			t.Errorf("chat script missing interaction contract %q", want)
		}
	}
}
