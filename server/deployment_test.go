package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDeploymentModeParsingAndLocalBindPolicy(t *testing.T) {
	for _, tc := range []struct {
		input string
		want  DeploymentMode
	}{
		{"", DeploymentLocal},
		{"local", DeploymentLocal},
		{"managed", DeploymentManaged},
		{"server", DeploymentManaged},
		{"browser", DeploymentBrowser},
		{"wasm-webgpu", DeploymentBrowser},
	} {
		got, err := ParseDeploymentMode(tc.input)
		if err != nil || got != tc.want {
			t.Fatalf("ParseDeploymentMode(%q) = %q, %v; want %q, nil", tc.input, got, err, tc.want)
		}
	}
	if _, err := ParseDeploymentMode("internet"); err == nil {
		t.Fatal("ParseDeploymentMode accepted an unknown mode")
	}
	for _, addr := range []string{"127.0.0.1:8080", "localhost:8080", "[::1]:8080"} {
		if err := validateDeploymentOptions(DeploymentLocal, "", addr, "", false, true); err != nil {
			t.Fatalf("local address %q rejected: %v", addr, err)
		}
	}
	for _, addr := range []string{":8080", "0.0.0.0:8080", "[::]:8080", "192.168.1.3:8080"} {
		if err := validateDeploymentOptions(DeploymentLocal, "", addr, "", false, true); err == nil {
			t.Fatalf("non-loopback local address %q was accepted", addr)
		}
	}
}

func TestManagedDeploymentProtectsServerControlsButNotGeneration(t *testing.T) {
	const token = "not-in-any-response"
	h := NewHandler(nil, HandlerOptions{DeploymentMode: DeploymentManaged, AdminToken: token, ChatUI: true})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	request := func(method, path, body string, admin bool) *http.Response {
		t.Helper()
		req, err := http.NewRequest(method, srv.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		if admin {
			req.Header.Set(AdminTokenHeader, token)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodPost, "/models/load", `{}`},
		{http.MethodPost, "/models/download", `{}`},
		{http.MethodGet, "/models/search?q=qwen", ""},
		{http.MethodPost, "/autotune/run", `{}`},
		{http.MethodPost, "/remote", `{}`},
		{http.MethodPost, "/agentos/execute", `{}`},
	} {
		resp := request(tc.method, tc.path, tc.body, false)
		if resp.StatusCode != http.StatusForbidden {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Errorf("%s %s without token = %d (%s), want 403", tc.method, tc.path, resp.StatusCode, body)
			continue
		}
		resp.Body.Close()
	}

	// The same protected route reaches its normal configuration response once
	// the token is present, proving the middleware is not merely hiding UI.
	resp := request(http.MethodPost, "/models/load", `{}`, true)
	if resp.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("authorized model load = %d (%s), want its ordinary 404", resp.StatusCode, body)
	}
	resp.Body.Close()

	// Public inference is intentionally not an admin-only operation in the
	// managed profile. With no model it fails for the regular 503 reason, not
	// with an authorization response.
	resp = request(http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, false)
	if resp.StatusCode != http.StatusServiceUnavailable {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("managed generation = %d (%s), want 503 without a loaded model", resp.StatusCode, body)
	}
	resp.Body.Close()

	resp = request(http.MethodGet, "/deployment", "", false)
	var public map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&public); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if public["mode"] != string(DeploymentManaged) || public["admin"] != false || strings.Contains(fmtJSON(public), token) {
		t.Fatalf("public deployment status leaked or was wrong: %+v", public)
	}

	req, err := http.NewRequest(http.MethodGet, srv.URL+"/deployment", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err = srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var private map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&private); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if private["admin"] != true || strings.Contains(fmtJSON(private), token) {
		t.Fatalf("authorized deployment status leaked or was wrong: %+v", private)
	}

	resp = request(http.MethodGet, "/chat", "", false)
	page, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || strings.Contains(string(page), token) || !strings.Contains(string(page), `data-admin-required="true"`) {
		t.Fatalf("managed chat template leaked token or lacks safe mode metadata: status=%d page=%s", resp.StatusCode, page)
	}
}

func TestBrowserDeploymentDisablesServerInferenceAndModelControls(t *testing.T) {
	wasmDir := t.TempDir()
	for _, name := range []string{"gopherllm.wasm", "wasm_exec.js"} {
		if err := os.WriteFile(filepath.Join(wasmDir, name), []byte("test"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	h := NewHandler(nil, HandlerOptions{
		DeploymentMode: DeploymentBrowser,
		ChatUI:         true,
		WasmDir:        wasmDir,
		ModelDir:       t.TempDir(),
	})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	for _, path := range []string{"/v1/chat/completions", "/models", "/models/search?q=qwen", "/autotune", "/agentos/status"} {
		req, err := http.NewRequest(http.MethodGet, srv.URL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusNotFound {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Errorf("browser %s = %d (%s), want 404", path, resp.StatusCode, body)
			continue
		}
		resp.Body.Close()
	}

	resp, err := srv.Client().Get(srv.URL + "/deployment")
	if err != nil {
		t.Fatal(err)
	}
	var status map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if status["mode"] != string(DeploymentBrowser) || status["browser_inference"] != true || status["server_inference"] != false {
		t.Fatalf("browser deployment status = %+v", status)
	}

	resp, err = srv.Client().Get(srv.URL + "/chat")
	if err != nil {
		t.Fatal(err)
	}
	page, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(page), `data-browser-only="true"`) || !strings.Contains(string(page), `data-deployment-mode="browser"`) {
		t.Fatalf("browser chat template has wrong deployment metadata: status=%d page=%s", resp.StatusCode, page)
	}
}

func fmtJSON(value any) string {
	data, _ := json.Marshal(value)
	return string(data)
}
