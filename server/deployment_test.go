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
	if err := validateDeploymentOptions(DeploymentLocal, "", "", false); err != nil {
		t.Fatalf("local deployment rejected: %v", err)
	}
	for _, addr := range []string{"127.0.0.1:8080", "localhost:8080", "[::1]:8080"} {
		if warning := NetworkExposureWarning(DeploymentLocal, addr, ""); warning != "" {
			t.Fatalf("loopback address %q warned: %q", addr, warning)
		}
	}
	for _, addr := range []string{":8080", "0.0.0.0:8080", "[::]:8080", "192.168.1.3:8080"} {
		warning := NetworkExposureWarning(DeploymentLocal, addr, "")
		if warning == "" {
			t.Fatalf("non-loopback address %q produced no warning", addr)
		}
		if !strings.Contains(warning, "reachable from your network") {
			t.Fatalf("warning for %q does not say what changes: %q", addr, warning)
		}
	}
	// A managed deployment with a token keeps its controls, so the warning
	// must not claim they were disabled.
	managed := NetworkExposureWarning(DeploymentManaged, "0.0.0.0:8080", "token")
	if managed == "" || strings.Contains(managed, "disabled while exposed") {
		t.Fatalf("managed warning should not claim a lockout: %q", managed)
	}
}

// TestLocalDeploymentOnNetworkLocksPrivilegedRoutes covers the security
// consequence of the loopback default being relaxed: exposing a local server
// to the network is allowed, but it must not hand every caller on the subnet
// the ability to load models, download GGUFs, or retune the process.
func TestLocalDeploymentOnNetworkLocksPrivilegedRoutes(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{
		NetworkExposed: true,
		Features:       AllFeatures(),
		ChatUI:         true,
	})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	for _, tc := range []struct{ method, path string }{
		{http.MethodPost, "/models/load"},
		{http.MethodPost, "/models/download"},
		{http.MethodGet, "/models/search"},
		{http.MethodPost, "/autotune/run"},
		{http.MethodPost, "/remote"},
	} {
		req, err := http.NewRequest(tc.method, srv.URL+tc.path, strings.NewReader("{}"))
		if err != nil {
			t.Fatalf("build %s %s: %v", tc.method, tc.path, err)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", tc.method, tc.path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("%s %s = %d, want 403 while exposed to the network", tc.method, tc.path, resp.StatusCode)
		}
	}
	// Generation itself stays available: exposing the server is the point.
	resp, err := srv.Client().Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /health = %d, want 200", resp.StatusCode)
	}
}

// TestDisabledFeaturesAreNotRegistered checks the stronger half of the
// promise: a capability the operator did not enable answers 404, so it is
// absent rather than merely guarded.
func TestDisabledFeaturesAreNotRegistered(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{ChatUI: true})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	for _, path := range []string{"/models", "/models/search", "/models/download", "/autotune", "/remote", "/batch/parse"} {
		resp, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("GET %s = %d, want 404 with no features enabled", path, resp.StatusCode)
		}
	}
	resp, err := srv.Client().Get(srv.URL + "/deployment")
	if err != nil {
		t.Fatalf("GET /deployment: %v", err)
	}
	defer resp.Body.Close()
	var status struct {
		Features map[string]bool `json:"features"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatalf("decode /deployment: %v", err)
	}
	if len(status.Features) != len(FeatureNames()) {
		t.Fatalf("/deployment reported %d features, want %d", len(status.Features), len(FeatureNames()))
	}
	for name, on := range status.Features {
		if on {
			t.Fatalf("/deployment reports %q enabled with a zero-value Features", name)
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

// TestCrossSiteWritesAreRejectedButAPIClientsAreNot pins both halves of the
// same-site check, because either half failing is silently bad: too strict and
// every non-browser API client breaks with no browser involved to explain it,
// too lax and any page the user visits can repoint POST /remote at a server of
// its choosing and harvest the whole conversation.
//
// The endpoint under test is /remote precisely because it is the worst case: it
// is a write whose effect outlives the request, and the attacker never needs to
// read the response.
func TestCrossSiteWritesAreRejectedButAPIClientsAreNot(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	do := func(method, path string, headers map[string]string) int {
		req, err := http.NewRequest(method, srv.URL+path, strings.NewReader(`{"base_url":"https://example.test/v1"}`))
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}

	for _, tc := range []struct {
		name    string
		method  string
		path    string
		headers map[string]string
		blocked bool
	}{
		// curl, the OpenAI SDKs and every agent framework send neither header.
		{"bare api client", http.MethodPost, "/remote", nil, false},
		{"our own page", http.MethodPost, "/remote", map[string]string{"Sec-Fetch-Site": "same-origin"}, false},
		{"typed url or bookmark", http.MethodPost, "/remote", map[string]string{"Sec-Fetch-Site": "none"}, false},

		{"drive-by write", http.MethodPost, "/remote", map[string]string{"Sec-Fetch-Site": "cross-site"}, true},
		// A sibling origin sharing a registrable domain is not a trust relationship.
		{"sibling origin", http.MethodPost, "/remote", map[string]string{"Sec-Fetch-Site": "same-site"}, true},
		// Browsers predating fetch metadata still label a cross-origin write.
		{"legacy origin header", http.MethodPost, "/remote", map[string]string{"Origin": "https://evil.example"}, true},
		// A text/plain body is a CORS simple request: no preflight is ever sent,
		// so the Content-Type must not be what the decision rests on.
		{"simple request smuggling json", http.MethodPost, "/remote", map[string]string{"Sec-Fetch-Site": "cross-site", "Content-Type": "text/plain"}, true},

		// Safe methods stay reachable cross-site; refusing them would break
		// ordinary linking without closing the forgery hole.
		{"cross-site read", http.MethodGet, "/deployment", map[string]string{"Sec-Fetch-Site": "cross-site"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := do(tc.method, tc.path, tc.headers)
			if blocked := got == http.StatusForbidden; blocked != tc.blocked {
				t.Fatalf("status %d: blocked=%v, want blocked=%v", got, blocked, tc.blocked)
			}
		})
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
