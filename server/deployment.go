package server

import (
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"strings"
)

// DeploymentMode describes where GopherLLM runs and who may change shared
// server state. It deliberately does not model end-user authentication: a
// managed deployment can put its normal users behind an identity-aware reverse
// proxy while this package keeps its own small, explicit admin boundary.
type DeploymentMode string

const (
	// DeploymentLocal is the single-user profile. Serve only accepts a loopback
	// address for it. Embedders that create their own listener are responsible
	// for binding it to loopback as well.
	DeploymentLocal DeploymentMode = "local"
	// DeploymentManaged is for a server shared by users. Generation remains
	// publicly available to the surrounding deployment, but shared settings and
	// host-side actions require the configured administrator token.
	DeploymentManaged DeploymentMode = "managed"
	// DeploymentBrowser serves the chat application and its WASM runtime, but
	// deliberately never runs a model in the server process. Each browser tab
	// loads its own GGUF and uses WebGPU when available.
	DeploymentBrowser DeploymentMode = "browser"
)

const (
	// AdminTokenHeader is the preferred header for API clients and the Web UI.
	// Authorization: Bearer <token> is accepted as well for standard tooling.
	AdminTokenHeader = "X-GopherLLM-Admin-Token"
)

// ParseDeploymentMode parses a stable deployment-mode value. The empty value
// intentionally preserves the original, local-only behavior for existing
// HandlerOptions users.
func ParseDeploymentMode(value string) (DeploymentMode, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", string(DeploymentLocal):
		return DeploymentLocal, nil
	case string(DeploymentManaged), "server":
		return DeploymentManaged, nil
	case string(DeploymentBrowser), "wasm", "wasm-webgpu":
		return DeploymentBrowser, nil
	default:
		return "", fmt.Errorf("deployment mode must be local, managed, or browser")
	}
}

func (m DeploymentMode) browserOnly() bool { return m == DeploymentBrowser }

func (m DeploymentMode) adminRequired() bool { return m == DeploymentManaged }

// validateDeploymentOptions catches configurations for which there is no safe
// or useful server behavior. NewHandler cannot return an error for historical
// API reasons, so Serve calls this before opening a listener; NewHandler still
// fails closed for privileged routes if it is called directly with bad input.
func validateDeploymentOptions(mode DeploymentMode, adminToken, addr, wasmDir string, chatUI bool, serving bool) error {
	if mode == DeploymentManaged && strings.TrimSpace(adminToken) == "" {
		return errors.New("managed deployment requires an admin token (use --admin-token-file or GOPHERLLM_ADMIN_TOKEN)")
	}
	if mode == DeploymentBrowser {
		if !chatUI {
			return errors.New("browser deployment requires the chat UI")
		}
		if !wasmRuntimeAvailable(wasmDir) {
			return errors.New("browser deployment requires a WasmDir containing gopherllm.wasm and wasm_exec.js")
		}
	}
	if serving && mode == DeploymentLocal && !isLoopbackListenAddress(addr) {
		return fmt.Errorf("local deployment only accepts a loopback --serve address (got %q)", addr)
	}
	return nil
}

func wasmRuntimeAvailable(dir string) bool {
	return strings.TrimSpace(dir) != "" && fileReadable(filepath.Join(dir, "gopherllm.wasm")) && fileReadable(filepath.Join(dir, "wasm_exec.js"))
}

// deploymentAccess owns request-time authorization. The token is never put in
// a template, status document, error, or log line.
type deploymentAccess struct {
	mode       DeploymentMode
	adminToken string
}

func newDeploymentAccess(raw DeploymentMode, adminToken string) deploymentAccess {
	mode, err := ParseDeploymentMode(string(raw))
	if err != nil {
		// Invalid direct-library input must not quietly become a permissive
		// profile. Managed without a token leaves all privileged routes denied.
		mode = DeploymentManaged
		adminToken = ""
	}
	return deploymentAccess{mode: mode, adminToken: strings.TrimSpace(adminToken)}
}

func (a deploymentAccess) adminAuthorized(req *http.Request) bool {
	if a.mode != DeploymentManaged {
		return true
	}
	if a.adminToken == "" || req == nil {
		return false
	}
	provided := strings.TrimSpace(req.Header.Get(AdminTokenHeader))
	if provided == "" {
		authorization := strings.TrimSpace(req.Header.Get("Authorization"))
		if len(authorization) >= len("Bearer ") && strings.EqualFold(authorization[:len("Bearer ")], "Bearer ") {
			provided = strings.TrimSpace(authorization[len("Bearer "):])
		}
	}
	// Hash both values first so the comparison runs over a fixed-size input;
	// a malformed token therefore does not disclose the configured token's
	// length before the constant-time equality check.
	expected := sha256.Sum256([]byte(a.adminToken))
	candidate := sha256.Sum256([]byte(provided))
	return subtle.ConstantTimeCompare(candidate[:], expected[:]) == 1
}

func (a deploymentAccess) status(req *http.Request) map[string]any {
	return map[string]any{
		"mode":              string(a.mode),
		"browser_inference": a.mode.browserOnly(),
		"server_inference":  !a.mode.browserOnly(),
		"admin_required":    a.mode.adminRequired(),
		"admin":             a.adminAuthorized(req),
	}
}

func (a deploymentAccess) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if a.mode.browserOnly() && browserDisabledPath(req.URL.Path) {
			http.Error(w, "server inference and model controls are disabled in browser deployment", http.StatusNotFound)
			return
		}
		if a.mode.adminRequired() && adminOnlyRequest(req) && !a.adminAuthorized(req) {
			w.Header().Set("Cache-Control", "no-store")
			http.Error(w, "administrator authorization is required for this server setting", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, req)
	})
}

// adminOnlyRequest covers changes to process-wide model/runtime state and
// every route that is exclusively used to make those changes. Per-request
// generation settings intentionally remain user-controlled in managed mode.
func adminOnlyRequest(req *http.Request) bool {
	if req == nil {
		return false
	}
	switch req.URL.Path {
	case "/models/load", "/models/embed/load", "/models/download", "/models/download/variants", "/models/search", "/autotune", "/autotune/run":
		return true
	case "/remote":
		return req.Method != http.MethodGet
	default:
		return strings.HasPrefix(req.URL.Path, "/agentos/")
	}
}

// browserDisabledPath is intentionally broader than the generation endpoints:
// browser profile users must not accidentally trigger a server model load or
// discover an abandoned server catalog. Static UI/WASM assets and local-only
// workspace helpers remain available.
func browserDisabledPath(path string) bool {
	if strings.HasPrefix(path, "/models") || strings.HasPrefix(path, "/autotune") || strings.HasPrefix(path, "/remote") || strings.HasPrefix(path, "/agentos") {
		return true
	}
	switch path {
	case "/generate", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/skills", "/api/generate", "/api/chat", "/api/embeddings", "/api/embed", "/api/tags", "/api/ps", "/api/show":
		return true
	default:
		return false
	}
}

func isLoopbackListenAddress(addr string) bool {
	host, _, err := net.SplitHostPort(strings.TrimSpace(addr))
	if err != nil {
		return false
	}
	host = strings.Trim(strings.TrimSpace(host), "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
