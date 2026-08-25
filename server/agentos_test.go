package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/agentos"
)

func agentOSTestHandler(t *testing.T, runner *agentos.Runner) http.Handler {
	t.Helper()
	m, err := gopherllm.OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })

	defaults := gopherllm.DefaultGenerationOptions()
	defaults.MaxTokens = 4
	defaults.Sampler.Temperature = 0
	defaults.Sampler.TopK = 1
	return HandlerForModel(m, HandlerOptions{Defaults: defaults, AgentOS: runner})
}

// The feature must not exist at all unless an operator opted in: no policy
// configured means the status endpoint says so, and the action endpoints
// refuse rather than silently doing nothing.
func TestAgentOSDisabledByDefault(t *testing.T) {
	handler := agentOSTestHandler(t, nil)

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/agentos/status", nil))
	var status map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if status["enabled"] != false {
		t.Fatalf("status = %#v, want enabled:false with no Runner configured", status)
	}

	for _, path := range []string{"/agentos/propose", "/agentos/execute"} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`))
		handler.ServeHTTP(rec, req)
		if rec.Code == http.StatusOK {
			t.Fatalf("%s: expected a non-200 refusal when the feature is disabled, got 200: %s", path, rec.Body.String())
		}
	}
}

func TestAgentOSStatusReportsThePolicyAndAllowList(t *testing.T) {
	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyWhitelist, Allowed: []string{"ls", "git"}})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/agentos/status", nil))
	var status struct {
		Enabled bool     `json:"enabled"`
		Policy  string   `json:"policy"`
		Allowed []string `json:"allowed"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if !status.Enabled || status.Policy != "whitelist" || len(status.Allowed) != 2 {
		t.Fatalf("status = %+v", status)
	}
}

func postJSON(t *testing.T, handler http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, bytes.NewReader(data)))
	return rec
}

// This is the pinned safety invariant from the agentos package: the model's
// own "safe" self-rating must never be able to authorize anything by itself.
// Deny mode must refuse a command it rates safe:2 exactly as it refuses one
// rated safe:0, until a human sets approved:true out of band.
func TestExecuteUnderDenyIgnoresTheModelsSafeClaimAndNeedsApproval(t *testing.T) {
	// A real "echo" binary does not exist standalone on Windows outside a
	// Git-Bash-flavored PATH, so this re-execs the test binary itself (see
	// TestAgentOSHelperProcess) instead of relying on an external program.
	t.Setenv("GO_WANT_SERVER_AGENTOS_HELPER_PROCESS", "1")
	helper := `"` + os.Args[0] + `" -test.run=^TestAgentOSHelperProcess$ -- echo:hi`

	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyDeny})

	for _, safe := range []int{0, 1, 2} {
		proposal := map[string]any{"cmd": helper, "dsc": "prints hi", "safe": safe}
		rec := postJSON(t, handler, "/agentos/execute", map[string]any{"proposal": proposal, "approved": false})
		if rec.Code == http.StatusOK {
			t.Fatalf("safe=%d: deny mode ran an unapproved command: %s", safe, rec.Body.String())
		}
	}

	// The same proposal, only a human approval flips it to true, now runs.
	proposal := map[string]any{"cmd": helper, "dsc": "prints hi", "safe": 0}
	rec := postJSON(t, handler, "/agentos/execute", map[string]any{"proposal": proposal, "approved": true})
	if rec.Code != http.StatusOK {
		t.Fatalf("approved command was refused: %s", rec.Body.String())
	}
	var decoded struct {
		Result struct {
			Output string `json:"output"`
		} `json:"result"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(decoded.Result.Output) != "hi" {
		t.Fatalf("output = %q", decoded.Result.Output)
	}
}

// Whitelist mode must refuse shell metacharacters outright — approved:true
// cannot force it through, because Blocked is checked before approval.
func TestExecuteUnderWhitelistBlocksMetacharactersEvenWhenApproved(t *testing.T) {
	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyWhitelist, Allowed: []string{"echo"}})
	proposal := map[string]any{"cmd": "echo hi; rm -rf /", "dsc": "innocuous, allegedly", "safe": 2}
	rec := postJSON(t, handler, "/agentos/execute", map[string]any{"proposal": proposal, "approved": true})
	if rec.Code == http.StatusOK {
		t.Fatalf("whitelist mode ran a command containing a shell metacharacter: %s", rec.Body.String())
	}
}

// A whitelisted program with no approval step needed still requires the
// handler to actually run it (AutoRun), proving the wiring reaches
// Runner.Execute and not just Runner.Evaluate.
//
// This must be a bare program name found via PATH lookup (Runner.Evaluate
// requires args[0] to have no path separators at all before it will
// AutoRun a whitelisted entry), which rules out the self-exec helper trick
// used by the deny-mode test above — an absolute re-exec path can never
// satisfy that check. "go" is used instead of "echo" because Windows has
// no standalone echo.exe, while a `go` binary is guaranteed to be on PATH
// in any environment that can run `go test` in the first place.
func TestExecuteUnderWhitelistAutoRunsAnAllowedProgram(t *testing.T) {
	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyWhitelist, Allowed: []string{"go"}})
	proposal := map[string]any{"cmd": "go version", "dsc": "prints the go toolchain version", "safe": 0}
	rec := postJSON(t, handler, "/agentos/execute", map[string]any{"proposal": proposal, "approved": false})
	if rec.Code != http.StatusOK {
		t.Fatalf("allow-listed program was refused: %s", rec.Body.String())
	}
}

// TestAgentOSHelperProcess is not a real test; it is re-exec'd as a child
// process (see TestExecuteUnderDenyIgnoresTheModelsSafeClaimAndNeedsApproval)
// so agentos tests can stand in for a real "echo" without depending on one
// being on PATH.
func TestAgentOSHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_SERVER_AGENTOS_HELPER_PROCESS") != "1" {
		return
	}
	arg := os.Args[len(os.Args)-1]
	if strings.HasPrefix(arg, "echo:") {
		fmt.Print(strings.TrimPrefix(arg, "echo:"))
		os.Exit(0)
	}
	os.Exit(2)
}

func TestProposeRejectsAnEmptyInstruction(t *testing.T) {
	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyDeny})
	rec := postJSON(t, handler, "/agentos/propose", map[string]any{"instruction": "   "})
	if rec.Code == http.StatusOK {
		t.Fatalf("expected an error for an empty instruction, got 200: %s", rec.Body.String())
	}
}

// The tiny synthetic model's raw output is not valid JSON, so a real
// end-to-end propose call is expected to fail parsing — this pins that the
// handler surfaces that failure clearly rather than, say, panicking or
// silently returning an empty 200.
func TestProposeReportsWhenTheModelDoesNotReturnAParsableProposal(t *testing.T) {
	handler := agentOSTestHandler(t, &agentos.Runner{Policy: agentos.PolicyDeny})
	rec := postJSON(t, handler, "/agentos/propose", map[string]any{"instruction": "list the current directory"})
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}
