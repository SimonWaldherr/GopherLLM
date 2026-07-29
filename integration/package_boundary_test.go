package integration_test

import (
	"os/exec"
	"strings"
	"testing"
)

// TestInferencePackageStaysFreeOfServerDependencies is the regression test for
// the package split: applications that import GopherLLM only to run inference
// must not drag in the HTTP server, the template engine, or the embedded web
// UI. Those live in the server subpackage.
//
// This is checked against the real dependency graph rather than by reading
// imports, so it also catches a transitive reintroduction (a new helper in the
// root package importing something that itself pulls in net/http).
func TestInferencePackageStaysFreeOfServerDependencies(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/SimonWaldherr/GopherLLM").CombinedOutput()
	if err != nil {
		t.Skipf("go list unavailable: %v (%s)", err, out)
	}
	deps := make(map[string]bool)
	for _, line := range strings.Split(string(out), "\n") {
		deps[strings.TrimSpace(line)] = true
	}
	// net/http is the expensive one (it pulls in crypto/tls and the whole
	// HTTP/2 stack); html/template and embed are what carried the web UI.
	for _, banned := range []string{"net/http", "html/template", "text/template"} {
		if deps[banned] {
			t.Errorf("inference package depends on %q — HTTP/templating belongs in the server subpackage", banned)
		}
	}
	if !deps["github.com/SimonWaldherr/GopherLLM"] {
		t.Fatalf("go list did not report the package itself; output was:\n%s", out)
	}
}

// TestServerPackageStillProvidesTheHTTPSurface guards the other direction: the
// split is only useful if the server subpackage remains a complete drop-in for
// what used to live in the root package.
func TestServerPackageStillProvidesTheHTTPSurface(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/SimonWaldherr/GopherLLM/server").CombinedOutput()
	if err != nil {
		t.Skipf("go list unavailable: %v (%s)", err, out)
	}
	deps := string(out)
	for _, want := range []string{"net/http", "html/template", "github.com/SimonWaldherr/GopherLLM"} {
		if !strings.Contains(deps, want) {
			t.Errorf("server package does not depend on %q", want)
		}
	}
}
