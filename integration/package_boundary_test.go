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

// TestInferencePackageStaysFreeOfTheCryptoAndRegexpTrees pins two closure
// reductions that are easy to undo by accident, because both are one innocuous
// import away and neither shows up as a build failure.
//
// crypto/sha256 is the expensive one by far: importing it anywhere in the root
// package pulls the FIPS-140 tree (aes, gcm, sha3, sha512, hmac, drbg,
// sysrand, ...) behind it, which was 27 of the root package's stdlib
// dependencies when it was used for nothing but two cache-key hashes. Those are
// FNV now — see autoTuneKey and metalProbeKey, which document why a
// non-cryptographic hash is the correct tool for naming a file in the caller's
// own cache directory. regexp is cheaper in package count but not in binary
// size (~458 KB), and it too had exactly one use, parsing the fixed-shape
// gguf-split filename that splitPathPrefix now reads by hand.
//
// Neither ban is absolute for the module: the server and huggingface packages
// use both freely, and test files may too, since neither reaches an embedding
// consumer's closure. This is only about what the inference library charges a
// program that links it.
func TestInferencePackageStaysFreeOfTheCryptoAndRegexpTrees(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/SimonWaldherr/GopherLLM").CombinedOutput()
	if err != nil {
		t.Skipf("go list unavailable: %v (%s)", err, out)
	}
	deps := make(map[string]bool)
	for _, line := range strings.Split(string(out), "\n") {
		deps[strings.TrimSpace(line)] = true
	}
	for _, banned := range []string{"crypto/sha256", "regexp"} {
		if deps[banned] {
			t.Errorf("inference package depends on %q again — see this test's comment for why that is more expensive than it looks", banned)
		}
	}
	// Catch the tree rather than only the entry point: any other crypto hash
	// reintroduces the same packages without naming crypto/sha256.
	for dep := range deps {
		if strings.HasPrefix(dep, "crypto/internal/fips140") {
			t.Errorf("inference package pulled the FIPS-140 crypto tree back in (via %q)", dep)
			break
		}
	}
	if !deps["github.com/SimonWaldherr/GopherLLM"] {
		t.Fatalf("go list did not report the package itself; output was:\n%s", out)
	}
}

// TestImageDecodersAreDroppable pins the other half of the same reduction: the
// PNG and JPEG decoders are registered from image_decoders.go so that
// -tags noimagedecoders can remove them, and with them compress/zlib,
// compress/flate, hash/adler32 and hash/crc32. That only works while the blank
// imports live in that one tagged file, and moving either of them back into an
// untagged file would silently make the tag a no-op.
//
// The image package itself must stay in both builds: image.Image is in
// DecodeImageBytes' and PreprocessImagePixtral's exported signatures, so
// dropping it would be an API break rather than a build-time knob.
func TestImageDecodersAreDroppable(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "-tags", "noimagedecoders", "github.com/SimonWaldherr/GopherLLM").CombinedOutput()
	if err != nil {
		t.Skipf("go list unavailable: %v (%s)", err, out)
	}
	deps := make(map[string]bool)
	for _, line := range strings.Split(string(out), "\n") {
		deps[strings.TrimSpace(line)] = true
	}
	for _, banned := range []string{"image/png", "image/jpeg", "compress/zlib", "compress/flate", "hash/adler32", "hash/crc32"} {
		if deps[banned] {
			t.Errorf("-tags noimagedecoders build still depends on %q; are the decoder imports still confined to image_decoders.go?", banned)
		}
	}
	if !deps["image"] {
		t.Error("image is missing from the tagged build, but image.Image is part of the exported API")
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

// TestHuggingFacePackageIsTheOptInNetworkSurface pins the reason the Hub
// client lives in its own package rather than in the root one: net/http is
// exactly what the root package must not carry, so Hub access has to be
// something a caller opts into by importing it.
//
// Without this, the natural "just re-export it from the root package for
// convenience" change silently reintroduces crypto/tls and the HTTP/2 stack
// into every program that only ever loads a local file.
func TestHuggingFacePackageIsTheOptInNetworkSurface(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/SimonWaldherr/GopherLLM/huggingface").CombinedOutput()
	if err != nil {
		t.Skipf("go list unavailable: %v (%s)", err, out)
	}
	deps := string(out)
	// It is the network surface, so it must actually have the client...
	if !strings.Contains(deps, "net/http") {
		t.Error("huggingface package does not depend on net/http; is it still a Hub client?")
	}
	// ...and it must not drag the whole inference engine along with it, so a
	// downloader-only tool stays small.
	if strings.Contains(deps, "html/template") {
		t.Error("huggingface package pulled in html/template; templating belongs in the server subpackage")
	}
}
