// This module simulates an external application importing GopherLLM as a
// dependency (via a replace directive pointing at the repo root). The
// TestExternalConsumerBuilds regression test in the root package builds it to
// prove the module is actually importable — the go tool ignores testdata/
// directories, so this nested module doesn't interfere with the main build.
module consumer

go 1.25

require github.com/SimonWaldherr/GopherLLM v0.0.0

replace github.com/SimonWaldherr/GopherLLM => ../..
