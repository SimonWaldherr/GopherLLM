package main

import (
	"testing"

	"github.com/SimonWaldherr/GopherLLM/agentos"
)

func TestParseCLISamplerFlags(t *testing.T) {
	cfg, err := parseCLI([]string{
		"model.gguf",
		"--temp", "0.5", "--top-p", "0.85", "--top-k", "10", "--min-p", "0.1",
		"--repeat-penalty", "1.2", "--max-tokens", "64", "--seed", "42", "--prompt", "hi",
	})
	if err != nil {
		t.Fatalf("parseCLI: %v", err)
	}
	if cfg.modelSelector == nil || *cfg.modelSelector != "model.gguf" {
		t.Fatalf("modelSelector = %v", cfg.modelSelector)
	}
	s := cfg.options.Sampler
	if s.Temperature != 0.5 || s.TopP != 0.85 || s.TopK != 10 || s.MinP != 0.1 || s.RepeatPenalty != 1.2 {
		t.Fatalf("sampler = %+v", s)
	}
	if cfg.options.MaxTokens != 64 || cfg.options.Seed != 42 || cfg.prompt != "hi" {
		t.Fatalf("maxtok=%d seed=%d prompt=%q", cfg.options.MaxTokens, cfg.options.Seed, cfg.prompt)
	}
}

func TestParseCLIServeAndStops(t *testing.T) {
	cfg, err := parseCLI([]string{
		"m", "--serve", "127.0.0.1:8080", "--chat",
		"--system-prompt", "be nice", "--stop", "END", "--stop", "STOP", "--repl",
		"--skills-dir", "/tmp/skills",
	})
	if err != nil {
		t.Fatalf("parseCLI: %v", err)
	}
	if cfg.serveAddr != "127.0.0.1:8080" || !cfg.chatUI || !cfg.repl {
		t.Fatalf("serve=%q chat=%v repl=%v", cfg.serveAddr, cfg.chatUI, cfg.repl)
	}
	if cfg.options.SystemPrompt != "be nice" {
		t.Fatalf("system = %q", cfg.options.SystemPrompt)
	}
	if len(cfg.options.StopSequences) != 2 || cfg.options.StopSequences[0] != "END" {
		t.Fatalf("stops = %v", cfg.options.StopSequences)
	}
	if cfg.skillsDir != "/tmp/skills" {
		t.Fatalf("skillsDir = %q", cfg.skillsDir)
	}
}

func TestParseCLIOutOfCore(t *testing.T) {
	cfg, err := parseCLI([]string{"model.gguf", "--out-of-core"})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.outOfCore {
		t.Fatal("--out-of-core was not recorded")
	}
}

func TestParseCLICompressFlags(t *testing.T) {
	cfg, err := parseCLI([]string{"model.gguf", "--compress", "--compress-format", "Q4_K", "--compress-out", "out.gguf", "--compress-uniform"})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.compress || cfg.compressFormat != "Q4_K" || cfg.compressOut != "out.gguf" || !cfg.compressUniform {
		t.Fatalf("compress flags not recorded correctly: %+v", cfg)
	}
}

func TestRunCompressRequiresOutPath(t *testing.T) {
	cfg, err := parseCLI([]string{"model.gguf", "--compress", "--compress-format", "Q4_K"})
	if err != nil {
		t.Fatal(err)
	}
	if err := runCompress("model.gguf", cfg); err == nil {
		t.Fatal("runCompress without --compress-out: want error, got nil")
	}
}

func TestRunCompressRejectsUnknownFormat(t *testing.T) {
	cfg, err := parseCLI([]string{"model.gguf", "--compress", "--compress-format", "bogus", "--compress-out", "out.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if err := runCompress("model.gguf", cfg); err == nil {
		t.Fatal("runCompress with an unsupported --compress-format: want error, got nil")
	}
}

func TestParseCLIHuggingFaceList(t *testing.T) {
	cfg, err := parseCLI([]string{"--hf-list", "bartowski/Qwen3-4B-GGUF@main", "--hf-offline"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.hfList != "bartowski/Qwen3-4B-GGUF@main" || !cfg.hfOffline {
		t.Fatalf("hf list/offline = %q/%v", cfg.hfList, cfg.hfOffline)
	}
}

func TestParseCLIPrivacyReport(t *testing.T) {
	cfg, err := parseCLI([]string{"--privacy"})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.privacyReport {
		t.Fatal("--privacy was not recorded")
	}
}

func TestParseCLIAutoFlags(t *testing.T) {
	// Each of the auto sub-flags implies --auto, so none of them can be passed
	// without actually turning tuning on.
	for _, tc := range []struct {
		args    []string
		refresh bool
		json    bool
		effort  string
	}{
		{args: []string{"m", "--auto"}},
		{args: []string{"m", "--auto-refresh"}, refresh: true},
		{args: []string{"m", "--auto-json"}, json: true},
		{args: []string{"m", "--auto-effort", "thorough"}, effort: "thorough"},
		{args: []string{"m", "--auto-effort", "quick"}, effort: "quick"},
	} {
		cfg, err := parseCLI(tc.args)
		if err != nil {
			t.Fatalf("parseCLI(%v): %v", tc.args, err)
		}
		if !cfg.autoTune {
			t.Fatalf("parseCLI(%v) did not enable auto tuning", tc.args)
		}
		if cfg.autoTuneRefresh != tc.refresh || cfg.autoTuneJSON != tc.json || cfg.autoTuneEffort != tc.effort {
			t.Fatalf("parseCLI(%v) = refresh:%v json:%v effort:%q, want %v/%v/%q",
				tc.args, cfg.autoTuneRefresh, cfg.autoTuneJSON, cfg.autoTuneEffort,
				tc.refresh, tc.json, tc.effort)
		}
	}
	// Every effort level must map to usable, non-zero calibration options.
	for _, effort := range []string{"", "quick", "balanced", "thorough"} {
		o := autoTuneOptions(effort)
		if o.Rounds < 1 || o.DecodeSteps < 1 || o.Context < 1 || o.MinGain <= 0 {
			t.Fatalf("autoTuneOptions(%q) = %+v, want usable values", effort, o)
		}
	}
	// quick deliberately skips prefill tuning: one prefill sample costs a whole
	// chunk of prompt processing.
	if autoTuneOptions("quick").TunePrefill {
		t.Fatal("quick effort should not tune prefill")
	}
	if !autoTuneOptions("balanced").TunePrefill {
		t.Fatal("balanced effort should tune prefill")
	}
}

func TestParseCLIErrors(t *testing.T) {
	cases := [][]string{
		{"--unknown-flag"},              // unknown option
		{"m", "--temp"},                 // missing value
		{"m", "--max-tokens", "xx"},     // invalid int
		{"m", "--temp", "notanum"},      // invalid float
		{"--chat"},                      // --chat without --serve
		{"m", "--auto-effort", "turbo"}, // unknown effort level
		{"m", "--auto-effort"},          // missing effort value
	}
	for _, args := range cases {
		if _, err := parseCLI(args); err == nil {
			t.Fatalf("parseCLI(%v) expected error", args)
		}
	}
}

func TestParseCLIOSCommandsFlags(t *testing.T) {
	cfg, err := parseCLI([]string{"m", "--os-commands", "whitelist", "--os-commands-allow", "ls, git ,cat"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.osCommandsPolicy != "whitelist" || cfg.osCommandsAllow != "ls, git ,cat" {
		t.Fatalf("policy=%q allow=%q", cfg.osCommandsPolicy, cfg.osCommandsAllow)
	}
}

// Absent --os-commands must leave the feature entirely unconfigured: no flag,
// no Runner, no endpoints — this is the "well-hidden by default" behavior the
// feature was built around, not merely an unset field.
func TestBuildAgentOSRunnerDisabledByDefault(t *testing.T) {
	runner, err := buildAgentOSRunner(cliConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if runner != nil {
		t.Fatalf("expected a nil Runner with no --os-commands flag, got %+v", runner)
	}
}

func TestBuildAgentOSRunnerRejectsAnUnknownPolicy(t *testing.T) {
	if _, err := buildAgentOSRunner(cliConfig{osCommandsPolicy: "yolo"}); err == nil {
		t.Fatal("expected an error for an invalid policy name")
	}
}

// Whitelist with no allow-list at all would auto-approve nothing yet still
// silently enable the feature; that is more likely an operator mistake than
// an intentional deny-everything whitelist, so it is rejected up front.
func TestBuildAgentOSRunnerRequiresAnAllowListForWhitelist(t *testing.T) {
	if _, err := buildAgentOSRunner(cliConfig{osCommandsPolicy: "whitelist"}); err == nil {
		t.Fatal("expected an error for whitelist with an empty allow-list")
	}
}

func TestBuildAgentOSRunnerParsesTheAllowList(t *testing.T) {
	runner, err := buildAgentOSRunner(cliConfig{osCommandsPolicy: "whitelist", osCommandsAllow: "ls, git ,, cat"})
	if err != nil {
		t.Fatal(err)
	}
	if runner == nil || runner.Policy != agentos.PolicyWhitelist {
		t.Fatalf("runner = %+v", runner)
	}
	want := []string{"ls", "git", "cat"}
	if len(runner.Allowed) != len(want) {
		t.Fatalf("allowed = %v, want %v", runner.Allowed, want)
	}
	for i, name := range want {
		if runner.Allowed[i] != name {
			t.Fatalf("allowed[%d] = %q, want %q", i, runner.Allowed[i], name)
		}
	}
}

func TestBuildAgentOSRunnerAllowAndDenyNeedNoAllowList(t *testing.T) {
	for _, policy := range []string{"deny", "allow"} {
		runner, err := buildAgentOSRunner(cliConfig{osCommandsPolicy: policy})
		if err != nil {
			t.Fatalf("policy %q: %v", policy, err)
		}
		if runner == nil || string(runner.Policy) != policy {
			t.Fatalf("policy %q: runner = %+v", policy, runner)
		}
	}
}
