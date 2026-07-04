package main

import "testing"

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

func TestParseCLIErrors(t *testing.T) {
	cases := [][]string{
		{"--unknown-flag"},          // unknown option
		{"m", "--temp"},             // missing value
		{"m", "--max-tokens", "xx"}, // invalid int
		{"m", "--temp", "notanum"},  // invalid float
		{"--chat"},                  // --chat without --serve
	}
	for _, args := range cases {
		if _, err := parseCLI(args); err == nil {
			t.Fatalf("parseCLI(%v) expected error", args)
		}
	}
}
