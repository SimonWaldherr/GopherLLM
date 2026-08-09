package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/server"
)

func writeTestConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gopherllm.json")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestConfigFileThenCLIOverrides(t *testing.T) {
	path := writeTestConfig(t, `{
  "version": 1,
  "preset": "creative",
  "model": "config.gguf",
  "model_dir": "/models/from-config",
  "generation": {
    "max_tokens": 384,
    "top_k": 64,
    "stop": ["CONFIG_STOP"],
    "context_window_mode": "recent"
  },
  "runtime": {
    "threads": 3,
    "metal": false,
    "timeout": "2m",
    "auto_effort": "quick"
  },
  "server": {
    "address": "127.0.0.1:8080",
    "chat": true,
    "chat_history_path": "/var/lib/gopherllm/chats.json.gz",
    "max_connections": 12
  },
  "huggingface": {"offline": true}
}`)

	cfg, err := parseCLI([]string{
		"--config", path,
		"--preset", "precise",
		"--temp", "0.33",
		"--context-window", "autoCompress",
		"--model", "cli.gguf",
		"--threads", "5",
		"--serve", "127.0.0.1:9090",
		"--stop", "CLI_STOP",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.preset != "precise" || cfg.options.Sampler.Temperature != 0.33 || cfg.options.Sampler.TopK != 30 {
		t.Fatalf("CLI preset/temperature were not applied after config: preset=%q sampler=%+v", cfg.preset, cfg.options.Sampler)
	}
	if cfg.options.MaxTokens != 384 || cfg.options.ContextWindowMode != "autoCompress" || len(cfg.options.StopSequences) != 2 || cfg.options.StopSequences[0] != "CONFIG_STOP" || cfg.options.StopSequences[1] != "CLI_STOP" {
		t.Fatalf("generation config = %+v", cfg.options)
	}
	if cfg.modelSelector == nil || *cfg.modelSelector != "cli.gguf" || cfg.modelSelectorFromConfig || cfg.modelDir != "/models/from-config" {
		t.Fatalf("model config = selector:%v configSource:%v dir:%q", cfg.modelSelector, cfg.modelSelectorFromConfig, cfg.modelDir)
	}
	if !cfg.threadsSet || cfg.threads != 5 || !cfg.metalExplicit || cfg.useMetal || cfg.timeout.String() != "2m0s" {
		t.Fatalf("runtime config = threads:%d/%v metal:%v/%v timeout:%s", cfg.threads, cfg.threadsSet, cfg.useMetal, cfg.metalExplicit, cfg.timeout)
	}
	if !cfg.autoTune || cfg.autoTuneEffort != "quick" || cfg.serveAddr != "127.0.0.1:9090" || !cfg.chatUI || cfg.chatHistoryPath != "/var/lib/gopherllm/chats.json.gz" || cfg.maxConn != 12 || !cfg.hfOffline {
		t.Fatalf("config sections were not applied: %+v", cfg)
	}
}

func TestConfigRejectsUnknownFieldsAndDuplicateConfigFlags(t *testing.T) {
	path := writeTestConfig(t, `{"version":1,"unexpected":true}`)
	if _, err := parseCLI([]string{"--config", path}); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown field error = %v", err)
	}
	valid := writeTestConfig(t, `{"version":1}`)
	if _, err := parseCLI([]string{"--config", valid, "--config", valid}); err == nil || !strings.Contains(err.Error(), "only once") {
		t.Fatalf("duplicate config error = %v", err)
	}
}

func TestConfigAndPresetScanIgnoreFlagLookingValues(t *testing.T) {
	cfg, err := parseCLI([]string{"--system-prompt", "--preset", "--stop", "--config", "local.gguf", "--chat-history", "/tmp/chats.gz"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.preset != "balanced" || cfg.options.SystemPrompt != "--preset" || len(cfg.options.StopSequences) != 1 || cfg.options.StopSequences[0] != "--config" {
		t.Fatalf("option scan treated a value as a flag: %+v", cfg)
	}
	if cfg.chatHistoryPath != "/tmp/chats.gz" {
		t.Fatalf("chat history path = %q", cfg.chatHistoryPath)
	}
}

func TestWriteEffectiveConfigIsReusableAndSecretFree(t *testing.T) {
	cfg, err := parseCLI([]string{"--preset", "deterministic", "--hf-offline", "--threads", "2", "--model", "local.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := writeEffectiveConfig(&out, cfg); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), "HF_TOKEN") {
		t.Fatalf("effective config exposed a token: %s", out.String())
	}
	path := writeTestConfig(t, out.String())
	reloaded, err := parseCLI([]string{"--config", path})
	if err != nil {
		t.Fatalf("printed config was not reusable: %v\n%s", err, out.String())
	}
	if reloaded.preset != "deterministic" || reloaded.options.Sampler.Temperature != 0 || !reloaded.hfOffline || !reloaded.threadsSet || reloaded.threads != 2 {
		t.Fatalf("reloaded effective config = %+v", reloaded)
	}
}

func TestWriteEffectiveConfigPreservesExplicitMetalDisable(t *testing.T) {
	path := writeTestConfig(t, `{"version":1,"runtime":{"metal":false}}`)
	cfg, err := parseCLI([]string{"--config", path})
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := writeEffectiveConfig(&out, cfg); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"metal": false`) {
		t.Fatalf("explicit Metal disable was omitted: %s", out.String())
	}
	reloadedPath := writeTestConfig(t, out.String())
	reloaded, err := parseCLI([]string{"--config", reloadedPath})
	if err != nil || !reloaded.metalExplicit || reloaded.useMetal {
		t.Fatalf("explicit Metal disable was not preserved: cfg=%+v err=%v", reloaded, err)
	}
}

func TestManagedDeploymentConfigKeepsTokenOutOfPrintedConfig(t *testing.T) {
	path := writeTestConfig(t, `{
  "version": 1,
  "server": {
    "address": "127.0.0.1:8080",
    "deployment": "managed",
    "admin_token_file": "/run/secrets/gopherllm-admin"
  }
}`)
	cfg, err := parseCLI([]string{"--config", path, "--admin-token", "never-print-this"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.deploymentMode != server.DeploymentManaged || cfg.adminTokenFile != "/run/secrets/gopherllm-admin" || cfg.adminToken != "never-print-this" {
		t.Fatalf("managed deployment config = %+v", cfg)
	}
	var out bytes.Buffer
	if err := writeEffectiveConfig(&out, cfg); err != nil {
		t.Fatal(err)
	}
	printed := out.String()
	if strings.Contains(printed, "never-print-this") {
		t.Fatalf("effective config leaked admin token: %s", printed)
	}
	if !strings.Contains(printed, `"deployment": "managed"`) || !strings.Contains(printed, `"admin_token_file": "/run/secrets/gopherllm-admin"`) {
		t.Fatalf("effective config lost safe managed deployment settings: %s", printed)
	}
}
