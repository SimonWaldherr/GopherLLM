package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/server"
)

const (
	configVersion  = 1
	maxConfigBytes = 1 << 20
)

// fileConfig is deliberately a small, stable JSON surface for deployments and
// repeatable local setups. Pointer fields distinguish an omitted value from an
// intentional zero or false, so defaults can evolve without changing an old
// config file's meaning.
type fileConfig struct {
	Version     int                    `json:"version"`
	Preset      string                 `json:"preset,omitempty"`
	Model       string                 `json:"model,omitempty"`
	ModelDir    string                 `json:"model_dir,omitempty"`
	Generation  *fileGenerationConfig  `json:"generation,omitempty"`
	Runtime     *fileRuntimeConfig     `json:"runtime,omitempty"`
	Server      *fileServerConfig      `json:"server,omitempty"`
	HuggingFace *fileHuggingFaceConfig `json:"huggingface,omitempty"`
}

type fileGenerationConfig struct {
	MaxTokens      *int     `json:"max_tokens,omitempty"`
	MTPDraftTokens *int     `json:"mtp_draft_tokens,omitempty"`
	Temperature    *float32 `json:"temperature,omitempty"`
	TopP           *float32 `json:"top_p,omitempty"`
	TopK           *int     `json:"top_k,omitempty"`
	MinP           *float32 `json:"min_p,omitempty"`
	RepeatPenalty  *float32 `json:"repeat_penalty,omitempty"`
	Seed           *uint64  `json:"seed,omitempty"`
	SystemPrompt   *string  `json:"system_prompt,omitempty"`
	Stop           []string `json:"stop,omitempty"`
	ContextWindow  *string  `json:"context_window_mode,omitempty"`
}

type fileRuntimeConfig struct {
	Threads      *int    `json:"threads,omitempty"`
	Metal        *bool   `json:"metal,omitempty"`
	PrepareQuant *bool   `json:"prepare_quant,omitempty"`
	OutOfCore    *bool   `json:"out_of_core,omitempty"`
	Timeout      *string `json:"timeout,omitempty"`
	AutoTune     *bool   `json:"auto_tune,omitempty"`
	AutoRefresh  *bool   `json:"auto_refresh,omitempty"`
	AutoEffort   *string `json:"auto_effort,omitempty"`
}

type fileServerConfig struct {
	Address    *string `json:"address,omitempty"`
	Deployment *string `json:"deployment,omitempty"`
	// AdminTokenFile is deliberately a file reference rather than the token
	// itself. Deployment secrets should live in an environment variable or a
	// permission-restricted file, and --print-config must never reproduce one.
	AdminTokenFile  *string `json:"admin_token_file,omitempty"`
	Chat            *bool   `json:"chat,omitempty"`
	ChatHistoryPath *string `json:"chat_history_path,omitempty"`
	MaxConnections  *int    `json:"max_connections,omitempty"`
	SkillsDir       *string `json:"skills_dir,omitempty"`
	OSCommands      *string `json:"os_commands,omitempty"`
	OSCommandsAllow *string `json:"os_commands_allow,omitempty"`
}

type fileHuggingFaceConfig struct {
	Offline *bool `json:"offline,omitempty"`
}

// cliValueOptions identifies flags whose following argument is data, not a
// second flag. The early config/preset scan uses it so a system prompt such as
// "--preset" cannot accidentally be interpreted as an option.
var cliValueOptions = map[string]bool{
	"--config":             true,
	"--preset":             true,
	"--model":              true,
	"--model-dir":          true,
	"--mmproj":             true,
	"--image":              true,
	"--hf-list":            true,
	"--prompt":             true,
	"-p":                   true,
	"--serve":              true,
	"--deployment":         true,
	"--admin-token":        true,
	"--admin-token-file":   true,
	"--chat-history":       true,
	"--wasm-dir":           true,
	"--max-connections":    true,
	"--max-tokens":         true,
	"-n":                   true,
	"--mtp-draft-tokens":   true,
	"--temp":               true,
	"-t":                   true,
	"--top-p":              true,
	"--top-k":              true,
	"--min-p":              true,
	"--repeat-penalty":     true,
	"--seed":               true,
	"--context-window":     true,
	"--threads":            true,
	"--system-prompt":      true,
	"--stop":               true,
	"--skills-dir":         true,
	"--os-commands":        true,
	"--os-commands-allow":  true,
	"--bench-runs":         true,
	"--kernel-bench-runs":  true,
	"--kernel-bench-layer": true,
	"--timeout":            true,
	"--auto-effort":        true,
	"--cpuprofile":         true,
	"--find-token":         true,
	"--token-neighbors":    true,
	"--neighbors":          true,
}

func configPathFromArgs(args []string) (string, error) { return singleCLIOptionValue(args, "--config") }

func presetFromArgs(args []string) (string, error) { return singleCLIOptionValue(args, "--preset") }

func singleCLIOptionValue(args []string, option string) (string, error) {
	var value string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg != option {
			if cliValueOptions[arg] && i+1 < len(args) {
				i++
			}
			continue
		}
		if i+1 >= len(args) {
			return "", fmt.Errorf("missing value for %s", option)
		}
		if value != "" {
			return "", fmt.Errorf("%s may be provided only once", option)
		}
		value = args[i+1]
		i++
	}
	return value, nil
}

func loadFileConfig(path string) (fileConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return fileConfig{}, fmt.Errorf("open config %s: %w", path, err)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxConfigBytes+1))
	if err != nil {
		return fileConfig{}, fmt.Errorf("read config %s: %w", path, err)
	}
	if len(data) > maxConfigBytes {
		return fileConfig{}, fmt.Errorf("config %s is larger than %d bytes", path, maxConfigBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var raw fileConfig
	if err := dec.Decode(&raw); err != nil {
		return fileConfig{}, fmt.Errorf("decode config %s: %w", path, err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fileConfig{}, fmt.Errorf("decode config %s: expected one JSON object", path)
		}
		return fileConfig{}, fmt.Errorf("decode config %s: %w", path, err)
	}
	if raw.Version != 0 && raw.Version != configVersion {
		return fileConfig{}, fmt.Errorf("config %s has unsupported version %d (supported: %d)", path, raw.Version, configVersion)
	}
	return raw, nil
}

func applyFileConfig(cfg *cliConfig, raw fileConfig) error {
	if cfg == nil {
		return fmt.Errorf("apply config: nil CLI configuration")
	}
	if raw.Preset != "" {
		if err := applyPreset(cfg, raw.Preset); err != nil {
			return fmt.Errorf("config preset: %w", err)
		}
	}
	if raw.Model != "" {
		model := raw.Model
		cfg.modelSelector = &model
		cfg.modelSelectorFromConfig = true
	}
	if raw.ModelDir != "" {
		cfg.modelDir = raw.ModelDir
	}
	if g := raw.Generation; g != nil {
		if g.MaxTokens != nil {
			cfg.options.MaxTokens = *g.MaxTokens
		}
		if g.MTPDraftTokens != nil {
			cfg.options.MTPDraftTokens = *g.MTPDraftTokens
		}
		if g.Temperature != nil {
			cfg.options.Sampler.Temperature = *g.Temperature
		}
		if g.TopP != nil {
			cfg.options.Sampler.TopP = *g.TopP
		}
		if g.TopK != nil {
			cfg.options.Sampler.TopK = *g.TopK
		}
		if g.MinP != nil {
			cfg.options.Sampler.MinP = *g.MinP
		}
		if g.RepeatPenalty != nil {
			cfg.options.Sampler.RepeatPenalty = *g.RepeatPenalty
		}
		if g.Seed != nil {
			cfg.options.Seed = *g.Seed
		}
		if g.SystemPrompt != nil {
			cfg.options.SystemPrompt = *g.SystemPrompt
		}
		if g.Stop != nil {
			cfg.options.StopSequences = append([]string(nil), g.Stop...)
		}
		if g.ContextWindow != nil {
			mode, err := parseContextWindowMode(*g.ContextWindow)
			if err != nil {
				return fmt.Errorf("config generation.context_window_mode: %w", err)
			}
			cfg.options.ContextWindowMode = mode
		}
	}
	if r := raw.Runtime; r != nil {
		if r.Threads != nil {
			cfg.threads, cfg.threadsSet = *r.Threads, true
		}
		if r.Metal != nil {
			cfg.useMetal, cfg.metalExplicit = *r.Metal, true
		}
		if r.PrepareQuant != nil {
			cfg.prepareQuant = *r.PrepareQuant
		}
		if r.OutOfCore != nil {
			cfg.outOfCore = *r.OutOfCore
		}
		if r.Timeout != nil {
			d, err := time.ParseDuration(*r.Timeout)
			if err != nil || d <= 0 {
				return fmt.Errorf("config runtime.timeout must be a duration greater than 0")
			}
			cfg.timeout = d
		}
		if r.AutoTune != nil {
			cfg.autoTune = *r.AutoTune
		}
		if r.AutoRefresh != nil {
			cfg.autoTuneRefresh = *r.AutoRefresh
			if *r.AutoRefresh {
				cfg.autoTune = true
			}
		}
		if r.AutoEffort != nil {
			if !validAutoEffort(*r.AutoEffort) {
				return fmt.Errorf("config runtime.auto_effort must be quick, balanced, or thorough")
			}
			cfg.autoTune, cfg.autoTuneEffort = true, *r.AutoEffort
		}
	}
	if s := raw.Server; s != nil {
		if s.Address != nil {
			cfg.serveAddr = *s.Address
		}
		if s.Deployment != nil {
			mode, err := server.ParseDeploymentMode(*s.Deployment)
			if err != nil {
				return fmt.Errorf("config server.deployment: %w", err)
			}
			cfg.deploymentMode = mode
		}
		if s.AdminTokenFile != nil {
			cfg.adminTokenFile = *s.AdminTokenFile
		}
		if s.Chat != nil {
			cfg.chatUI = *s.Chat
		}
		if s.ChatHistoryPath != nil {
			cfg.chatHistoryPath = *s.ChatHistoryPath
		}
		if s.MaxConnections != nil {
			cfg.maxConn = *s.MaxConnections
		}
		if s.SkillsDir != nil {
			cfg.skillsDir = *s.SkillsDir
		}
		if s.OSCommands != nil {
			cfg.osCommandsPolicy = *s.OSCommands
		}
		if s.OSCommandsAllow != nil {
			cfg.osCommandsAllow = *s.OSCommandsAllow
		}
	}
	if hf := raw.HuggingFace; hf != nil && hf.Offline != nil {
		cfg.hfOffline = *hf.Offline
	}
	return nil
}

func applyPreset(cfg *cliConfig, value string) error {
	if cfg == nil {
		return fmt.Errorf("nil CLI configuration")
	}
	preset := strings.ToLower(strings.TrimSpace(value))
	sampler := gopherllm.DefaultSamplerConfig()
	switch preset {
	case "balanced":
	case "precise":
		sampler.Temperature, sampler.TopP, sampler.TopK, sampler.MinP, sampler.RepeatPenalty = 0.2, 0.9, 30, 0, 1.05
	case "creative":
		sampler.Temperature, sampler.TopP, sampler.TopK, sampler.MinP, sampler.RepeatPenalty = 0.95, 0.95, 80, 0.05, 1.05
	case "deterministic":
		sampler.Temperature, sampler.TopP, sampler.TopK, sampler.MinP, sampler.RepeatPenalty = 0, 1, 0, 0, 1
	default:
		return fmt.Errorf("preset must be balanced, precise, creative, or deterministic")
	}
	cfg.preset, cfg.options.Sampler = preset, sampler
	return nil
}

func validAutoEffort(value string) bool {
	switch value {
	case "quick", "balanced", "thorough":
		return true
	default:
		return false
	}
}

func parseContextWindowMode(value string) (gopherllm.ContextWindowMode, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "full":
		return gopherllm.ContextWindowFull, nil
	case "recent":
		return gopherllm.ContextWindowRecent, nil
	case "autocompress":
		return gopherllm.ContextWindowAutoCompress, nil
	default:
		return "", fmt.Errorf("context window mode must be full, recent, or autoCompress")
	}
}

func effectiveContextWindowMode(mode gopherllm.ContextWindowMode) string {
	if mode == "" {
		return string(gopherllm.ContextWindowFull)
	}
	return string(mode)
}

// writeEffectiveConfig emits a ready-to-edit JSON config without access tokens.
// HF_TOKEN and transient user prompts are intentionally not part of the schema.
func writeEffectiveConfig(w io.Writer, cfg cliConfig) error {
	type effectiveGenerationConfig struct {
		MaxTokens      int      `json:"max_tokens"`
		MTPDraftTokens int      `json:"mtp_draft_tokens"`
		Temperature    float32  `json:"temperature"`
		TopP           float32  `json:"top_p"`
		TopK           int      `json:"top_k"`
		MinP           float32  `json:"min_p"`
		RepeatPenalty  float32  `json:"repeat_penalty"`
		Seed           uint64   `json:"seed"`
		SystemPrompt   string   `json:"system_prompt"`
		Stop           []string `json:"stop,omitempty"`
		ContextWindow  string   `json:"context_window_mode"`
	}
	type effectiveRuntimeConfig struct {
		Threads      int    `json:"threads,omitempty"`
		Metal        *bool  `json:"metal,omitempty"`
		PrepareQuant bool   `json:"prepare_quant,omitempty"`
		OutOfCore    bool   `json:"out_of_core,omitempty"`
		Timeout      string `json:"timeout,omitempty"`
		AutoTune     bool   `json:"auto_tune,omitempty"`
		AutoRefresh  bool   `json:"auto_refresh,omitempty"`
		AutoEffort   string `json:"auto_effort,omitempty"`
	}
	type effectiveServerConfig struct {
		Address         string `json:"address,omitempty"`
		Deployment      string `json:"deployment,omitempty"`
		AdminTokenFile  string `json:"admin_token_file,omitempty"`
		Chat            bool   `json:"chat,omitempty"`
		ChatHistoryPath string `json:"chat_history_path,omitempty"`
		MaxConnections  int    `json:"max_connections,omitempty"`
		SkillsDir       string `json:"skills_dir,omitempty"`
		OSCommands      string `json:"os_commands,omitempty"`
		OSCommandsAllow string `json:"os_commands_allow,omitempty"`
	}
	type effectiveConfig struct {
		Version     int                       `json:"version"`
		Preset      string                    `json:"preset"`
		Model       string                    `json:"model,omitempty"`
		ModelDir    string                    `json:"model_dir"`
		Generation  effectiveGenerationConfig `json:"generation"`
		Runtime     effectiveRuntimeConfig    `json:"runtime,omitempty"`
		Server      *effectiveServerConfig    `json:"server,omitempty"`
		HuggingFace *fileHuggingFaceConfig    `json:"huggingface,omitempty"`
	}
	result := effectiveConfig{
		Version:  configVersion,
		Preset:   cfg.preset,
		ModelDir: cfg.modelDir,
		Generation: effectiveGenerationConfig{
			MaxTokens:      cfg.options.MaxTokens,
			MTPDraftTokens: cfg.options.MTPDraftTokens,
			Temperature:    cfg.options.Sampler.Temperature,
			TopP:           cfg.options.Sampler.TopP,
			TopK:           cfg.options.Sampler.TopK,
			MinP:           cfg.options.Sampler.MinP,
			RepeatPenalty:  cfg.options.Sampler.RepeatPenalty,
			Seed:           cfg.options.Seed,
			SystemPrompt:   cfg.options.SystemPrompt,
			Stop:           append([]string(nil), cfg.options.StopSequences...),
			ContextWindow:  effectiveContextWindowMode(cfg.options.ContextWindowMode),
		},
		Runtime: effectiveRuntimeConfig{
			PrepareQuant: cfg.prepareQuant,
			OutOfCore:    cfg.outOfCore,
			AutoTune:     cfg.autoTune,
			AutoRefresh:  cfg.autoTuneRefresh,
			AutoEffort:   cfg.autoTuneEffort,
		},
	}
	if cfg.threadsSet {
		result.Runtime.Threads = cfg.threads
	}
	if cfg.metalExplicit {
		result.Runtime.Metal = boolPtr(cfg.useMetal)
	}
	if cfg.timeout > 0 {
		result.Runtime.Timeout = cfg.timeout.String()
	}
	if cfg.modelSelector != nil {
		result.Model = *cfg.modelSelector
	}
	if cfg.serveAddr != "" || cfg.deploymentMode != server.DeploymentLocal || cfg.chatUI || cfg.chatHistoryPath != "" || cfg.maxConn != 8 || cfg.skillsDir != "" || cfg.osCommandsPolicy != "" || cfg.osCommandsAllow != "" {
		result.Server = &effectiveServerConfig{
			Address:         cfg.serveAddr,
			Deployment:      string(cfg.deploymentMode),
			AdminTokenFile:  cfg.adminTokenFile,
			Chat:            cfg.chatUI,
			ChatHistoryPath: cfg.chatHistoryPath,
			MaxConnections:  cfg.maxConn,
			SkillsDir:       cfg.skillsDir,
			OSCommands:      cfg.osCommandsPolicy,
			OSCommandsAllow: cfg.osCommandsAllow,
		}
	}
	if cfg.hfOffline {
		result.HuggingFace = &fileHuggingFaceConfig{Offline: boolPtr(true)}
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

func boolPtr(value bool) *bool { return &value }
