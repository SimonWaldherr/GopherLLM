package main

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"

	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func printUsage(name string) {
	fmt.Fprintf(os.Stderr, "gopherllm %s\n\n", gopherllm.Version)
	fmt.Fprintf(os.Stderr, "Usage: %s [model.gguf|model-name|model-dir] [options]\n\n", name)
	fmt.Fprintln(os.Stderr, "Options:")
	fmt.Fprintln(os.Stderr, "  --model <name>           Select a GGUF from --model-dir by repo, file, or metadata name")
	fmt.Fprintln(os.Stderr, "  --model-dir <path>       Directory to recursively scan for GGUF files")
	fmt.Fprintln(os.Stderr, "                           Passing a directory as model selector opens an interactive model picker")
	fmt.Fprintln(os.Stderr, "  --list-models            List GGUF files in --model-dir and exit")
	fmt.Fprintln(os.Stderr, "  --prompt <text>           Input prompt (interactive if omitted)")
	fmt.Fprintln(os.Stderr, "  --repl                    Start an interactive REPL session")
	fmt.Fprintln(os.Stderr, "  --serve <addr>            Start HTTP API server, e.g. 127.0.0.1:8080")
	fmt.Fprintln(os.Stderr, "  --chat                    Enable the minimal Web UI at /chat with --serve")
	fmt.Fprintln(os.Stderr, "  --max-connections <N>     Max concurrent server connections")
	fmt.Fprintln(os.Stderr, "  --max-tokens <N>          Max tokens to generate (default: 256)")
	fmt.Fprintln(os.Stderr, "  --temp <F>                Temperature (default: 0.7, 0=greedy)")
	fmt.Fprintln(os.Stderr, "  --top-p <F>               Nucleus sampling threshold (default: 0.9)")
	fmt.Fprintln(os.Stderr, "  --top-k <N>               Top-K filtering (default: 40)")
	fmt.Fprintln(os.Stderr, "  --min-p <F>               Min-P sampling threshold (default: 0, disabled)")
	fmt.Fprintln(os.Stderr, "  --repeat-penalty <F>      Repetition penalty (default: 1.1)")
	fmt.Fprintln(os.Stderr, "  --seed <N>                RNG seed (default: time-based)")
	fmt.Fprintln(os.Stderr, "  --threads <N>             Override thread count")
	fmt.Fprintln(os.Stderr, "  --metal                   Use selective Metal Q4_K/Q6_K matvec offload when available")
	fmt.Fprintln(os.Stderr, "  --prepare-quant           Precompute supported quantized scale data during load for faster matvecs")
	fmt.Fprintln(os.Stderr, "  --system-prompt <T>       Override the default system prompt")
	fmt.Fprintln(os.Stderr, "  --stop <text>             Stop generation when this string appears")
	fmt.Fprintln(os.Stderr, "  --skills-dir <path>       Directory of SKILL.md files offered via a load_skill tool")
	fmt.Fprintln(os.Stderr, "  --embed                   Embed prompt and print the vector")
	fmt.Fprintln(os.Stderr, "  --bench                   Run a non-streaming generation benchmark")
	fmt.Fprintln(os.Stderr, "  --bench-json              Run benchmark and emit machine-readable JSON")
	fmt.Fprintln(os.Stderr, "  --bench-output            Include generated text for each benchmark run")
	fmt.Fprintln(os.Stderr, "  --bench-runs <N>          Number of benchmark runs (default: 3)")
	fmt.Fprintln(os.Stderr, "  --kernel-bench            Run isolated kernel benchmark")
	fmt.Fprintln(os.Stderr, "  --kernel-bench-json       Emit isolated kernel benchmark JSON")
	fmt.Fprintln(os.Stderr, "  --kernel-bench-runs <N>   Number of kernel benchmark runs (default: 25)")
	fmt.Fprintln(os.Stderr, "  --kernel-bench-layer <N>  Transformer layer to benchmark (default: 0)")
	fmt.Fprintln(os.Stderr, "  --timeout <duration>      Abort generation or each benchmark run after a duration, e.g. 2m")
	fmt.Fprintln(os.Stderr, "  --cpuprofile <path>       Write a CPU profile for the full command run")
	fmt.Fprintln(os.Stderr, "  --inspect                 Inspect GGUF metadata and compatibility without loading weights")
	fmt.Fprintln(os.Stderr, "  --list-metadata           Print GGUF metadata with --inspect")
	fmt.Fprintln(os.Stderr, "  --list-tensors            Print GGUF tensor inventory and exit")
	fmt.Fprintln(os.Stderr, "  --analyze                 Print a structural analysis report (params, quant mix, geometry)")
	fmt.Fprintln(os.Stderr, "  --find-token <text>       Search the vocabulary for tokens containing <text>")
	fmt.Fprintln(os.Stderr, "  --token-neighbors <t>     Show embedding-space nearest neighbors of a token (id or text)")
	fmt.Fprintln(os.Stderr, "  --neighbors <N>           Neighbor count for --token-neighbors (default: 12)")
}

type cliConfig struct {
	modelSelector    *string
	modelDir         string
	prompt           string
	options          gopherllm.GenerationOptions
	listModels       bool
	listTensors      bool
	repl             bool
	serveAddr        string
	chatUI           bool
	maxConn          int
	embed            bool
	bench            bool
	benchJSON        bool
	benchOutput      bool
	benchRuns        int
	kernelBench      bool
	kernelBenchJSON  bool
	kernelBenchRuns  int
	kernelBenchLayer int
	timeout          time.Duration
	cpuProfile       string
	useMetal         bool
	prepareQuant     bool
	inspect          bool
	listMetadata     bool
	skillsDir        string
	analyze          bool
	findToken        string
	tokenNeighbors   string
	neighborCount    int
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	args := os.Args
	if len(args) < 2 {
		printUsage(args[0])
		return fmt.Errorf("missing model selector; use --list-models to inspect the configured model directory")
	}
	for _, arg := range args[1:] {
		if arg == "--help" || arg == "-h" {
			printUsage(args[0])
			return nil
		}
		if arg == "--version" || arg == "-V" {
			fmt.Println(gopherllm.Version)
			return nil
		}
	}
	cfg, err := parseCLI(args[1:])
	if err != nil {
		return err
	}
	if cfg.listModels {
		entries, err := gopherllm.DiscoverModels(cfg.modelDir, os.Stderr)
		if err != nil {
			return err
		}
		gopherllm.PrintModelList(entries)
		return nil
	}
	if cfg.options.MaxTokens <= 0 {
		return fmt.Errorf("--max-tokens must be greater than 0")
	}
	if err := cfg.options.Validate(); err != nil {
		return err
	}
	if cfg.benchRuns <= 0 {
		return fmt.Errorf("--bench-runs must be greater than 0")
	}
	if cfg.kernelBenchRuns <= 0 {
		return fmt.Errorf("--kernel-bench-runs must be greater than 0")
	}
	stopProfile, err := startCPUProfile(cfg.cpuProfile)
	if err != nil {
		return err
	}
	defer stopProfile()
	fmt.Fprintf(os.Stderr, "System: %d threads\n", runtime.GOMAXPROCS(0))
	metalAvailable := gopherllm.MetalAvailable()
	if cfg.useMetal {
		if !metalAvailable {
			if errText := gopherllm.MetalError(); errText != "" {
				return fmt.Errorf("Metal requested but unavailable: %s", errText)
			}
			return fmt.Errorf("Metal requested but unavailable")
		}
		fmt.Fprintln(os.Stderr, "Metal: enabled (selective Q4_K/Q6_K matvec offload)")
	} else if metalAvailable {
		fmt.Fprintln(os.Stderr, "Metal: available (disabled; pass --metal)")
	} else {
		fmt.Fprintln(os.Stderr, "Metal: unavailable (pure Go / no CGO)")
	}
	if cfg.prepareQuant {
		fmt.Fprintln(os.Stderr, "Quant prep: enabled for supported quantized weights")
	}
	modelPath, err := gopherllm.ResolveModelPath(cfg.modelSelector, cfg.modelDir)
	if err != nil {
		return err
	}
	if cfg.inspect || cfg.listTensors || cfg.listMetadata {
		return inspectGGUF(modelPath, cfg.listMetadata, cfg.listTensors)
	}
	if cfg.analyze || cfg.findToken != "" {
		// Header-only analysis: no weights are loaded, so this is instant
		// even for multi-gigabyte files.
		return analyzeGGUF(modelPath, cfg)
	}
	model, err := gopherllm.Open(
		context.Background(),
		modelPath,
		gopherllm.WithLogWriter(os.Stderr),
		gopherllm.WithPrepareQuantized(cfg.prepareQuant),
		gopherllm.WithMetal(cfg.useMetal),
	)
	if err != nil {
		return err
	}
	defer model.Close()
	runner := model.Runner()
	info := model.Info()
	fmt.Fprintf(os.Stderr, "Loaded %s (%.2f GB) in %.2fs\n", filepath.Base(modelPath), float64(info.FileSizeBytes)/(1024*1024*1024), info.LoadTime.Seconds())
	if name := model.Name(); name != "" {
		fmt.Fprintf(os.Stderr, "Model: %s\n", name)
	}
	fmt.Fprintf(os.Stderr, "Architecture: %s\n", runner.Architecture())
	if cfg.serveAddr != "" {
		return gopherllm.Serve(runner, gopherllm.ServeOptions{Addr: cfg.serveAddr, Defaults: cfg.options, MaxConcurrentConnections: cfg.maxConn, ChatUI: cfg.chatUI, ChatHistoryLock: &sync.Mutex{}, ModelDir: cfg.modelDir, ModelPath: modelPath, SkillsDir: cfg.skillsDir})
	}
	if cfg.embed {
		prompt, err := promptText(cfg.prompt)
		if err != nil {
			return err
		}
		result, err := runner.Embed(prompt)
		if err != nil {
			return err
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"token_count": result.TokenCount, "embedding": result.Embedding})
	}
	if cfg.tokenNeighbors != "" {
		matches, err := model.NearestTokens(cfg.tokenNeighbors, cfg.neighborCount)
		if err != nil {
			return err
		}
		fmt.Printf("nearest neighbors of %q in embedding space:\n", cfg.tokenNeighbors)
		for _, m := range matches {
			fmt.Printf("  %6d  %-24q cos=%.4f\n", m.ID, m.Text, m.Score)
		}
		return nil
	}
	if cfg.bench {
		return runBench(runner, cfg)
	}
	if cfg.kernelBench {
		return gopherllm.RunKernelBench(runner, modelPath, cfg.kernelBenchRuns, cfg.kernelBenchLayer, cfg.kernelBenchJSON)
	}
	skills, err := gopherllm.LoadSkills(cfg.skillsDir)
	if err != nil {
		return err
	}
	if len(skills) > 0 {
		fmt.Fprintf(os.Stderr, "Skills: loaded %d from %s\n", len(skills), cfg.skillsDir)
	}
	if cfg.repl || cfg.prompt == "" {
		return runREPL(runner, cfg.options, skills)
	}
	result, err := runWithTimeout(cfg.timeout, func() (gopherllm.GenerationResult, error) {
		return gopherllm.RunAgenticChat(runner, []gopherllm.ChatMessage{gopherllm.UserMessage(cfg.prompt)}, cfg.options, skills, func(s string) bool {
			fmt.Print(s)
			return true
		})
	})
	if err != nil {
		return err
	}
	fmt.Println()
	printReasoningAndToolCalls(result)
	printStats(result.Stats)
	return nil
}

// printReasoningAndToolCalls surfaces the parts of a gopherllm.GenerationResult that the
// CLI's plain stdout stream doesn't otherwise show: any chain-of-thought the
// model separated out, and any tool call the model wants that the CLI (unlike
// the HTTP server) has no client to hand it back to. Both go to stderr so
// stdout stays just the visible answer for piping.
func printReasoningAndToolCalls(result gopherllm.GenerationResult) {
	if result.ReasoningText != "" {
		fmt.Fprintf(os.Stderr, "[reasoning]\n%s\n[/reasoning]\n", result.ReasoningText)
	}
	for _, call := range result.ToolCalls {
		fmt.Fprintf(os.Stderr, "[tool_call] %s(%s) — no --skills-dir tool executor is configured to answer this\n", call.Function.Name, call.Function.Arguments)
	}
}

func parseCLI(args []string) (cliConfig, error) {
	cfg := cliConfig{modelDir: gopherllm.DefaultModelDir(), options: gopherllm.DefaultGenerationOptions(), benchRuns: 3, kernelBenchRuns: 25, maxConn: 8}
	setSelector := func(value string) error {
		if cfg.modelSelector != nil {
			return fmt.Errorf("multiple model selectors were provided")
		}
		cfg.modelSelector = &value
		return nil
	}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		next := func(flag string) (string, error) {
			i++
			if i >= len(args) {
				return "", fmt.Errorf("missing value for %s", flag)
			}
			return args[i], nil
		}
		switch arg {
		case "--model":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			if cfg.modelSelector != nil {
				if st, err := os.Stat(*cfg.modelSelector); err == nil && st.IsDir() {
					cfg.modelDir = *cfg.modelSelector
					cfg.modelSelector = &v
				} else {
					return cfg, fmt.Errorf("multiple model selectors were provided")
				}
			} else {
				cfg.modelSelector = &v
			}
		case "--model-dir":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.modelDir = v
		case "--list-models":
			cfg.listModels = true
		case "--prompt", "-p":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.prompt = v
		case "--repl":
			cfg.repl = true
		case "--serve":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.serveAddr = v
		case "--chat":
			cfg.chatUI = true
		case "--max-connections":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.maxConn = v
		case "--max-tokens", "-n":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.MaxTokens = v
		case "--temp", "-t":
			v, err := parseNextFloat(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.Sampler.Temperature = v
		case "--top-p":
			v, err := parseNextFloat(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.Sampler.TopP = v
		case "--top-k":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.Sampler.TopK = v
		case "--min-p":
			v, err := parseNextFloat(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.Sampler.MinP = v
		case "--repeat-penalty":
			v, err := parseNextFloat(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.Sampler.RepeatPenalty = v
		case "--seed":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			n, err := strconv.ParseUint(v, 10, 64)
			if err != nil {
				return cfg, err
			}
			cfg.options.Seed = n
		case "--threads":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			gopherllm.SetNumThreads(v)
			runtime.GOMAXPROCS(v)
		case "--metal":
			cfg.useMetal = true
		case "--prepare-quant":
			cfg.prepareQuant = true
		case "--system-prompt":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.SystemPrompt = v
		case "--stop":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.options.StopSequences = append(cfg.options.StopSequences, v)
		case "--skills-dir":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.skillsDir = v
		case "--embed":
			cfg.embed = true
		case "--bench":
			cfg.bench = true
		case "--bench-json", "—bench-json":
			cfg.bench = true
			cfg.benchJSON = true
			cfg.benchOutput = true
		case "--bench-output":
			cfg.benchOutput = true
		case "--bench-runs":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.benchRuns = v
		case "--kernel-bench":
			cfg.kernelBench = true
		case "--kernel-bench-json":
			cfg.kernelBench = true
			cfg.kernelBenchJSON = true
		case "--kernel-bench-runs":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.kernelBenchRuns = v
		case "--kernel-bench-layer":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.kernelBenchLayer = v
		case "--timeout":
			v, err := parseNextDuration(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.timeout = v
		case "--cpuprofile":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.cpuProfile = v
		case "--inspect":
			cfg.inspect = true
		case "--list-metadata":
			cfg.inspect = true
			cfg.listMetadata = true
		case "--list-tensors":
			cfg.listTensors = true
		case "--analyze":
			cfg.analyze = true
		case "--find-token":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.findToken = v
		case "--token-neighbors":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.tokenNeighbors = v
		case "--neighbors":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.neighborCount = v
		default:
			if strings.HasPrefix(arg, "-") {
				return cfg, fmt.Errorf("unknown option: %s", arg)
			}
			if cfg.modelSelector != nil {
				if st, err := os.Stat(arg); err == nil && st.IsDir() {
					cfg.modelDir = arg
				} else {
					return cfg, fmt.Errorf("multiple model selectors were provided")
				}
			} else if err := setSelector(arg); err != nil {
				return cfg, err
			}
		}
	}
	if cfg.chatUI && cfg.serveAddr == "" {
		return cfg, fmt.Errorf("--chat requires --serve <addr>")
	}
	if cfg.maxConn <= 0 {
		return cfg, fmt.Errorf("--max-connections must be greater than 0")
	}
	return cfg, nil
}

func parseNextInt(next func(string) (string, error), flag string) (int, error) {
	v, err := next(flag)
	if err != nil {
		return 0, err
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s value %q: %w", flag, v, err)
	}
	return n, nil
}

func parseNextFloat(next func(string) (string, error), flag string) (float32, error) {
	v, err := next(flag)
	if err != nil {
		return 0, err
	}
	n, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid %s value %q: %w", flag, v, err)
	}
	return float32(n), nil
}

func parseNextDuration(next func(string) (string, error), flag string) (time.Duration, error) {
	v, err := next(flag)
	if err != nil {
		return 0, err
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s value %q: %w", flag, v, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("%s must be greater than 0", flag)
	}
	return d, nil
}

// analyzeGGUF handles --analyze and --find-token: header-only work over the
// mmap'd file, no weight loading.
func analyzeGGUF(path string, cfg cliConfig) error {
	mmap, err := gopherllm.OpenMmap(path)
	if err != nil {
		return err
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUF(mmap.Bytes())
	if err != nil {
		return err
	}
	tok, err := gopherllm.TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		tok = nil // analysis still works without a tokenizer
	}
	if cfg.analyze {
		fmt.Printf("file:           %s (%.2f GB)\n", path, float64(mmap.Len())/(1024*1024*1024))
		gopherllm.AnalyzeGGUF(gguf, tok).WriteText(os.Stdout)
	}
	if cfg.findToken != "" {
		if tok == nil {
			return fmt.Errorf("--find-token requires a tokenizer, which this file lacks")
		}
		matches := gopherllm.SearchTokens(tok, cfg.findToken, 50)
		fmt.Printf("%d tokens match %q:\n", len(matches), cfg.findToken)
		for _, m := range matches {
			fmt.Printf("  %6d  %q\n", m.ID, m.Text)
		}
	}
	return nil
}

func inspectGGUF(path string, listMetadata, listTensors bool) error {
	mmap, err := gopherllm.OpenMmap(path)
	if err != nil {
		return err
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUF(mmap.Bytes())
	if err != nil {
		return err
	}
	arch, _ := gguf.GetString("general.architecture")
	name, _ := gguf.GetString("general.name")
	fmt.Printf("file: %s\nname: %s\narchitecture: %s\nsupported: %v\nmetadata: %d\ntensors: %d\ndata_offset: %d\n", path, name, arch, gopherllm.ArchitectureSupported(arch), len(gguf.Metadata), len(gguf.Tensors), gguf.DataOffset)
	if listMetadata {
		keys := make([]string, 0, len(gguf.Metadata))
		for key := range gguf.Metadata {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			fmt.Printf("%-48s %s\n", key, formatMetaValue(gguf.Metadata[key], 0))
		}
	}
	if listTensors {
		for _, t := range gguf.Tensors {
			fmt.Printf("%-56s %-8s dims=%v offset=%d\n", t.Name, t.DType, t.Dims, t.Offset)
		}
	}
	return nil
}

func formatMetaValue(v gopherllm.MetaValue, depth int) string {
	switch x := v.Value.(type) {
	case []gopherllm.MetaValue:
		if depth > 0 {
			return fmt.Sprintf("[%d values]", len(x))
		}
		limit := min(len(x), 8)
		parts := make([]string, 0, limit+1)
		for i := range limit {
			parts = append(parts, formatMetaValue(x[i], depth+1))
		}
		if len(x) > limit {
			parts = append(parts, fmt.Sprintf("... (%d total)", len(x)))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case string:
		return strconv.Quote(x)
	default:
		return fmt.Sprint(x)
	}
}

func promptText(prompt string) (string, error) {
	if prompt != "" {
		return prompt, nil
	}
	b, err := io.ReadAll(os.Stdin)
	return string(b), err
}

func runREPL(r *gopherllm.Runner, options gopherllm.GenerationOptions, skills []gopherllm.Skill) error {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Fprintln(os.Stderr, "Enter prompts; empty line exits.")
	for {
		fmt.Fprint(os.Stderr, "> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		prompt := strings.TrimSpace(scanner.Text())
		if prompt == "" {
			return nil
		}
		result, err := gopherllm.RunAgenticChat(r, []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)}, options, skills, func(s string) bool { fmt.Print(s); return true })
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		fmt.Println()
		printReasoningAndToolCalls(result)
	}
}

func runBench(r *gopherllm.Runner, cfg cliConfig) error {
	prompt := cfg.prompt
	if prompt == "" {
		prompt = "Write a concise explanation of local LLM inference."
	}
	results := []gopherllm.GenerationResult{}
	for range cfg.benchRuns {
		result, err := runWithTimeout(cfg.timeout, func() (gopherllm.GenerationResult, error) {
			return r.Generate(prompt, cfg.options)
		})
		if err != nil {
			return err
		}
		results = append(results, result)
	}
	if cfg.benchJSON {
		type row struct {
			PromptTokens          int     `json:"prompt_tokens"`
			GeneratedTokens       int     `json:"generated_tokens"`
			TTFTMS                int64   `json:"ttft_ms"`
			PrefillMS             int64   `json:"prefill_ms"`
			DecodeMS              int64   `json:"decode_ms"`
			TotalMS               int64   `json:"total_ms"`
			PrefillTokensPerSec   float64 `json:"prefill_tokens_per_second"`
			GeneratedTokensPerSec float64 `json:"generated_tokens_per_second"`
			Text                  string  `json:"text,omitempty"`
		}
		rows := []row{}
		for _, r := range results {
			prefillTPS := float64(r.Stats.PromptTokens) / max(1e-9, r.Stats.PrefillTime.Seconds())
			decodeTPS := float64(r.Stats.GeneratedTokens) / max(1e-9, r.Stats.DecodeTime.Seconds())
			item := row{
				PromptTokens:          r.Stats.PromptTokens,
				GeneratedTokens:       r.Stats.GeneratedTokens,
				TTFTMS:                r.Stats.TTFT.Milliseconds(),
				PrefillMS:             r.Stats.PrefillTime.Milliseconds(),
				DecodeMS:              r.Stats.DecodeTime.Milliseconds(),
				TotalMS:               r.Stats.TotalTime.Milliseconds(),
				PrefillTokensPerSec:   prefillTPS,
				GeneratedTokensPerSec: decodeTPS,
			}
			if cfg.benchOutput {
				item.Text = r.Text
			}
			rows = append(rows, item)
		}
		return json.NewEncoder(os.Stdout).Encode(rows)
	}
	for i, result := range results {
		fmt.Printf("run %d: ", i+1)
		printStats(result.Stats)
		if cfg.benchOutput {
			fmt.Printf("run %d output: %s\n", i+1, result.Text)
		}
	}
	return nil
}

func startCPUProfile(path string) (func(), error) {
	if strings.TrimSpace(path) == "" {
		return func() {}, nil
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create CPU profile: %w", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("start CPU profile: %w", err)
	}
	return func() {
		pprof.StopCPUProfile()
		_ = f.Close()
	}, nil
}

func runWithTimeout(timeout time.Duration, fn func() (gopherllm.GenerationResult, error)) (gopherllm.GenerationResult, error) {
	if timeout <= 0 {
		return fn()
	}
	type response struct {
		result gopherllm.GenerationResult
		err    error
	}
	done := make(chan response, 1)
	go func() {
		result, err := fn()
		done <- response{result: result, err: err}
	}()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case res := <-done:
		return res.result, res.err
	case <-timer.C:
		return gopherllm.GenerationResult{}, fmt.Errorf("generation timed out after %s", timeout)
	}
}

func printStats(s gopherllm.GenerationStats) {
	tps := float64(s.GeneratedTokens) / max(1e-9, s.DecodeTime.Seconds())
	fmt.Fprintf(os.Stderr, "prompt=%d generated=%d ttft=%s prefill=%s decode=%s total=%s tok/s=%.2f\n", s.PromptTokens, s.GeneratedTokens, s.TTFT.Round(time.Millisecond), s.PrefillTime.Round(time.Millisecond), s.DecodeTime.Round(time.Millisecond), s.TotalTime.Round(time.Millisecond), tps)
}
