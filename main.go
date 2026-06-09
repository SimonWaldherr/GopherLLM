package main

import (
	"bufio"
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
	fmt.Fprintf(os.Stderr, "gopherllm %s\n\n", Version)
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
	fmt.Fprintln(os.Stderr, "  --repeat-penalty <F>      Repetition penalty (default: 1.1)")
	fmt.Fprintln(os.Stderr, "  --seed <N>                RNG seed (default: time-based)")
	fmt.Fprintln(os.Stderr, "  --threads <N>             Override thread count")
	fmt.Fprintln(os.Stderr, "  --system-prompt <T>       Override the default system prompt")
	fmt.Fprintln(os.Stderr, "  --stop <text>             Stop generation when this string appears")
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
}

type cliConfig struct {
	modelSelector    *string
	modelDir         string
	prompt           string
	options          GenerationOptions
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
	inspect          bool
	listMetadata     bool
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
			fmt.Println(Version)
			return nil
		}
	}
	cfg, err := parseCLI(args[1:])
	if err != nil {
		return err
	}
	if cfg.listModels {
		entries, err := DiscoverModels(cfg.modelDir)
		if err != nil {
			return err
		}
		PrintModelList(entries)
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
	fmt.Fprintf(stderr(), "System: %d threads\n", runtime.GOMAXPROCS(0))
	if MetalAvailable() {
		fmt.Fprintln(stderr(), "Metal: enabled")
	} else {
		fmt.Fprintln(stderr(), "Metal: unavailable (pure Go / no CGO)")
	}
	modelPath, err := ResolveModelPath(cfg.modelSelector, cfg.modelDir)
	if err != nil {
		return err
	}
	if cfg.inspect || cfg.listTensors || cfg.listMetadata {
		return inspectGGUF(modelPath, cfg.listMetadata, cfg.listTensors)
	}
	runner, info, err := RunnerFromPath(modelPath)
	if err != nil {
		return err
	}
	defer runner.Close()
	fmt.Fprintf(stderr(), "Loaded %s (%.2f GB) in %.2fs\n", filepath.Base(modelPath), float64(info.FileSizeBytes)/(1024*1024*1024), info.LoadTime.Seconds())
	if name, ok := runner.ModelName(); ok {
		fmt.Fprintf(stderr(), "Model: %s\n", name)
	}
	fmt.Fprintf(stderr(), "Architecture: %s\n", runner.Architecture())
	if cfg.serveAddr != "" {
		return Serve(runner, ServeOptions{Addr: cfg.serveAddr, Defaults: cfg.options, MaxConcurrentConnections: cfg.maxConn, ChatUI: cfg.chatUI, ChatHistoryLock: &sync.Mutex{}, ModelDir: cfg.modelDir, ModelPath: modelPath})
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
	if cfg.bench {
		return runBench(runner, cfg)
	}
	if cfg.kernelBench {
		return RunKernelBench(runner, modelPath, cfg.kernelBenchRuns, cfg.kernelBenchLayer, cfg.kernelBenchJSON)
	}
	if cfg.repl || cfg.prompt == "" {
		return runREPL(runner, cfg.options)
	}
	result, err := runWithTimeout(cfg.timeout, func() (GenerationResult, error) {
		return runner.GenerateStream(cfg.prompt, cfg.options, func(s string) {
			fmt.Print(s)
		})
	})
	if err != nil {
		return err
	}
	fmt.Println()
	printStats(result.Stats)
	return nil
}

func parseCLI(args []string) (cliConfig, error) {
	cfg := cliConfig{modelDir: DefaultModelDir(), options: DefaultGenerationOptions(), benchRuns: 3, kernelBenchRuns: 25, maxConn: 8}
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
			SetNumThreads(v)
			runtime.GOMAXPROCS(v)
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

func inspectGGUF(path string, listMetadata, listTensors bool) error {
	mmap, err := OpenMmap(path)
	if err != nil {
		return err
	}
	defer mmap.Close()
	gguf, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		return err
	}
	arch, _ := gguf.GetString("general.architecture")
	name, _ := gguf.GetString("general.name")
	fmt.Printf("file: %s\nname: %s\narchitecture: %s\nsupported: %v\nmetadata: %d\ntensors: %d\ndata_offset: %d\n", path, name, arch, ArchitectureSupported(arch), len(gguf.Metadata), len(gguf.Tensors), gguf.DataOffset)
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

func formatMetaValue(v MetaValue, depth int) string {
	switch x := v.Value.(type) {
	case []MetaValue:
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

func runREPL(r *Runner, options GenerationOptions) error {
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
		_, err := r.GenerateStream(prompt, options, func(s string) { fmt.Print(s) })
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		fmt.Println()
	}
}

func runBench(r *Runner, cfg cliConfig) error {
	prompt := cfg.prompt
	if prompt == "" {
		prompt = "Write a concise explanation of local LLM inference."
	}
	results := []GenerationResult{}
	for range cfg.benchRuns {
		result, err := runWithTimeout(cfg.timeout, func() (GenerationResult, error) {
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

func runWithTimeout(timeout time.Duration, fn func() (GenerationResult, error)) (GenerationResult, error) {
	if timeout <= 0 {
		return fn()
	}
	type response struct {
		result GenerationResult
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
		return GenerationResult{}, fmt.Errorf("generation timed out after %s", timeout)
	}
}

func printStats(s GenerationStats) {
	tps := float64(s.GeneratedTokens) / max(1e-9, s.DecodeTime.Seconds())
	fmt.Fprintf(os.Stderr, "prompt=%d generated=%d prefill=%s decode=%s total=%s tok/s=%.2f\n", s.PromptTokens, s.GeneratedTokens, s.PrefillTime.Round(time.Millisecond), s.DecodeTime.Round(time.Millisecond), s.TotalTime.Round(time.Millisecond), tps)
}
