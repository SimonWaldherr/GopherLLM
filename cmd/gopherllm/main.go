package main

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/agentos"
	"github.com/SimonWaldherr/GopherLLM/internal/huggingface"
	"github.com/SimonWaldherr/GopherLLM/server"

	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
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
	fmt.Fprintf(os.Stderr, "Usage: %s [model.gguf|model-name|model-dir|hf:owner/repo:quant@revision] [options]\n\n", name)
	fmt.Fprintln(os.Stderr, "If the model is omitted, the last successfully loaded local model is reused.")
	fmt.Fprintln(os.Stderr, "Hugging Face: hf:owner/repo:Q4_K_M[@revision] downloads and reuses GGUFs from the HF cache.")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, "Options:")
	fmt.Fprintln(os.Stderr, "  --model <name>           Select a GGUF from --model-dir by repo, file, or metadata name")
	fmt.Fprintln(os.Stderr, "  --model-dir <path>       Directory to recursively scan for GGUF files")
	fmt.Fprintln(os.Stderr, "                           Passing a directory as model selector opens an interactive model picker")
	fmt.Fprintln(os.Stderr, "  --config <file>          Load a strict JSON configuration (defaults < config < CLI flags)")
	fmt.Fprintln(os.Stderr, "  --print-config           Print the effective JSON configuration without access tokens and exit")
	fmt.Fprintln(os.Stderr, "  --preset <name>          Generation preset: balanced | precise | creative | deterministic")
	fmt.Fprintln(os.Stderr, "  --list-models            List GGUF files in --model-dir and exit")
	fmt.Fprintln(os.Stderr, "  --hf-list <repo>         List GGUF variants in a Hugging Face repo, e.g. bartowski/Qwen3-4B-GGUF")
	fmt.Fprintln(os.Stderr, "  --hf-offline             Use complete Hugging Face cache entries only; never make a network request")
	fmt.Fprintln(os.Stderr, "  --privacy                Print the local-first privacy and network-use report, then exit")
	fmt.Fprintln(os.Stderr, "  --prompt <text>           Input prompt (interactive if omitted)")
	fmt.Fprintln(os.Stderr, "  --repl                    Start an interactive REPL session")
	fmt.Fprintln(os.Stderr, "  --serve <addr>            Start HTTP API server, e.g. 127.0.0.1:8080")
	fmt.Fprintln(os.Stderr, "  --chat                    Enable the minimal Web UI at /chat with --serve")
	fmt.Fprintln(os.Stderr, "  --chat-history <path>     Enable compressed server-side chat history at <path>")
	fmt.Fprintln(os.Stderr, "                           Explicit model selectors always override the remembered model")
	fmt.Fprintln(os.Stderr, "  --wasm-dir <path>         Serve gopherllm.wasm+wasm_exec.js from <path> for the chat UI's")
	fmt.Fprintln(os.Stderr, "                           local-browser-inference mode (default: auto-detect ./bin)")
	fmt.Fprintln(os.Stderr, "  --max-connections <N>     Max concurrent server connections")
	fmt.Fprintln(os.Stderr, "  --max-tokens <N>          Max tokens to generate (default: 256)")
	fmt.Fprintln(os.Stderr, "  --temp <F>                Temperature (default: 0.7, 0=greedy)")
	fmt.Fprintln(os.Stderr, "  --top-p <F>               Nucleus sampling threshold (default: 0.9)")
	fmt.Fprintln(os.Stderr, "  --top-k <N>               Top-K filtering (default: 40)")
	fmt.Fprintln(os.Stderr, "  --min-p <F>               Min-P sampling threshold (default: 0, disabled)")
	fmt.Fprintln(os.Stderr, "  --repeat-penalty <F>      Repetition penalty (default: 1.1)")
	fmt.Fprintln(os.Stderr, "  --seed <N>                RNG seed (default: time-based)")
	fmt.Fprintln(os.Stderr, "  --context-window <mode>  Chat overflow: full | recent | autoCompress (default: full)")
	fmt.Fprintln(os.Stderr, "  --threads <N>             Override thread count")
	fmt.Fprintln(os.Stderr, "  --metal                   Use selective Metal Q4_K/Q6_K matvec offload when available")
	fmt.Fprintln(os.Stderr, "  --prepare-quant           Precompute supported quantized scale data during load for faster matvecs")
	fmt.Fprintln(os.Stderr, "  --out-of-core             CPU-only mmap inference; keep sparse MoE experts and scalar weights demand-paged")
	fmt.Fprintln(os.Stderr, "  --system-prompt <T>       Override the default system prompt")
	fmt.Fprintln(os.Stderr, "  --stop <text>             Stop generation when this string appears")
	fmt.Fprintln(os.Stderr, "  --skills-dir <path>       Directory of SKILL.md files offered via a load_skill tool")
	fmt.Fprintln(os.Stderr, "  --os-commands <policy>    Enable /agentos endpoints: deny | whitelist | allow (default: disabled)")
	fmt.Fprintln(os.Stderr, "                            deny still lets a model propose a command, but a human must approve every one")
	fmt.Fprintln(os.Stderr, "  --os-commands-allow <l>   Comma-separated program names auto-approved under whitelist, e.g. ls,git,cat")
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
	fmt.Fprintln(os.Stderr, "  --auto                    Measure this model on this machine at startup and use the fastest settings")
	fmt.Fprintln(os.Stderr, "                            (result is cached per model+hardware, so later runs start instantly)")
	fmt.Fprintln(os.Stderr, "  --auto-effort <level>     Calibration depth: quick | balanced | thorough (default: balanced)")
	fmt.Fprintln(os.Stderr, "  --auto-refresh            Re-measure even if a cached tuning exists")
	fmt.Fprintln(os.Stderr, "  --auto-json               Print the tuning result as JSON and exit")
	fmt.Fprintln(os.Stderr, "  --cpuprofile <path>       Write a CPU profile for the full command run")
	fmt.Fprintln(os.Stderr, "  --inspect                 Inspect GGUF metadata and compatibility without loading weights")
	fmt.Fprintln(os.Stderr, "  --list-metadata           Print GGUF metadata with --inspect")
	fmt.Fprintln(os.Stderr, "  --list-tensors            Print GGUF tensor inventory and exit")
	fmt.Fprintln(os.Stderr, "  --analyze                 Print a structural analysis report (params, quant mix, geometry)")
	fmt.Fprintln(os.Stderr, "  --find-token <text>       Search the vocabulary for tokens containing <text>")
	fmt.Fprintln(os.Stderr, "  --token-neighbors <t>     Show embedding-space nearest neighbors of a token (id or text)")
	fmt.Fprintln(os.Stderr, "  --neighbors <N>           Neighbor count for --token-neighbors (default: 12)")
	fmt.Fprintln(os.Stderr, "  --compress                Requantize the GGUF to --compress-format via round-to-nearest and exit")
	fmt.Fprintln(os.Stderr, "  --compress-format <F>     Target format: Q8_0 | Q4_0 | Q4_K | Q5_K | Q6_K")
	fmt.Fprintln(os.Stderr, "  --compress-out <path>     Output path for --compress (required, must differ from the source)")
	fmt.Fprintln(os.Stderr, "  --compress-uniform        Quantize token_embd/output tensors to --compress-format too")
	fmt.Fprintln(os.Stderr, "                            (default: floor them at Q6_K, matching llama.cpp's quantize tool)")
}

type cliConfig struct {
	modelSelector           *string
	modelSelectorFromConfig bool
	modelDir                string
	preset                  string
	prompt                  string
	options                 gopherllm.GenerationOptions
	threads                 int
	threadsSet              bool
	printConfig             bool
	listModels              bool
	hfList                  string
	hfOffline               bool
	privacyReport           bool
	listTensors             bool
	repl                    bool
	serveAddr               string
	chatUI                  bool
	chatHistoryPath         string
	// wasmDir points at a directory containing gopherllm.wasm + wasm_exec.js
	// (see `make wasm-build`), enabling the chat UI's local-browser-inference
	// mode. Empty disables it; resolveWasmDir applies the "bin" auto-detect
	// default unless --wasm-dir was given explicitly (including explicitly
	// empty, via --wasm-dir "", to opt out of auto-detection).
	wasmDir          string
	wasmDirSet       bool
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
	autoTune         bool
	autoTuneJSON     bool
	autoTuneRefresh  bool
	autoTuneEffort   string
	useMetal         bool
	metalExplicit    bool
	prepareQuant     bool
	outOfCore        bool
	inspect          bool
	listMetadata     bool
	skillsDir        string
	analyze          bool
	findToken        string
	tokenNeighbors   string
	neighborCount    int
	// mmprojPath is a companion Pixtral-style vision-encoder GGUF loaded
	// alongside modelSelector, enabling --image on the one-shot prompt path
	// and image_url content on the server's chat endpoints.
	mmprojPath    string
	mmprojPathSet bool
	// imagePath attaches one image to the one-shot --prompt request. Only
	// meaningful together with --mmproj (or a model directory pairing one
	// automatically, see catalog.go's pairProjectors).
	imagePath string
	// compress requantizes modelSelector's GGUF to compressFormat via plain
	// round-to-nearest and writes it to compressOut. See compress.go.
	compress        bool
	compressFormat  string
	compressOut     string
	compressUniform bool
	// osCommandsPolicy is the raw --os-commands value: "" disables the
	// agentic OS-command feature entirely (no endpoints registered); any
	// other value must parse via agentos.ParsePolicy.
	osCommandsPolicy string
	// osCommandsAllow is a comma-separated program allow-list, only
	// meaningful under the whitelist policy.
	osCommandsAllow string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	args := os.Args
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
	if cfg.printConfig {
		if err := cfg.options.Validate(); err != nil {
			return err
		}
		return writeEffectiveConfig(os.Stdout, cfg)
	}
	commandCtx, stopCommand := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stopCommand()
	if cfg.hfList != "" {
		return huggingface.ListGGUFContextWithOptions(commandCtx, cfg.hfList, os.Stdout, huggingface.Options{Offline: cfg.hfOffline})
	}
	if cfg.privacyReport {
		return gopherllm.WritePrivacyReport(os.Stdout)
	}
	if cfg.listModels {
		entries, err := gopherllm.DiscoverModels(cfg.modelDir, os.Stderr)
		if err != nil {
			return err
		}
		gopherllm.PrintModelList(entries)
		return nil
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
	if cfg.outOfCore && cfg.useMetal {
		return fmt.Errorf("--out-of-core cannot be combined with --metal")
	}
	if cfg.outOfCore && cfg.prepareQuant {
		return fmt.Errorf("--out-of-core cannot be combined with --prepare-quant")
	}
	if cfg.outOfCore && cfg.autoTune {
		return fmt.Errorf("--out-of-core cannot be combined with --auto: calibration intentionally streams the full model")
	}
	if cfg.threadsSet {
		gopherllm.SetNumThreads(cfg.threads)
		runtime.GOMAXPROCS(cfg.threads)
	}
	agentOSRunner, err := buildAgentOSRunner(cfg)
	if err != nil {
		return err
	}
	// A successfully used local model is the default for every later command,
	// including a bare invocation and server restarts. An explicit selector
	// always wins. An unusable saved path leaves a server in its model-picker
	// state rather than preventing it from starting.
	resumeLastModel(&cfg, os.Stderr)
	if cfg.modelSelector == nil && len(args) < 2 {
		printUsage(args[0])
		return fmt.Errorf("no model has been used yet; provide a model once or use --list-models")
	}
	// A server with a model catalog is useful before any weights are loaded:
	// the browser can show its model picker immediately and POST /models/load
	// performs the first load. Non-server commands retain the existing model
	// picker behavior below.
	if cfg.modelSelector == nil && cfg.serveAddr != "" {
		if cfg.inspect || cfg.listTensors || cfg.listMetadata || cfg.analyze || cfg.findToken != "" || cfg.tokenNeighbors != "" || cfg.embed || cfg.bench || cfg.kernelBench || cfg.repl || cfg.prompt != "" || cfg.autoTune || cfg.compress {
			return fmt.Errorf("this command needs a model selector; start only --serve (optionally --chat) to choose a model later")
		}
		fmt.Fprintln(os.Stderr, "No model loaded. Choose one in the Web UI or POST /models/load.")
		return server.Serve(nil, server.ServeOptions{
			Context:                  commandCtx,
			Addr:                     cfg.serveAddr,
			Defaults:                 cfg.options,
			MaxConcurrentConnections: cfg.maxConn,
			ChatUI:                   cfg.chatUI,
			ChatHistoryPath:          cfg.chatHistoryPath,
			ChatHistoryLock:          &sync.Mutex{},
			ModelDir:                 cfg.modelDir,
			WasmDir:                  resolveWasmDir(cfg),
			SkillsDir:                cfg.skillsDir,
			ModelLoadOptions:         serverModelLoadOptions(cfg),
			AgentOS:                  agentOSRunner,
		})
	}
	stopProfile, err := startCPUProfile(cfg.cpuProfile)
	if err != nil {
		return err
	}
	defer stopProfile()
	fmt.Fprintf(os.Stderr, "System: %d threads\n", runtime.GOMAXPROCS(0))
	metalAvailable := gopherllm.MetalAvailable()
	// --compress never loads a Runner (CompressModel works directly off the
	// source file's mmap, always single-file, always CPU), so none of
	// --metal/--out-of-core/--auto/--prepare-quant apply to it. Printing
	// their banners anyway would tell the operator those features are
	// active for a run where they silently do nothing — skip the banners
	// and say so once, instead.
	if cfg.compress {
		if cfg.outOfCore || cfg.useMetal || cfg.autoTune || cfg.prepareQuant {
			fmt.Fprintln(os.Stderr, "compress: --metal/--out-of-core/--auto/--prepare-quant do not apply to compression; ignoring")
		}
	} else if cfg.outOfCore {
		fmt.Fprintln(os.Stderr, "Out-of-core: enabled (CPU mmap; sparse experts stay demand-paged)")
	} else if cfg.useMetal {
		if !metalAvailable {
			if errText := gopherllm.MetalError(); errText != "" {
				return fmt.Errorf("Metal requested but unavailable: %s", errText)
			}
			return fmt.Errorf("Metal requested but unavailable")
		}
		fmt.Fprintln(os.Stderr, "Metal: enabled (selective Q4_K/Q6_K matvec offload)")
	} else if metalAvailable && cfg.autoTune {
		// Saying "pass --metal" here would be wrong: --auto is about to decide
		// it by measurement, and may well turn it on.
		fmt.Fprintln(os.Stderr, "Metal: available (--auto will measure whether to use it)")
	} else if metalAvailable {
		fmt.Fprintln(os.Stderr, "Metal: available (disabled; pass --metal or --auto to measure)")
	} else {
		fmt.Fprintln(os.Stderr, "Metal: unavailable (pure Go / no CGO)")
	}
	if cfg.prepareQuant && !cfg.compress {
		fmt.Fprintln(os.Stderr, "Quant prep: enabled for supported quantized weights")
	}
	var modelPath string
	if cfg.modelSelector != nil && strings.HasPrefix(strings.ToLower(*cfg.modelSelector), "hf:") {
		modelPath, err = huggingface.ResolveHuggingFaceModelContextWithOptions(commandCtx, *cfg.modelSelector, os.Stderr, huggingface.Options{Offline: cfg.hfOffline})
	} else {
		modelPath, err = gopherllm.ResolveModelPath(cfg.modelSelector, cfg.modelDir)
	}
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
	if cfg.compress {
		return runCompress(modelPath, cfg)
	}
	// A nearby compatible Pixtral projector is part of a multimodal model
	// release, not an optional UI-only extra. Make the normal initial CLI
	// path behave like /models/load: use the catalog's conservative pairing
	// when --mmproj was not explicitly supplied. Do not put this automatic
	// path into cfg.mmprojPath: later server hot-swaps must discover each
	// selected model's own companion, while an explicit --mmproj intentionally
	// remains an override for all loads.
	visionProjectorPath := cfg.mmprojPath
	if !cfg.mmprojPathSet {
		paired, pairErr := gopherllm.PairedVisionProjectorPath(modelPath)
		if pairErr != nil {
			fmt.Fprintf(os.Stderr, "Vision: automatic projector discovery skipped: %v\n", pairErr)
		} else if paired != "" {
			visionProjectorPath = paired
			fmt.Fprintf(os.Stderr, "Vision: automatically paired %s\n", filepath.Base(paired))
		}
	}
	// Metal is fixed when the weights load, so auto-tuning (which runs against
	// an already-loaded Runner) cannot reach it. Decide it here instead, by
	// actually measuring both ways. An explicit --metal is the operator's call
	// and is never second-guessed.
	if cfg.autoTune && !cfg.metalExplicit && metalAvailable {
		base := gopherllm.LoadOptions{PrepareQuantized: cfg.prepareQuant}
		probe, cached, probeErr := gopherllm.ProbeMetalOrCached(modelPath, base, 0, cfg.autoTuneRefresh)
		switch {
		case probeErr != nil:
			fmt.Fprintf(os.Stderr, "Auto: could not measure Metal (%v); leaving it off\n", probeErr)
		case cached:
			fmt.Fprintf(os.Stderr, "Auto: reusing Metal measurement — %s\n", probe.SummaryLine())
			cfg.useMetal = probe.UseMetal
		default:
			fmt.Fprintf(os.Stderr, "Auto: %s\n", probe.SummaryLine())
			cfg.useMetal = probe.UseMetal
		}
	}
	openOpts := []gopherllm.Option{
		gopherllm.WithLogWriter(os.Stderr),
		gopherllm.WithPrepareQuantized(cfg.prepareQuant),
		gopherllm.WithMetal(cfg.useMetal),
		gopherllm.WithOutOfCore(cfg.outOfCore),
	}
	if visionProjectorPath != "" {
		openOpts = append(openOpts, gopherllm.WithVisionProjector(visionProjectorPath))
	}
	model, err := gopherllm.Open(
		context.Background(),
		modelPath,
		openOpts...,
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
	recordLastModel(modelPath)
	// Auto-tuning runs before every generation path (one-shot, REPL, server,
	// bench) so all of them share the measured configuration.
	var appliedAutoTune *gopherllm.AutoTuneResult
	var baselineRuntimeTuning *gopherllm.RuntimeTuning
	if cfg.autoTune {
		baseline := gopherllm.CaptureRuntimeTuning()
		baselineRuntimeTuning = &baseline
		res, done, err := runAutoTune(runner, cfg)
		if err != nil {
			return err
		}
		if done {
			return nil
		}
		appliedAutoTune = &res
	}
	if cfg.serveAddr != "" {
		return server.Serve(runner, server.ServeOptions{Context: commandCtx, Addr: cfg.serveAddr, Defaults: cfg.options, MaxConcurrentConnections: cfg.maxConn, ChatUI: cfg.chatUI, ChatHistoryPath: cfg.chatHistoryPath, ChatHistoryLock: &sync.Mutex{}, ModelDir: cfg.modelDir, ModelPath: modelPath, WasmDir: resolveWasmDir(cfg), SkillsDir: cfg.skillsDir, ModelLoadOptions: serverModelLoadOptions(cfg), AppliedAutoTune: appliedAutoTune, BaselineRuntimeTuning: baselineRuntimeTuning, ModelLoaded: recordLastModel, AgentOS: agentOSRunner})
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
	promptMessage := gopherllm.UserMessage(cfg.prompt)
	if cfg.imagePath != "" {
		imgBytes, err := os.ReadFile(cfg.imagePath)
		if err != nil {
			return fmt.Errorf("reading --image: %w", err)
		}
		promptMessage = gopherllm.UserMessageWithImages(cfg.prompt, gopherllm.ImageContent{Bytes: imgBytes})
	}
	result, err := runWithTimeout(cfg.timeout, func() (gopherllm.GenerationResult, error) {
		return gopherllm.RunAgenticChat(runner, []gopherllm.ChatMessage{promptMessage}, cfg.options, skills, func(s string) bool {
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

func resumeLastModel(cfg *cliConfig, logw io.Writer) bool {
	if cfg == nil || cfg.modelSelector != nil {
		return false
	}
	if logw == nil {
		logw = io.Discard
	}
	path, err := loadLastModel()
	if err != nil {
		fmt.Fprintf(logw, "Warning: could not resume the last model: %v\n", err)
		return false
	}
	if path == "" {
		return false
	}
	cfg.modelSelector = &path
	fmt.Fprintf(logw, "Resuming last model: %s\n", path)
	return true
}

// resolveWasmDir returns the directory server.ServeOptions.WasmDir should
// use: cfg.wasmDir verbatim if --wasm-dir was given explicitly (including
// empty, to opt out of auto-detection), else "bin" if it already contains
// both files `make wasm-build` produces, else "" (the feature simply isn't
// offered -- see HandlerOptions.WasmDir's doc comment).
func resolveWasmDir(cfg cliConfig) string {
	if cfg.wasmDirSet {
		return cfg.wasmDir
	}
	const dir = "bin"
	if fileExists(filepath.Join(dir, "gopherllm.wasm")) && fileExists(filepath.Join(dir, "wasm_exec.js")) {
		return dir
	}
	return ""
}

// serverModelLoadOptions keeps catalog hot-swaps on the same runtime path as
// the initial CLI load. The explicit --mmproj value belongs here because it
// is a user override; an automatically discovered projector is deliberately
// not retained, so each later catalog selection can receive its own verified
// companion in the /models/load handler.
func serverModelLoadOptions(cfg cliConfig) gopherllm.LoadOptions {
	options := gopherllm.LoadOptions{
		PrepareQuantized: cfg.prepareQuant,
		UseMetal:         cfg.useMetal,
		OutOfCore:        cfg.outOfCore,
	}
	if cfg.mmprojPathSet && cfg.mmprojPath != "" {
		options.VisionProjectorPath = cfg.mmprojPath
	}
	return options
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

// runAutoTune calibrates the runtime settings for this model on this machine.
// It reports whether the command is finished (--auto-json prints the result and
// exits). A cached tuning is reused unless --auto-refresh was passed, so the
// startup cost is paid once per model+hardware combination rather than per run.
func runAutoTune(runner *gopherllm.Runner, cfg cliConfig) (gopherllm.AutoTuneResult, bool, error) {
	opts := autoTuneOptions(cfg.autoTuneEffort)
	if !cfg.autoTuneJSON {
		opts.LogWriter = os.Stderr
	}
	res, cached, err := runner.AutoTuneOrCached(opts, cfg.autoTuneRefresh)
	if err != nil {
		return res, false, err
	}
	if cfg.autoTuneJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return res, true, enc.Encode(res)
	}
	if cached {
		fmt.Fprintf(os.Stderr, "Auto: reusing tuning measured %s (--auto-refresh re-measures)\n",
			res.MeasuredAt.Local().Format("2006-01-02 15:04"))
	} else {
		fmt.Fprintf(os.Stderr, "Auto: calibrated in %.1fs\n", res.ElapsedMs/1000)
	}
	fmt.Fprintf(os.Stderr, "  %s\n", res.SettingsLine())
	if g := res.GainsLine(); g != "" {
		fmt.Fprintf(os.Stderr, "  %s\n", g)
	}
	return res, false, nil
}

// autoTuneOptions maps the effort level onto calibration cost; see
// gopherllm.AutoTuneOptionsForEffort, which the HTTP API's POST /autotune/run
// shares so the CLI and the web UI can never silently drift apart.
func autoTuneOptions(effort string) gopherllm.AutoTuneOptions {
	return gopherllm.AutoTuneOptionsForEffort(effort)
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

// buildAgentOSRunner turns --os-commands/--os-commands-allow into a Runner
// for the server, or nil if the flag was never given: the feature does not
// exist on a server started without it, matching the operator-configured,
// off-by-default gate the agentos package's threat model calls for.
func buildAgentOSRunner(cfg cliConfig) (*agentos.Runner, error) {
	if strings.TrimSpace(cfg.osCommandsPolicy) == "" {
		return nil, nil
	}
	policy, err := agentos.ParsePolicy(cfg.osCommandsPolicy)
	if err != nil {
		return nil, err
	}
	var allowed []string
	for _, name := range strings.Split(cfg.osCommandsAllow, ",") {
		if name = strings.TrimSpace(name); name != "" {
			allowed = append(allowed, name)
		}
	}
	if policy == agentos.PolicyWhitelist && len(allowed) == 0 {
		return nil, fmt.Errorf("--os-commands whitelist needs --os-commands-allow with at least one program name")
	}
	return &agentos.Runner{Policy: policy, Allowed: allowed}, nil
}

func parseCLI(args []string) (cliConfig, error) {
	cfg := cliConfig{modelDir: gopherllm.DefaultModelDir(), preset: "balanced", options: gopherllm.DefaultGenerationOptions(), benchRuns: 3, kernelBenchRuns: 25, maxConn: 8}
	configPath, err := configPathFromArgs(args)
	if err != nil {
		return cfg, err
	}
	if configPath != "" {
		raw, err := loadFileConfig(configPath)
		if err != nil {
			return cfg, err
		}
		if err := applyFileConfig(&cfg, raw); err != nil {
			return cfg, err
		}
	}
	cliPreset, err := presetFromArgs(args)
	if err != nil {
		return cfg, err
	}
	if cliPreset != "" {
		if err := applyPreset(&cfg, cliPreset); err != nil {
			return cfg, err
		}
	}
	setSelector := func(value string) error {
		if cfg.modelSelector != nil && !cfg.modelSelectorFromConfig {
			return fmt.Errorf("multiple model selectors were provided")
		}
		cfg.modelSelector = &value
		cfg.modelSelectorFromConfig = false
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
		case "--config":
			if _, err := next(arg); err != nil {
				return cfg, err
			}
		case "--print-config":
			cfg.printConfig = true
		case "--preset":
			if _, err := next(arg); err != nil {
				return cfg, err
			}
		case "--model":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			if cfg.modelSelectorFromConfig {
				cfg.modelSelector, cfg.modelSelectorFromConfig = &v, false
			} else if cfg.modelSelector != nil {
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
		case "--mmproj":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.mmprojPath, cfg.mmprojPathSet = v, true
		case "--image":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.imagePath = v
		case "--list-models":
			cfg.listModels = true
		case "--hf-list":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.hfList = v
		case "--hf-offline":
			cfg.hfOffline = true
		case "--privacy":
			cfg.privacyReport = true
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
		case "--chat-history":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.chatHistoryPath = v
		case "--wasm-dir":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.wasmDir, cfg.wasmDirSet = v, true
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
		case "--context-window":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			mode, err := parseContextWindowMode(v)
			if err != nil {
				return cfg, err
			}
			cfg.options.ContextWindowMode = mode
		case "--threads":
			v, err := parseNextInt(next, arg)
			if err != nil {
				return cfg, err
			}
			cfg.threads, cfg.threadsSet = v, true
		case "--metal":
			cfg.useMetal = true
			cfg.metalExplicit = true
		case "--prepare-quant":
			cfg.prepareQuant = true
		case "--out-of-core":
			cfg.outOfCore = true
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
		case "--os-commands":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.osCommandsPolicy = v
		case "--os-commands-allow":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.osCommandsAllow = v
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
		case "--auto":
			cfg.autoTune = true
		case "--auto-refresh":
			cfg.autoTune = true
			cfg.autoTuneRefresh = true
		case "--auto-json":
			cfg.autoTune = true
			cfg.autoTuneJSON = true
		case "--auto-effort":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			if !validAutoEffort(v) {
				return cfg, fmt.Errorf("--auto-effort must be quick, balanced, or thorough")
			}
			cfg.autoTune = true
			cfg.autoTuneEffort = v
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
		case "--compress":
			cfg.compress = true
		case "--compress-format":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.compressFormat = v
		case "--compress-out":
			v, err := next(arg)
			if err != nil {
				return cfg, err
			}
			cfg.compressOut = v
		case "--compress-uniform":
			cfg.compressUniform = true
		default:
			if strings.HasPrefix(arg, "-") {
				return cfg, fmt.Errorf("unknown option: %s", arg)
			}
			if cfg.modelSelector != nil {
				if cfg.modelSelectorFromConfig {
					if err := setSelector(arg); err != nil {
						return cfg, err
					}
					continue
				}
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
	if cfg.threadsSet && cfg.threads <= 0 {
		return cfg, fmt.Errorf("--threads must be greater than 0")
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

// runCompress requantizes modelPath's GGUF to cfg.compressFormat via plain
// round-to-nearest (gopherllm.CompressModel) and writes it to
// cfg.compressOut. No weights are loaded through the normal Runner path —
// CompressModel reads tensor bytes directly from the source file's mmap.
func runCompress(modelPath string, cfg cliConfig) error {
	if cfg.compressOut == "" {
		return fmt.Errorf("--compress requires --compress-out <path>")
	}
	format, ok := gopherllm.ParseCompressFormat(cfg.compressFormat)
	if !ok {
		return fmt.Errorf("--compress-format %q is not supported (want one of: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K)", cfg.compressFormat)
	}
	fmt.Fprintf(os.Stderr, "Compressing %s -> %s (%s)\n", modelPath, cfg.compressOut, format)
	if err := gopherllm.CompressModel(modelPath, cfg.compressOut, gopherllm.CompressOptions{
		TargetFormat: format,
		Uniform:      cfg.compressUniform,
		LogWriter:    os.Stderr,
	}); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "Wrote %s\n", cfg.compressOut)
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
