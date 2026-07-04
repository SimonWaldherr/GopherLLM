BINARY       ?= gopherllm
BUILD_DIR    ?= bin
CACHE_DIR    ?= $(CURDIR)/.cache/go-build
GO           ?= go
GOFLAGS      ?=
CGO_ENABLED  ?= 0
CROSS_CGO_ENABLED ?= 0
GOCACHE      ?= $(CACHE_DIR)
METAL_BIN    ?= $(BUILD_DIR)/$(BINARY)-metal
METAL_TAGS   ?= metal

MODEL_DIR     ?= $(HOME)/.cache/lm-studio/models/lmstudio-community
MODEL         ?=
PROMPT        ?= Wer war Albert Einstein?
SYNONYM_PROMPT ?= Nenne ein Synonym für Synonym und antworte nur mit diesem einen Wort.
NATO_PROMPT ?= Output exactly the 26 NATO phonetic alphabet code words from A to Z, one word per line. No letters, numbers, punctuation, parentheses, or explanation.
MAX_TOKENS    ?= 32
TEMP          ?= 0
TOP_P         ?= 0.9
TOP_K         ?= 40
BENCH_RUNS    ?= 3
KERNEL_BENCH_RUNS ?= 25
KERNEL_BENCH_LAYER ?= 0
MODEL_TIMEOUT ?= 2m
PREPARE_QUANT ?= 0
ADDR          ?= 127.0.0.1:8080
SERVE_ADDR    ?= $(ADDR)
CHAT          ?= 1
ARGS          ?=

export CGO_ENABLED
export GOCACHE

BIN        := $(BUILD_DIR)/$(BINARY)
METAL_RUN_ARGS = --metal $(_RUN_ARGS)
CHAT_FLAG  = $(if $(filter 1 true yes on,$(CHAT)),--chat,)
PREPARE_FLAG = $(if $(filter 1 true yes on,$(PREPARE_QUANT)),--prepare-quant,)
_MODEL_ARG = $(if $(MODEL),--model "$(MODEL)",)
_BASE_RUN_ARGS = $(if $(ARGS),$(ARGS),--model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" --top-p "$(TOP_P)" --top-k "$(TOP_K)")
_RUN_ARGS  = $(PREPARE_FLAG) $(_BASE_RUN_ARGS)

.PHONY: all build release build-metal cross-build run run-normal run-prep run-metal run-full run-full-prep run-full-metal run-full-metal-prep compare-run compare-run-metal repl serve serve-metal https list-models inspect list-tensors bench bench-model bench-model-prep bench-model-metal compare-bench synonym-bench nato-bench kernel-bench kernel-bench-prep kernel-bench-metal compare-kernel-bench compare-kernel-bench-metal fmt test test-small-models vet check clean help

all: check release

build:
	@mkdir -p $(BUILD_DIR) $(GOCACHE)
	@$(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BIN) .

release: build

build-metal:
	@mkdir -p $(BUILD_DIR) $(GOCACHE)
	@CGO_ENABLED=1 $(GO) build $(GOFLAGS) -tags "$(METAL_TAGS)" -trimpath -ldflags="-s -w" -o $(METAL_BIN) .

cross-build:
	@mkdir -p $(BUILD_DIR) $(GOCACHE)
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=darwin GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-darwin-amd64 .
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=darwin GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-darwin-arm64 .
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=linux GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-linux-amd64 .
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=linux GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-linux-arm64 .
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=windows GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-windows-amd64.exe .
	CGO_ENABLED=$(CROSS_CGO_ENABLED) GOOS=windows GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-windows-arm64.exe .

run: release
	@$(BIN) $(_RUN_ARGS)

run-normal: run

run-prep: PREPARE_QUANT=1
run-prep: run

run-metal: build-metal
	@$(METAL_BIN) $(METAL_RUN_ARGS)

run-full: MAX_TOKENS=256
run-full: run

run-full-prep: MAX_TOKENS=256
run-full-prep: PREPARE_QUANT=1
run-full-prep: run

run-full-metal: MAX_TOKENS=256
run-full-metal: run-metal

run-full-metal-prep: MAX_TOKENS=256
run-full-metal-prep: PREPARE_QUANT=1
run-full-metal-prep: run-metal

compare-run: release
	@printf "\n== normal ==\n"
	@$(BIN) $(_BASE_RUN_ARGS)
	@printf "\n== prepare-quant ==\n"
	@$(BIN) --prepare-quant $(_BASE_RUN_ARGS)

compare-run-metal: release build-metal
	@printf "\n== normal ==\n"
	@$(BIN) $(_BASE_RUN_ARGS)
	@printf "\n== metal ==\n"
	@$(METAL_BIN) --metal $(_BASE_RUN_ARGS)
	@printf "\n== metal + prepare-quant ==\n"
	@$(METAL_BIN) --metal --prepare-quant $(_BASE_RUN_ARGS)

repl: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --repl

serve: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --serve "$(SERVE_ADDR)" $(CHAT_FLAG)

serve-metal:
	@printf "serve-metal is not wired yet; Metal currently accelerates selected CLI matvecs.\n"
	@printf "Use: make run-metal MODEL=...\n"
	@exit 2

https:
	@printf "https is not available in the Go port yet.\n"
	@printf "Use RustyLLM's make https target for TLS serving.\n"
	@exit 2

list-models: release
	@$(BIN) --model-dir "$(MODEL_DIR)" --list-models

inspect: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --inspect

list-tensors: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --list-tensors

bench:
	@mkdir -p $(GOCACHE)
	$(GO) test $(GOFLAGS) -run '^$$' -bench=. -benchmem .

bench-model: release
	@$(BIN) $(PREPARE_FLAG) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

bench-model-prep: PREPARE_QUANT=1
bench-model-prep: bench-model

bench-model-metal: build-metal
	@$(METAL_BIN) --metal $(PREPARE_FLAG) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

compare-bench: release
	@printf "\n== normal bench ==\n"
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"
	@printf "\n== prepare-quant bench ==\n"
	@$(BIN) --prepare-quant --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

synonym-bench: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(SYNONYM_PROMPT)" --max-tokens "8" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

nato-bench: release
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(NATO_PROMPT)" --max-tokens "128" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --repeat-penalty "1" --bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

kernel-bench: release
	@$(BIN) $(PREPARE_FLAG) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

kernel-bench-prep: PREPARE_QUANT=1
kernel-bench-prep: kernel-bench

kernel-bench-metal: build-metal
	@$(METAL_BIN) --metal $(PREPARE_FLAG) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

compare-kernel-bench: release
	@printf "\n== normal kernel bench ==\n"
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"
	@printf "\n== prepare-quant kernel bench ==\n"
	@$(BIN) --prepare-quant --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

compare-kernel-bench-metal: release build-metal
	@printf "\n== normal kernel bench ==\n"
	@$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"
	@printf "\n== metal kernel bench ==\n"
	@$(METAL_BIN) --metal --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--kernel-bench-json --kernel-bench-runs "$(KERNEL_BENCH_RUNS)" --kernel-bench-layer "$(KERNEL_BENCH_LAYER)"

fmt:
	$(GO) fmt ./...

test:
	@mkdir -p $(GOCACHE)
	$(GO) test $(GOFLAGS) ./...

test-small-models: release
	GOPHERLLM_RUN_MODEL_SWEEP=1 GOPHERLLM_MODEL_DIR="$(MODEL_DIR)" GOPHERLLM_SWEEP_BINARY="$(CURDIR)/$(BIN)" GOPHERLLM_MODEL_SWEEP_TIMEOUT="$(MODEL_TIMEOUT)" \
		$(GO) test $(GOFLAGS) -run TestSmallLocalModelsAnswerEinsteinPrompt -count=1 -timeout=20m -v .

vet:
	@mkdir -p $(GOCACHE)
	$(GO) vet $(GOFLAGS) ./...

check: fmt test vet

clean:
	rm -rf $(BUILD_DIR) .cache

help:
	@printf "Targets:\n"
	@printf "  make all                             Run check and release build\n"
	@printf "  make build/release                   Build ./$(BIN)\n"
	@printf "  make build-metal                     Build ./$(METAL_BIN) with CGO+Metal tag\n"
	@printf "  make cross-build                     Build darwin/linux/windows for amd64 and arm64\n"
	@printf "  make run MODEL=... PROMPT='...'      Generate from a one-shot prompt\n"
	@printf "  make run-prep MODEL=...              Generate with --prepare-quant\n"
	@printf "  make run-metal MODEL=...             Generate with experimental --metal\n"
	@printf "  make run-full MODEL=...              Generate 256 tokens, matching CLI default\n"
	@printf "  make run-full-prep MODEL=...         Generate 256 tokens with --prepare-quant\n"
	@printf "  make run-full-metal MODEL=...        Generate 256 tokens with --metal\n"
	@printf "  make run-full-metal-prep MODEL=...   Generate 256 tokens with --metal and --prepare-quant\n"
	@printf "  make compare-run MODEL=...           Run normal, then --prepare-quant\n"
	@printf "  make compare-run-metal MODEL=...     Run normal, --metal, then --metal --prepare-quant\n"
	@printf "  make run ARGS='...'                  Run the CLI with custom args\n"
	@printf "  make repl MODEL=...                  Start interactive REPL mode\n"
	@printf "  make serve MODEL=... CHAT=1          Start HTTP API / optional web UI\n"
	@printf "  make serve-metal                     Explain why Metal is unavailable\n"
	@printf "  make https                           Explain TLS status for the Go port\n"
	@printf "  make list-models                     List GGUFs in MODEL_DIR\n"
	@printf "  make inspect MODEL=...               Inspect GGUF metadata and compatibility\n"
	@printf "  make list-tensors MODEL=...          Print tensor inventory\n"
	@printf "  make bench                           Run Go microbenchmarks\n"
	@printf "  make bench-model MODEL=...           Run CLI generation benchmark JSON with per-run output\n"
	@printf "  make bench-model-prep MODEL=...      Run generation benchmark with --prepare-quant\n"
	@printf "  make bench-model-metal MODEL=...     Run generation benchmark with --metal\n"
	@printf "  make compare-bench MODEL=...         Benchmark normal and --prepare-quant\n"
	@printf "  make synonym-bench MODEL=...         Run fixed one-word synonym prompt benchmark\n"
	@printf "  make nato-bench MODEL=...            Run fixed NATO alphabet prompt benchmark\n"
	@printf "  make kernel-bench MODEL=...          Run isolated kernel benchmark JSON\n"
	@printf "  make kernel-bench-prep MODEL=...     Run isolated kernel benchmark with --prepare-quant\n"
	@printf "  make kernel-bench-metal MODEL=...    Run isolated kernel benchmark with --metal\n"
	@printf "  make compare-kernel-bench MODEL=...  Kernel-benchmark normal and --prepare-quant\n"
	@printf "  make compare-kernel-bench-metal MODEL=...  Kernel-benchmark normal and --metal\n"
	@printf "  make fmt/test/vet/check              Format, test, vet, or all three\n"
	@printf "  make test-small-models               Run local <5GB model prompt sweep\n"
	@printf "  make clean                           Remove build artifacts\n"
	@printf "\nVariables:\n"
	@printf "  MODEL_DIR=%s\n" "$(MODEL_DIR)"
	@printf "  MODEL=%s  (name fragment or absolute .gguf path)\n" "$(MODEL)"
	@printf "  METAL_BIN=%s METAL_TAGS=%s\n" "$(METAL_BIN)" "$(METAL_TAGS)"
	@printf "  PROMPT=%s\n" "$(PROMPT)"
	@printf "  SYNONYM_PROMPT=%s\n" "$(SYNONYM_PROMPT)"
	@printf "  NATO_PROMPT=%s\n" "$(NATO_PROMPT)"
	@printf "  MAX_TOKENS=%s TEMP=%s TOP_P=%s TOP_K=%s\n" "$(MAX_TOKENS)" "$(TEMP)" "$(TOP_P)" "$(TOP_K)"
	@printf "  BENCH_RUNS=%s MODEL_TIMEOUT=%s SERVE_ADDR=%s CHAT=%s\n" "$(BENCH_RUNS)" "$(MODEL_TIMEOUT)" "$(SERVE_ADDR)" "$(CHAT)"
	@printf "  KERNEL_BENCH_RUNS=%s KERNEL_BENCH_LAYER=%s\n" "$(KERNEL_BENCH_RUNS)" "$(KERNEL_BENCH_LAYER)"
	@printf "  PREPARE_QUANT=%s\n" "$(PREPARE_QUANT)"
