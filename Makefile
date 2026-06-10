BINARY       ?= gopherllm
BUILD_DIR    ?= bin
CACHE_DIR    ?= $(CURDIR)/.cache/go-build
GO           ?= go
GOFLAGS      ?=
CGO_ENABLED  ?= 0
GOCACHE      ?= $(CACHE_DIR)

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
ADDR          ?= 127.0.0.1:8080
SERVE_ADDR    ?= $(ADDR)
CHAT          ?= 1
ARGS          ?=

export CGO_ENABLED
export GOCACHE

BIN        := $(BUILD_DIR)/$(BINARY)
CHAT_FLAG  := $(if $(filter 1 true yes on,$(CHAT)),--chat,)
_MODEL_ARG := $(if $(MODEL),--model "$(MODEL)",)
_RUN_ARGS  := $(if $(ARGS),$(ARGS),--model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" --top-p "$(TOP_P)" --top-k "$(TOP_K)")

.PHONY: all build release cross-build run repl serve serve-metal https list-models inspect list-tensors bench bench-model synonym-bench nato-bench kernel-bench fmt test test-small-models vet check clean help

all: check release

build:
	@mkdir -p $(BUILD_DIR) $(GOCACHE)
	$(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BIN) .

release: build

cross-build:
	@mkdir -p $(BUILD_DIR) $(GOCACHE)
	GOOS=darwin GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-darwin-amd64 .
	GOOS=darwin GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-darwin-arm64 .
	GOOS=linux GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-linux-amd64 .
	GOOS=linux GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-linux-arm64 .
	GOOS=windows GOARCH=amd64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-windows-amd64.exe .
	GOOS=windows GOARCH=arm64 $(GO) build $(GOFLAGS) -trimpath -ldflags="-s -w" -o $(BUILD_DIR)/$(BINARY)-windows-arm64.exe .

run: release
	$(BIN) $(_RUN_ARGS)

repl: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --repl

serve: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --serve "$(SERVE_ADDR)" $(CHAT_FLAG)

serve-metal:
	@printf "serve-metal is not available: the Go port currently runs pure Go / no CGO only.\n"
	@printf "Use: make serve\n"
	@exit 2

https:
	@printf "https is not available in the Go port yet.\n"
	@printf "Use RustyLLM's make https target for TLS serving.\n"
	@exit 2

list-models: release
	$(BIN) --model-dir "$(MODEL_DIR)" --list-models

inspect: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --inspect

list-tensors: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) --list-tensors

bench:
	@mkdir -p $(GOCACHE)
	$(GO) test $(GOFLAGS) -run '^$$' -bench=. -benchmem .

bench-model: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(PROMPT)" --max-tokens "$(MAX_TOKENS)" --temp "$(TEMP)" \
		--bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

synonym-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(SYNONYM_PROMPT)" --max-tokens "8" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

nato-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
		--prompt "$(NATO_PROMPT)" --max-tokens "128" --temp "0" \
		--top-p "$(TOP_P)" --top-k "$(TOP_K)" --repeat-penalty "1" --bench --bench-json --bench-runs "$(BENCH_RUNS)" --timeout "$(MODEL_TIMEOUT)"

kernel-bench: release
	$(BIN) --model-dir "$(MODEL_DIR)" $(_MODEL_ARG) \
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
	@printf "  make cross-build                     Build darwin/linux/windows for amd64 and arm64\n"
	@printf "  make run MODEL=... PROMPT='...'      Generate from a one-shot prompt\n"
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
	@printf "  make synonym-bench MODEL=...         Run fixed one-word synonym prompt benchmark\n"
	@printf "  make nato-bench MODEL=...            Run fixed NATO alphabet prompt benchmark\n"
	@printf "  make kernel-bench MODEL=...          Run isolated kernel benchmark JSON\n"
	@printf "  make fmt/test/vet/check              Format, test, vet, or all three\n"
	@printf "  make test-small-models               Run local <5GB model prompt sweep\n"
	@printf "  make clean                           Remove build artifacts\n"
	@printf "\nVariables:\n"
	@printf "  MODEL_DIR=%s\n" "$(MODEL_DIR)"
	@printf "  MODEL=%s\n" "$(MODEL)"
	@printf "  PROMPT=%s\n" "$(PROMPT)"
	@printf "  SYNONYM_PROMPT=%s\n" "$(SYNONYM_PROMPT)"
	@printf "  NATO_PROMPT=%s\n" "$(NATO_PROMPT)"
	@printf "  MAX_TOKENS=%s TEMP=%s TOP_P=%s TOP_K=%s\n" "$(MAX_TOKENS)" "$(TEMP)" "$(TOP_P)" "$(TOP_K)"
	@printf "  BENCH_RUNS=%s MODEL_TIMEOUT=%s SERVE_ADDR=%s CHAT=%s\n" "$(BENCH_RUNS)" "$(MODEL_TIMEOUT)" "$(SERVE_ADDR)" "$(CHAT)"
	@printf "  KERNEL_BENCH_RUNS=%s KERNEL_BENCH_LAYER=%s\n" "$(KERNEL_BENCH_RUNS)" "$(KERNEL_BENCH_LAYER)"
