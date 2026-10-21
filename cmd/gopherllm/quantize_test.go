package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestParseQuantizeCommand(t *testing.T) {
	for _, args := range [][]string{
		{"input.gguf", "output.gguf", "Q4_K_M", "--threads", "3"},
		{"--threads=3", "input.gguf", "output.gguf", "q4_k_m"},
		{"input.gguf", "--threads", "3", "output.gguf", "Q4_K_M"},
	} {
		cmd, err := parseQuantizeCommand(args)
		if err != nil {
			t.Fatalf("parseQuantizeCommand(%v): %v", args, err)
		}
		if cmd.source != "input.gguf" || cmd.output != "output.gguf" || cmd.threads != 3 {
			t.Fatalf("parseQuantizeCommand(%v) = %+v", args, cmd)
		}
		if cmd.options.TargetFormat != gopherllm.GGMLTypeQ4_K || cmd.options.Preset != "Q4_K_M" || cmd.options.Uniform {
			t.Fatalf("mixed preset lost: %+v", cmd.options)
		}
	}
	cmd, err := parseQuantizeCommand([]string{"--uniform", "--", "-input.gguf", "output.gguf", "Q5_K"})
	if err != nil || !cmd.options.Uniform || cmd.source != "-input.gguf" || cmd.options.TargetFormat != gopherllm.GGMLTypeQ5_K {
		t.Fatalf("uniform/end-of-options = %+v, %v", cmd, err)
	}
	cmd, err = parseQuantizeCommand([]string{"--", "--help", "output.gguf", "Q4_K"})
	if err != nil || cmd.help || cmd.source != "--help" {
		t.Fatalf("literal help filename = %+v, %v", cmd, err)
	}
}

func TestParseQuantizeCommandRejectsInvalidArguments(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
	}{
		{nil, "expected <input.gguf>"},
		{[]string{"a", "b"}, "expected <input.gguf>"},
		{[]string{"a", "b", "Q4_K_M", "extra"}, "expected <input.gguf>"},
		{[]string{"a", "b", "Q9"}, "unsupported type"},
		{[]string{"a", "b", "Q4_K_M", "--threads"}, "requires a positive integer"},
		{[]string{"a", "b", "Q4_K_M", "--threads", "0"}, "positive integer"},
		{[]string{"a", "b", "Q4_K_M", "--threads=-1"}, "positive integer"},
		{[]string{"a", "b", "Q4_K_M", "--threads="}, "positive integer"},
		{[]string{"a", "b", "Q4_K_M", "--threads", "many"}, "positive integer"},
		{[]string{"a", "b", "Q4_K_M", "--bogus"}, "unknown option"},
		{[]string{"a", "b", "Q4_K_M", "--uniform"}, "cannot be combined with mixed preset"},
		{[]string{"", "b", "Q4_K_M"}, "paths must not be empty"},
	} {
		if _, err := parseQuantizeCommand(tc.args); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("parseQuantizeCommand(%v) error = %v, want %q", tc.args, err, tc.want)
		}
	}
}

func TestRunQuantizeHelp(t *testing.T) {
	for _, flag := range []string{"--help", "-h"} {
		var out bytes.Buffer
		if err := runQuantize(context.Background(), []string{flag}, &out); err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"gopherllm quantize", "Q4_K_M", "Q5_K_M", "--threads N"} {
			if !strings.Contains(out.String(), want) {
				t.Fatalf("help missing %q: %s", want, out.String())
			}
		}
	}
}

func TestRunDispatchesQuantizeBeforeGlobalHelp(t *testing.T) {
	stderr, err := os.CreateTemp(t.TempDir(), "stderr")
	if err != nil {
		t.Fatal(err)
	}
	defer stderr.Close()
	previousArgs, previousStderr := os.Args, os.Stderr
	os.Args = []string{"gopherllm", "quantize", "--help"}
	os.Stderr = stderr
	defer func() { os.Args, os.Stderr = previousArgs, previousStderr }()
	if err := run(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(stderr.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(data, []byte("round-to-nearest (RTN)")) || bytes.Contains(data, []byte("--serve")) {
		t.Fatalf("expected quantize-specific help, got %s", data)
	}
}

func writeQuantizeCLIFixture(t *testing.T) string {
	t.Helper()
	plans := []gopherllm.PlannedTensor{
		{Name: "blk.0.ffn_up.weight", Dims: []uint64{256, 2}, DType: gopherllm.GGMLTypeF16},
		{Name: "output.weight", Dims: []uint64{256, 2}, DType: gopherllm.GGMLTypeF16},
	}
	var buf bytes.Buffer
	w, err := gopherllm.NewGGUFWriter(&buf, map[string]gopherllm.MetaValue{
		"general.architecture":  {Kind: "str", Value: "llama"},
		"llama.block_count":     {Kind: "u32", Value: uint32(1)},
		"tokenizer.ggml.tokens": {Kind: "array", Value: []string{"hello", "world"}},
	}, plans, 32)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]byte, 256*2*2)
	for i := range 256 * 2 {
		binary.LittleEndian.PutUint16(values[i*2:], gopherllm.F32ToF16(float32(i%31-15)/16))
	}
	for range plans {
		if err := w.WriteTensor(values); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "input-f16.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunQuantizeF16ToMixedPreset(t *testing.T) {
	input := writeQuantizeCLIFixture(t)
	output := filepath.Join(t.TempDir(), "output-Q4_K_M.gguf")
	var log bytes.Buffer
	if err := runQuantize(context.Background(), []string{input, output, "Q4_K_M"}, &log); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	model, err := gopherllm.ParseGGUF(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Tensors) != 2 || model.Tensors[0].DType != gopherllm.GGMLTypeQ4_K || model.Tensors[1].DType != gopherllm.GGMLTypeQ6_K {
		t.Fatalf("output tensor types: %+v", model.Tensors)
	}
	if _, ok := model.Metadata["tokenizer.ggml.tokens"]; !ok {
		t.Fatal("tokenizer metadata lost")
	}
	if !strings.Contains(log.String(), "Q4_K_M") || !strings.Contains(log.String(), "Wrote "+output) {
		t.Fatalf("command log: %s", log.String())
	}
}

func TestRunQuantizePreservesInputAndDestinationOnFailure(t *testing.T) {
	input := writeQuantizeCLIFixture(t)
	original, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := runQuantize(context.Background(), []string{input, input, "Q4_K_M"}, nil); err == nil {
		t.Fatal("in-place quantization accepted")
	}
	after, err := os.ReadFile(input)
	if err != nil || !bytes.Equal(original, after) {
		t.Fatal("in-place quantization changed the source")
	}
	output := filepath.Join(t.TempDir(), "existing.gguf")
	const sentinel = "preserve existing file"
	if err := os.WriteFile(output, []byte(sentinel), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = runQuantize(ctx, []string{input, output, "Q4_K_M"}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled quantization: %v", err)
	}
	after, err = os.ReadFile(output)
	if err != nil || string(after) != sentinel {
		t.Fatalf("cancelled quantization changed destination: %q, %v", after, err)
	}
}
