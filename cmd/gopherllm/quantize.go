package main

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const quantizeFormats = "Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q4_K_M, Q5_K, Q5_K_M, Q6_K"

type quantizeCommand struct {
	source, output string
	options        gopherllm.CompressOptions
	threads        int
	help           bool
}

func printQuantizeUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage: gopherllm quantize <input.gguf> <output.gguf> <type> [--threads N]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Quantize a local GGUF with round-to-nearest (RTN) encoders.")
	fmt.Fprintln(w, "Types: "+quantizeFormats)
	fmt.Fprintln(w, "  --threads N    Number of CPU workers (default: automatic)")
	fmt.Fprintln(w, "  --uniform      Use the base type for every eligible tensor")
	fmt.Fprintln(w, "  --help, -h     Show this help")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Example: gopherllm quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M")
}

func parseQuantizeCommand(args []string) (quantizeCommand, error) {
	var cmd quantizeCommand
	for _, arg := range args {
		if arg == "--" {
			break
		}
		if arg == "--help" || arg == "-h" {
			cmd.help = true
			return cmd, nil
		}
	}
	var positional []string
	var uniform bool
	options := true
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if options && arg == "--" {
			options = false
			continue
		}
		switch {
		case options && (arg == "--threads" || strings.HasPrefix(arg, "--threads=")):
			value := strings.TrimPrefix(arg, "--threads=")
			if arg == "--threads" {
				i++
				if i == len(args) {
					return cmd, fmt.Errorf("quantize: --threads requires a positive integer")
				}
				value = args[i]
			}
			n, err := strconv.Atoi(value)
			if err != nil || n <= 0 {
				return cmd, fmt.Errorf("quantize: --threads must be a positive integer, got %q", value)
			}
			cmd.threads = n
		case options && arg == "--uniform":
			uniform = true
		case options && strings.HasPrefix(arg, "-"):
			return cmd, fmt.Errorf("quantize: unknown option %q (see 'gopherllm quantize --help')", arg)
		default:
			positional = append(positional, arg)
		}
	}
	if len(positional) != 3 {
		return cmd, fmt.Errorf("quantize: expected <input.gguf> <output.gguf> <type> (see 'gopherllm quantize --help')")
	}
	cmd.source, cmd.output = positional[0], positional[1]
	if cmd.source == "" || cmd.output == "" {
		return cmd, fmt.Errorf("quantize: input and output paths must not be empty")
	}
	var ok bool
	cmd.options, ok = gopherllm.ParseCompressOptions(positional[2])
	if !ok {
		return cmd, fmt.Errorf("quantize: unsupported type %q (want one of: %s)", positional[2], quantizeFormats)
	}
	if uniform && cmd.options.Preset != "" {
		return cmd, fmt.Errorf("quantize: --uniform cannot be combined with mixed preset %s; use %s for uniform quantization", cmd.options.Preset, cmd.options.TargetFormat)
	}
	cmd.options.Uniform = uniform
	return cmd, nil
}

func compressTargetName(opts gopherllm.CompressOptions) string {
	if opts.Preset != "" {
		return opts.Preset
	}
	return opts.TargetFormat.String()
}

func runQuantize(ctx context.Context, args []string, log io.Writer) error {
	cmd, err := parseQuantizeCommand(args)
	if err != nil {
		return err
	}
	if log == nil {
		log = io.Discard
	}
	if cmd.help {
		printQuantizeUsage(log)
		return nil
	}
	if cmd.threads > 0 {
		gopherllm.SetNumThreads(cmd.threads)
		runtime.GOMAXPROCS(cmd.threads)
	}
	cmd.options.Context = ctx
	cmd.options.LogWriter = log
	fmt.Fprintf(log, "Quantizing %s -> %s (%s)\n", cmd.source, cmd.output, compressTargetName(cmd.options))
	if err := gopherllm.CompressModel(cmd.source, cmd.output, cmd.options); err != nil {
		return fmt.Errorf("quantize: %w", err)
	}
	fmt.Fprintf(log, "Wrote %s\n", cmd.output)
	return nil
}
