// Embedded Speech streams raw PCM recordings through one reusable local model.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "local Voxtral Realtime GGUF")
	flag.Parse()
	if *modelPath == "" || flag.NArg() == 0 {
		return errors.New("usage: embedded-speech -model voxtral.gguf recording.pcm [another.pcm ...]; PCM16LE, mono, 16000 Hz")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	model, err := gopherllm.OpenVoxtral(ctx, *modelPath)
	if err != nil {
		return err
	}
	defer model.Close()
	for _, path := range flag.Args() {
		if err := transcribe(ctx, model, path); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

func transcribe(ctx context.Context, model *gopherllm.VoxtralModel, path string) error {
	input, err := os.Open(path)
	if err != nil {
		return err
	}
	defer input.Close()
	session, err := model.NewSession(ctx)
	if err != nil {
		return err
	}
	defer session.Close()
	buffer := make([]byte, 6400) // 200 ms of PCM16; reader chunks need not be this size.
	previous := ""
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, readErr := io.ReadFull(input, buffer)
		if readErr != nil && readErr != io.EOF && readErr != io.ErrUnexpectedEOF {
			return readErr
		}
		if n > 0 {
			text, err := session.PushPCM16(ctx, buffer[:n], false)
			if err != nil {
				return err
			}
			// Push returns accumulated text, not a delta. A UI should replace its
			// live transcript. No sleeps are needed; capture provides backpressure.
			if text != previous {
				fmt.Printf("partial: %s\n", text)
				previous = text
			}
		}
		if readErr != nil {
			break
		}
	}
	text, err := session.Flush(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("final (%s): %s\n", path, text)
	return nil
}
