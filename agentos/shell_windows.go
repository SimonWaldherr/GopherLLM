//go:build windows

package agentos

import (
	"context"
	"os"
	"os/exec"
)

func shellCommandContext(ctx context.Context, command string) *exec.Cmd {
	shell := os.Getenv("COMSPEC")
	if shell == "" {
		shell = "cmd.exe"
	}
	return exec.CommandContext(ctx, shell, "/d", "/s", "/c", command)
}
