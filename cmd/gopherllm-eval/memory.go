package main

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// RSS sampling is optional and explicitly local to a supplied server PID.
// A missing/unsupported ps reading remains unavailable, never zero.
func sampleRSS(parent context.Context, pid int) func() *uint64 {
	if pid <= 0 {
		return func() *uint64 { return nil }
	}
	ctx, cancel := context.WithCancel(parent)
	var peak atomic.Uint64
	done := make(chan struct{})
	read := func() {
		callCtx, stop := context.WithTimeout(ctx, time.Second)
		defer stop()
		b, err := exec.CommandContext(callCtx, "ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
		if err != nil {
			return
		}
		kb, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
		if err != nil {
			return
		}
		value := kb * 1024
		for old := peak.Load(); value > old; old = peak.Load() {
			if peak.CompareAndSwap(old, value) {
				break
			}
		}
	}
	read()
	go func() {
		defer close(done)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				read()
			}
		}
	}()
	return func() *uint64 {
		cancel()
		<-done
		value := peak.Load()
		if value == 0 {
			return nil
		}
		return &value
	}
}
