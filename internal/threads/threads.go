// Package threads holds the process-wide worker override shared by internal
// subsystems. Zero means use the current GOMAXPROCS value.
package threads

import (
	"runtime"
	"sync/atomic"
)

var configured atomic.Int64

// Set configures the shared worker count. Values below one restore the
// GOMAXPROCS default.
func Set(n int) {
	if n < 1 {
		n = 0
	}
	configured.Store(int64(n))
}

// Count returns the configured worker count or the current GOMAXPROCS value.
func Count() int {
	if n := int(configured.Load()); n > 0 {
		return n
	}
	return max(1, runtime.GOMAXPROCS(0))
}
