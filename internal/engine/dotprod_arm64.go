//go:build arm64

package engine

import "github.com/SimonWaldherr/GopherLLM/internal/cpuinfo"

// Cache the feature gate before the optional ARM kernel self-checks run.
var hasDotProd = cpuinfo.HasDotProd()
