//go:build unix

package mmapfile

import (
	"os"
	"runtime"
	"syscall"
)

// File exposes a model file as one immutable byte slice, memory-mapped
// where the platform allows so multi-gigabyte weights are paged in on demand
// rather than copied. Quantized Weights borrow sub-slices of it directly
// (loadWeight's borrow mode), so it must stay open for the Runner's lifetime;
// Runner.Close unmaps it.
type File struct {
	data   []byte
	mapped bool
}

// hugepageThreshold is the smallest mapping for which we request Linux
// transparent hugepages.  The kernel's khugepaged already collapses eligible
// regions over time, but an explicit MADV_HUGEPAGE makes it happen eagerly for
// the hot weight bytes and avoids TLB misses on the decode path. 2 MiB is the
// common hugepage size on x86_64 and a reasonable lower bound on arm64 as well.
const hugepageThreshold = 2 << 20

// Open uses the platform mmap syscall without CGO. If mmap is unavailable
// for a specific file, it falls back to os.ReadFile while preserving the same
// immutable byte-slice API.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return &File{}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err == nil {
		adviseHugepage(data)
		return &File{data: data, mapped: true}, nil
	}

	data, err = os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &File{data: data}, nil
}

// adviseHugepage asks the kernel to back the mapping with transparent hugepages
// on Linux. This is best-effort: if the kernel cannot satisfy the request the
// mapping still works with normal pages. The opt-out variable
// GOPHERLLM_NO_HUGEPAGE skips the advice for A/B testing or environments where
// hugepages hurt latency predictability.
func adviseHugepage(data []byte) {
	if runtime.GOOS != "linux" || len(data) < hugepageThreshold || os.Getenv("GOPHERLLM_NO_HUGEPAGE") != "" {
		return
	}
	// syscall.Madvise is defined on Linux only; guard by build tag would split
	// this file, so use runtime.GOOS instead and ignore errors.
	_ = syscall.Madvise(data, syscall.MADV_HUGEPAGE)
}

func (m *File) Bytes() []byte  { return m.data }
func (m *File) Len() int       { return len(m.data) }
func (m *File) IsMapped() bool { return m.mapped }
func (m *File) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	data := m.data
	m.data = nil
	if m.mapped {
		m.mapped = false
		return syscall.Munmap(data)
	}
	return nil
}
