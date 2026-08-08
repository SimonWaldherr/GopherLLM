//go:build unix

package mmapfile

import (
	"os"
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
		return &File{data: data, mapped: true}, nil
	}

	data, err = os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &File{data: data}, nil
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
