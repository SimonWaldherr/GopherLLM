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

func open(path string, requireMapping bool) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if size := st.Size(); size == 0 || size > int64(^uint(0)>>1) {
		return checkMappingSize(path, size, requireMapping)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err == nil {
		return &File{data: data, mapped: true}, nil
	}
	if requireMapping {
		return nil, &os.PathError{Op: "mmap", Path: path, Err: err}
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
