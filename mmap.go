//go:build !windows

package main

import (
	"os"
	"syscall"
)

type MmapFile struct {
	data []byte
	mmap bool
}

// OpenMmap uses the platform mmap syscall without CGO. If mmap is unavailable
// for a specific file, it falls back to os.ReadFile while preserving the same
// immutable byte-slice API.
func OpenMmap(path string) (*MmapFile, error) {
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
		return &MmapFile{}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err == nil {
		return &MmapFile{data: data, mmap: true}, nil
	}

	data, err = os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &MmapFile{data: data}, nil
}

func (m *MmapFile) Bytes() []byte { return m.data }
func (m *MmapFile) Len() int      { return len(m.data) }
func (m *MmapFile) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	data := m.data
	m.data = nil
	if m.mmap {
		m.mmap = false
		return syscall.Munmap(data)
	}
	return nil
}
