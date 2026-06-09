//go:build windows

package main

import "os"

type MmapFile struct {
	data []byte
}

func OpenMmap(path string) (*MmapFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &MmapFile{data: data}, nil
}

func (m *MmapFile) Bytes() []byte { return m.data }
func (m *MmapFile) Len() int      { return len(m.data) }
func (m *MmapFile) Close() error {
	m.data = nil
	return nil
}
