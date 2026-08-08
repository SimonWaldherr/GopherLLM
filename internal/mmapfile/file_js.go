//go:build js || wasip1

package mmapfile

import "fmt"

// File is the js/wasip1 variant of the Unix/Windows types: the same
// immutable byte-slice API shape, but there is no OS mapping and no
// filesystem to fall back to (a browser tab has neither) — Open always
// fails. Model loading under GOOS=js goes through the byte-based
// RunnerFromGGUFBytes path instead, which never calls into this package;
// this file exists solely so the package type-checks on these platforms.
type File struct{}

// Open unconditionally reports that memory-mapped files are unsupported.
// It does not attempt to touch the filesystem: under GOOS=js a browser tab
// has none, and failing fast avoids a confusing not-exist error standing in
// for the real reason.
func Open(path string) (*File, error) {
	return nil, fmt.Errorf("mmapfile: memory-mapped files are not supported in the browser; load the model from bytes via RunnerFromGGUFBytes instead")
}

func (m *File) Bytes() []byte  { return nil }
func (m *File) Len() int       { return 0 }
func (m *File) IsMapped() bool { return false }
func (m *File) Close() error   { return nil }
