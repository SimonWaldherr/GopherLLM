package engine

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// readModelJSON reads one bounded metadata document. Check before opening so
// ordinary paths to pipes cannot block, then check the descriptor and actual
// bytes read as the file may have changed since Stat.
func readModelJSON(path string, maxBytes int64, value any) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > maxBytes {
		return fmt.Errorf("metadata must be a regular file no larger than %d bytes: %s", maxBytes, path)
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err = f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > maxBytes {
		return fmt.Errorf("metadata must be a regular file no larger than %d bytes: %s", maxBytes, path)
	}
	data, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	if int64(len(data)) > maxBytes {
		return fmt.Errorf("metadata exceeds %d bytes: %s", maxBytes, path)
	}
	if err := json.Unmarshal(data, value); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
