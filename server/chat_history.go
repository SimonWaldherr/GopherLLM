package server

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const maxChatHistoryBytes = 64 << 20

var errChatHistoryConflict = errors.New("chat history changed on the server")

// chatHistoryStore is an opt-in, single-workspace server store. The browser
// remains the default so a server never starts collecting conversations just
// because its chat UI was enabled. Files are gzip-compressed and replaced
// atomically, which keeps large histories small and prevents partial writes
// after a process or machine interruption.
type chatHistoryStore struct {
	path string
	mu   sync.RWMutex
}

func newChatHistoryStore(path string) *chatHistoryStore {
	return &chatHistoryStore{path: strings.TrimSpace(path)}
}

func (s *chatHistoryStore) enabled() bool { return s != nil && s.path != "" }

func (s *chatHistoryStore) read(ctx context.Context) ([]byte, string, error) {
	if !s.enabled() {
		return nil, "", os.ErrNotExist
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.readLocked(ctx)
}

func (s *chatHistoryStore) readLocked(ctx context.Context) ([]byte, string, error) {
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	f, err := os.Open(s.path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	data, err := readCompressedHistory(ctx, f)
	if err != nil {
		return nil, "", fmt.Errorf("read chat history: %w", err)
	}
	normalized, err := normalizeChatWorkspace(data)
	if err != nil {
		return nil, "", fmt.Errorf("validate chat history: %w", err)
	}
	return normalized, chatHistoryETag(normalized), nil
}

func (s *chatHistoryStore) write(ctx context.Context, data []byte, expectedETag string) (string, error) {
	if !s.enabled() {
		return "", os.ErrNotExist
	}
	normalized, err := normalizeChatWorkspace(data)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if expectedETag != "" {
		_, currentETag, readErr := s.readLocked(ctx)
		if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
			return "", readErr
		}
		if readErr == nil && strings.Trim(expectedETag, "\"") != currentETag {
			return currentETag, errChatHistoryConflict
		}
		if errors.Is(readErr, os.ErrNotExist) && strings.Trim(expectedETag, "\"") != "" {
			return "", errChatHistoryConflict
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("create chat history directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".gopherllm-chat-*.tmp")
	if err != nil {
		return "", fmt.Errorf("create chat history temporary file: %w", err)
	}
	tmpName := tmp.Name()
	removeTemp := true
	defer func() {
		_ = tmp.Close()
		if removeTemp {
			_ = os.Remove(tmpName)
		}
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return "", fmt.Errorf("protect chat history temporary file: %w", err)
	}
	zw := gzip.NewWriter(tmp)
	zw.Header.ModTime = time.Unix(0, 0)
	if _, err := zw.Write(normalized); err != nil {
		_ = zw.Close()
		return "", fmt.Errorf("compress chat history: %w", err)
	}
	if err := zw.Close(); err != nil {
		return "", fmt.Errorf("finish chat history compression: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return "", fmt.Errorf("sync chat history: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", fmt.Errorf("close chat history: %w", err)
	}
	if err := os.Rename(tmpName, s.path); err != nil {
		return "", fmt.Errorf("replace chat history: %w", err)
	}
	// Keep the final path private as well as the temporary file. Some
	// filesystems apply their default mode while materializing a renamed file,
	// so relying on the temporary file's mode can expose conversation data.
	if err := os.Chmod(s.path, 0o600); err != nil {
		return "", fmt.Errorf("protect chat history: %w", err)
	}
	removeTemp = false
	return chatHistoryETag(normalized), nil
}

func (s *chatHistoryStore) clear(ctx context.Context) error {
	if !s.enabled() {
		return os.ErrNotExist
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	err := os.Remove(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func readCompressedHistory(ctx context.Context, r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxChatHistoryBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxChatHistoryBytes {
		return nil, fmt.Errorf("chat history exceeds %d bytes", maxChatHistoryBytes)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return data, nil
	}
	gr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer gr.Close()
	decoded, err := io.ReadAll(io.LimitReader(gr, maxChatHistoryBytes+1))
	if err != nil {
		return nil, err
	}
	if len(decoded) > maxChatHistoryBytes {
		return nil, fmt.Errorf("decompressed chat history exceeds %d bytes", maxChatHistoryBytes)
	}
	return decoded, nil
}

func normalizeChatWorkspace(data []byte) ([]byte, error) {
	if len(data) == 0 || len(data) > maxChatHistoryBytes {
		return nil, fmt.Errorf("chat history must be between 1 and %d bytes", maxChatHistoryBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	var workspace map[string]json.RawMessage
	if err := dec.Decode(&workspace); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, errors.New("chat history must contain exactly one JSON object")
		}
		return nil, fmt.Errorf("invalid trailing JSON: %w", err)
	}
	var format string
	if raw := workspace["format"]; len(raw) > 0 {
		if err := json.Unmarshal(raw, &format); err != nil {
			return nil, errors.New("format must be a string")
		}
	}
	if format != "gopherllm-chat-workspace" {
		return nil, errors.New("format must be gopherllm-chat-workspace")
	}
	if raw := workspace["conversations"]; len(raw) > 0 {
		var conversations []json.RawMessage
		if err := json.Unmarshal(raw, &conversations); err != nil {
			return nil, errors.New("conversations must be an array")
		}
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, data); err != nil {
		return nil, err
	}
	return compact.Bytes(), nil
}

func chatHistoryETag(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
