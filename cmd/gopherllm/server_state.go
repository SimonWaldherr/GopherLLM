package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const serverStatePathEnv = "GOPHERLLM_SERVER_STATE_PATH"

type serverState struct {
	ModelPath string `json:"model_path"`
}

// lastServerModelStatePath returns the per-user state file used only by the
// CLI server. The environment override makes isolated deployments and tests
// possible without changing the user's normal configuration.
func lastServerModelStatePath() (string, error) {
	if path := strings.TrimSpace(os.Getenv(serverStatePathEnv)); path != "" {
		return path, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("find user configuration directory: %w", err)
	}
	return filepath.Join(dir, "gopherllm", "last-server-model.json"), nil
}

// loadLastServerModel returns a usable saved model path. A missing state file
// is the normal first-run case; a deleted model is treated the same way so a
// server can still start and expose its model picker.
func loadLastServerModel() (string, error) {
	statePath, err := lastServerModelStatePath()
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(statePath)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("read %s: %w", statePath, err)
	}
	var state serverState
	if err := json.Unmarshal(data, &state); err != nil {
		return "", fmt.Errorf("parse %s: %w", statePath, err)
	}
	path := strings.TrimSpace(state.ModelPath)
	if path == "" {
		return "", nil
	}
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("inspect saved model %s: %w", path, err)
	}
	if !info.Mode().IsRegular() {
		return "", nil
	}
	return path, nil
}

// recordLastServerModel persists a successful local server load. It deliberately
// only warns on failure: the model is ready and a persistence problem should
// not make a successful HTTP request fail.
func recordLastServerModel(modelPath string) {
	if err := saveLastServerModel(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not remember server model: %v\n", err)
	}
}

func saveLastServerModel(modelPath string) error {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return errors.New("model path is empty")
	}
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return fmt.Errorf("make model path absolute: %w", err)
	}
	statePath, err := lastServerModelStatePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(statePath), 0o700); err != nil {
		return fmt.Errorf("create state directory: %w", err)
	}
	data, err := json.Marshal(serverState{ModelPath: absPath})
	if err != nil {
		return fmt.Errorf("encode server state: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(statePath), ".last-server-model-*")
	if err != nil {
		return fmt.Errorf("create temporary state file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return fmt.Errorf("set state file permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write server state: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close server state: %w", err)
	}
	if err := os.Rename(tmpPath, statePath); err != nil {
		return fmt.Errorf("replace server state: %w", err)
	}
	return nil
}
