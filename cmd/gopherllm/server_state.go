package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	modelStatePathEnv        = "GOPHERLLM_MODEL_STATE_PATH"
	legacyServerStatePathEnv = "GOPHERLLM_SERVER_STATE_PATH"
)

type modelState struct {
	ModelPath string `json:"model_path"`
}

// lastModelStatePath returns the per-user state file shared by every CLI mode.
// The old server-specific environment override remains accepted so existing
// deployments and tests keep their isolated state location.
func lastModelStatePath() (string, error) {
	if path := strings.TrimSpace(os.Getenv(modelStatePathEnv)); path != "" {
		return path, nil
	}
	if path := strings.TrimSpace(os.Getenv(legacyServerStatePathEnv)); path != "" {
		return path, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("find user configuration directory: %w", err)
	}
	return filepath.Join(dir, "gopherllm", "last-model.json"), nil
}

// loadLastModel returns a usable saved model path. A missing state file is the
// normal first-run case; a deleted model is treated the same way so the server
// can still start and expose its model picker.
func loadLastModel() (string, error) {
	statePath, err := lastModelStatePath()
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(statePath)
	if errors.Is(err, os.ErrNotExist) {
		// Before model selection became a CLI-wide default, server sessions
		// used last-server-model.json. Read it once as a migration fallback;
		// the next successful load writes the new generalized state file.
		if strings.TrimSpace(os.Getenv(modelStatePathEnv)) == "" && strings.TrimSpace(os.Getenv(legacyServerStatePathEnv)) == "" {
			legacyPath := filepath.Join(filepath.Dir(statePath), "last-server-model.json")
			data, err = os.ReadFile(legacyPath)
			if errors.Is(err, os.ErrNotExist) {
				return "", nil
			}
			if err != nil {
				return "", fmt.Errorf("read %s: %w", legacyPath, err)
			}
			statePath = legacyPath
		} else {
			return "", nil
		}
	}
	if err != nil {
		return "", fmt.Errorf("read %s: %w", statePath, err)
	}
	var state modelState
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

// recordLastModel persists every successful local model load. It deliberately
// only warns on failure: the model is ready and a persistence problem should
// not make a successful CLI command or HTTP request fail.
func recordLastModel(modelPath string) {
	if err := saveLastModel(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not remember model: %v\n", err)
	}
}

func saveLastModel(modelPath string) error {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return errors.New("model path is empty")
	}
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return fmt.Errorf("make model path absolute: %w", err)
	}
	statePath, err := lastModelStatePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(statePath), 0o700); err != nil {
		return fmt.Errorf("create state directory: %w", err)
	}
	data, err := json.Marshal(modelState{ModelPath: absPath})
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
