package hestia

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is Hestia's validated JSON configuration (CONCEPT.md section 13).
// This first milestone only needs listen address and data directory; TLS,
// model paths, knowledge backends and device adapters follow with voice
// and retrieval integration.
type Config struct {
	Listen           string `json:"listen"`
	DataDir          string `json:"data_dir"`
	VoxtralModelPath string `json:"voxtral_model_path,omitempty"` // optional; voice disabled without it
	EmbedModelPath   string `json:"embed_model_path,omitempty"`   // optional; document search disabled without it
}

// DefaultConfig returns sane loopback-only defaults (CONCEPT.md section 12:
// "Der Prozess bindet standardmäßig an Loopback.").
func DefaultConfig() Config {
	return Config{Listen: "127.0.0.1:8787", DataDir: "./hestia-data"}
}

// LoadConfig reads and validates a JSON config file. Unknown fields are an
// error, matching CONCEPT.md section 13's "unbekannte Felder führen zu
// einer verständlichen Fehlermeldung".
func LoadConfig(path string) (Config, error) {
	cfg := DefaultConfig()
	f, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("opening config: %w", err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("parsing config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if c.Listen == "" {
		return fmt.Errorf("config: listen must not be empty")
	}
	if c.DataDir == "" {
		return fmt.Errorf("config: data_dir must not be empty")
	}
	return nil
}
