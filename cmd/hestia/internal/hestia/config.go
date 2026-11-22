package hestia

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/netip"
	"os"
	"strconv"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
	"github.com/SimonWaldherr/GopherLLM/speech"
)

// Config is Hestia's validated JSON configuration.
type Config struct {
	Listen           string         `json:"listen"`
	DataDir          string         `json:"data_dir"`
	ListenWord       string         `json:"listen_word"`
	PersistHistory   bool           `json:"persist_history"`
	Speech           SpeechConfig   `json:"speech"`
	Research         ResearchConfig `json:"research"`
	HomeAssistant    HomeConfig     `json:"home_assistant"`
	VoxtralModelPath string         `json:"voxtral_model_path,omitempty"`
	EmbedModelPath   string         `json:"embed_model_path,omitempty"`
	ChatModelPath    string         `json:"chat_model_path,omitempty"`
}

type ResearchConfig struct {
	Wikipedia    bool   `json:"wikipedia"`
	WikiLanguage string `json:"wiki_language"`
	WebEndpoint  string `json:"web_endpoint"`
}

type HomeConfig struct {
	Endpoint string            `json:"endpoint"`
	Lights   []adapters.Entity `json:"lights"`
}

// DefaultConfig returns sane loopback-only defaults (CONCEPT.md section 12:
// "Der Prozess bindet standardmäßig an Loopback.").
func DefaultConfig() Config {
	return Config{Listen: "127.0.0.1:8787", DataDir: "./hestia-data", ListenWord: "Computer"}
}

// LoadConfig reads and validates a JSON config file. Unknown fields are an
// error, matching CONCEPT.md section 13's "unbekannte Felder führen zu
// einer verständlichen Fehlermeldung".
func LoadConfig(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("opening config: %w", err)
	}
	defer f.Close()
	return DecodeConfig(f)
}

func DecodeConfig(r io.Reader) (Config, error) {
	cfg := DefaultConfig()
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, fmt.Errorf("parsing config: %w", err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return Config{}, fmt.Errorf("config: expected a single JSON object")
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	host, port, err := net.SplitHostPort(c.Listen)
	portNumber, portErr := strconv.ParseUint(port, 10, 16)
	if err != nil || !loopbackHost(host) || portErr != nil || portNumber == 0 {
		return fmt.Errorf("config: listen must be a loopback address with a port")
	}
	if strings.TrimSpace(c.DataDir) == "" {
		return fmt.Errorf("config: data_dir must not be empty")
	}
	if _, err := speech.ParseWakePhrase(c.ListenWord); err != nil {
		return fmt.Errorf("config: listen_word: %w", err)
	}
	if (c.HomeAssistant.Endpoint != "") != (len(c.HomeAssistant.Lights) > 0) {
		return fmt.Errorf("config: home_assistant requires both endpoint and explicitly allowed lights")
	}
	return c.EffectiveSpeech().Validate()
}

func (c Config) EffectiveSpeech() SpeechConfig {
	if c.Speech == (SpeechConfig{}) && c.VoxtralModelPath != "" {
		return SpeechConfig{Backend: "voxtral", ModelPath: c.VoxtralModelPath}
	}
	return c.Speech
}

func loopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip, err := netip.ParseAddr(host)
	return err == nil && ip.Unmap().IsLoopback()
}
