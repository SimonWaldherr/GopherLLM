package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	g "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/huggingface"
)

// Only catalog IDs may be supplied by the browser; paths and Hub references
// are registered at startup, never interpreted from HTTP input.
type modelOption struct {
	ID        string `json:"id"`
	Kind      string `json:"kind"`
	Name      string `json:"name"`
	Source    string `json:"source"`
	Subfolder string `json:"subfolder,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}
type loadedModel struct {
	predictor g.DecisionPredictor
	generate  generator
	close     func() error
}
type modelLoader func(context.Context, modelOption) (loadedModel, error)
type modelStatus struct {
	Laya             bool          `json:"laya"`
	Ministral        bool          `json:"ministral"`
	LayaID           string        `json:"laya_id"`
	MinistralID      string        `json:"ministral_id"`
	LayaVersion      uint64        `json:"laya_version"`
	MinistralVersion uint64        `json:"ministral_version"`
	Options          []modelOption `json:"options"`
}

func option(kind, name, source, subfolder string) modelOption {
	sum := sha256.Sum256([]byte(kind + "\x00" + source + "\x00" + subfolder))
	return modelOption{ID: fmt.Sprintf("%x", sum[:12]), Kind: kind, Name: name, Source: source, Subfolder: subfolder}
}
func modelCatalog(root string) ([]modelOption, error) {
	options := []modelOption{
		option("laya", "Laya · Mehrsprachig (auch Deutsch)", "hf:convaiinnovations/laya", "multilingual"),
		option("laya", "Laya · Englisch", "hf:convaiinnovations/laya", ""),
		option("laya", "Laya · Typed Decisions", "hf:convaiinnovations/laya", "typed-decisions"),
	}
	if root == "" {
		return options, nil
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	entries, err := g.DiscoverModels(abs, os.Stderr)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if !entry.IsSupported || entry.IsProjector || entry.IsEmbedding {
			continue
		}
		rel, _ := filepath.Rel(abs, entry.Path)
		name := rel
		if entry.ModelName != "" {
			name = entry.ModelName + " · " + rel
		}
		item := option("ministral", name, entry.Path, "")
		item.SizeBytes = entry.SizeBytes
		options = append(options, item)
	}
	err = filepath.WalkDir(abs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || d.Name() != "rl_agent_config.json" {
			return nil
		}
		dir := filepath.Dir(path)
		for _, file := range []string{"encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"} {
			if info, err := os.Stat(filepath.Join(dir, file)); err != nil || info.IsDir() {
				return nil
			}
		}
		rel, _ := filepath.Rel(abs, dir)
		options = append(options, option("laya", "Laya · "+rel, dir, ""))
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.SliceStable(options, func(i, j int) bool {
		if options[i].Kind != options[j].Kind {
			return options[i].Kind < options[j].Kind
		}
		return options[i].Name < options[j].Name
	})
	return options, nil
}
func nativeLoader(offline bool) modelLoader {
	return func(ctx context.Context, selected modelOption) (loadedModel, error) {
		if selected.Kind == "laya" {
			dir := selected.Source
			if strings.HasPrefix(dir, "hf:") {
				var err error
				dir, err = huggingface.DownloadLaya(ctx, strings.TrimPrefix(dir, "hf:"), selected.Subfolder, os.Stderr, huggingface.Options{Offline: offline})
				if err != nil {
					return loadedModel{}, err
				}
			}
			model, err := g.OpenLaya(ctx, dir)
			if err != nil {
				return loadedModel{}, err
			}
			return loadedModel{predictor: model, close: model.Close}, nil
		}
		model, err := g.Open(ctx, selected.Source, g.WithLogWriter(os.Stderr))
		if err != nil {
			return loadedModel{}, err
		}
		generate := func(ctx context.Context, prompt, system string) (string, error) {
			result, err := model.Generate(ctx, prompt, g.WithSystemPrompt(system), g.WithMaxTokens(600), g.WithTemperature(.2))
			if err != nil {
				return "", err
			}
			if result.FinishReason == "length" {
				return "", fmt.Errorf("Antwort wurde abgeschnitten; bitte kürzere Eingabe verwenden")
			}
			return result.Text, nil
		}
		return loadedModel{generate: generate, close: model.Close}, nil
	}
}
func (a *app) snapshot() modelStatus {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return modelStatus{Laya: a.predictor != nil, Ministral: a.generate != nil, LayaID: a.layaID, MinistralID: a.ministralID, LayaVersion: a.layaVersion, MinistralVersion: a.ministralVersion, Options: a.options}
}

// Call while holding gate, which serializes loading, inference and closing.
func (a *app) switchModel(ctx context.Context, kind, id string) error {
	if kind != "laya" && kind != "ministral" {
		return fmt.Errorf("Unbekannte Modellart")
	}
	var chosen modelOption
	if id != "" {
		for _, entry := range a.options {
			if entry.ID == id && entry.Kind == kind {
				chosen = entry
				break
			}
		}
		if chosen.ID == "" {
			return fmt.Errorf("Modell ist nicht im angebotenen Katalog")
		}
	}
	a.mu.RLock()
	same := (kind == "laya" && a.layaID == id) || (kind == "ministral" && a.ministralID == id)
	a.mu.RUnlock()
	if same {
		return nil
	}
	var next loadedModel
	if id != "" {
		var err error
		next, err = a.load(ctx, chosen)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			if next.close != nil {
				_ = next.close()
			}
			return err
		}
		if (kind == "laya" && next.predictor == nil) || (kind == "ministral" && next.generate == nil) {
			if next.close != nil {
				_ = next.close()
			}
			return fmt.Errorf("Modell hat nicht die benötigte Funktion")
		}
	}
	a.mu.Lock()
	var closeOld func() error
	if kind == "laya" {
		closeOld = a.closeLaya
		a.predictor = next.predictor
		a.closeLaya = next.close
		a.layaID = id
		a.layaVersion++
	} else {
		closeOld = a.closeMinistral
		a.generate = next.generate
		a.closeMinistral = next.close
		a.ministralID = id
		a.ministralVersion++
	}
	a.mu.Unlock()
	if closeOld != nil {
		_ = closeOld()
	}
	return nil
}
func (a *app) selectModel(w http.ResponseWriter, r *http.Request) {
	// Model selections are small; reject large requests before allocating them.
	r.Body = http.MaxBytesReader(w, r.Body, 4096)
	var in struct {
		Kind string `json:"kind"`
		ID   string `json:"id"`
	}
	if err := decode(w, r, &in); err != nil {
		fail(w, err, 400)
		return
	}
	valid := in.ID == ""
	for _, entry := range a.options {
		if entry.ID == in.ID && entry.Kind == in.Kind {
			valid = true
		}
	}
	if !valid || (in.Kind != "laya" && in.Kind != "ministral") {
		http.Error(w, "Bitte ein angebotenes Modell auswählen", 422)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()
	if err := a.acquire(ctx); err != nil {
		fail(w, err, 408)
		return
	}
	defer func() { <-a.gate }()
	if err := a.switchModel(ctx, in.Kind, in.ID); err != nil {
		fail(w, err, 503)
		return
	}
	respond(w, a.snapshot())
}
func (a *app) closeModels() {
	_ = a.acquire(context.Background())
	defer func() { <-a.gate }()
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closeLaya != nil {
		_ = a.closeLaya()
	}
	if a.closeMinistral != nil {
		_ = a.closeMinistral()
	}
}

// Startup selections are also listed in the browser even outside -model-dir.
func (a *app) register(item modelOption) {
	for _, existing := range a.options {
		if existing.ID == item.ID {
			return
		}
	}
	a.options = append(a.options, item)
}
