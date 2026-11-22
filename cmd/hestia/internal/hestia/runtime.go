package hestia

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
)

// Runtime owns the assistant and the models and storage it opens.
type Runtime struct {
	App     *App
	closers []func() error
}

func OpenRuntime(ctx context.Context, cfg Config, token string, logw io.Writer) (_ *Runtime, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	wiki, web, err := openSearch(cfg.Research)
	if err != nil {
		return nil, err
	}
	device, inventory, err := openDevices(ctx, cfg.HomeAssistant, token)
	if err != nil {
		return nil, err
	}
	if cfg.PersistHistory || cfg.EmbedModelPath != "" {
		if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
			return nil, err
		}
	}
	journalPath := ""
	if cfg.PersistHistory {
		journalPath = filepath.Join(cfg.DataDir, "journal.jsonl")
	}
	journal, err := OpenJournal(journalPath)
	if err != nil {
		return nil, err
	}
	runtime := &Runtime{closers: []func() error{journal.Close}}
	defer func() {
		if err != nil {
			_ = runtime.Close()
		}
	}()
	voice, err := OpenConfiguredSpeechHost(ctx, cfg.EffectiveSpeech(), logw)
	if err != nil {
		return nil, err
	}
	if voice != nil {
		runtime.closers = append(runtime.closers, voice.Close)
	}
	var knowledge *KnowledgeBase
	if cfg.EmbedModelPath != "" {
		knowledge, err = OpenKnowledgeBase(cfg.EmbedModelPath, filepath.Join(cfg.DataDir, "knowledge.db"))
		if err != nil {
			return nil, err
		}
		runtime.closers = append(runtime.closers, knowledge.Close)
	}
	app := NewApp(journal, device, inventory, voice, knowledge)
	app.ListenWord, app.Wikipedia, app.WebSearch = cfg.ListenWord, wiki, web
	if cfg.ChatModelPath != "" {
		chat, err := OpenAssistantChat(cfg.ChatModelPath)
		if err != nil {
			return nil, err
		}
		runtime.closers = append(runtime.closers, chat.Close)
		app.Chat = chat
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	runtime.App = app
	return runtime, nil
}

func (r *Runtime) Close() error {
	var errs []error
	for i := len(r.closers) - 1; i >= 0; i-- {
		errs = append(errs, r.closers[i]())
	}
	r.closers = nil
	return errors.Join(errs...)
}

func openSearch(cfg ResearchConfig) (wiki, web SearchProvider, err error) {
	if cfg.Wikipedia {
		wiki, err = NewWikipediaSearch(cfg.WikiLanguage)
		if err != nil {
			return nil, nil, err
		}
	}
	if cfg.WebEndpoint != "" {
		web, err = NewWebSearch(cfg.WebEndpoint)
		if err != nil {
			return nil, nil, err
		}
	}
	return wiki, web, nil
}

func openDevices(ctx context.Context, cfg HomeConfig, token string) (adapters.Adapter, Inventory, error) {
	var adapter adapters.Adapter
	entities := cfg.Lights
	if cfg.Endpoint == "" {
		entities = []adapters.Entity{
			{ID: "light_living_floor", Name: "Stehlampe", Room: "Wohnzimmer", Kind: "light"},
			{ID: "light_living_ceiling", Name: "Deckenlampe", Room: "Wohnzimmer", Kind: "light"},
			{ID: "light_kitchen", Name: "Lampe", Room: "Küche", Kind: "light"},
		}
		adapter = adapters.NewSimulator(entities...)
	} else {
		var err error
		adapter, err = adapters.NewHomeAssistant(cfg.Endpoint, token, entities)
		if err != nil {
			return nil, Inventory{}, err
		}
		entities, err = adapter.Inventory(ctx)
		if err != nil {
			return nil, Inventory{}, err
		}
	}
	devices := make([]Device, 0, len(entities))
	for _, e := range entities {
		devices = append(devices, Device{ID: e.ID, Name: e.Name, Room: e.Room, Kind: e.Kind})
	}
	return adapter, Inventory{Devices: devices}, nil
}
