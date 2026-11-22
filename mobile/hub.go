package mobile

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/huggingface"
)

type hubOptions struct {
	CacheDir    string `json:"cache_dir"`
	Token       string `json:"token,omitempty"`
	Offline     bool   `json:"offline,omitempty"`
	PipelineTag string `json:"pipeline_tag,omitempty"`
	Format      string `json:"format,omitempty"`
	Limit       int    `json:"limit,omitempty"`
}

func parseHubOptions(raw string) (hubOptions, error) {
	var opts hubOptions
	if err := decodeStrictJSON(raw, &opts); err != nil {
		return opts, fmt.Errorf("invalid Hub options: %w", err)
	}
	if !filepath.IsAbs(opts.CacheDir) || strings.ContainsRune(opts.CacheDir, 0) {
		return opts, fmt.Errorf("Hub cache_dir must be an absolute local path")
	}
	return opts, nil
}

func (o hubOptions) hubOptions() huggingface.Options {
	// Apps use their own cache and credential, never the process token or a
	// mirror selected by the environment.
	return huggingface.Options{CacheDir: o.CacheDir, Endpoint: "https://huggingface.co", Token: &o.Token, Offline: o.Offline}
}

func HubSearch(ctx context.Context, query, optionsJSON string) (string, error) {
	opts, err := parseHubOptions(optionsJSON)
	if err != nil {
		return "", err
	}
	results, err := huggingface.SearchModels(ctx, huggingface.SearchOptions{
		Query: query, Limit: opts.Limit, PipelineTag: opts.PipelineTag, Format: opts.Format,
	}, opts.hubOptions())
	if err != nil {
		return "", err
	}
	return marshal(results), nil
}

func HubInspect(ctx context.Context, reference, optionsJSON string) (string, error) {
	opts, err := parseHubOptions(optionsJSON)
	if err != nil {
		return "", err
	}
	manifest, err := huggingface.Inspect(ctx, reference, opts.hubOptions())
	if err != nil {
		return "", err
	}
	return marshal(manifest), nil
}

// HubDownload returns paths in the private cache. Empty filesJSON resolves a
// GGUF including all its shards. A file list selects expected model artifacts.
func HubDownload(ctx context.Context, reference, filesJSON, optionsJSON string, progress huggingface.ProgressFunc) (string, error) {
	config, err := parseHubOptions(optionsJSON)
	if err != nil {
		return "", err
	}
	opts := config.hubOptions()
	opts.OnProgress = progress
	var paths []string
	if filesJSON == "" {
		paths, err = huggingface.ResolveFiles(ctx, reference, io.Discard, opts)
	} else {
		var files []string
		if err := decodeStrictJSON(filesJSON, &files); err != nil {
			return "", fmt.Errorf("invalid Hub file selection: %w", err)
		}
		if len(files) == 0 || len(files) > 128 {
			return "", fmt.Errorf("select between 1 and 128 model files")
		}
		for _, file := range files {
			if !hubModelArtifact(file) {
				return "", fmt.Errorf("unsupported model artifact %q", file)
			}
		}
		paths, err = huggingface.DownloadFiles(ctx, reference, files, io.Discard, opts)
	}
	if err != nil {
		return "", err
	}
	return marshal(paths), nil
}

func hubModelArtifact(file string) bool {
	name := strings.ToLower(filepath.Base(file))
	switch filepath.Ext(name) {
	case ".gguf", ".fermion", ".safetensors", ".json":
		return true
	case ".bin":
		return strings.HasPrefix(name, "ggml-")
	}
	return false
}
