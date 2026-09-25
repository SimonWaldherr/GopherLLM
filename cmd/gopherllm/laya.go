package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	publichf "github.com/SimonWaldherr/GopherLLM/huggingface"
	"github.com/SimonWaldherr/GopherLLM/server"
)

func runLayaCommand(ctx context.Context, cfg cliConfig) error {
	if cfg.layaModel == "" {
		return fmt.Errorf("Laya classification, CSV and download options require --laya-model")
	}
	if cfg.deploymentMode == server.DeploymentBrowser {
		return fmt.Errorf("native Laya inference is unavailable in browser deployment")
	}
	if cfg.modelSelector != nil || cfg.prompt != "" || cfg.repl || cfg.embed || cfg.bench || cfg.kernelBench || cfg.compress || cfg.autoTune || cfg.imagePath != "" || cfg.mmprojPathSet || cfg.chatUI || cfg.inspect || cfg.listMetadata || cfg.listTensors || cfg.analyze || cfg.findToken != "" || cfg.tokenNeighbors != "" || cfg.metalExplicit || cfg.prepareQuant || cfg.outOfCore {
		return fmt.Errorf("--laya-model is a dedicated classification mode; remove chat, generation, image, embedding, compression or benchmark options")
	}
	modes := 0
	for _, enabled := range []bool{cfg.classifyPath != "", cfg.classifyCSV != "", cfg.serveAddr != "", cfg.layaDownloadOnly} {
		if enabled {
			modes++
		}
	}
	if modes != 1 {
		return fmt.Errorf("--laya-model requires exactly one of --classify, --classify-csv, --serve, or --laya-download-only")
	}
	var csvOpts gopherllm.DecisionCSVOptions
	if cfg.hasCSVFlags() {
		if cfg.classifyCSV == "" {
			return fmt.Errorf("CSV options require --classify-csv")
		}
		var err error
		csvOpts, err = cfg.decisionCSVOptions()
		if err != nil {
			return err
		}
	}
	if cfg.threadsSet {
		gopherllm.SetNumThreads(cfg.threads)
		runtime.GOMAXPROCS(cfg.threads)
	}
	if cfg.timeout > 0 && cfg.serveAddr == "" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}
	dir := cfg.layaModel
	if strings.HasPrefix(dir, "hf:") {
		var err error
		dir, err = publichf.DownloadLaya(ctx, dir, cfg.layaSubfolder, os.Stderr, publichf.Options{Offline: cfg.hfOffline})
		if err != nil {
			return err
		}
	} else if cfg.layaSubfolder != "" {
		if !filepath.IsLocal(cfg.layaSubfolder) {
			return fmt.Errorf("--laya-subfolder must be relative to the model directory")
		}
		dir = filepath.Join(dir, cfg.layaSubfolder)
	}
	if cfg.layaDownloadOnly {
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
			return err
		}
		fmt.Fprintln(os.Stdout, dir)
		return nil
	}
	model, err := gopherllm.OpenLaya(ctx, dir)
	if err != nil {
		return err
	}
	defer model.Close()
	if cfg.classifyCSV != "" {
		return runLayaCSV(model, cfg, csvOpts, ctx)
	}
	if cfg.serveAddr != "" {
		id := strings.TrimPrefix(cfg.layaModel, "hf:")
		if !strings.HasPrefix(cfg.layaModel, "hf:") {
			id = filepath.Base(filepath.Clean(cfg.layaModel))
		}
		if cfg.layaSubfolder != "" {
			id += "/" + cfg.layaSubfolder
		}
		return server.Serve(nil, server.ServeOptions{Context: ctx, Addr: cfg.serveAddr, DecisionModel: model, DecisionModelID: id, DeploymentMode: cfg.deploymentMode, AdminToken: cfg.adminToken, RequestTimeout: cfg.requestTimeout, MaxConcurrentConnections: cfg.maxConn, LogWriter: os.Stderr})
	}
	var in io.Reader = os.Stdin
	if cfg.classifyPath != "-" {
		f, e := os.Open(cfg.classifyPath)
		if e != nil {
			return e
		}
		defer f.Close()
		in = f
	}
	data, err := io.ReadAll(io.LimitReader(in, (2<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 2<<20 {
		return fmt.Errorf("classification request exceeds 2 MiB")
	}
	req, err := gopherllm.DecodeDecisionRequest(bytes.NewReader(data))
	if err != nil {
		return err
	}
	result, err := model.Predict(ctx, req)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}
