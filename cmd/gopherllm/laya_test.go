package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
)

func TestParseLayaCLI(t *testing.T) {
	cfg, e := parseCLI([]string{"--laya-model", "hf:convaiinnovations/laya", "--laya-subfolder", "multilingual", "--classify", "-", "--hf-offline"})
	if e != nil {
		t.Fatal(e)
	}
	if cfg.layaModel != "hf:convaiinnovations/laya" || cfg.layaSubfolder != "multilingual" || cfg.classifyPath != "-" || !cfg.hfOffline {
		t.Fatalf("bad classifier flags")
	}
	for _, args := range [][]string{{"--laya-model"}, {"--classify"}, {"--laya-subfolder", ""}} {
		if _, e := parseCLI(args); e == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestLayaModeValidation(t *testing.T) {
	for _, args := range [][]string{
		{"--classify", "-"},
		{"--laya-model", "missing"},
		{"--laya-model", "missing", "--classify", "-", "--serve"},
		{"--laya-model", "missing", "--serve", "--chat"},
		{"--laya-model", "missing", "--serve", "--deployment", "browser"},
	} {
		cfg, e := parseCLI(args)
		if e != nil {
			continue
		}
		if e = runLayaCommand(context.Background(), cfg); e == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestLayaConfigRoundTrip(t *testing.T) {
	cfg, e := parseCLI([]string{"--laya-model", "hf:owner/fine-tune", "--laya-subfolder", "multilingual"})
	if e != nil {
		t.Fatal(e)
	}
	var out bytes.Buffer
	if e = writeEffectiveConfig(&out, cfg); e != nil {
		t.Fatal(e)
	}
	var raw fileConfig
	if e = json.Unmarshal(out.Bytes(), &raw); e != nil {
		t.Fatal(e)
	}
	var again cliConfig
	if e = applyFileConfig(&again, raw); e != nil {
		t.Fatal(e)
	}
	if again.layaModel != cfg.layaModel || again.layaSubfolder != cfg.layaSubfolder {
		t.Fatal("classifier config lost on round trip")
	}
}

func TestParseCSVCLI(t *testing.T) {
	cfg, e := parseCLI([]string{"--laya-model", "local", "--classify-csv", "-", "--instruction", "Route", "--criteria", `["billing","tech"]`, "--csv-column", "text", "--result-column", "category", "--csv-delimiter", `;`})
	if e != nil {
		t.Fatal(e)
	}
	opts, e := cfg.decisionCSVOptions()
	if e != nil {
		t.Fatal(e)
	}
	if cfg.classifyCSV != "-" || opts.Question.Type != "choice" || opts.InputColumn != "text" || opts.ResultColumn != "category" || opts.Delimiter != ";" {
		t.Fatal(cfg, opts)
	}
	cfg, e = parseCLI([]string{"--laya-model", "local", "--classify-csv", "-", "--instruction", "Yes?", "--csv-no-header", "--csv-result-json", "--csv-delimiter", `\t`})
	if e != nil {
		t.Fatal(e)
	}
	opts, e = cfg.decisionCSVOptions()
	if e != nil || opts.Question.Type != "noul" || !opts.NoHeader || !opts.ResultJSON || opts.Delimiter != "\t" {
		t.Fatal(opts, e)
	}
	for _, args := range [][]string{
		{"--laya-model", "missing", "--classify-csv", "-"},
		{"--laya-model", "missing", "--instruction", "Yes?", "--serve"},
		{"--laya-model", "missing", "--classify-csv", "-", "--instruction", "Route", "--criteria", "bad JSON"},
		{"--laya-model", "missing", "--classify-csv", "-", "--instruction", "Yes?", "--classify", "request.json"},
	} {
		cfg, e := parseCLI(args)
		if e == nil {
			e = runLayaCommand(context.Background(), cfg)
		}
		if e == nil {
			t.Fatalf("accepted invalid flags %v", args)
		}
	}
}
