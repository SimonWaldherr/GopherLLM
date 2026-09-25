package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func (cfg cliConfig) decisionCSVOptions() (gopherllm.DecisionCSVOptions, error) {
	typ := cfg.csvType
	if typ == "" {
		typ = "noul"
		if cfg.csvCriteria != "" {
			typ = "choice"
		}
	}
	opts := gopherllm.DecisionCSVOptions{
		Question:    gopherllm.DecisionQuestion{Type: typ, Instructions: cfg.csvInstruction},
		InputColumn: cfg.csvColumn, ResultColumn: cfg.csvResultColumn, Delimiter: cfg.csvDelimiter,
		NoHeader: cfg.csvNoHeader, ResultJSON: cfg.csvResultJSON,
	}
	if cfg.csvCriteria != "" {
		if !json.Valid([]byte(cfg.csvCriteria)) {
			return opts, fmt.Errorf("--criteria must be a JSON object or list")
		}
		opts.Question.Criteria = json.RawMessage(cfg.csvCriteria)
	}
	if opts.Delimiter == `\t` {
		opts.Delimiter = "\t"
	}
	return opts, opts.Validate()
}
func (cfg cliConfig) hasCSVFlags() bool {
	return cfg.classifyCSV != "" || cfg.csvInstruction != "" || cfg.csvType != "" || cfg.csvCriteria != "" || cfg.csvColumn != "" || cfg.csvResultColumn != "" || cfg.csvDelimiter != "" || cfg.csvNoHeader || cfg.csvResultJSON
}
func runLayaCSV(model *gopherllm.LayaModel, cfg cliConfig, opts gopherllm.DecisionCSVOptions, ctx context.Context) error {
	in := os.Stdin
	if cfg.classifyCSV != "-" {
		f, err := os.Open(cfg.classifyCSV)
		if err != nil {
			return err
		}
		defer f.Close()
		in = f
	}
	_, err := model.ClassifyCSV(ctx, in, os.Stdout, opts)
	return err
}
