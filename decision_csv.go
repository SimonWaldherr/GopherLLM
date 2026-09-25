package gopherllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"
)

// DecisionPredictor allows record processing to use any native decision model.
type DecisionPredictor interface {
	Predict(context.Context, DecisionRequest) (DecisionResult, error)
}

// DecisionCSVOptions describes the single question applied to every CSV record.
// With headers (the default), InputColumn names a column; without headers it is
// a one-based column number. Empty InputColumn sends the complete record as a
// JSON object (headers) or array (no headers). The result is always appended.
type DecisionCSVOptions struct {
	Question     DecisionQuestion `json:"question"`
	InputColumn  string           `json:"input_column,omitempty"`
	ResultColumn string           `json:"result_column,omitempty"`
	Delimiter    string           `json:"delimiter,omitempty"`
	NoHeader     bool             `json:"no_header,omitempty"`
	ResultJSON   bool             `json:"result_json,omitempty"`
	MaxLen       int              `json:"max_len,omitempty"`
	HeadMaxLen   int              `json:"head_max_len,omitempty"`
}

type DecisionCSVStats struct {
	Rows        int `json:"rows"`
	InputTokens int `json:"input_tokens"`
}

// Validate checks the CSV format and question without loading a model.
func (o DecisionCSVOptions) Validate() error {
	invalid := func(err error) error { return fmt.Errorf("%w: %v", ErrInvalidDecision, err) }
	if strings.TrimSpace(o.Question.Instructions) == "" {
		return invalid(fmt.Errorf("question.instructions is required"))
	}
	if _, err := prepareLayaQuestion(o.Question); err != nil {
		return invalid(err)
	}
	if _, err := o.separator(); err != nil {
		return invalid(err)
	}
	if o.NoHeader && o.InputColumn != "" {
		n, err := strconv.Atoi(o.InputColumn)
		if err != nil || n < 1 {
			return invalid(fmt.Errorf("input_column must be a one-based number without a header"))
		}
	}
	if o.MaxLen < 0 || o.HeadMaxLen < 0 {
		return invalid(fmt.Errorf("token budgets cannot be negative"))
	}
	return nil
}
func (o DecisionCSVOptions) separator() (rune, error) {
	if o.Delimiter == "" {
		return ',', nil
	}
	if !utf8.ValidString(o.Delimiter) || utf8.RuneCountInString(o.Delimiter) != 1 {
		return 0, fmt.Errorf("delimiter must be one character")
	}
	r, _ := utf8.DecodeRuneInString(o.Delimiter)
	if r == 0 || r == '"' || r == '\r' || r == '\n' || r == utf8.RuneError {
		return 0, fmt.Errorf("invalid CSV delimiter")
	}
	return r, nil
}

// ClassifyCSV is a record-at-a-time io.Reader -> io.Writer filter. It loads no
// model, buffers no whole file, preserves record order and flushes every output
// record. Quoted fields and embedded newlines follow encoding/csv semantics.
// On an error it stops immediately and returns completed-row stats; previously
// written records remain in the output. The caller owns both streams and model.
func ClassifyCSV(ctx context.Context, model DecisionPredictor, in io.Reader, out io.Writer, opts DecisionCSVOptions) (DecisionCSVStats, error) {
	var stats DecisionCSVStats
	if err := opts.Validate(); err != nil {
		return stats, err
	}
	if model == nil || in == nil || out == nil {
		return stats, fmt.Errorf("model, input and output are required")
	}
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	sep, _ := opts.separator()
	buffered := bufio.NewReader(in)
	if prefix, _ := buffered.Peek(3); bytes.Equal(prefix, []byte{0xef, 0xbb, 0xbf}) {
		_, _ = buffered.Discard(3)
	}
	reader := csv.NewReader(buffered)
	reader.Comma = sep
	writer := csv.NewWriter(out)
	writer.Comma = sep
	write := func(row []string) error {
		if err := writer.Write(row); err != nil {
			return err
		}
		writer.Flush()
		return writer.Error()
	}
	invalid := func(err error) error { return fmt.Errorf("%w: %v", ErrInvalidDecision, err) }
	var header []string
	var headerKeys [][]byte
	column := -1
	if opts.NoHeader {
		if opts.InputColumn != "" {
			n, _ := strconv.Atoi(opts.InputColumn)
			column = n - 1
		}
	} else {
		var err error
		header, err = reader.Read()
		if err == io.EOF {
			return stats, nil
		}
		if err != nil {
			return stats, invalid(fmt.Errorf("CSV header: %w", err))
		}
		header[0] = strings.TrimPrefix(header[0], "\ufeff")
		seen := map[string]bool{}
		name := opts.ResultColumn
		if name == "" {
			name = "result"
		}
		for i, h := range header {
			if h == "" || seen[h] {
				return stats, invalid(fmt.Errorf("CSV headers must be nonempty and unique (column %d)", i+1))
			}
			if h == name {
				return stats, invalid(fmt.Errorf("result column %q already exists", name))
			}
			seen[h] = true
			if h == opts.InputColumn {
				column = i
			}
		}
		if opts.InputColumn != "" && column < 0 {
			return stats, invalid(fmt.Errorf("input column %q not found", opts.InputColumn))
		}
		if column < 0 {
			headerKeys = make([][]byte, len(header))
			for i, key := range header {
				headerKeys[i], _ = json.Marshal(key)
			}
		}
		if err := write(append(append([]string(nil), header...), name)); err != nil {
			return stats, err
		}
	}
	var outputRow []string // Reuse output storage without mutating the predictor's state.
	request := DecisionRequest{Questions: map[string]DecisionQuestion{"result": opts.Question}, MaxLen: opts.MaxLen, HeadMaxLen: opts.HeadMaxLen}
	for {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		row, err := reader.Read()
		if err == io.EOF {
			return stats, nil
		}
		record := stats.Rows + 1
		if err != nil {
			return stats, invalid(fmt.Errorf("CSV data record %d: %w", record, err))
		}
		if column >= len(row) {
			return stats, invalid(fmt.Errorf("CSV data record %d: input column %d is out of range", record, column+1))
		}
		switch {
		case column >= 0:
			request.State = row[column]
		case opts.NoHeader:
			request.State = row
		default:
			// Preserve the original column order in the prompt, including numeric labels.
			var b strings.Builder
			b.WriteByte('{')
			for i, key := range headerKeys {
				if i > 0 {
					b.WriteByte(',')
				}
				v, _ := json.Marshal(row[i])
				b.Write(key)
				b.WriteByte(':')
				b.Write(v)
			}
			b.WriteByte('}')
			request.State = json.RawMessage(b.String())
		}
		result, err := model.Predict(ctx, request)
		if err != nil {
			return stats, fmt.Errorf("CSV data record %d: %w", record, err)
		}
		answer, ok := result.Answers["result"]
		if !ok {
			return stats, fmt.Errorf("CSV data record %d: model returned no result", record)
		}
		value, err := decisionCSVValue(answer, opts.ResultJSON)
		if err != nil {
			return stats, fmt.Errorf("CSV data record %d: %w", record, err)
		}
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		if cap(outputRow) < len(row)+1 {
			outputRow = make([]string, len(row)+1)
		}
		outputRow = outputRow[:len(row)+1]
		copy(outputRow, row)
		outputRow[len(row)] = value
		if err := write(outputRow); err != nil {
			return stats, fmt.Errorf("CSV data record %d output: %w", record, err)
		}
		stats.Rows++
		stats.InputTokens += result.Usage.InputTokens
	}
}
func decisionCSVValue(a DecisionAnswer, asJSON bool) (string, error) {
	if asJSON {
		b, err := json.Marshal(a)
		return string(b), err
	}
	switch a.Type {
	case "choice":
		if a.Choice != nil {
			return *a.Choice, nil
		}
	case "score":
		if a.Score != nil {
			return strconv.FormatFloat(*a.Score, 'g', -1, 64), nil
		}
	case "noul":
		if a.Noul != nil {
			return strconv.FormatFloat(*a.Noul, 'g', -1, 64), nil
		}
	}
	return "", fmt.Errorf("model returned an incomplete %q answer", a.Type)
}

func (m *LayaModel) ClassifyCSV(ctx context.Context, in io.Reader, out io.Writer, opts DecisionCSVOptions) (DecisionCSVStats, error) {
	return ClassifyCSV(ctx, m, in, out, opts)
}
