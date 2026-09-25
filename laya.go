package gopherllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

// DecisionQuestion defines a choice, an ordinal score, or a noul (yes/no
// probability). Criteria is a label-to-description object for choice, a list
// of level descriptions for score, or optional false/true descriptions for noul.
// Choice also accepts a list of labels. json.RawMessage preserves option order;
// ordinary Go maps use encoding/json's deterministic key order.
type DecisionQuestion struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     any               `json:"criteria,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
}

func (q *DecisionQuestion) UnmarshalJSON(b []byte) error {
	var wire struct {
		Type         string            `json:"type"`
		Instructions *string           `json:"instructions"`
		Criteria     json.RawMessage   `json:"criteria"`
		Labels       map[string]string `json:"labels"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		return err
	}
	if wire.Instructions == nil {
		return fmt.Errorf("decision question requires instructions")
	}
	*q = DecisionQuestion{Type: wire.Type, Instructions: *wire.Instructions, Labels: wire.Labels}
	if len(wire.Criteria) > 0 {
		q.Criteria = wire.Criteria
	}
	return nil
}

// DecisionRequest is the Jev-style System One request. Model is an identifier
// for a preloaded checkpoint, not a path or a request to download weights.
// MaxLen and HeadMaxLen optionally override the checkpoint's token budgets.
type DecisionRequest struct {
	Model      string                      `json:"model,omitempty"`
	State      any                         `json:"state"`
	Questions  map[string]DecisionQuestion `json:"questions"`
	MaxLen     int                         `json:"max_len,omitempty"`
	HeadMaxLen int                         `json:"head_max_len,omitempty"`
}

func (r *DecisionRequest) UnmarshalJSON(b []byte) error {
	var w struct {
		Model      string                      `json:"model"`
		State      json.RawMessage             `json:"state"`
		Questions  map[string]DecisionQuestion `json:"questions"`
		MaxLen     int                         `json:"max_len"`
		HeadMaxLen int                         `json:"head_max_len"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	*r = DecisionRequest{Model: w.Model, State: w.State, Questions: w.Questions, MaxLen: w.MaxLen, HeadMaxLen: w.HeadMaxLen}
	return nil
}

type DecisionAnswer struct {
	Type             string             `json:"type"`
	Choice           *string            `json:"choice,omitempty"`
	Score            *float64           `json:"score,omitempty"`
	Noul             *float64           `json:"noul,omitempty"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	Legend           map[string]any     `json:"legend,omitempty"`
	Confidence       float64            `json:"confidence"`
	AnswerConfidence float64            `json:"answer_confidence"`
	Action           DecisionAction     `json:"action"`
}
type DecisionAction struct {
	ActProbability float64 `json:"act_probability"`
}
type DecisionUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}
type DecisionResult struct {
	Model   string                    `json:"model"`
	Answers map[string]DecisionAnswer `json:"answers"`
	Usage   DecisionUsage             `json:"usage"`
}

// ErrInvalidDecision identifies invalid input, including questions that exceed
// the configured prompt budget. Callers can use errors.Is for HTTP 422 handling.
var ErrInvalidDecision = errors.New("invalid decision request")

type layaQuestion struct {
	kind              int
	typ, instructions string
	keys, options     []string
	legend            map[string]any
}

func layaJSON(v any) ([]byte, error) {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	err := e.Encode(v)
	return bytes.TrimSpace(b.Bytes()), err
}
func layaText(b []byte) (string, error) {
	var text string
	if json.Unmarshal(b, &text) == nil {
		return text, nil
	}
	return layaJSONText(b)
}

// Re-render structured state/rubrics in insertion order with decoded Unicode,
// matching json.dumps(..., ensure_ascii=False) used by the reference runtime.
func layaJSONText(b []byte) (string, error) {
	b = bytes.TrimSpace(b)
	if !json.Valid(b) {
		return "", fmt.Errorf("invalid JSON value")
	}
	switch b[0] {
	case '"':
		var text string
		if err := json.Unmarshal(b, &text); err != nil {
			return "", err
		}
		encoded, err := layaJSON(text)
		return string(encoded), err
	case '{':
		keys, values, err := layaObject(b)
		if err != nil {
			return "", err
		}
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			encoded, err := layaJSON(key)
			if err != nil {
				return "", err
			}
			v, err := layaJSONText(values[key])
			if err != nil {
				return "", err
			}
			parts = append(parts, string(encoded)+": "+v)
		}
		return "{" + strings.Join(parts, ", ") + "}", nil
	case '[':
		var values []json.RawMessage
		if err := json.Unmarshal(b, &values); err != nil {
			return "", err
		}
		parts := make([]string, len(values))
		for i, v := range values {
			s, err := layaJSONText(v)
			if err != nil {
				return "", err
			}
			parts[i] = s
		}
		return "[" + strings.Join(parts, ", ") + "]", nil
	default:
		return string(b), nil
	}
}
func layaObject(b []byte) (keys []string, values map[string]json.RawMessage, err error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	tok, e := dec.Token()
	if e != nil || tok != json.Delim('{') {
		return nil, nil, fmt.Errorf("criteria must be an object")
	}
	values = map[string]json.RawMessage{}
	for dec.More() {
		tok, e = dec.Token()
		if e != nil {
			return nil, nil, e
		}
		key := tok.(string)
		if _, exists := values[key]; exists {
			return nil, nil, fmt.Errorf("duplicate criterion %q", key)
		}
		var v json.RawMessage
		if e = dec.Decode(&v); e != nil {
			return nil, nil, e
		}
		keys = append(keys, key)
		values[key] = v
	}
	if _, e = dec.Token(); e != nil {
		return nil, nil, e
	}
	return keys, values, nil
}
func prepareLayaQuestion(q DecisionQuestion) (layaQuestion, error) {
	p := layaQuestion{typ: q.Type, instructions: q.Instructions}
	raw, err := layaJSON(q.Criteria)
	if len(raw) > 256<<10 {
		return p, fmt.Errorf("criteria exceed 256 KiB")
	}
	if err != nil {
		return p, err
	}
	if len(q.Instructions) > 32768 {
		return p, fmt.Errorf("instructions too long")
	}
	if q.Type != "noul" && q.Labels != nil {
		return p, fmt.Errorf("labels only apply to noul")
	}
	switch q.Type {
	case "choice":
		p.kind = 0
		var values map[string]json.RawMessage
		if len(raw) > 0 && raw[0] == '[' {
			var list []json.RawMessage
			if err = json.Unmarshal(raw, &list); err != nil {
				return p, err
			}
			values = map[string]json.RawMessage{}
			for _, b := range list {
				var label string
				if err = json.Unmarshal(b, &label); err != nil {
					return p, fmt.Errorf("choice list labels must be strings")
				}
				if _, ok := values[label]; ok {
					return p, fmt.Errorf("duplicate choice label %q", label)
				}
				p.keys = append(p.keys, label)
				values[label] = json.RawMessage(`""`)
			}
		} else {
			p.keys, values, err = layaObject(raw)
			if err != nil {
				return p, err
			}
		}
		for _, key := range p.keys {
			desc, e := layaText(values[key])
			if e != nil {
				return p, e
			}
			if desc == "" || bytes.Equal(values[key], []byte("null")) {
				p.options = append(p.options, key)
			} else {
				p.options = append(p.options, key+": "+desc)
			}
		}
	case "score":
		p.kind = 1
		p.legend = map[string]any{}
		var values []json.RawMessage
		if err = json.Unmarshal(raw, &values); err != nil {
			return p, fmt.Errorf("score criteria must be a list")
		}
		for i, b := range values {
			if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
				return p, fmt.Errorf("score levels cannot be null")
			}
			s, e := layaText(b)
			if e != nil {
				return p, e
			}
			key := strconv.Itoa(i)
			p.keys = append(p.keys, key)
			p.options = append(p.options, "level "+key+": "+s)
			p.legend[key] = b
		}
	case "noul":
		p.kind = 2
		p.keys = []string{"false", "true"}
		values := map[string]json.RawMessage{}
		if string(raw) != "null" {
			_, values, err = layaObject(raw)
			if err != nil {
				return p, err
			}
		}
		labels := map[string]string{"false": "false", "true": "true"}
		if q.Labels != nil {
			labels = q.Labels
			if len(labels) != 2 || labels["false"] == "" || labels["true"] == "" || labels["false"] == labels["true"] {
				return p, fmt.Errorf("noul labels require distinct nonempty false and true strings")
			}
		}
		for i, key := range p.keys {
			desc := []string{"no, the statement does not hold", "yes, the statement holds"}[i]
			if b := values[key]; len(b) > 0 && string(b) != "null" {
				s, e := layaText(b)
				if e != nil {
					return p, e
				}
				if s != "" {
					desc = s
				}
			}
			p.options = append(p.options, labels[key]+": "+desc)
		}
	default:
		return p, fmt.Errorf("unknown question type %q; use choice, score or noul", q.Type)
	}
	if len(p.options) == 0 || len(p.options) > 100 {
		return p, fmt.Errorf("questions need 1 to 100 options")
	}
	return p, nil
}
func (m *LayaModel) sequence(state []uint32, q layaQuestion, maxLen, headLen int) ([]uint32, []int, error) {
	encode := func(s string) ([]uint32, error) { return m.tok.Encode(strings.ReplaceAll(s, m.tok.MaskText, " ")) }
	head, e := encode(q.typ + " question: " + q.instructions)
	if e != nil {
		return nil, nil, e
	}
	opts := make([][]uint32, len(q.options))
	used := 0
	for i, o := range q.options {
		v, err := encode(" " + o)
		if err != nil {
			return nil, nil, err
		}
		v = v[:min(48, len(v))]
		opts[i] = append([]uint32{m.tok.Mask}, v...)
		used += len(opts[i])
	}
	if headLen-used < 16 {
		per := max(4, (headLen-16)/len(opts))
		used = 0
		for i, o := range opts {
			opts[i] = o[:min(per, len(o))]
			used += len(opts[i])
		}
	}
	head = head[:min(len(head), max(8, headLen-used))]
	ids := append([]uint32{m.tok.CLS}, head...)
	ids = append(ids, m.tok.SEP)
	markers := make([]int, len(opts))
	for i, o := range opts {
		markers[i] = len(ids)
		ids = append(ids, o...)
	}
	ids = append(ids, m.tok.SEP)
	if len(ids)+1 > maxLen {
		return nil, nil, fmt.Errorf("question options exceed max_len; increase the budget or reduce the options")
	}
	room := maxLen - len(ids) - 1
	ids = append(ids, state[:min(len(state), room)]...)
	ids = append(ids, m.tok.SEP)
	return ids, markers, nil
}

// Predict evaluates all questions against State. Each question gets a complete
// bidirectional forward pass; CPU execution processes questions sequentially.
// Long states are truncated at the checkpoint token budget, as in Laya.
func (m *LayaModel) Predict(ctx context.Context, req DecisionRequest) (DecisionResult, error) {
	if m == nil || m.gate == nil {
		return DecisionResult{}, fmt.Errorf("laya: model is not initialized")
	}
	select {
	case m.gate <- struct{}{}:
		defer func() { <-m.gate }()
	case <-ctx.Done():
		return DecisionResult{}, ctx.Err()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return DecisionResult{}, fmt.Errorf("laya: model is closed")
	}
	if err := ctx.Err(); err != nil {
		return DecisionResult{}, err
	}
	invalid := func(e error) (DecisionResult, error) {
		return DecisionResult{}, fmt.Errorf("%w: %v", ErrInvalidDecision, e)
	}
	stateJSON, e := layaJSON(req.State)
	if e != nil {
		return invalid(e)
	}
	if len(stateJSON) == 0 || bytes.Equal(stateJSON, []byte("null")) {
		return invalid(fmt.Errorf("state is required"))
	}
	if len(stateJSON) > 256<<10 {
		return invalid(fmt.Errorf("state exceeds 256 KiB"))
	}
	if req.Questions == nil || len(req.Questions) > 64 {
		return invalid(fmt.Errorf("questions must be an object with at most 64 entries"))
	}
	maxLen, headLen := m.cfg.MaxLen, m.cfg.HeadMaxLen
	if req.MaxLen != 0 {
		maxLen = req.MaxLen
	}
	if req.HeadMaxLen != 0 {
		headLen = req.HeadMaxLen
	}
	if maxLen < 8 || maxLen > min(m.enc.MaxPosition, 8192) || headLen < 8 || headLen > maxLen-4 {
		return invalid(fmt.Errorf("invalid max_len/head_max_len budgets"))
	}
	result := DecisionResult{Model: "laya-rl-agent", Answers: map[string]DecisionAnswer{}}
	if len(req.Questions) == 0 {
		return result, nil
	}
	text, e := layaText(stateJSON)
	if e != nil {
		return invalid(e)
	}
	state, e := m.tok.Encode(strings.ReplaceAll(text, m.tok.MaskText, " "))
	if e != nil {
		return invalid(e)
	}
	names := make([]string, 0, len(req.Questions))
	for name := range req.Questions {
		names = append(names, name)
	}
	sort.Strings(names)
	type item struct {
		q       layaQuestion
		ids     []uint32
		markers []int
	}
	items := make([]item, len(names))
	// Validate every question before running the first expensive forward pass.
	for i, name := range names {
		q, err := prepareLayaQuestion(req.Questions[name])
		if err != nil {
			return invalid(fmt.Errorf("question %q: %w", name, err))
		}
		ids, markers, err := m.sequence(state, q, maxLen, headLen)
		if err != nil {
			return invalid(fmt.Errorf("question %q: %w", name, err))
		}
		items[i] = item{q, ids, markers}
	}
	for i, it := range items {
		logits, act, err := m.forward(ctx, it.ids, it.markers, it.q.kind)
		if err != nil {
			return DecisionResult{}, err
		}
		for _, v := range append(logits, act) {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return DecisionResult{}, fmt.Errorf("laya: inference produced non-finite values")
			}
		}
		result.Answers[names[i]] = m.decodeLaya(it.q, logits, act)
		result.Usage.InputTokens += len(it.ids)
	}
	return result, nil
}
func (m *LayaModel) decodeLaya(q layaQuestion, logits []float32, act float32) DecisionAnswer {
	k := len(logits)
	bucket := "11+"
	switch {
	case k <= 2:
		bucket = "2"
	case k <= 5:
		bucket = "3-5"
	case k <= 10:
		bucket = "6-10"
	}
	temp := m.cfg.Temperature[q.kind]
	if t, ok := m.cfg.TemperatureByOptions[q.typ+":"+bucket]; ok {
		temp = t
	}
	for i := range logits {
		logits[i] /= float32(temp)
	}
	layaSoftmax(logits)
	best := 0
	entropy, score := float64(0), float64(0)
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
		entropy -= float64(v) * math.Log(max(float64(v), 1e-12))
		score += float64(i) * float64(v)
	}
	round := func(x float64) float64 { return math.Round(x*10000) / 10000 }
	confidence := float64(1)
	if k > 1 {
		confidence = min(1, max(0, 1-entropy/math.Log(float64(k))))
	}
	answer := DecisionAnswer{Type: q.typ, Confidence: round(confidence), AnswerConfidence: round(float64(logits[best])), Action: DecisionAction{round(float64(act))}}
	if q.kind == 2 {
		v := round(float64(logits[1]))
		answer.Noul = &v
		answer.Confidence = answer.AnswerConfidence
		return answer
	}
	answer.Probabilities = map[string]float64{}
	for i, key := range q.keys {
		answer.Probabilities[key] = round(float64(logits[i]))
	}
	if q.kind == 0 {
		key := q.keys[best]
		answer.Choice = &key
	} else {
		v := round(score)
		answer.Score = &v
		answer.Legend = q.legend
	}
	return answer
}

// DecodeDecisionRequest decodes a single request, rejecting trailing JSON.
func DecodeDecisionRequest(r io.Reader) (DecisionRequest, error) {
	var req DecisionRequest
	d := json.NewDecoder(r)
	if e := d.Decode(&req); e != nil {
		return req, e
	}
	var extra any
	if e := d.Decode(&extra); e != io.EOF {
		return req, fmt.Errorf("expected one JSON request")
	}
	return req, nil
}
