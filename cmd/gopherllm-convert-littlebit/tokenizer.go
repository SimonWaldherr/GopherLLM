package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
)

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}
type tokenizerJSON struct {
	Model struct {
		Type         string            `json:"type"`
		Vocab        map[string]int    `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		ByteFallback bool              `json:"byte_fallback"`
	} `json:"model"`
	Decoder struct {
		Type string `json:"type"`
	} `json:"decoder"`
	Normalizer struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	Pre struct {
		Type   string `json:"type"`
		Stages []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior  string `json:"behavior"`
			Invert    bool   `json:"invert"`
			AddPrefix bool   `json:"add_prefix_space"`
			UseRegex  bool   `json:"use_regex"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	Added []addedToken `json:"added_tokens"`
}
type tokenizerConfig struct {
	BOS      specialToken `json:"bos_token"`
	EOS      specialToken `json:"eos_token"`
	UNK      specialToken `json:"unk_token"`
	Pad      specialToken `json:"pad_token"`
	AddBOS   *bool        `json:"add_bos_token"`
	AddEOS   bool         `json:"add_eos_token"`
	Template string       `json:"chat_template"`
}

type specialToken string

func (s *specialToken) UnmarshalJSON(raw []byte) error {
	if raw[0] != '{' {
		return json.Unmarshal(raw, (*string)(s))
	}
	var token addedToken
	if err := json.Unmarshal(raw, &token); err != nil {
		return err
	}
	*s = specialToken(token.Content)
	return nil
}

func (cfg hfConfig) tokenizer(dir string, meta map[string]gguf.MetaValue) error {
	var info tokenizerConfig
	for _, name := range []string{"special_tokens_map.json", "tokenizer_config.json"} {
		if err := readJSON(dir, name, &info); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	var gen struct {
		BOS *uint32  `json:"bos_token_id"`
		EOS tokenIDs `json:"eos_token_id"`
		Pad *uint32  `json:"pad_token_id"`
	}
	if err := readJSON(dir, "generation_config.json", &gen); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	var jsonTok tokenizerJSON
	tokenizerErr := readJSON(dir, "tokenizer.json", &jsonTok)
	if tokenizerErr != nil && !errors.Is(tokenizerErr, os.ErrNotExist) {
		return tokenizerErr
	}
	// Use the original SentencePiece scores when tokenizer.model is present.
	// Transformers Llama 3 and Qwen3 save byte-level BPE as tokenizer.json.
	useByteBPE := jsonTok.Decoder.Type == "ByteLevel"
	if cfg.ModelType == "qwen3" && !useByteBPE {
		return fmt.Errorf("Qwen3 requires a byte-level BPE tokenizer.json")
	}
	model := "llama"
	var tokens tokenTable
	if !useByteBPE {
		raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.model"))
		if err == nil {
			tokens, err = loadSentencePiece(raw)
			if err != nil {
				return fmt.Errorf("tokenizer.model: %w", err)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	if len(tokens.pieces) == 0 {
		if tokenizerErr != nil {
			return fmt.Errorf("checkpoint needs tokenizer.model or tokenizer.json")
		}
		if jsonTok.Model.Type != "BPE" || !useByteBPE && !jsonTok.Model.ByteFallback {
			return fmt.Errorf("unsupported tokenizer.json model %q", jsonTok.Model.Type)
		}
		size := len(jsonTok.Model.Vocab)
		for _, token := range jsonTok.Added {
			if token.ID < 0 || token.ID >= cfg.Vocab {
				return fmt.Errorf("added tokenizer ID %d is outside model vocabulary", token.ID)
			}
			size = max(size, token.ID+1)
		}
		if size == 0 || size > cfg.Vocab || size > len(jsonTok.Model.Vocab)+len(jsonTok.Added) {
			return fmt.Errorf("tokenizer vocabulary does not match model vocabulary")
		}
		tokens = newTokenTable(size)
		for token, index := range jsonTok.Model.Vocab {
			typ := uint32(1)
			if !useByteBPE && strings.HasPrefix(token, "<0x") && len(token) == 6 && token[5] == '>' {
				typ = 6
			}
			if err := tokens.set(index, token, typ); err != nil {
				return err
			}
		}
		merges, err := jsonTok.merges()
		if err != nil {
			return err
		}
		if useByteBPE {
			model = "gpt2"
			pre, err := jsonTok.byteBPEPre(cfg.ModelType)
			if err != nil {
				return err
			}
			meta["tokenizer.ggml.pre"] = gguf.MetaValue{Kind: "str", Value: pre}
			if jsonTok.Normalizer.Type == "NFC" {
				meta["tokenizer.ggml.normalizer.nfc"] = gguf.MetaValue{Kind: "bool", Value: true}
			}
			meta["tokenizer.ggml.merges"] = gguf.MetaValue{Kind: "array", Value: merges}
		} else {
			for i, pair := range merges {
				left, right, _ := strings.Cut(pair, " ")
				if id, ok := jsonTok.Model.Vocab[left+right]; ok {
					tokens.scores[id] = -float32(i + 1)
				}
			}
		}
	}
	if err := tokens.add(jsonTok.Added, cfg.Vocab); err != nil {
		return err
	}
	if slices.Contains(tokens.pieces, "") || len(tokens.pieces) > cfg.Vocab || !useByteBPE && len(tokens.pieces) != cfg.Vocab {
		return fmt.Errorf("tokenizer contains %d entries for model vocabulary %d, or has unassigned IDs", len(tokens.pieces), cfg.Vocab)
	}
	activeVocab := len(tokens.pieces)
	if useByteBPE {
		// Qwen rounds embedding rows up past the tokenizer vocabulary.
		// The padded rows still occupy real model IDs.
		if cfg.Vocab-activeVocab > 1024 {
			return fmt.Errorf("tokenizer is missing %d model vocabulary entries", cfg.Vocab-activeVocab)
		}
		for n := activeVocab; n < cfg.Vocab; n++ {
			tokens.pieces = append(tokens.pieces, fmt.Sprintf("[PAD%d]", n))
			tokens.types = append(tokens.types, 5)
		}
	}
	meta["tokenizer.ggml.model"] = gguf.MetaValue{Kind: "str", Value: model}
	meta["tokenizer.ggml.tokens"] = gguf.MetaValue{Kind: "array", Value: tokens.pieces}
	meta["tokenizer.ggml.token_type"] = gguf.MetaValue{Kind: "array", Value: metaU32Array(tokens.types)}
	if !useByteBPE {
		meta["tokenizer.ggml.scores"] = gguf.MetaValue{Kind: "array", Value: tokens.scores}
	}
	addBOS := cfg.ModelType != "qwen3"
	if info.AddBOS != nil {
		addBOS = *info.AddBOS
	}
	meta["tokenizer.ggml.add_bos_token"] = gguf.MetaValue{Kind: "bool", Value: addBOS}
	meta["tokenizer.ggml.add_eos_token"] = gguf.MetaValue{Kind: "bool", Value: info.AddEOS}

	eos := cfg.EOS
	if len(gen.EOS) > 0 {
		eos = gen.EOS
	}
	for _, id := range eos {
		if uint64(id) >= uint64(activeVocab) {
			return fmt.Errorf("eos_token_id %d out of tokenizer range", id)
		}
	}
	if gen.BOS != nil {
		cfg.BOS = gen.BOS
	}
	if gen.Pad != nil {
		cfg.Pad = gen.Pad
	}
	var firstEOS *uint32
	if len(eos) > 0 {
		firstEOS = &eos[0]
	}
	for _, special := range []struct {
		key  string
		text specialToken
		id   *uint32
	}{
		{"bos", info.BOS, cfg.BOS},
		{"eos", info.EOS, firstEOS},
		{"unknown", info.UNK, nil},
		{"padding", info.Pad, cfg.Pad},
	} {
		var id uint32
		if i := slices.Index(tokens.pieces[:activeVocab], string(special.text)); special.text != "" && i >= 0 {
			id = uint32(i)
		} else if special.id != nil {
			id = *special.id
		} else if special.text == "" && special.key != "eos" && (special.key != "bos" || !addBOS) {
			continue
		} else {
			return fmt.Errorf("tokenizer has no %s token", special.key)
		}
		if uint64(id) >= uint64(activeVocab) {
			return fmt.Errorf("%s_token_id %d out of tokenizer range", special.key, id)
		}
		meta["tokenizer.ggml."+special.key+"_token_id"] = gguf.MetaValue{Kind: "u32", Value: id}
	}
	if len(eos) > 1 {
		meta["tokenizer.ggml.eog_token_ids"] = gguf.MetaValue{Kind: "array", Value: metaU32Array(eos)}
	}

	template, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja"))
	if err == nil && len(template) > 0 {
		info.Template = string(template)
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if info.Template != "" {
		meta["tokenizer.chat_template"] = gguf.MetaValue{Kind: "str", Value: info.Template}
	}
	return nil
}

func metaU32Array(values []uint32) []gguf.MetaValue {
	result := make([]gguf.MetaValue, len(values))
	for i, v := range values {
		result[i] = gguf.MetaValue{Kind: "u32", Value: v}
	}
	return result
}

func (t tokenizerJSON) byteBPEPre(arch string) (string, error) {
	// This is the common Llama 3 / Qwen split expression. The digit
	// alternative is the models' only difference in pre-tokenization.
	const front = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}`
	const back = `| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	if t.Pre.Type != "Sequence" || len(t.Pre.Stages) != 2 ||
		t.Normalizer.Type != "" && t.Normalizer.Type != "NFC" {
		return "", fmt.Errorf("unsupported byte-BPE tokenizer normalization or pre-tokenization")
	}
	a, b := t.Pre.Stages[0], t.Pre.Stages[1]
	if a.Type != "Split" || a.Behavior != "Isolated" || a.Invert || b.Type != "ByteLevel" || b.AddPrefix || b.UseRegex {
		return "", fmt.Errorf("unsupported byte-BPE tokenizer split stages")
	}
	switch {
	case arch == "qwen3" && a.Pattern.Regex == front+back:
		return "qwen3", nil
	case arch == "llama" && a.Pattern.Regex == front+`{1,3}`+back:
		return "llama-bpe", nil
	default:
		return "", fmt.Errorf("unsupported %s byte-BPE tokenizer split pattern", arch)
	}
}

func (t tokenizerJSON) merges() ([]string, error) {
	merges := make([]string, len(t.Model.Merges))
	for i, raw := range t.Model.Merges {
		var pair []string
		if raw[0] == '[' {
			if err := json.Unmarshal(raw, &pair); err != nil || len(pair) != 2 {
				return nil, fmt.Errorf("invalid tokenizer merge at %d", i)
			}
		} else {
			var text string
			if err := json.Unmarshal(raw, &text); err != nil {
				return nil, fmt.Errorf("invalid tokenizer merge at %d", i)
			}
			pair = strings.Split(text, " ")
		}
		if len(pair) != 2 || pair[0] == "" || pair[1] == "" || strings.ContainsAny(pair[0]+pair[1], " ") {
			return nil, fmt.Errorf("invalid tokenizer merge at %d", i)
		}
		merges[i] = strings.Join(pair, " ")
	}
	return merges, nil
}

type tokenTable struct {
	pieces []string
	scores []float32
	types  []uint32
}

func newTokenTable(size int) tokenTable {
	return tokenTable{make([]string, size), make([]float32, size), make([]uint32, size)}
}

func (t *tokenTable) set(index int, piece string, typ uint32) error {
	if piece == "" || index < 0 || index >= len(t.pieces) || t.pieces[index] != "" && t.pieces[index] != piece {
		return fmt.Errorf("invalid or conflicting tokenizer token %q at %d", piece, index)
	}
	t.pieces[index], t.types[index] = piece, typ
	return nil
}

func (t *tokenTable) add(added []addedToken, vocab int) error {
	for _, a := range added {
		if a.ID < 0 || a.ID >= vocab {
			return fmt.Errorf("added tokenizer ID %d is outside model vocabulary", a.ID)
		}
		if a.ID >= len(t.pieces) {
			if a.ID-len(t.pieces) >= len(added) {
				return fmt.Errorf("noncontiguous tokenizer vocabulary")
			}
			gap := a.ID + 1 - len(t.pieces)
			t.pieces = append(t.pieces, make([]string, gap)...)
			t.scores = append(t.scores, make([]float32, gap)...)
			t.types = append(t.types, make([]uint32, gap)...)
		}
		typ := uint32(4)
		if a.Special {
			typ = 3
		}
		if err := t.set(a.ID, a.Content, typ); err != nil {
			return err
		}
	}
	return nil
}
