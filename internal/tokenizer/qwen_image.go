package tokenizer

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const qwenImagePattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// QwenImageTokenizer accepts the NFC/byte-BPE pipeline shipped with
// Qwen-Image-2.1's Qwen3-VL encoder. Its regex differs from both GPT-2 and
// Qwen3.5, especially at digits, combining marks and whitespace boundaries.
type QwenImageTokenizer struct{ bpe LayaTokenizer }

func NewQwenImageTokenizer(data []byte) (*QwenImageTokenizer, error) {
	var raw struct {
		Model struct {
			Type         string            `json:"type"`
			Vocab        map[string]uint32 `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			ByteFallback bool              `json:"byte_fallback"`
			Dropout      *float64          `json:"dropout"`
			IgnoreMerges bool              `json:"ignore_merges"`
			Continuing   string            `json:"continuing_subword_prefix"`
			End          string            `json:"end_of_word_suffix"`
		} `json:"model"`
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
		Added []layaAddedToken `json:"added_tokens"`
	}
	if len(data) > 32<<20 {
		return nil, fmt.Errorf("qwen image tokenizer: JSON exceeds 32 MiB")
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw.Model.Type != "BPE" || len(raw.Model.Vocab) < 256 || len(raw.Model.Vocab) > 1<<20 || raw.Model.ByteFallback || raw.Model.IgnoreMerges || raw.Model.Continuing != "" || raw.Model.End != "" || raw.Model.Dropout != nil && *raw.Model.Dropout != 0 {
		return nil, fmt.Errorf("qwen image tokenizer: unsupported BPE model")
	}
	if raw.Normalizer.Type != "NFC" || raw.Pre.Type != "Sequence" || len(raw.Pre.Stages) != 2 {
		return nil, fmt.Errorf("qwen image tokenizer: unsupported normalization/pretokenization")
	}
	a, b := raw.Pre.Stages[0], raw.Pre.Stages[1]
	if a.Type != "Split" || a.Pattern.Regex != qwenImagePattern || a.Behavior != "Isolated" || a.Invert || b.Type != "ByteLevel" || b.AddPrefix || b.UseRegex {
		return nil, fmt.Errorf("qwen image tokenizer: unsupported split rule")
	}
	t := &QwenImageTokenizer{bpe: LayaTokenizer{vocab: raw.Model.Vocab, ranks: make(map[Pair]int, len(raw.Model.Merges)), added: raw.Added, nfc: true, split: qwenImagePieces}}
	t.bpe.bytes, _ = buildByteMaps()
	ids := make(map[uint32]string, len(raw.Model.Vocab)+len(raw.Added))
	for token, id := range t.bpe.vocab {
		if token == "" || id >= 1<<20 {
			return nil, fmt.Errorf("qwen image tokenizer: invalid vocabulary entry")
		}
		if _, ok := ids[id]; ok {
			return nil, fmt.Errorf("qwen image tokenizer: duplicate token ID %d", id)
		}
		ids[id] = token
	}
	for _, r := range t.bpe.bytes {
		if _, ok := t.bpe.vocab[string(r)]; !ok {
			return nil, fmt.Errorf("qwen image tokenizer: missing byte token")
		}
	}
	for i, m := range raw.Model.Merges {
		var pair []string
		if json.Unmarshal(m, &pair) != nil {
			var s string
			if err := json.Unmarshal(m, &s); err != nil {
				return nil, err
			}
			pair = strings.Split(s, " ")
		}
		if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
			return nil, fmt.Errorf("qwen image tokenizer: invalid merge %d", i)
		}
		if _, ok := t.bpe.vocab[pair[0]+pair[1]]; !ok {
			return nil, fmt.Errorf("qwen image tokenizer: merge %d has no vocabulary entry", i)
		}
		key := Pair{pair[0], pair[1]}
		if _, ok := t.bpe.ranks[key]; ok {
			return nil, fmt.Errorf("qwen image tokenizer: duplicate merge %d", i)
		}
		t.bpe.ranks[key] = i
	}
	for _, a := range t.bpe.added {
		if a.Content == "" || a.SingleWord || a.LStrip || a.RStrip || a.ID >= 1<<20 {
			return nil, fmt.Errorf("qwen image tokenizer: unsupported added token")
		}
		if old, ok := ids[a.ID]; ok && old != a.Content {
			return nil, fmt.Errorf("qwen image tokenizer: conflicting added ID %d", a.ID)
		}
		if old, ok := t.bpe.vocab[a.Content]; ok && old != a.ID {
			return nil, fmt.Errorf("qwen image tokenizer: conflicting added token %q", a.Content)
		}
		t.bpe.vocab[a.Content] = a.ID
		ids[a.ID] = a.Content
	}
	sort.SliceStable(t.bpe.added, func(i, j int) bool { return len(t.bpe.added[i].Content) > len(t.bpe.added[j].Content) })
	for _, s := range []string{"<|im_start|>", "<|im_end|>"} {
		if _, ok := t.bpe.vocab[s]; !ok {
			return nil, fmt.Errorf("qwen image tokenizer: missing %s", s)
		}
	}
	return t, nil
}

func (t *QwenImageTokenizer) Encode(text string) ([]uint32, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("qwen image tokenizer: invalid UTF-8")
	}
	return t.bpe.Encode(text)
}

func (t *QwenImageTokenizer) ValidateVocab(size int) error { return t.bpe.ValidateVocab(size) }

func qwenImagePieces(text string) []string {
	r := []rune(text)
	out := make([]string, 0, len(r)/3+1)
	for i := 0; i < len(r); {
		end := i
		if e, ok := matchQwen35Contraction(r, i); ok {
			end = e
		} else {
			start := i
			if r[i] != '\r' && r[i] != '\n' && !unicode.IsLetter(r[i]) && !unicode.IsNumber(r[i]) {
				start++
			}
			if start < len(r) && unicode.IsLetter(r[start]) {
				end = start + 1
				for end < len(r) && unicode.IsLetter(r[end]) {
					end++
				}
			}
		}
		if end == i && unicode.IsNumber(r[i]) {
			end = i + 1
		}
		if end == i {
			start := i
			if r[i] == ' ' {
				start++
			}
			end = start
			for end < len(r) && !unicode.IsSpace(r[end]) && !unicode.IsLetter(r[end]) && !unicode.IsNumber(r[end]) {
				end++
			}
			if end > start {
				for end < len(r) && (r[end] == '\r' || r[end] == '\n') {
					end++
				}
			} else {
				end = i
			}
		}
		if end == i && unicode.IsSpace(r[i]) {
			lastNewline := -1
			end = i
			for end < len(r) && unicode.IsSpace(r[end]) {
				if r[end] == '\r' || r[end] == '\n' {
					lastNewline = end
				}
				end++
			}
			if lastNewline >= 0 {
				end = lastNewline + 1
			} else if end < len(r) && end-i > 1 {
				end--
			}
		}
		if end == i {
			end++
		}
		out = append(out, string(r[i:end]))
		i = end
	}
	return out
}
