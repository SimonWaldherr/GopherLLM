package tokenizer

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// LayaTokenizer reads the two BPE formats shipped by Laya: ModernBERT's
// byte-level tokenizer and mmBERT's Gemma-style metaspace tokenizer.
// It deliberately rejects unknown pipelines rather than silently changing IDs.
type LayaTokenizer struct {
	vocab          map[string]uint32
	ranks          map[Pair]int
	added          []layaAddedToken
	bytes          map[byte]rune
	metaspace      bool
	nfc            bool
	fallback       bool
	CLS, SEP, Mask uint32
	MaskText       string
}
type layaAddedToken struct {
	ID         uint32 `json:"id"`
	Content    string `json:"content"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
}

func NewLayaTokenizer(data, config []byte) (*LayaTokenizer, error) {
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
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizer"`
		Pre struct {
			Type        string `json:"type"`
			AddPrefix   bool   `json:"add_prefix_space"`
			UseRegex    bool   `json:"use_regex"`
			Replacement string `json:"replacement"`
			Prepend     string `json:"prepend_scheme"`
			Split       bool   `json:"split"`
		} `json:"pre_tokenizer"`
		Added []layaAddedToken `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw.Model.Type != "BPE" || len(raw.Model.Vocab) == 0 || raw.Model.IgnoreMerges || raw.Model.Continuing != "" || raw.Model.End != "" || raw.Model.Dropout != nil && *raw.Model.Dropout != 0 {
		return nil, fmt.Errorf("laya: unsupported tokenizer model")
	}
	t := &LayaTokenizer{vocab: raw.Model.Vocab, ranks: make(map[Pair]int), added: raw.Added, fallback: raw.Model.ByteFallback}
	switch {
	case raw.Pre.Type == "ByteLevel" && !raw.Pre.AddPrefix && raw.Pre.UseRegex && raw.Normalizer.Type == "NFC":
		t.bytes, _ = buildByteMaps()
		t.nfc = true
	case raw.Pre.Type == "Metaspace" && raw.Pre.Replacement == "▁" && raw.Pre.Prepend == "always" && raw.Pre.Split && raw.Normalizer.Type == "Replace" && raw.Normalizer.Pattern.String == " " && raw.Normalizer.Content == "▁":
		t.metaspace = true
	default:
		return nil, fmt.Errorf("laya: unsupported tokenizer normalization/pre-tokenization pipeline")
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
		if len(pair) != 2 {
			return nil, fmt.Errorf("laya: invalid BPE merge %d", i)
		}
		t.ranks[Pair{pair[0], pair[1]}] = i
	}
	for _, a := range t.added {
		if a.Content == "" || a.SingleWord {
			return nil, fmt.Errorf("laya: unsupported added token %q", a.Content)
		}
		t.vocab[a.Content] = a.ID
	}
	sort.SliceStable(t.added, func(i, j int) bool { return len(t.added[i].Content) > len(t.added[j].Content) })
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(config, &cfg); err != nil {
		return nil, err
	}
	special := func(key string) (string, uint32, error) {
		var s string
		if json.Unmarshal(cfg[key], &s) != nil {
			var v struct {
				Content string `json:"content"`
			}
			if err := json.Unmarshal(cfg[key], &v); err != nil {
				return "", 0, fmt.Errorf("laya: missing %s", key)
			}
			s = v.Content
		}
		id, ok := t.vocab[s]
		if !ok {
			return "", 0, fmt.Errorf("laya: unknown %s %q", key, s)
		}
		return s, id, nil
	}
	var err error
	if _, t.CLS, err = special("cls_token"); err != nil {
		return nil, err
	}
	if _, t.SEP, err = special("sep_token"); err != nil {
		return nil, err
	}
	if t.MaskText, t.Mask, err = special("mask_token"); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *LayaTokenizer) ValidateVocab(size int) error {
	for _, id := range t.vocab {
		if uint64(id) >= uint64(size) {
			return fmt.Errorf("laya: token ID %d exceeds vocabulary %d", id, size)
		}
	}
	return nil
}

// Encode adds no boundary tokens; the decision prompt places them explicitly.
func (t *LayaTokenizer) Encode(text string) ([]uint32, error) {
	if t.nfc {
		text = normalizeNFC(text)
	}
	var out []uint32
	for len(text) > 0 {
		pos := len(text)
		var match *layaAddedToken
		for i := range t.added {
			a := &t.added[i]
			p := strings.Index(text, a.Content)
			if p >= 0 && (p < pos || p == pos && match == nil) {
				pos = p
				match = a
			}
		}
		plain := text[:pos]
		if match != nil && match.LStrip {
			plain = strings.TrimRightFunc(plain, unicode.IsSpace)
		}
		var err error
		out, err = t.encodePlain(out, plain)
		if err != nil {
			return nil, err
		}
		if match == nil {
			break
		}
		out = append(out, match.ID)
		text = text[pos+len(match.Content):]
		if match.RStrip {
			text = strings.TrimLeftFunc(text, unicode.IsSpace)
		}
	}
	return out, nil
}
func (t *LayaTokenizer) encodePlain(out []uint32, text string) ([]uint32, error) {
	if text == "" {
		return out, nil
	}
	var pieces []string
	if t.metaspace {
		text = strings.ReplaceAll(text, " ", "▁")
		if !strings.HasPrefix(text, "▁") {
			text = "▁" + text
		}
		// Metaspace splits before every marker, including consecutive markers.
		for len(text) > 0 {
			idx := strings.Index(text[len("▁"):], "▁")
			if idx < 0 {
				pieces = append(pieces, text)
				break
			}
			idx += len("▁")
			pieces = append(pieces, text[:idx])
			text = text[idx:]
		}
	} else {
		pieces = layaByteLevelPieces(text)
	}
	var syms []bpeSymbol
	var heap bpeHeap
	for _, piece := range pieces {
		if !t.metaspace {
			var b strings.Builder
			for i := range len(piece) {
				b.WriteRune(t.bytes[piece[i]])
			}
			piece = b.String()
		}
		syms = bpeSymbolsFromRunes(syms, piece)
		mergeBPESymbols(syms, &heap, func(l, r *bpeSymbol) (float64, uint32, bool) {
			rank, ok := t.ranks[Pair{l.text, r.text}]
			return float64(rank), 0, ok
		}, func(l, r *bpeSymbol, _ uint32) string { return l.text + r.text })
		for i := int32(0); i >= 0; i = syms[i].next {
			s := syms[i].text
			if id, ok := t.vocab[s]; ok {
				out = append(out, id)
				continue
			}
			if !t.fallback {
				return nil, fmt.Errorf("laya: BPE symbol %q missing from vocabulary", s)
			}
			for j := range len(s) {
				id, ok := t.vocab[spByteTokens[s[j]]]
				if !ok {
					return nil, fmt.Errorf("laya: missing byte fallback")
				}
				out = append(out, id)
			}
		}
	}
	return out, nil
}

// layaByteLevelPieces implements the exact tokenizers ByteLevel GPT-2 regex:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// The legacy GGUF splitter has different contraction/whitespace behavior.
func layaByteLevelPieces(text string) []string {
	r := []rune(text)
	var out []string
	category := func(c rune) int {
		if unicode.IsLetter(c) {
			return 1
		}
		if unicode.IsNumber(c) {
			return 2
		}
		if unicode.IsSpace(c) {
			return 0
		}
		return 3
	}
	for i := 0; i < len(r); {
		end := i
		if r[i] == '\'' {
			for _, suffix := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
				if i+1+len(suffix) <= len(r) && string(r[i+1:i+1+len(suffix)]) == suffix {
					end = i + 1 + len(suffix)
					break
				}
			}
		}
		if end == i {
			j := i
			if r[j] == ' ' && j+1 < len(r) && category(r[j+1]) != 0 {
				j++
			}
			kind := category(r[j])
			if kind != 0 {
				end = j + 1
				for end < len(r) && category(r[end]) == kind {
					end++
				}
			} else {
				end = i + 1
				for end < len(r) && unicode.IsSpace(r[end]) {
					end++
				}
				if end < len(r) && end-i > 1 {
					end--
				}
			}
		}
		out = append(out, string(r[i:end]))
		i = end
	}
	return out
}
