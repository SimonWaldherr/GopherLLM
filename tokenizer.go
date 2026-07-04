package main

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type TokenizerMode int

const (
	TokenizerSentencePiece TokenizerMode = iota
	TokenizerGPT2BPE
)

type Pair struct {
	Left  string
	Right string
}

type Tokenizer struct {
	Vocab       []string
	Scores      []float32
	TokenToID   map[string]uint32
	MergeRanks  map[Pair]int
	ByteEncoder map[byte]rune
	ByteDecoder map[rune]byte
	Mode        TokenizerMode
	Pre         string
	AddBOS      bool
	BOSID       uint32
	EOSID       uint32
}

func TokenizerFromMetadata(metadata map[string]MetaValue) (*Tokenizer, error) {
	tokensValue, ok := metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens")
	}
	vocab, ok := tokensValue.AsStringArray()
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not a string array")
	}
	scores := make([]float32, len(vocab))
	if v, ok := metadata["tokenizer.ggml.scores"]; ok {
		if arr, ok := v.AsF32Array(); ok {
			copy(scores, arr)
		}
	}
	tokenToID := make(map[string]uint32, len(vocab))
	for i, tok := range vocab {
		tokenToID[tok] = uint32(i)
	}
	mergeRanks := map[Pair]int{}
	if v, ok := metadata["tokenizer.ggml.merges"]; ok {
		if merges, ok := v.AsStringArray(); ok {
			for rank, merge := range merges {
				if left, right, ok := strings.Cut(merge, " "); ok {
					mergeRanks[Pair{left, right}] = rank
				}
			}
		}
	}
	model, pre := "", ""
	if v, ok := metadata["tokenizer.ggml.model"]; ok {
		model, _ = v.AsString()
	}
	if v, ok := metadata["tokenizer.ggml.pre"]; ok {
		pre, _ = v.AsString()
	}
	mode := TokenizerSentencePiece
	preLower := strings.ToLower(pre)
	if strings.EqualFold(model, "gpt2") || strings.Contains(preLower, "qwen") || strings.Contains(preLower, "gpt") {
		mode = TokenizerGPT2BPE
	}
	enc, dec := buildByteMaps()
	bosID, eosID := uint32(1), uint32(2)
	if v, ok := metadata["tokenizer.ggml.bos_token_id"]; ok {
		if n, ok := v.AsU32(); ok {
			bosID = n
		}
	}
	if v, ok := metadata["tokenizer.ggml.eos_token_id"]; ok {
		if n, ok := v.AsU32(); ok {
			eosID = n
		}
	}
	addBOS := true
	if v, ok := metadata["tokenizer.ggml.add_bos_token"]; ok {
		if b, ok := v.AsBool(); ok {
			addBOS = b
		}
	}
	return &Tokenizer{Vocab: vocab, Scores: scores, TokenToID: tokenToID, MergeRanks: mergeRanks, ByteEncoder: enc, ByteDecoder: dec, Mode: mode, Pre: preLower, AddBOS: addBOS, BOSID: bosID, EOSID: eosID}, nil
}

func (t *Tokenizer) Encode(text string) []uint32 {
	out := make([]uint32, 0, len(text)/3+1)
	if t.AddBOS {
		out = append(out, t.BOSID)
	}
	return append(out, t.EncodeWithoutBOS(text)...)
}

func (t *Tokenizer) EncodeWithoutBOS(text string) []uint32 {
	if text == "" {
		return nil
	}
	if t.Mode == TokenizerGPT2BPE {
		return t.encodeGPT2BPE(text)
	}
	return t.encodeSentencePiece(text)
}

func (t *Tokenizer) decodeRaw(id uint32) string {
	if int(id) < len(t.Vocab) {
		return t.Vocab[id]
	}
	return ""
}

func (t *Tokenizer) DecodeToken(id uint32) string {
	raw := t.decodeRaw(id)
	if t.Mode == TokenizerGPT2BPE {
		return t.decodeGPT2Bytes(raw)
	}
	if strings.HasPrefix(raw, "<0x") && strings.HasSuffix(raw, ">") && len(raw) == 6 {
		if b, err := strconv.ParseUint(raw[3:5], 16, 8); err == nil {
			return string([]byte{byte(b)})
		}
	}
	return strings.ReplaceAll(raw, "\u2581", " ")
}

func (t *Tokenizer) VocabSize() int { return len(t.Vocab) }

func (t *Tokenizer) SpecialID(token string) (uint32, bool) {
	id, ok := t.TokenToID[token]
	return id, ok
}

func (t *Tokenizer) encodeSentencePiece(text string) []uint32 {
	processed := strings.ReplaceAll(" "+text, " ", "\u2581")
	current := t.encodeFromPieces(strings.Split(processed, ""))
	for len(current) >= 2 {
		bestScore := float32(-3.4e38)
		bestIdx := -1
		var bestID uint32
		for i := 0; i+1 < len(current); i++ {
			merged := t.decodeRaw(current[i]) + t.decodeRaw(current[i+1])
			if id, ok := t.TokenToID[merged]; ok {
				score := float32(0)
				if int(id) < len(t.Scores) {
					score = t.Scores[id]
				}
				if score > bestScore {
					bestScore, bestIdx, bestID = score, i, id
				}
			}
		}
		if bestIdx < 0 {
			break
		}
		current[bestIdx] = bestID
		current = append(current[:bestIdx+1], current[bestIdx+2:]...)
	}
	return current
}

func (t *Tokenizer) encodeGPT2BPE(text string) []uint32 {
	out := []uint32{}
	for _, piece := range t.pretokenize(text) {
		var encoded strings.Builder
		for _, b := range []byte(piece) {
			if ch, ok := t.ByteEncoder[b]; ok {
				encoded.WriteRune(ch)
			}
		}
		symbols := strings.Split(encoded.String(), "")
		for len(symbols) > 1 {
			bestRank := int(^uint(0) >> 1)
			bestIdx := -1
			for i := 0; i+1 < len(symbols); i++ {
				if rank, ok := t.MergeRanks[Pair{symbols[i], symbols[i+1]}]; ok && rank < bestRank {
					bestRank, bestIdx = rank, i
				}
			}
			if bestIdx < 0 {
				break
			}
			symbols[bestIdx] += symbols[bestIdx+1]
			symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
		}
		for _, symbol := range symbols {
			if id, ok := t.TokenToID[symbol]; ok {
				out = append(out, id)
			} else {
				out = append(out, t.encodeFromPieces([]string{symbol})...)
			}
		}
	}
	return out
}

func (t *Tokenizer) encodeFromPieces(pieces []string) []uint32 {
	out := []uint32{}
	for _, piece := range pieces {
		if id, ok := t.TokenToID[piece]; ok {
			out = append(out, id)
			continue
		}
		for _, b := range []byte(piece) {
			byteTok := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.TokenToID[byteTok]; ok {
				out = append(out, id)
			}
		}
	}
	return out
}

func (t *Tokenizer) decodeGPT2Bytes(raw string) string {
	bytes := make([]byte, 0, len(raw))
	for _, ch := range raw {
		if b, ok := t.ByteDecoder[ch]; ok {
			bytes = append(bytes, b)
		} else {
			bytes = append(bytes, []byte(string(ch))...)
		}
	}
	return string(bytes)
}

func (t *Tokenizer) pretokenize(text string) []string {
	if strings.Contains(t.Pre, "tekken") {
		return pretokenizeTekken(text)
	}
	return pretokenizeGPT2(text)
}

// pretokenizeTekken splits text following Mistral's Tekken pre-tokenizer
// (the tiktoken-style pattern shipped with Ministral and other Tekken models):
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//	| [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//	| \p{N}
//	|  ?[^\s\p{L}\p{N}]+[\r\n/]*
//	| \s*[\r\n]+ | \s+(?!\S) | \s+
//
// The key differences from the generic GPT-2 splitter are that each numeric
// character becomes its own piece and words split on upper/lower-case
// boundaries, matching how the Tekken merges were trained.
func pretokenizeTekken(text string) []string {
	r := []rune(text)
	n := len(r)
	pieces := make([]string, 0, len(r)/3+1)
	i := 0
	for i < n {
		if end, ok := matchTekkenWord(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if isTekkenNumber(r[i]) {
			pieces = append(pieces, string(r[i:i+1]))
			i++
			continue
		}
		if end, ok := matchTekkenPunct(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsSpace(r[i]) {
			end := tekkenWhitespaceEnd(r, n, i)
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		pieces = append(pieces, string(r[i:i+1]))
		i++
	}
	return pieces
}

func isTekkenNumber(c rune) bool { return unicode.IsNumber(c) }

// tekkenUpperClass matches [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}].
func tekkenUpperClass(c rune) bool {
	return (unicode.IsLetter(c) && !unicode.IsLower(c)) || unicode.Is(unicode.M, c)
}

// tekkenLowerClass matches [\p{Ll}\p{Lm}\p{Lo}\p{M}].
func tekkenLowerClass(c rune) bool {
	return (unicode.IsLetter(c) && !unicode.IsUpper(c) && !unicode.IsTitle(c)) || unicode.Is(unicode.M, c)
}

// matchTekkenWord handles the two letter alternatives (optional leading
// non-letter/digit character, then case-split letter runs).
func matchTekkenWord(r []rune, n, i int) (int, bool) {
	opt := i
	if opt < n && r[opt] != '\r' && r[opt] != '\n' && !unicode.IsLetter(r[opt]) && !isTekkenNumber(r[opt]) {
		opt++
	}
	// Alt 1: U* L+
	k := opt
	for k < n && tekkenUpperClass(r[k]) {
		k++
	}
	lStart := k
	for k < n && tekkenLowerClass(r[k]) {
		k++
	}
	if k > lStart {
		return k, true
	}
	// Alt 2: U+ L*
	k = opt
	for k < n && tekkenUpperClass(r[k]) {
		k++
	}
	if k > opt {
		for k < n && tekkenLowerClass(r[k]) {
			k++
		}
		return k, true
	}
	return i, false
}

// matchTekkenPunct handles " ?[^\s\p{L}\p{N}]+[\r\n/]*".
func matchTekkenPunct(r []rune, n, i int) (int, bool) {
	j := i
	if j < n && r[j] == ' ' {
		j++
	}
	pStart := j
	for j < n && !unicode.IsSpace(r[j]) && !unicode.IsLetter(r[j]) && !isTekkenNumber(r[j]) {
		j++
	}
	if j == pStart {
		return i, false
	}
	for j < n && (r[j] == '\r' || r[j] == '\n' || r[j] == '/') {
		j++
	}
	return j, true
}

// tekkenWhitespaceEnd handles "\s*[\r\n]+ | \s+(?!\S) | \s+": a whitespace run
// that ends at its final newline is cut there, otherwise the whole run is one
// piece.
func tekkenWhitespaceEnd(r []rune, n, i int) int {
	k := i
	lastNL := -1
	for k < n && unicode.IsSpace(r[k]) {
		if r[k] == '\r' || r[k] == '\n' {
			lastNL = k
		}
		k++
	}
	if lastNL >= 0 {
		return lastNL + 1
	}
	return k
}

func pretokenizeGPT2(text string) []string {
	chars := []rune(text)
	pieces := []string{}
	for i := 0; i < len(chars); {
		start := i
		hadSpace := false
		for i < len(chars) && unicode.IsSpace(chars[i]) {
			hadSpace = true
			i++
		}
		if i >= len(chars) {
			if hadSpace {
				pieces = append(pieces, string(chars[start:i]))
			}
			break
		}
		j := i
		c := chars[i]
		switch {
		case unicode.IsLetter(c):
			for j < len(chars) && unicode.IsLetter(chars[j]) {
				j++
			}
		case unicode.IsDigit(c):
			for j < len(chars) && unicode.IsDigit(chars[j]) {
				j++
			}
		default:
			for j < len(chars) && !unicode.IsSpace(chars[j]) && !unicode.IsLetter(chars[j]) && !unicode.IsDigit(chars[j]) {
				j++
			}
		}
		pieceStart := i
		if hadSpace {
			pieceStart = start
		}
		pieces = append(pieces, string(chars[pieceStart:j]))
		i = j
	}
	return pieces
}

func buildByteMaps() (map[byte]rune, map[rune]byte) {
	bs := []uint32{}
	for b := byte('!'); b <= byte('~'); b++ {
		bs = append(bs, uint32(b))
	}
	for b := byte(0xA1); b <= byte(0xAC); b++ {
		bs = append(bs, uint32(b))
	}
	for b := byte(0xAE); b != 0; b++ {
		bs = append(bs, uint32(b))
		if b == 0xFF {
			break
		}
	}
	cs := append([]uint32(nil), bs...)
	n := uint32(0)
	contains := func(v uint32) bool {
		for _, x := range bs {
			if x == v {
				return true
			}
		}
		return false
	}
	for b := uint32(0); b <= 255; b++ {
		if !contains(b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	enc := map[byte]rune{}
	dec := map[rune]byte{}
	for i, b := range bs {
		ch := rune(cs[i])
		enc[byte(b)] = ch
		dec[ch] = byte(b)
	}
	return enc, dec
}
