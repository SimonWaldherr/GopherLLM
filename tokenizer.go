package gopherllm

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"github.com/SimonWaldherr/GopherLLM/internal/wordpiece"
)

type TokenizerMode int

const (
	TokenizerSentencePiece TokenizerMode = iota
	TokenizerGPT2BPE
	TokenizerWordPiece
)

type Pair struct {
	Left  string
	Right string
}

// Tokenizer implements the GGUF tokenizer families:
//
//   - TokenizerSentencePiece (tokenizer.ggml.model = "llama"): text is mapped
//     to "▁"-prefixed pieces and greedily merged by vocabulary Scores, with
//     <0xNN> byte tokens as the fallback for uncovered bytes.
//   - TokenizerGPT2BPE ("gpt2", incl. Mistral's Tekken, Qwen, and Kimi):
//     text is pre-tokenized by a regex-equivalent splitter (Pre selects the
//     family-specific variant), bytes are mapped through the GPT-2
//     printable-byte alphabet (ByteEncoder/ByteDecoder), and pieces are
//     merged by MergeRanks.
//   - TokenizerWordPiece ("bert"): text is normalized and split at whitespace,
//     punctuation, and CJK characters, then each word is greedily segmented.
//     Both llama.cpp's phantom-space GGUF vocabulary and raw "##" WordPiece
//     vocabularies are accepted.
//
// AddBOS mirrors tokenizer.ggml.add_bos_token: Encode prepends BOSID when
// set. BERT tokenizers also append SEP/EOS according to their metadata.
// EncodeWithoutBOS never adds any boundary tokens (chat renderers place those
// themselves).
type Tokenizer struct {
	Vocab             []string
	Scores            []float32
	TokenToID         map[string]uint32
	MergeRanks        map[Pair]int
	ByteEncoder       map[byte]rune
	ByteDecoder       map[rune]byte
	Mode              TokenizerMode
	Pre               string
	AddBOS            bool
	AddEOS            bool
	AddSEP            bool
	BOSID             uint32
	EOSID             uint32
	UNKID             uint32
	SEPID             uint32
	Lowercase         bool
	StripAccents      bool
	RawWordPieceVocab bool
}

// TokenizerFromMetadata builds a Tokenizer from a GGUF's tokenizer.* keys:
// vocabulary, merge ranks/scores, BOS/EOS ids, and the model/pre strings that
// select the encoding mode.
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
			// Pre-sized: BPE merge lists commonly run to six figures, and an
			// unsized map rehashes (full-table copy) on every doubling as it
			// grows, which showed up as real load-time cost under profiling.
			mergeRanks = make(map[Pair]int, len(merges))
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
	// Several GPT-2 BPE families do not set tokenizer.ggml.model to the literal
	// "gpt2". In particular, Kimi K2 GGUFs identify their tiktoken vocabulary
	// through tokenizer.ggml.pre = "kimi-k2". Classify the pre-tokenizer rather
	// than silently treating such vocabularies as SentencePiece.
	if strings.EqualFold(model, "gpt2") || strings.Contains(preLower, "qwen") || strings.Contains(preLower, "gpt") || strings.Contains(preLower, "tekken") || strings.Contains(preLower, "kimi") {
		mode = TokenizerGPT2BPE
	} else if strings.EqualFold(model, "bert") {
		mode = TokenizerWordPiece
	}
	enc, dec := buildByteMaps()
	bosID, eosID, unkID, sepID := uint32(1), uint32(2), uint32(0), uint32(2)
	addBOS, addEOS, addSEP := true, false, false
	sepFromVocab := false
	if mode == TokenizerWordPiece {
		bosID, eosID, unkID, sepID = 101, 102, 100, 102
		addSEP = true
		// Small and custom BERT vocabularies do not necessarily use the
		// conventional 100-103 IDs. Token text is a useful fallback when the
		// explicit metadata is absent.
		if id, ok := tokenToID["[CLS]"]; ok {
			bosID = id
		}
		if id, ok := tokenToID["[UNK]"]; ok {
			unkID = id
		} else if id, ok := tokenToID["<unk>"]; ok {
			unkID = id
		}
		if id, ok := tokenToID["[SEP]"]; ok {
			sepID, eosID = id, id
			sepFromVocab = true
		}
	}
	readID := func(current uint32, keys ...string) uint32 {
		for _, key := range keys {
			if v, ok := metadata[key]; ok {
				if n, ok := v.AsU32(); ok {
					return n
				}
			}
		}
		return current
	}
	readBool := func(current bool, key string) bool {
		if v, ok := metadata[key]; ok {
			if b, ok := v.AsBool(); ok {
				return b
			}
		}
		return current
	}
	bosID = readID(bosID, "tokenizer.ggml.bos_token_id", "tokenizer.ggml.cls_token_id")
	eosID = readID(eosID, "tokenizer.ggml.eos_token_id")
	unkID = readID(unkID, "tokenizer.ggml.unknown_token_id")
	// llama.cpp's GGUF key has historically used the "seperator" spelling.
	// Accept the corrected spelling too for files produced by other writers.
	sepKeys := []string{
		"tokenizer.ggml.seperator_token_id",
		"tokenizer.ggml.separator_token_id",
		"tokenizer.ggml.sep_token_id",
	}
	hasExplicitSEP := false
	for _, key := range sepKeys {
		if v, ok := metadata[key]; ok {
			if n, ok := v.AsU32(); ok {
				sepID, hasExplicitSEP = n, true
				break
			}
		}
	}
	if mode == TokenizerWordPiece && !hasExplicitSEP && !sepFromVocab {
		// Some GGUF writers only expose the BERT separator as EOS.
		sepID = eosID
	}
	addBOS = readBool(addBOS, "tokenizer.ggml.add_bos_token")
	addEOS = readBool(addEOS, "tokenizer.ggml.add_eos_token")
	addSEP = readBool(addSEP, "tokenizer.ggml.add_sep_token")
	// llama.cpp's BertNormalizer defaults to lowercase=true. Keep that default
	// scoped to WordPiece; the field is unused by the other tokenizer modes.
	lowercase := readBool(mode == TokenizerWordPiece, "tokenizer.ggml.normalizer.lowercase")
	// This matches llama.cpp's BERT default: uncased tokenizers strip accents
	// unless the GGUF carries an explicit override.
	stripAccents := readBool(lowercase, "tokenizer.ggml.normalizer.strip_accents")
	rawWordPieceVocab := false
	if mode == TokenizerWordPiece {
		hasPhantomSpace := false
		for _, token := range vocab {
			if strings.HasPrefix(token, "##") {
				rawWordPieceVocab = true
				break
			}
			hasPhantomSpace = hasPhantomSpace || strings.HasPrefix(token, "\u2581")
		}
		if !rawWordPieceVocab {
			rawWordPieceVocab = !hasPhantomSpace
		}
	}
	return &Tokenizer{
		Vocab: vocab, Scores: scores, TokenToID: tokenToID, MergeRanks: mergeRanks,
		ByteEncoder: enc, ByteDecoder: dec, Mode: mode, Pre: preLower,
		AddBOS: addBOS, AddEOS: addEOS, AddSEP: addSEP,
		BOSID: bosID, EOSID: eosID, UNKID: unkID, SEPID: sepID,
		Lowercase: lowercase, StripAccents: stripAccents,
		RawWordPieceVocab: rawWordPieceVocab,
	}, nil
}

func (t *Tokenizer) Encode(text string) []uint32 {
	out := make([]uint32, 0, len(text)/3+2)
	if t.AddBOS {
		out = append(out, t.BOSID)
	}
	out = append(out, t.EncodeWithoutBOS(text)...)
	if t.Mode == TokenizerWordPiece {
		if t.AddSEP {
			out = append(out, t.SEPID)
		}
		if t.AddEOS && (!t.AddSEP || t.EOSID != t.SEPID) {
			out = append(out, t.EOSID)
		}
	}
	return out
}

func (t *Tokenizer) EncodeWithoutBOS(text string) []uint32 {
	if text == "" {
		return nil
	}
	if t.Mode == TokenizerGPT2BPE {
		return t.encodeGPT2BPE(text)
	}
	if t.Mode == TokenizerWordPiece {
		return t.encodeWordPiece(text)
	}
	return t.encodeSentencePiece(text)
}

func (t *Tokenizer) decodeRaw(id uint32) string {
	if int(id) < len(t.Vocab) {
		return t.Vocab[id]
	}
	return ""
}

// DecodeToken renders one token id back to text: GPT-2 tokens go through the
// byte alphabet, SentencePiece <0xNN> byte tokens decode to their raw byte,
// and "▁" markers become spaces. Byte tokens may produce partial UTF-8; the
// generation loop buffers output until it is valid (validUTF8PrefixLen).
func (t *Tokenizer) DecodeToken(id uint32) string {
	raw := t.decodeRaw(id)
	if t.Mode == TokenizerGPT2BPE {
		return t.decodeGPT2Bytes(raw)
	}
	if t.Mode == TokenizerWordPiece && t.RawWordPieceVocab {
		if strings.HasPrefix(raw, "##") {
			return strings.TrimPrefix(raw, "##")
		}
		if (strings.HasPrefix(raw, "[") && strings.HasSuffix(raw, "]")) ||
			(strings.HasPrefix(raw, "<") && strings.HasSuffix(raw, ">")) {
			return raw
		}
		// Raw Hugging Face vocabularies use the absence of ## to mark a new
		// word. Mirror the leading ▁ carried by converted GGUF vocabularies so
		// token-by-token decoding preserves boundaries.
		return " " + raw
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

func (t *Tokenizer) encodeWordPiece(text string) []uint32 {
	words := t.preprocessWordPiece(text)
	out := make([]uint32, 0, len(words))
	for _, word := range words {
		wordStart := len(out)
		pieces := []rune(word)
		if !t.RawWordPieceVocab {
			pieces = append([]rune{'\u2581'}, pieces...)
		}
		for start := 0; start < len(pieces); {
			matchEnd := -1
			var matchID uint32
			for end := len(pieces); end > start; end-- {
				candidate := string(pieces[start:end])
				if t.RawWordPieceVocab && start > 0 {
					candidate = "##" + candidate
				}
				if id, ok := t.TokenToID[candidate]; ok {
					matchEnd, matchID = end, id
					break
				}
			}
			if matchEnd < 0 {
				// WordPiece treats a partially covered word as wholly unknown;
				// retaining its earlier pieces would change every later
				// position compared with the model's training tokenizer.
				out = out[:wordStart]
				out = append(out, t.UNKID)
				break
			}
			out = append(out, matchID)
			start = matchEnd
		}
	}
	return out
}

func (t *Tokenizer) preprocessWordPiece(text string) []string {
	words := make([]string, 0, len(text)/5+1)
	current := make([]rune, 0, 16)
	flush := func() {
		if len(current) > 0 {
			words = append(words, string(current))
			current = current[:0]
		}
	}
	for _, r := range text {
		if unicode.IsSpace(r) || unicode.Is(unicode.Z, r) {
			flush()
			continue
		}
		if r == 0 || r == unicode.ReplacementChar || unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			continue
		}
		if t.Lowercase {
			r = unicode.ToLower(r)
		}
		if t.StripAccents {
			if unicode.Is(unicode.Mn, r) {
				continue
			}
			r = wordpiece.StripAccent(r)
		}
		if unicode.IsPunct(r) || (r < unicode.MaxASCII && unicode.IsSymbol(r)) || isWordPieceCJK(r) {
			flush()
			words = append(words, string(r))
			continue
		}
		current = append(current, r)
	}
	flush()
	return words
}

func isWordPieceCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B920 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
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
	if strings.Contains(t.Pre, "qwen35") {
		return pretokenizeQwen35(text)
	}
	if strings.Contains(t.Pre, "kimi") {
		return pretokenizeKimi(text)
	}
	if strings.Contains(t.Pre, "tekken") {
		return pretokenizeTekken(text)
	}
	return pretokenizeGPT2(text)
}

// pretokenizeQwen35 implements Qwen3.5/3.6's tokenizer.json pattern:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// It is deliberately separate from Qwen2: Qwen35 keeps combining marks with
// their word and emits each numeric rune as a piece. Both details affect BPE
// merge boundaries and therefore prompt token IDs.
func pretokenizeQwen35(text string) []string {
	r := []rune(text)
	pieces := make([]string, 0, len(r)/3+1)
	for i := 0; i < len(r); {
		if end, ok := matchQwen35Contraction(r, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if end, ok := matchQwen35Word(r, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsNumber(r[i]) {
			pieces = append(pieces, string(r[i:i+1]))
			i++
			continue
		}
		if end, ok := matchQwen35Punct(r, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsSpace(r[i]) {
			end := qwen35WhitespaceEnd(r, i)
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		pieces = append(pieces, string(r[i:i+1]))
		i++
	}
	return pieces
}

func qwen35WordRune(c rune) bool {
	return unicode.IsLetter(c) || unicode.Is(unicode.M, c)
}

func matchQwen35Contraction(r []rune, i int) (int, bool) {
	if i >= len(r) || r[i] != '\'' {
		return i, false
	}
	for _, suffix := range []string{"re", "ve", "ll", "s", "t", "m", "d"} {
		if i+1+len(suffix) > len(r) {
			continue
		}
		match := true
		for j := range suffix {
			c := r[i+1+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != rune(suffix[j]) {
				match = false
				break
			}
		}
		if match {
			return i + 1 + len(suffix), true
		}
	}
	return i, false
}

func matchQwen35Word(r []rune, i int) (int, bool) {
	wordEnd := func(start int) int {
		end := start
		for end < len(r) && qwen35WordRune(r[end]) {
			end++
		}
		return end
	}
	// Prefer a word beginning at i. This preserves a leading combining mark,
	// which is a member of [\p{L}\p{M}]+ even though it can also satisfy the
	// optional prefix's broad negated class.
	if end := wordEnd(i); end > i {
		return end, true
	}
	if i < len(r) && r[i] != '\r' && r[i] != '\n' && !unicode.IsLetter(r[i]) && !unicode.IsNumber(r[i]) {
		if end := wordEnd(i + 1); end > i+1 {
			return end, true
		}
	}
	return i, false
}

func matchQwen35Punct(r []rune, i int) (int, bool) {
	j := i
	if j < len(r) && r[j] == ' ' {
		j++
	}
	start := j
	for j < len(r) && !unicode.IsSpace(r[j]) && !qwen35WordRune(r[j]) && !unicode.IsNumber(r[j]) {
		j++
	}
	if j == start {
		return i, false
	}
	for j < len(r) && (r[j] == '\r' || r[j] == '\n') {
		j++
	}
	return j, true
}

func qwen35WhitespaceEnd(r []rune, i int) int {
	end, lastNewline := i, -1
	for end < len(r) && unicode.IsSpace(r[end]) {
		if r[end] == '\r' || r[end] == '\n' {
			lastNewline = end
		}
		end++
	}
	if lastNewline >= 0 {
		return lastNewline + 1
	}
	return end
}

// pretokenizeKimi implements the tiktoken pattern bundled with Kimi K2:
//
//	[\p{Han}]+|[^\r\n\p{L}\p{N}]?[upper]*[lower]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
//	|[^\r\n\p{L}\p{N}]?[upper]+[lower]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
//	|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// where upper/lower have the Unicode classes from Moonshot's
// tokenization_kimi.py. It is intentionally separate from Tekken: Kimi groups
// numbers in threes and treats Han runs as their own pieces, while Tekken
// splits every numeric rune and has different punctuation handling.
func pretokenizeKimi(text string) []string {
	r := []rune(text)
	n := len(r)
	pieces := make([]string, 0, len(r)/3+1)
	for i := 0; i < n; {
		if isKimiHan(r[i]) {
			end := i + 1
			for end < n && isKimiHan(r[end]) {
				end++
			}
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if end, ok := matchKimiWord(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsNumber(r[i]) {
			end := i + 1
			for end < n && end-i < 3 && unicode.IsNumber(r[end]) {
				end++
			}
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if end, ok := matchKimiPunct(r, n, i); ok {
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		if unicode.IsSpace(r[i]) {
			end := kimiWhitespaceEnd(r, n, i)
			pieces = append(pieces, string(r[i:end]))
			i = end
			continue
		}
		pieces = append(pieces, string(r[i]))
		i++
	}
	return pieces
}

func isKimiHan(c rune) bool { return unicode.Is(unicode.Han, c) }

// kimiUpperClass and kimiLowerClass mirror the two case-aware tiktoken
// alternatives. Lm/Lo/marks belong to both classes in the source pattern;
// Han is deliberately excluded because it is handled by the first branch.
func kimiUpperClass(c rune) bool {
	return !isKimiHan(c) && (unicode.IsUpper(c) || unicode.IsTitle(c) || unicode.Is(unicode.Lm, c) || unicode.Is(unicode.Lo, c) || unicode.Is(unicode.M, c))
}

func kimiLowerClass(c rune) bool {
	return !isKimiHan(c) && (unicode.IsLower(c) || unicode.Is(unicode.Lm, c) || unicode.Is(unicode.Lo, c) || unicode.Is(unicode.M, c))
}

func matchKimiWord(r []rune, n, i int) (int, bool) {
	start := i
	if start < n && r[start] != '\r' && r[start] != '\n' && !unicode.IsLetter(r[start]) && !unicode.IsNumber(r[start]) {
		start++
	}
	// First alternative: [upper]*[lower]+.
	k := start
	for k < n && kimiUpperClass(r[k]) {
		k++
	}
	lowerStart := k
	for k < n && kimiLowerClass(r[k]) {
		k++
	}
	if k > lowerStart {
		return kimiContractionEnd(r, n, k), true
	}
	// Second alternative: [upper]+[lower]*.
	k = start
	for k < n && kimiUpperClass(r[k]) {
		k++
	}
	if k > start {
		for k < n && kimiLowerClass(r[k]) {
			k++
		}
		return kimiContractionEnd(r, n, k), true
	}
	return i, false
}

func kimiContractionEnd(r []rune, n, i int) int {
	if i >= n || r[i] != '\'' {
		return i
	}
	for _, suffix := range []string{"re", "ve", "ll", "s", "t", "m", "d"} {
		if i+1+len(suffix) > n {
			continue
		}
		match := true
		for j := range suffix {
			c := r[i+1+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != rune(suffix[j]) {
				match = false
				break
			}
		}
		if match {
			return i + 1 + len(suffix)
		}
	}
	return i
}

func matchKimiPunct(r []rune, n, i int) (int, bool) {
	j := i
	if j < n && r[j] == ' ' {
		j++
	}
	start := j
	for j < n && !unicode.IsSpace(r[j]) && !unicode.IsLetter(r[j]) && !unicode.IsNumber(r[j]) {
		j++
	}
	if j == start {
		return i, false
	}
	for j < n && (r[j] == '\r' || r[j] == '\n') {
		j++
	}
	return j, true
}

func kimiWhitespaceEnd(r []rune, n, i int) int {
	j := i
	lastNewline := -1
	for j < n && unicode.IsSpace(r[j]) {
		if r[j] == '\r' || r[j] == '\n' {
			lastNewline = j
		}
		j++
	}
	if lastNewline >= 0 {
		return lastNewline + 1
	}
	return j
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
