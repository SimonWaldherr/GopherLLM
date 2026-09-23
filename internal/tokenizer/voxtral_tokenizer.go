package tokenizer

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	gguf "github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
)

// voxtralBuildBPEVocab builds the shared vocabulary/merge-rank structures
// this package's two tiktoken/tekken-style Voxtral tokenizer sources
// (voxtralTokenizerFromMetadata's GGUF "voxtral.tokenizer.*" convention, and
// voxtralTokenizerFromTekkenJSON's official tekken.json release format) both
// need, so the intricate merge-reconstruction algorithm exists once.
//
// Layout, verified empirically against a working "stt.*"-convention GGUF of
// the identical mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint (not
// assumed): the full vocabulary is two concatenated blocks, SPECIALS FIRST
// -- specialStrings (global IDs 0..len(specialStrings)-1, e.g.
// "[STREAMING_PAD]" at a fixed ID verified against a working file's own
// tokenizer.ggml.tokens at that same ID), then rawByteVocab (global IDs
// starting right after) -- one raw byte string per token, a tiktoken-style
// byte-level BPE vocabulary. IDs 0..255 within this block are the 256 raw
// single bytes in order; every ID from 256 up is a merged multi-byte token.
// There is no explicit merge list in either source format -- merge priority
// is implicit in ID order, exactly like tiktoken's mergeable_ranks dict, so
// this reconstructs an equivalent MergeRanks table by replaying each
// multi-byte token's construction against only the merges discovered so far
// (lower IDs): a well-formed vocabulary collapses to exactly one remaining
// pair per token past the first 256, and that pair is the token's merge
// rule, with its ID doubling as the rule's rank since mergeBPESymbols only
// compares priorities relatively (lower wins) and IDs are already assigned
// in merge order.
func voxtralBuildBPEVocab(specialStrings []string, rawByteVocab [][]byte) (vocab []string, tokenToID map[string]uint32, mergeRanks map[Pair]int, enc map[byte]rune, dec map[rune]byte) {
	enc, dec = buildByteMaps()
	encodeBytes := func(b []byte) string {
		var sb strings.Builder
		sb.Grow(len(b))
		for _, c := range b {
			sb.WriteRune(enc[c])
		}
		return sb.String()
	}

	total := len(specialStrings) + len(rawByteVocab)
	vocab = make([]string, 0, total)
	tokenToID = make(map[string]uint32, total)
	for _, s := range specialStrings {
		// Pure ASCII control-token strings ("<s>", "[INST]", ...) are their
		// own byte encoding under the GPT-2 alphabet (printable ASCII maps
		// to itself), so this is a no-op for the tokens actually seen in
		// either source and only matters if a future export used non-ASCII
		// special-token text.
		text := encodeBytes([]byte(s))
		id := uint32(len(vocab))
		vocab = append(vocab, text)
		if _, exists := tokenToID[text]; !exists {
			tokenToID[text] = id
		}
	}
	byteVocabBase := len(vocab)
	for _, raw := range rawByteVocab {
		text := encodeBytes(raw)
		id := uint32(len(vocab))
		vocab = append(vocab, text)
		tokenToID[text] = id
	}

	mergeRanks = make(map[Pair]int, len(rawByteVocab))
	var syms []bpeSymbol
	var heap bpeHeap
	rate := func(l, r *bpeSymbol) (float64, uint32, bool) {
		rank, ok := mergeRanks[Pair{l.text, r.text}]
		if !ok {
			return 0, 0, false
		}
		return float64(rank), 0, true
	}
	joined := func(l, r *bpeSymbol, _ uint32) string { return l.text + r.text }
	for localID, raw := range rawByteVocab {
		if len(raw) <= 1 {
			continue
		}
		syms = bpeSymbolsFromRunes(syms, vocab[byteVocabBase+localID])
		mergeBPESymbols(syms, &heap, rate, joined)
		first := int32(0)
		second := syms[first].next
		if second < 0 || syms[second].next >= 0 {
			// Did not collapse to exactly two symbols under the merges
			// known so far -- not reconstructible as a single pairwise
			// merge. Leave it without a merge rule; it stays reachable by
			// direct vocabulary lookup for an exact match, only BPE
			// fallback segmentation through it is lost.
			continue
		}
		mergeRanks[Pair{syms[first].text, syms[second].text}] = localID
	}
	return vocab, tokenToID, mergeRanks, enc, dec
}

// voxtralTokenizerFromMetadata builds a Tokenizer from the "voxtral.tokenizer.*"
// metadata convention -- see voxtralTensorNames' doc comment in
// voxtral_realtime.go for why a second GGUF naming convention exists for the
// identical mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint.
// TokenizerFromMetadata falls back to this builder when the standard
// tokenizer.ggml.tokens key is absent.
//
// voxtral.token.bos/eos/audio/begin_audio/streaming_pad are already global
// vocabulary IDs (verified: voxtral.token.streaming_pad=32 against a working
// file's own tokenizer.ggml.tokens[32]=="[STREAMING_PAD]", the identical ID,
// not one needing an offset). 1000 (specials) + 130072 (byte vocab) =
// 131072 = voxtral.vocab_size, confirming voxtralBuildBPEVocab's two-block
// layout is the whole vocabulary here, not a subset missing something else.
func voxtralTokenizerFromMetadata(metadata map[string]gguf.MetaValue) (*Tokenizer, error) {
	vocabB64Value, ok := metadata["voxtral.tokenizer.vocab_token_bytes_b64"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens")
	}
	vocabB64, ok := vocabB64Value.AsStringArray()
	if !ok {
		return nil, fmt.Errorf("voxtral.tokenizer.vocab_token_bytes_b64 is not a string array")
	}
	specialStrings, _ := metadata["voxtral.tokenizer.special_token_strings"].AsStringArray()

	rawByteVocab := make([][]byte, len(vocabB64))
	for i, s := range vocabB64 {
		raw, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			return nil, fmt.Errorf("voxtral.tokenizer.vocab_token_bytes_b64[%d]: %w", i, err)
		}
		rawByteVocab[i] = raw
	}
	vocab, tokenToID, mergeRanks, enc, dec := voxtralBuildBPEVocab(specialStrings, rawByteVocab)

	specialID := func(key string) (uint32, bool) {
		v, ok := metadata[key]
		if !ok {
			return 0, false
		}
		n, ok := v.AsU32()
		if !ok || int(n) >= len(specialStrings) {
			return 0, false
		}
		return n, true
	}
	bosID, haveBOS := specialID("voxtral.token.bos")
	eosID, haveEOS := specialID("voxtral.token.eos")
	if !haveBOS {
		bosID = 1
	}
	if !haveEOS {
		eosID = 2
	}
	unkID := uint32(0)
	if id, ok := tokenToID["<unk>"]; ok {
		unkID = id
	}

	return &Tokenizer{
		Vocab: vocab, TokenToID: tokenToID, MergeRanks: mergeRanks,
		ByteEncoder: enc, ByteDecoder: dec, Mode: TokenizerGPT2BPE, Pre: "tekken",
		AddBOS: true, AddEOS: false,
		BOSID: bosID, EOSID: eosID, UNKID: unkID,
	}, nil
}

// voxtralTekkenFile is tekken.json's schema, as shipped in
// mistralai/Voxtral-Mini-4B-Realtime-2602's official Hugging Face release
// (and every other Mistral model using the Tekken tokenizer). Fetched and
// inspected directly: {config:{pattern, num_vocab_tokens, default_vocab_size,
// default_num_special_tokens, version}, vocab:[{rank, token_bytes (base64),
// token_str}], special_tokens:[{rank, token_str, is_control}]}. vocab has
// num_vocab_tokens entries (150000 in the checked release) but only the
// first default_vocab_size-default_num_special_tokens are part of the
// active vocabulary -- the rest is unused headroom reserved for future
// growth, confirmed by default_vocab_size (131072) exactly matching
// params.json's vocab_size.
type voxtralTekkenFile struct {
	Config struct {
		DefaultVocabSize        int `json:"default_vocab_size"`
		DefaultNumSpecialTokens int `json:"default_num_special_tokens"`
	} `json:"config"`
	Vocab []struct {
		TokenBytes string `json:"token_bytes"`
	} `json:"vocab"`
	SpecialTokens []struct {
		TokenStr string `json:"token_str"`
	} `json:"special_tokens"`
}

// voxtralTokenizerFromTekkenJSON builds a Tokenizer from the official
// tekken.json release format, structurally the same tiktoken-style
// byte-vocabulary as the GGUF "voxtral.tokenizer.*" convention (see
// voxtralBuildBPEVocab) but with special tokens and byte-vocab entries each
// carrying an explicit "rank" field (redundant with array position in every
// file inspected, so not separately consulted) rather than needing this
// package's own ID-offset bookkeeping.
func voxtralTokenizerFromTekkenJSON(data []byte) (*Tokenizer, error) {
	var tek voxtralTekkenFile
	if err := json.Unmarshal(data, &tek); err != nil {
		return nil, fmt.Errorf("parsing tekken.json: %w", err)
	}
	numSpecial := tek.Config.DefaultNumSpecialTokens
	if numSpecial <= 0 || numSpecial > len(tek.SpecialTokens) {
		numSpecial = len(tek.SpecialTokens)
	}
	vocabSize := tek.Config.DefaultVocabSize
	byteVocabLen := vocabSize - numSpecial
	if byteVocabLen <= 0 || byteVocabLen > len(tek.Vocab) {
		return nil, fmt.Errorf("tekken.json: vocab size mismatch (default_vocab_size=%d default_num_special_tokens=%d vocab entries=%d special entries=%d)",
			vocabSize, numSpecial, len(tek.Vocab), len(tek.SpecialTokens))
	}

	specialStrings := make([]string, numSpecial)
	for i, s := range tek.SpecialTokens[:numSpecial] {
		specialStrings[i] = s.TokenStr
	}
	rawByteVocab := make([][]byte, byteVocabLen)
	for i := 0; i < byteVocabLen; i++ {
		raw, err := base64.StdEncoding.DecodeString(tek.Vocab[i].TokenBytes)
		if err != nil {
			return nil, fmt.Errorf("tekken.json vocab[%d]: %w", i, err)
		}
		rawByteVocab[i] = raw
	}
	vocab, tokenToID, mergeRanks, enc, dec := voxtralBuildBPEVocab(specialStrings, rawByteVocab)

	// Every Mistral tekken.json/GGUF export inspected so far fixes <unk>=0,
	// <s>=1 (BOS), </s>=2 (EOS) as the first three special-token slots; this
	// file format carries no separate bos_token_id/eos_token_id field of
	// its own the way params.json/config.json do; verified directly:
	// tek.SpecialTokens[1].TokenStr == "<s>", [2] == "</s>".
	bosID, eosID := uint32(1), uint32(2)
	unkID := uint32(0)
	if id, ok := tokenToID["<unk>"]; ok {
		unkID = id
	}

	return &Tokenizer{
		Vocab: vocab, TokenToID: tokenToID, MergeRanks: mergeRanks,
		ByteEncoder: enc, ByteDecoder: dec, Mode: TokenizerGPT2BPE, Pre: "tekken",
		AddBOS: true, AddEOS: false,
		BOSID: bosID, EOSID: eosID, UNKID: unkID,
	}, nil
}
