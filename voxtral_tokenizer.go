package gopherllm

import (
	"encoding/base64"
	"fmt"
	"strings"
)

// voxtralTokenizerFromMetadata builds a Tokenizer from the "voxtral.tokenizer.*"
// metadata convention -- see voxtralTensorNames' doc comment in
// voxtral_realtime.go for why a second GGUF naming convention exists for the
// identical mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint.
// TokenizerFromMetadata falls back to this builder when the standard
// tokenizer.ggml.tokens key is absent.
//
// Layout, verified empirically against a working "stt.*"-convention file of
// the identical checkpoint (not assumed): the full 131072-entry vocabulary
// is two concatenated blocks, SPECIALS FIRST --
//
//   - voxtral.tokenizer.special_token_strings (1000 entries, global IDs
//     0..999): atomic control/special tokens, plain display strings
//     ("<unk>", "<s>", "[INST]", ...; unused slots are named
//     "<SPECIAL_N>"). voxtral.token.bos/eos/audio/begin_audio/
//     streaming_pad are already global IDs into this block directly, e.g.
//     voxtral.token.streaming_pad=32 (verified against the working file's
//     own tokenizer.ggml.tokens[32]=="[STREAMING_PAD]" -- the identical
//     ID, not an ID needing any offset).
//   - voxtral.tokenizer.vocab_token_bytes_b64 (130072 entries, global IDs
//     1000..131071): one base64-encoded raw byte string per token, a
//     tiktoken/tekken-style byte-level BPE vocabulary appended right after
//     the specials block (verified: this array's local index 0 decodes to
//     raw byte 0x00 and lines up with the working file's global ID 1000,
//     whose token text is the corresponding byte-encoded glyph; local
//     index 256 decodes to two raw 0x20 bytes and lines up with the
//     working file's global ID 1256 == "ĠĠ", its byte-encoded double
//     space -- both cross-checks land exactly on the 1000-token offset,
//     not on 0). IDs 0..255 within this block (global 1000..1255) are the
//     256 raw single bytes in order; every local ID from 256 up (global
//     1256+) is a merged multi-byte token. There is no explicit merge list
//     anywhere in the file -- merge priority is implicit in local ID
//     order, exactly like tiktoken's mergeable_ranks dict, so this builder
//     reconstructs an equivalent MergeRanks table (see below).
//
// 1000 + 130072 = 131072 = voxtral.vocab_size, confirming this two-block
// layout is the whole vocabulary, not a subset missing something else.
func voxtralTokenizerFromMetadata(metadata map[string]MetaValue) (*Tokenizer, error) {
	vocabB64Value, ok := metadata["voxtral.tokenizer.vocab_token_bytes_b64"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens")
	}
	vocabB64, ok := vocabB64Value.AsStringArray()
	if !ok {
		return nil, fmt.Errorf("voxtral.tokenizer.vocab_token_bytes_b64 is not a string array")
	}
	specialStrings, _ := metadata["voxtral.tokenizer.special_token_strings"].AsStringArray()

	enc, dec := buildByteMaps()
	encodeBytes := func(b []byte) string {
		var sb strings.Builder
		sb.Grow(len(b))
		for _, c := range b {
			sb.WriteRune(enc[c])
		}
		return sb.String()
	}

	total := len(specialStrings) + len(vocabB64)
	vocab := make([]string, 0, total)
	tokenToID := make(map[string]uint32, total)
	for _, s := range specialStrings {
		// Pure ASCII control-token strings ("<s>", "[INST]", ...) are their
		// own byte encoding under the GPT-2 alphabet (printable ASCII maps
		// to itself), so this is a no-op for the tokens actually seen in
		// this file and only matters if a future export used non-ASCII
		// special-token text.
		text := encodeBytes([]byte(s))
		id := uint32(len(vocab))
		vocab = append(vocab, text)
		if _, exists := tokenToID[text]; !exists {
			tokenToID[text] = id
		}
	}
	byteVocabBase := len(vocab)
	rawBytes := make([][]byte, 0, len(vocabB64))
	for i, s := range vocabB64 {
		raw, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			return nil, fmt.Errorf("voxtral.tokenizer.vocab_token_bytes_b64[%d]: %w", i, err)
		}
		text := encodeBytes(raw)
		id := uint32(len(vocab))
		vocab = append(vocab, text)
		rawBytes = append(rawBytes, raw)
		tokenToID[text] = id
	}

	// Reconstruct BPE merge ranks by replaying each multi-byte token's
	// construction against only the merges discovered so far (i.e. lower
	// local IDs). A well-formed tiktoken-style vocabulary collapses to
	// exactly one remaining pair per token past the first 256 (the raw
	// single bytes, which have no merge to record) -- that pair is this
	// token's merge rule, and its local ID doubles as the rule's rank,
	// since mergeBPESymbols only compares priorities relatively (lower
	// wins) and local IDs are already assigned in merge order.
	mergeRanks := make(map[Pair]int, len(vocabB64))
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
	for localID, raw := range rawBytes {
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
