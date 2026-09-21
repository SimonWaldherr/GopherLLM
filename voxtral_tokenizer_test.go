package gopherllm

import (
	"encoding/base64"
	"encoding/json"
	"testing"
)

// buildTestTekkenJSON assembles a minimal but schema-accurate tekken.json
// fixture: 3 special tokens, then a byte vocabulary of 'a','b','c' as single
// bytes followed by one merged two-byte token "ab" -- enough to exercise
// vocab layout, BOS/EOS resolution, and one real BPE merge without needing
// the full 256-byte alphabet a real release ships.
func buildTestTekkenJSON(t *testing.T) []byte {
	t.Helper()
	b64 := func(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }
	doc := map[string]any{
		"config": map[string]any{
			"default_vocab_size":         7, // 3 specials + 4 byte-vocab entries
			"default_num_special_tokens": 3,
		},
		"special_tokens": []map[string]any{
			{"rank": 0, "token_str": "<unk>", "is_control": true},
			{"rank": 1, "token_str": "<s>", "is_control": true},
			{"rank": 2, "token_str": "</s>", "is_control": true},
		},
		"vocab": []map[string]any{
			{"rank": 0, "token_bytes": b64("a"), "token_str": "a"},
			{"rank": 1, "token_bytes": b64("b"), "token_str": "b"},
			{"rank": 2, "token_bytes": b64("c"), "token_str": "c"},
			{"rank": 3, "token_bytes": b64("ab"), "token_str": "ab"},
		},
	}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestVoxtralTokenizerFromTekkenJSONLayoutAndSpecialIDs(t *testing.T) {
	tok, err := voxtralTokenizerFromTekkenJSON(buildTestTekkenJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(tok.Vocab) != 7 {
		t.Fatalf("len(Vocab) = %d, want 7 (3 specials + 4 byte-vocab)", len(tok.Vocab))
	}
	if tok.BOSID != 1 || tok.EOSID != 2 {
		t.Fatalf("BOSID=%d EOSID=%d, want 1/2", tok.BOSID, tok.EOSID)
	}
	if tok.UNKID != 0 {
		t.Fatalf("UNKID = %d, want 0", tok.UNKID)
	}
	// Specials occupy IDs 0..2, byte vocab starts right after at ID 3.
	if tok.Vocab[0] != "<unk>" || tok.Vocab[1] != "<s>" || tok.Vocab[2] != "</s>" {
		t.Fatalf("special tokens not at expected IDs: %q", tok.Vocab[:3])
	}
	if tok.Vocab[3] != "a" || tok.Vocab[4] != "b" || tok.Vocab[5] != "c" {
		t.Fatalf("byte vocab not at expected IDs: %q", tok.Vocab[3:6])
	}
}

func TestVoxtralTokenizerFromTekkenJSONReconstructsMerge(t *testing.T) {
	tok, err := voxtralTokenizerFromTekkenJSON(buildTestTekkenJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	// "ab" must merge into the single dedicated token (ID 6), not stay as
	// two single-byte tokens ('a'=ID3, 'b'=ID4) -- this is the actual thing
	// voxtralBuildBPEVocab's merge-rank reconstruction exists to get right.
	got := tok.EncodeWithoutBOS("ab")
	want := []uint32{6}
	if len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("EncodeWithoutBOS(\"ab\") = %v, want %v (the merged token, not [3 4])", got, want)
	}
	// "c" alone has no merge partner and must fall back to its own token.
	got = tok.EncodeWithoutBOS("c")
	if len(got) != 1 || got[0] != 5 {
		t.Fatalf("EncodeWithoutBOS(\"c\") = %v, want [5]", got)
	}
}

func TestVoxtralTokenizerFromTekkenJSONRejectsMismatchedSizes(t *testing.T) {
	if _, err := voxtralTokenizerFromTekkenJSON([]byte(`{"config":{"default_vocab_size":1000,"default_num_special_tokens":3},"special_tokens":[],"vocab":[]}`)); err == nil {
		t.Fatal("expected an error when default_vocab_size exceeds the available vocab/special entries")
	}
}
