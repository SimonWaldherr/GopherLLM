package tokenizer

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestLayaTokenizerReference(t *testing.T) {
	b, e := os.ReadFile("../../testdata/laya-tiny/tokenizer/tokenizer.json")
	if e != nil {
		t.Fatal(e)
	}
	c, e := os.ReadFile("../../testdata/laya-tiny/tokenizer/tokenizer_config.json")
	if e != nil {
		t.Fatal(e)
	}
	tok, e := NewLayaTokenizer(b, c)
	if e != nil {
		t.Fatal(e)
	}
	a, e := tok.Encode("Café")
	if e != nil {
		t.Fatal(e)
	}
	d, e := tok.Encode("Cafe\u0301")
	if e != nil || !reflect.DeepEqual(a, d) {
		t.Fatalf("NFC: %v %v %v", a, d, e)
	}
}
func TestLayaRealTokenizers(t *testing.T) {
	for _, name := range []string{"ENGLISH", "MULTILINGUAL"} {
		t.Run(name, func(t *testing.T) {
			prefix := os.Getenv("GOPHERLLM_LAYA_TOKENIZER_" + name)
			if prefix == "" {
				t.Skip("optional upstream tokenizer parity test")
			}
			b, e := os.ReadFile(prefix + ".json")
			if e != nil {
				t.Fatal(e)
			}
			c, e := os.ReadFile(prefix + "-config.json")
			if e != nil {
				t.Fatal(e)
			}
			tok, e := NewLayaTokenizer(b, c)
			if e != nil {
				t.Fatal(e)
			}
			golden, e := os.ReadFile(prefix + "-golden.json")
			if e != nil {
				t.Fatal(e)
			}
			var refs []struct {
				Text string
				IDs  []uint32
			}
			if e = json.Unmarshal(golden, &refs); e != nil {
				t.Fatal(e)
			}
			for _, r := range refs {
				ids, e := tok.Encode(r.Text)
				if e != nil || !reflect.DeepEqual(ids, r.IDs) {
					t.Errorf("%q: got %v, want %v (%v)", r.Text, ids, r.IDs, e)
				}
			}
		})
	}
}
func TestNormalizeNFC(t *testing.T) {
	for _, p := range [][2]string{{"Cafe\u0301", "Café"}, {"A\u030a\u0301", "Ǻ"}, {"\u1100\u1161\u11a8", "각"}, {"\u212b", "Å"}, {"a\u0315\u0300", "à\u0315"}, {"\u0344", "\u0308\u0301"}} {
		if got := normalizeNFC(p[0]); got != p[1] {
			t.Errorf("%q => %q, want %q", p[0], got, p[1])
		}
	}
}

func TestLayaByteLevelRegex(t *testing.T) {
	for _, tc := range []struct {
		text   string
		pieces []string
	}{
		{"I can't do it. We'll check.", []string{"I", " can", "'t", " do", " it", ".", " We", "'ll", " check", "."}},
		{"hello\n\nworld", []string{"hello", "\n", "\n", "world"}},
		{"foo\r\nbar", []string{"foo", "\r", "\n", "bar"}},
		{"a   b ", []string{"a", "  ", " b", " "}},
		{"你好123!", []string{"你好", "123", "!"}},
	} {
		if got := layaByteLevelPieces(tc.text); !reflect.DeepEqual(got, tc.pieces) {
			t.Errorf("%q: %q, want %q", tc.text, got, tc.pieces)
		}
	}
}
func TestLayaMetaspaceAndByteFallback(t *testing.T) {
	raw := []byte(`{"model":{"type":"BPE","vocab":{"▁":0,"a":1,"b":2,"▁a":3,"<cls>":4,"<sep>":5,"<mask>":6,"<0xF0>":7,"<0x9F>":8,"<0x99>":9,"<0x82>":10},"merges":[["▁","a"]],"byte_fallback":true},"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"added_tokens":[{"id":6,"content":"<mask>","lstrip":true}]}`)
	tok, e := NewLayaTokenizer(raw, []byte(`{"cls_token":"<cls>","sep_token":"<sep>","mask_token":"<mask>"}`))
	if e != nil {
		t.Fatal(e)
	}
	for _, tc := range []struct {
		text string
		ids  []uint32
	}{{"a b", []uint32{3, 0, 2}}, {"🙂", []uint32{0, 7, 8, 9, 10}}, {"a  <mask>b", []uint32{3, 6, 0, 2}}} {
		ids, e := tok.Encode(tc.text)
		if e != nil || !reflect.DeepEqual(ids, tc.ids) {
			t.Errorf("%q: %v want %v (%v)", tc.text, ids, tc.ids, e)
		}
	}
}
