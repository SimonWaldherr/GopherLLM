package gopherllm

import "testing"

func TestEncodeWithoutBOSEmpty(t *testing.T) {
	tok := newInstTestTokenizer()
	if got := tok.EncodeWithoutBOS(""); got != nil {
		t.Fatalf("empty encode = %v, want nil", got)
	}
}

func TestSpecialIDLookup(t *testing.T) {
	tok := newInstTestTokenizer()
	if id, ok := tok.SpecialID("[INST]"); !ok || id != tok.TokenToID["[INST]"] {
		t.Fatalf("SpecialID([INST]) = %d,%v", id, ok)
	}
	if _, ok := tok.SpecialID("[NOPE]"); ok {
		t.Fatal("SpecialID for missing token should be false")
	}
}
