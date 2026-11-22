package engine

import "testing"

func TestPhononTextPiecesKeepsTokenIDs(t *testing.T) {
	labels := []string{"<unk>", "<|de|>", "▁Guten", "<pad>", "▁Morgen", "!"}
	got, err := ParakeetDecodeTokens(phononTextPieces(labels), []int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if got != "Guten Morgen!" {
		t.Fatalf("decoded %q", got)
	}
}
