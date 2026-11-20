package tokenizer

import (
	"slices"
	"testing"
)

func TestPretokenizeLlamaBPE(t *testing.T) {
	for _, tc := range []struct {
		text string
		want []string
	}{
		{"ABC1234567 we're  !word", []string{"ABC", "123", "456", "7", " we", "'re", " ", " !", "word"}},
		{"it's\t \n\n Hello?\n", []string{"it", "'s", "\t \n\n", " Hello", "?\n"}},
		{"\n  hello   ", []string{"\n", " ", " hello", "   "}},
		{"any  ", []string{"any", "  "}},
	} {
		if got := pretokenizeLlamaQwenBPE(tc.text, 3); !slices.Equal(got, tc.want) {
			t.Errorf("%q: got %q, want %q", tc.text, got, tc.want)
		}
	}
}
