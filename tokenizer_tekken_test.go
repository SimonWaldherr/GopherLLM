package main

import (
	"reflect"
	"testing"
)

func TestPretokenizeTekken(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"iPhone", []string{"i", "Phone"}},
		{"HELLO", []string{"HELLO"}},
		{" 42", []string{" ", "4", "2"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		{"Hello, World!", []string{"Hello", ",", " World", "!"}},
		{"hi ", []string{"hi", " "}},
		{"  ", []string{"  "}},
		{"3.14", []string{"3", ".", "1", "4"}},
	}
	for _, c := range cases {
		got := pretokenizeTekken(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("pretokenizeTekken(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPretokenizeDispatch(t *testing.T) {
	tek := &Tokenizer{Pre: "tekken"}
	if got := tek.pretokenize("a1"); !reflect.DeepEqual(got, []string{"a", "1"}) {
		t.Fatalf("tekken dispatch = %q, want [a 1]", got)
	}
	// Non-tekken GPT-2 keeps grouped digits.
	gpt := &Tokenizer{Pre: "qwen2"}
	if got := gpt.pretokenize("a12"); !reflect.DeepEqual(got, []string{"a", "12"}) {
		t.Fatalf("gpt2 dispatch = %q, want [a 12]", got)
	}
}
