package main

import "testing"

func TestEvaluateArithmetic(t *testing.T) {
	cases := []struct {
		expr string
		want float64
	}{
		{"1 + 2", 3},
		{"1234 * 5678 + 42", 7006694},
		{"(3 + 4) * 2", 14},
		{"10 - 3 - 2", 5},   // left-associative
		{"2 + 3 * 4", 14},   // precedence: * before +
		{"(2 + 3) * 4", 20}, // parens override precedence
		{"10 / 4", 2.5},     // float result
		{"-5 + 3", -2},      // leading unary minus
		{"3 - -2", 5},       // unary minus as an operand
		{"+5", 5},           // leading unary plus
		{"0.5 * 4", 2},      // float literal
		{"((1 + 2) * (3 + 4))", 21},
	}
	for _, c := range cases {
		got, err := evaluate(c.expr)
		if err != nil {
			t.Fatalf("evaluate(%q) returned error: %v", c.expr, err)
		}
		if got != c.want {
			t.Fatalf("evaluate(%q) = %v, want %v", c.expr, got, c.want)
		}
	}
}

func TestEvaluateRejectsNonArithmeticSyntax(t *testing.T) {
	cases := []string{
		"os.Exit(1)",     // selector + call must not execute anything
		"someIdentifier", // bare identifier has no numeric value
		`"a string"`,     // not a number
		"1 + ",           // malformed: parser.ParseExpr itself rejects this
		"[]int{1}",       // composite literal
		"func() {}",      // function literal
		"1 << 2",         // bitwise ops are not in the supported set
		"1 % 2",          // modulo is not in the supported set
	}
	for _, expr := range cases {
		if _, err := evaluate(expr); err == nil {
			t.Fatalf("evaluate(%q) = nil error, want a rejection", expr)
		}
	}
}

func TestEvaluateDivisionByZero(t *testing.T) {
	if _, err := evaluate("1 / 0"); err == nil {
		t.Fatal("evaluate(\"1 / 0\") = nil error, want a division-by-zero error")
	}
}

func TestFormatResultAvoidsScientificNotation(t *testing.T) {
	cases := []struct {
		in   float64
		want string
	}{
		{7006694, "7006694"},
		{2.5, "2.5"},
		{-12, "-12"},
		{0, "0"},
		{1234567890, "1234567890"},
	}
	for _, c := range cases {
		if got := formatResult(c.in); got != c.want {
			t.Errorf("formatResult(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestEvaluateMatchesToolDescription(t *testing.T) {
	// The tool's own description promises +, -, *, /, and parentheses; pin
	// that every one of those actually works end to end, not just in
	// isolation from the cases above.
	got, err := evaluate("(3 + 4) * 2 - 8 / 4")
	if err != nil {
		t.Fatal(err)
	}
	if want := float64(12); got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}
