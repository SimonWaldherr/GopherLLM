// Package jsonconstraint implements a bounded incremental JSON-object grammar.
// State values are copyable, allowing candidate tokens to be tested without
// changing the accepted prefix. It has no tokenizer or protocol dependency.
package jsonconstraint

import "unicode/utf8"

type State struct {
	stack   [64]byte
	depth   int
	started bool
	lex     byte
	key     bool
	literal string
	at      int
	number  byte
	escape  bool
	hex     int
	utf     [4]byte
	utfN    int
}

func (s State) Complete() bool { return s.started && s.depth == 0 && s.lex == 0 }
func space(c byte) bool        { return c == ' ' || c == '\n' || c == '\r' || c == '\t' }
func digit(c byte) bool        { return c >= '0' && c <= '9' }
func (s State) Advance(text string) (State, bool) {
	for i := 0; i < len(text); i++ {
		if !s.consume(text[i]) {
			return State{}, false
		}
	}
	return s, true
}
func (s *State) push(c byte) bool {
	if s.depth == len(s.stack) {
		return false
	}
	s.stack[s.depth] = c
	s.depth++
	return true
}
func (s *State) consume(c byte) bool {
	if s.lex == 's' {
		if s.hex > 0 {
			if !(digit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
				return false
			}
			s.hex--
			return true
		}
		if s.escape {
			s.escape = false
			if c == 'u' {
				s.hex = 4
				return true
			}
			return c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't'
		}
		if s.utfN > 0 || c >= 128 {
			if s.utfN == 4 {
				return false
			}
			s.utf[s.utfN] = c
			s.utfN++
			if utf8.FullRune(s.utf[:s.utfN]) {
				r, n := utf8.DecodeRune(s.utf[:s.utfN])
				if r == utf8.RuneError && n == 1 {
					return false
				}
				s.utfN = 0
			}
			return true
		}
		if c < 32 {
			return false
		}
		if c == '\\' {
			s.escape = true
			return true
		}
		if c == '"' {
			s.lex = 0
			if s.key {
				s.stack[s.depth-1] = ':'
			}
		}
		return true
	}
	if s.lex == 'l' {
		if c != s.literal[s.at] {
			return false
		}
		s.at++
		if s.at == len(s.literal) {
			s.lex = 0
		}
		return true
	}
	if s.lex == 'n' {
		switch s.number {
		case '-':
			if c == '0' {
				s.number = '0'
				return true
			}
			if c >= '1' && c <= '9' {
				s.number = 'i'
				return true
			}
			return false
		case '0', 'i':
			if digit(c) {
				if s.number == '0' {
					return false
				}
				return true
			}
			if c == '.' {
				s.number = '.'
				return true
			}
			if c == 'e' || c == 'E' {
				s.number = 'e'
				return true
			}
		case '.':
			if digit(c) {
				s.number = 'f'
				return true
			}
			return false
		case 'f':
			if digit(c) {
				return true
			}
			if c == 'e' || c == 'E' {
				s.number = 'e'
				return true
			}
		case 'e':
			if c == '+' || c == '-' {
				s.number = '+'
				return true
			}
			if digit(c) {
				s.number = 'x'
				return true
			}
			return false
		case '+':
			if digit(c) {
				s.number = 'x'
				return true
			}
			return false
		case 'x':
			if digit(c) {
				return true
			}
		}
		s.lex = 0 // delimiter belongs to container
	}
	if space(c) {
		return true
	}
	if !s.started {
		if c != '{' {
			return false
		}
		s.started = true
		return s.push('k')
	}
	if s.depth == 0 {
		return false
	}
	top := &s.stack[s.depth-1]
	switch *top {
	case 'k', 'K':
		if c == '}' && *top == 'k' {
			s.depth--
			return true
		}
		if c != '"' {
			return false
		}
		s.lex = 's'
		s.key = true
		return true
	case ':':
		if c != ':' {
			return false
		}
		*top = 'v'
		return true
	case 'o':
		if c == '}' {
			s.depth--
			return true
		}
		if c != ',' {
			return false
		}
		*top = 'K'
		return true
	case 'a':
		if c == ']' {
			s.depth--
			return true
		}
		if c != ',' {
			return false
		}
		*top = 'V'
		return true
	case 'v', 'V', 'A':
		if *top == 'A' && c == ']' {
			s.depth--
			return true
		}
		if *top == 'v' {
			*top = 'o'
		} else {
			*top = 'a'
		}
		switch c {
		case '{':
			return s.push('k')
		case '[':
			return s.push('A')
		case '"':
			s.lex = 's'
			s.key = false
			return true
		case 't':
			s.lex = 'l'
			s.literal = "true"
			s.at = 1
			return true
		case 'f':
			s.lex = 'l'
			s.literal = "false"
			s.at = 1
			return true
		case 'n':
			s.lex = 'l'
			s.literal = "null"
			s.at = 1
			return true
		case '-':
			s.lex = 'n'
			s.number = '-'
			return true
		case '0':
			s.lex = 'n'
			s.number = '0'
			return true
		default:
			if c >= '1' && c <= '9' {
				s.lex = 'n'
				s.number = 'i'
				return true
			}
		}
	}
	return false
}
