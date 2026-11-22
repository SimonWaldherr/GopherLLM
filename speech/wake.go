package speech

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// WakePhrase matches complete leading words, ignoring casing and punctuation
// inserted by speech recognition. A zero phrase never matches.
type WakePhrase struct {
	words [5]string
	count int
}

// ParseWakePhrase accepts one to five words with at most 64 characters.
// An empty phrase disables matching.
func ParseWakePhrase(text string) (WakePhrase, error) {
	var phrase WakePhrase
	if text == "" {
		return phrase, nil
	}
	if len(text) > 64*utf8.UTFMax || utf8.RuneCountInString(text) > 64 {
		return WakePhrase{}, errors.New("activation phrase exceeds 64 characters")
	}
	for _, r := range text {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != ' ' && r != '-' {
			return WakePhrase{}, errors.New("activation phrase can contain letters, digits, spaces and hyphens")
		}
	}
	for text != "" {
		word, rest := nextWord(text)
		if word == "" {
			break
		}
		if phrase.count == len(phrase.words) {
			return WakePhrase{}, errors.New("activation phrase can contain at most five words")
		}
		phrase.words[phrase.count] = word
		phrase.count++
		text = rest
	}
	if phrase.count == 0 {
		return WakePhrase{}, errors.New("activation phrase must contain at least one word")
	}
	return phrase, nil
}

// Match returns the command after a matching activation phrase. Unmatched
// transcripts and wake-only utterances produce an empty command.
func (p WakePhrase) Match(transcript string) (command string, matched bool) {
	if p.count == 0 {
		return "", false
	}
	for _, want := range p.words[:p.count] {
		word, rest := nextWord(transcript)
		if !strings.EqualFold(word, want) {
			return "", false
		}
		transcript = rest
	}
	return strings.TrimSpace(strings.TrimLeftFunc(transcript, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})), true
}

func nextWord(text string) (word, rest string) {
	start := -1
	for i, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			return text[start:i], text[i:]
		}
	}
	if start >= 0 {
		return text[start:], ""
	}
	return "", ""
}
