package tokenizer

import (
	"strconv"
	"strings"
	"sync"
)

var nfcOnce sync.Once
var nfcDecomp map[rune][]rune
var nfcCCC map[rune]int
var nfcCompose map[[2]rune]rune

func loadNFCTables() {
	nfcDecomp = map[rune][]rune{}
	nfcCCC = map[rune]int{}
	nfcCompose = map[[2]rune]rune{}
	number := func(s string) rune {
		v, e := strconv.ParseInt(s, 16, 32)
		if e != nil {
			panic(e)
		}
		return rune(v)
	}
	for _, entry := range strings.Fields(nfcDecompositions) {
		a, b, _ := strings.Cut(entry, ":")
		for _, v := range strings.Split(b, ",") {
			nfcDecomp[number(a)] = append(nfcDecomp[number(a)], number(v))
		}
	}
	for _, entry := range strings.Fields(nfcClasses) {
		a, b, _ := strings.Cut(entry, ":")
		nfcCCC[number(a)] = int(number(b))
	}
	for _, entry := range strings.Fields(nfcCompositions) {
		a, b, _ := strings.Cut(entry, ":")
		x, y, _ := strings.Cut(a, ",")
		nfcCompose[[2]rune{number(x), number(y)}] = number(b)
	}
}
func normalizeNFC(s string) string {
	ascii := true
	for i := range len(s) {
		if s[i] >= 128 {
			ascii = false
			break
		}
	}
	if ascii {
		return s
	}
	nfcOnce.Do(loadNFCTables)
	var d []rune
	var expand func(rune)
	expand = func(r rune) {
		if r >= 0xac00 && r < 0xac00+11172 {
			v := r - 0xac00
			expand(0x1100 + v/588)
			expand(0x1161 + (v%588)/28)
			if v%28 != 0 {
				expand(0x11a7 + v%28)
			}
			return
		}
		if seq, ok := nfcDecomp[r]; ok {
			for _, v := range seq {
				expand(v)
			}
			return
		}
		d = append(d, r)
		c := nfcCCC[r]
		if c != 0 {
			for i := len(d) - 1; i > 0 && nfcCCC[d[i-1]] > c; i-- {
				d[i], d[i-1] = d[i-1], d[i]
			}
		}
	}
	for _, r := range s {
		expand(r)
	}
	if len(d) == 0 {
		return ""
	}
	out := []rune{d[0]}
	starter := 0
	last := nfcCCC[d[0]]
	for _, r := range d[1:] {
		c := nfcCCC[r]
		a := out[starter]
		joined, ok := nfcCompose[[2]rune{a, r}]
		if a >= 0x1100 && a < 0x1100+19 && r >= 0x1161 && r < 0x1161+21 {
			joined = 0xac00 + ((a-0x1100)*21+r-0x1161)*28
			ok = true
		}
		if a >= 0xac00 && a < 0xac00+11172 && (a-0xac00)%28 == 0 && r > 0x11a7 && r < 0x11a7+28 {
			joined = a + r - 0x11a7
			ok = true
		}
		if ok && (last == 0 || last < c) {
			out[starter] = joined
			continue
		}
		if c == 0 {
			starter = len(out)
		}
		out = append(out, r)
		last = c
	}
	return string(out)
}
