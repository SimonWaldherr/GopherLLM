package server

import (
	"strings"
	"testing"
)

// The model writes these queries, and a model working from fetched text is
// working from text a stranger can influence. A guard that only checked the
// leading keyword let an update ride along after a semicolon.
func TestSPARQLGuardRejectsUpdatesAfterALeadingSelect(t *testing.T) {
	for _, query := range []string{
		"SELECT ?s WHERE { ?s ?p ?o } ; DROP GRAPH <g>",
		"SELECT ?s WHERE { ?s ?p ?o } ; CLEAR ALL",
		"SELECT ?s WHERE { ?s ?p ?o } ; CREATE GRAPH <g>",
		"SELECT ?s WHERE { ?s ?p ?o } ; COPY <a> TO <b>",
		"SELECT ?s WHERE { ?s ?p ?o } ; MOVE <a> TO <b>",
		"SELECT ?s WHERE { ?s ?p ?o } ; ADD <a> TO <b>",
		"SELECT ?s WHERE { ?s ?p ?o } ; INSERT DATA { <a> <b> <c> }",
		"SELECT ?s WHERE { ?s ?p ?o } ; DELETE WHERE { ?s ?p ?o }",
		"SELECT ?s WHERE { ?s ?p ?o } LOAD <http://evil/>",
		"SELECT ?s WHERE { SERVICE <http://evil/sparql> { ?s ?p ?o } }",
	} {
		if err := checkReadOnlySPARQL(query); err == nil {
			t.Errorf("accepted %q", query)
		}
	}
}

// A keyword hidden behind a comment must not slip past a substring scan.
func TestSPARQLGuardStripsCommentsBeforeScanning(t *testing.T) {
	for _, query := range []string{
		"# harmless\nDROP GRAPH <g>",
		"SELECT ?s WHERE { ?s ?p ?o } # ok\n; DELETE WHERE { ?s ?p ?o }",
	} {
		if err := checkReadOnlySPARQL(query); err == nil {
			t.Errorf("accepted %q", query)
		}
	}
}

func TestSPARQLGuardAcceptsOrdinaryReadOnlyQueries(t *testing.T) {
	for _, query := range []string{
		"SELECT ?label WHERE { wd:Q1 rdfs:label ?label . FILTER(LANG(?label) = \"de\") }",
		"ASK { wd:Q1 ?p ?o }",
		"PREFIX wd: <http://www.wikidata.org/entity/>\nSELECT ?x WHERE { ?x ?p ?o } LIMIT 5",
		// A variable that merely contains a banned keyword is fine, because
		// matching is whole-word.
		"SELECT ?address ?adder WHERE { ?address ?p ?adder }",
		// SERVICE wikibase:label is the standard, near-essential pattern for
		// human-readable labels instead of bare Q-IDs -- it must not be
		// banned along with genuinely federated (SSRF-risk) SERVICE calls.
		"SELECT ?countryLabel WHERE { wd:Q183 wdt:P463 ?country. SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } }",
		"SELECT ?s WHERE { SERVICE SILENT wikibase:label { ?s ?p ?o } }",
	} {
		if err := checkReadOnlySPARQL(query); err != nil {
			t.Errorf("rejected legitimate query %q: %v", query, err)
		}
	}
}

// TestSPARQLGuardStillRejectsExternalServiceFederation guards the boundary
// of the SERVICE allowance above: only Wikidata's own wikibase: services are
// permitted. A SERVICE clause naming an arbitrary external URI can make
// Wikidata's own query engine issue a request to a server the caller
// controls (SSRF), so that must stay blocked even though wikibase:label now
// is not.
func TestSPARQLGuardStillRejectsExternalServiceFederation(t *testing.T) {
	for _, query := range []string{
		"SELECT ?s WHERE { SERVICE <http://evil.example/sparql> { ?s ?p ?o } }",
		"SELECT ?s WHERE { SERVICE ?endpoint { ?s ?p ?o } }",
	} {
		if err := checkReadOnlySPARQL(query); err == nil {
			t.Errorf("accepted external SERVICE federation %q", query)
		}
	}
}

// The language code becomes a hostname, so only vetted ones are accepted.
func TestWikiLanguageIsAllowListed(t *testing.T) {
	for _, lang := range wikiLanguages {
		if got, err := wikiLanguage(lang); err != nil || got != lang {
			t.Errorf("wikiLanguage(%q) = %q, %v", lang, got, err)
		}
	}
	if got, err := wikiLanguage(""); err != nil || got != "de" {
		t.Errorf("empty language should default to de, got %q, %v", got, err)
	}
	for _, lang := range []string{"ru", "zz", "en-evil", "commons", "a-very-long-code"} {
		if _, err := wikiLanguage(lang); err == nil {
			t.Errorf("accepted unvetted language %q", lang)
		} else if !strings.Contains(err.Error(), "unsupported") {
			t.Errorf("unexpected error for %q: %v", lang, err)
		}
	}
}
