package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const wikimediaUserAgent = "GopherLLM/1.0 (local research tool; https://github.com/SimonWaldherr/GopherLLM)"

var wikidataID = regexp.MustCompile(`^Q[1-9][0-9]{0,9}$`)

type wikimediaClient struct {
	httpClient      *http.Client
	wikipediaBase   string
	wikidataAPIBase string
	sparqlBase      string
}

type wikidataEntityResponse struct {
	Entities map[string]wikidataEntity `json:"entities"`
}

type wikidataEntity struct {
	Labels       map[string]wikidataText        `json:"labels"`
	Descriptions map[string]wikidataText        `json:"descriptions"`
	Claims       map[string][]wikidataStatement `json:"claims"`
}

type wikidataText struct {
	Value string `json:"value"`
}

type wikidataStatement struct {
	Mainsnak struct {
		Datavalue struct {
			Value any `json:"value"`
		} `json:"datavalue"`
	} `json:"mainsnak"`
}

type sparqlResponse struct {
	Boolean *bool `json:"boolean"`
	Results struct {
		Bindings []map[string]sparqlValue `json:"bindings"`
	} `json:"results"`
}

type sparqlValue struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

func newWikimediaClient(client *http.Client) *wikimediaClient {
	if client == nil {
		client = &http.Client{Timeout: 12 * time.Second}
	}
	return &wikimediaClient{
		httpClient:      client,
		wikipediaBase:   "https://%s.wikipedia.org",
		wikidataAPIBase: "https://www.wikidata.org/w/api.php",
		sparqlBase:      "https://query.wikidata.org/sparql",
	}
}

func (c *wikimediaClient) tools() []gopherllm.AgenticTool {
	return []gopherllm.AgenticTool{
		{Definition: wikiToolDefinition("wikipedia_search", "Search Wikipedia for relevant articles. Use this first when you need factual background or a suitable article title.", `{"type":"object","properties":{"query":{"type":"string","description":"Search terms"},"language":{"type":"string","description":"Wikipedia language code, e.g. de or en"}},"required":["query"]}`), Execute: c.search},
		{Definition: wikiToolDefinition("wikipedia_summary", "Retrieve a Wikipedia article's short LEAD PARAGRAPH ONLY (like a search-engine snippet) — not the article body, and not any list, table, or enumeration the article contains. For a \"List of ...\" article this typically returns just a sentence stating how many items exist, none of their names. Use wikidata_sparql instead when the question needs the actual members of a list, category, or set.", `{"type":"object","properties":{"title":{"type":"string","description":"Exact article title"},"language":{"type":"string","description":"Wikipedia language code, e.g. de or en"}},"required":["title"]}`), Execute: c.summary},
		{Definition: wikiToolDefinition("wikidata_entity", "Retrieve structured labels and claims for one Wikidata entity Q-ID.", `{"type":"object","properties":{"id":{"type":"string","description":"Wikidata Q-ID, e.g. Q64"},"language":{"type":"string","description":"Preferred language code, e.g. de or en"}},"required":["id"]}`), Execute: c.entity},
		{Definition: wikiToolDefinition("wikidata_sparql", "Run a bounded read-only SELECT or ASK query against the Wikidata Query Service. Use it for structured comparisons, lists, dates, identifiers, or relations that article summaries cannot answer.", `{"type":"object","properties":{"query":{"type":"string","description":"Read-only SPARQL SELECT or ASK query; include LIMIT when useful"}},"required":["query"]}`), Execute: c.sparql},
	}
}

func wikiToolDefinition(name, description, parameters string) gopherllm.ToolDefinition {
	return gopherllm.ToolDefinition{Type: "function", Function: gopherllm.ToolFunctionDef{Name: name, Description: description, Parameters: json.RawMessage(parameters)}}
}

func (c *wikimediaClient) search(ctx context.Context, call gopherllm.ToolCall) (string, error) {
	var args struct{ Query, Language string }
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	query := boundedText(args.Query, 400)
	if query == "" {
		return "", fmt.Errorf("query is required")
	}
	lang, err := wikiLanguage(args.Language)
	if err != nil {
		return "", err
	}
	u := c.wikipediaURL(lang) + "/w/rest.php/v1/search/page?q=" + url.QueryEscape(query) + "&limit=5"
	var data struct {
		Pages []struct{ Title, Description, Excerpt string } `json:"pages"`
	}
	if err := c.getJSON(ctx, u, &data, "application/json"); err != nil {
		return "", err
	}
	type result struct {
		Title       string `json:"title"`
		Description string `json:"description,omitempty"`
		Excerpt     string `json:"excerpt,omitempty"`
		URL         string `json:"url"`
	}
	results := make([]result, 0, len(data.Pages))
	for _, p := range data.Pages {
		results = append(results, result{p.Title, p.Description, trimText(stripTags(p.Excerpt), 400), fmt.Sprintf("https://%s.wikipedia.org/wiki/%s", lang, url.PathEscape(strings.ReplaceAll(p.Title, " ", "_")))})
	}
	return toolJSON(map[string]any{"source": "Wikipedia search", "language": lang, "query": query, "results": results})
}

func (c *wikimediaClient) summary(ctx context.Context, call gopherllm.ToolCall) (string, error) {
	var args struct{ Title, Language string }
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	title := boundedText(args.Title, 200)
	if title == "" {
		return "", fmt.Errorf("title is required")
	}
	lang, err := wikiLanguage(args.Language)
	if err != nil {
		return "", err
	}
	u := c.wikipediaURL(lang) + "/api/rest_v1/page/summary/" + url.PathEscape(strings.ReplaceAll(title, " ", "_"))
	var data struct {
		Title, Description, Extract string
		ContentURLs                 struct {
			Desktop struct{ Page string } `json:"desktop"`
		} `json:"content_urls"`
	}
	if err := c.getJSON(ctx, u, &data, "application/json"); err != nil {
		return "", err
	}
	return toolJSON(map[string]any{"source": "Wikipedia article summary", "language": lang, "title": data.Title, "description": data.Description, "extract": trimText(data.Extract, 3000), "url": data.ContentURLs.Desktop.Page})
}

func (c *wikimediaClient) entity(ctx context.Context, call gopherllm.ToolCall) (string, error) {
	var args struct{ ID, Language string }
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	id := strings.ToUpper(strings.TrimSpace(args.ID))
	if !wikidataID.MatchString(id) {
		return "", fmt.Errorf("id must be a Wikidata Q-ID")
	}
	lang, err := wikiLanguage(args.Language)
	if err != nil {
		return "", err
	}
	u := c.wikidataAPIBase + "?action=wbgetentities&format=json&props=labels%7Cdescriptions%7Cclaims&languages=" + url.QueryEscape(lang+"|en") + "&ids=" + id
	var data wikidataEntityResponse
	if err := c.getJSON(ctx, u, &data, "application/json"); err != nil {
		return "", err
	}
	entity, ok := data.Entities[id]
	if !ok {
		return "", fmt.Errorf("entity %s was not found", id)
	}
	claims := map[string][]any{}
	for property, statements := range entity.Claims {
		for _, statement := range statements {
			if statement.Mainsnak.Datavalue.Value != nil {
				claims[property] = append(claims[property], statement.Mainsnak.Datavalue.Value)
				if len(claims[property]) == 3 {
					break
				}
			}
		}
	}
	return toolJSON(map[string]any{"source": "Wikidata entity", "id": id, "url": "https://www.wikidata.org/wiki/" + id, "label": localizedValue(entity.Labels, lang), "description": localizedValue(entity.Descriptions, lang), "claims": claims})
}

func (c *wikimediaClient) sparql(ctx context.Context, call gopherllm.ToolCall) (string, error) {
	var args struct{ Query string }
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	query := boundedText(args.Query, 4000)
	if err := checkReadOnlySPARQL(query); err != nil {
		return "", err
	}
	u := c.sparqlBase + "?format=json&query=" + url.QueryEscape(query)
	var data sparqlResponse
	if err := c.getJSON(ctx, u, &data, "application/sparql-results+json"); err != nil {
		return "", err
	}
	if data.Boolean != nil {
		return toolJSON(map[string]any{"source": "Wikidata SPARQL", "endpoint": c.sparqlBase, "boolean": *data.Boolean})
	}
	rows := make([]map[string]string, 0, min(len(data.Results.Bindings), 25))
	for _, binding := range data.Results.Bindings {
		row := map[string]string{}
		for key, value := range binding {
			row[key] = value.Value
		}
		rows = append(rows, row)
		if len(rows) == 25 {
			break
		}
	}
	return toolJSON(map[string]any{"source": "Wikidata SPARQL", "endpoint": c.sparqlBase, "rows": rows, "truncated": len(data.Results.Bindings) > len(rows)})
}

func (c *wikimediaClient) getJSON(ctx context.Context, endpoint string, out any, accept string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", wikimediaUserAgent)
	req.Header.Set("Accept", accept)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("Wikimedia returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 128<<10)).Decode(out); err != nil {
		return fmt.Errorf("decode Wikimedia response: %w", err)
	}
	return nil
}

func (c *wikimediaClient) wikipediaURL(language string) string {
	if strings.Contains(c.wikipediaBase, "%") {
		return fmt.Sprintf(c.wikipediaBase, language)
	}
	return c.wikipediaBase
}

// wikiLanguages is closed on purpose. The language code becomes a hostname
// ("<lang>.wikipedia.org"), and the model — which may be working from text an
// attacker influenced — chooses it. An allow-list keeps that choice from
// steering requests at hosts nobody vetted.
var wikiLanguages = []string{"de", "en", "fr", "it"}

func wikiLanguage(value string) (string, error) {
	lang := strings.ToLower(strings.TrimSpace(value))
	if lang == "" {
		return "de", nil
	}
	for _, allowed := range wikiLanguages {
		if lang == allowed {
			return lang, nil
		}
	}
	return "", fmt.Errorf("unsupported Wikipedia language %q (want one of %s)", lang, strings.Join(wikiLanguages, ", "))
}

// sparqlUpdateKeywords covers every SPARQL 1.1 Update form, not just the few
// that are easy to think of. A prefix check alone is not enough: a query may
// start with SELECT and still carry an update after a semicolon.
var sparqlUpdateKeywords = []string{
	"insert", "delete", "drop", "clear", "load", "create", "copy", "move", "add", "service", "with", "using",
}

// checkReadOnlySPARQL rejects anything that is not a plain SELECT or ASK.
// Comments are stripped first, because "# note\nDROP GRAPH <g>" would
// otherwise hide the keyword from a substring scan, and matching is done on
// whole words so a legitimate variable like ?address is not mistaken for ADD.
func checkReadOnlySPARQL(query string) error {
	var stripped strings.Builder
	for _, line := range strings.Split(query, "\n") {
		if hash := strings.IndexByte(line, '#'); hash >= 0 {
			line = line[:hash]
		}
		stripped.WriteString(line)
		stripped.WriteByte('\n')
	}
	lower := strings.ToLower(stripped.String())
	trimmed := strings.TrimSpace(lower)
	if !strings.HasPrefix(trimmed, "select") && !strings.HasPrefix(trimmed, "ask") && !strings.HasPrefix(trimmed, "prefix") {
		return fmt.Errorf("only read-only SELECT or ASK queries are allowed")
	}
	for _, word := range strings.FieldsFunc(lower, func(r rune) bool { return r < 'a' || r > 'z' }) {
		for _, banned := range sparqlUpdateKeywords {
			if word == banned {
				return fmt.Errorf("query contains the disallowed SPARQL operation %q", word)
			}
		}
	}
	if !strings.Contains(lower, "select") && !strings.Contains(lower, "ask") {
		return fmt.Errorf("only read-only SELECT or ASK queries are allowed")
	}
	return nil
}
func boundedText(value string, limit int) string { return trimText(strings.TrimSpace(value), limit) }
func trimText(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit]
}
func stripTags(value string) string { return regexp.MustCompile(`<[^>]*>`).ReplaceAllString(value, "") }
func localizedValue(values map[string]wikidataText, language string) string {
	if value, ok := values[language]; ok {
		return value.Value
	}
	if value, ok := values["en"]; ok {
		return value.Value
	}
	for _, value := range values {
		return value.Value
	}
	return ""
}
func toolJSON(value any) (string, error) { data, err := json.Marshal(value); return string(data), err }
