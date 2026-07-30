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
	"sync"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// OSMUsageNotice must be shown to anyone enabling the public Nominatim
// endpoint. It is intentionally prominent because the public service permits
// only user-triggered, low-volume lookups and must not receive personal data.
const OSMUsageNotice = "OpenStreetMap place search sends only the model-requested place query to Nominatim. Do not submit personal or confidential data. The public endpoint is rate-limited to one request per second, must not be used for autocomplete or bulk geocoding, and is subject to https://operations.osmfoundation.org/policies/nominatim/."

// ResearchOptions selects bounded, server-owned factual source tools for Go
// applications. Every source defaults to disabled; callers choose explicitly
// which sources their product and privacy policy permit.
type ResearchOptions struct {
	Wikimedia     bool
	OpenStreetMap bool
	HTTPClient    *http.Client
	OSMSearchURL  string
}

// NewResearchTools returns opt-in factual research tools for direct Go use
// with RunAgenticChatWithTools. A caller that does not enable any source gets
// nil, so importing server never causes network activity on its own.
func NewResearchTools(opts ResearchOptions) []gopherllm.AgenticTool {
	var tools []gopherllm.AgenticTool
	if opts.Wikimedia {
		tools = append(tools, newWikimediaClient(opts.HTTPClient).tools()...)
	}
	if opts.OpenStreetMap {
		tools = append(tools, newOSMClient(opts.HTTPClient, opts.OSMSearchURL).tools()...)
	}
	return tools
}

const defaultOSMSearchURL = "https://nominatim.openstreetmap.org/search"
const osmUserAgent = "GopherLLM/1.0 (local research tool; https://github.com/SimonWaldherr/GopherLLM)"

type osmClient struct {
	httpClient *http.Client
	searchURL  string
	mu         sync.Mutex
	next       time.Time
}

func newOSMClient(client *http.Client, searchURL string) *osmClient {
	if client == nil {
		client = &http.Client{Timeout: 12 * time.Second}
	}
	if strings.TrimSpace(searchURL) == "" {
		searchURL = defaultOSMSearchURL
	}
	return &osmClient{httpClient: client, searchURL: strings.TrimRight(searchURL, "?")}
}

func (c *osmClient) tools() []gopherllm.AgenticTool {
	return []gopherllm.AgenticTool{
		{Definition: wikiToolDefinition("openstreetmap_search", "Search OpenStreetMap/Nominatim for a specific place, address, or nearby named amenity. Use only for a direct user-requested place lookup; never autocomplete, bulk-geocode, enumerate places, or send private/personal data. This external request is rate-limited. Results include OpenStreetMap attribution and a source URL.", `{"type":"object","properties":{"query":{"type":"string","description":"A specific place, address, or named amenity search"},"country_code":{"type":"string","description":"Optional ISO 3166-1 alpha-2 country filter, e.g. DE"}},"required":["query"]}`), Execute: c.search},
	}
}

var osmCountryCode = regexp.MustCompile(`^[a-zA-Z]{2}$`)

func (c *osmClient) search(ctx context.Context, call gopherllm.ToolCall) (string, error) {
	var args struct {
		Query       string `json:"query"`
		CountryCode string `json:"country_code"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	query := boundedText(args.Query, 250)
	if query == "" {
		return "", fmt.Errorf("query is required")
	}
	country := strings.ToLower(strings.TrimSpace(args.CountryCode))
	if country != "" && !osmCountryCode.MatchString(country) {
		return "", fmt.Errorf("country_code must be a two-letter ISO code")
	}
	if err := c.waitForSlot(ctx); err != nil {
		return "", err
	}
	values := url.Values{"q": {query}, "format": {"jsonv2"}, "addressdetails": {"1"}, "limit": {"5"}}
	if country != "" {
		values.Set("countrycodes", country)
	}
	u := c.searchURL + "?" + values.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", osmUserAgent)
	req.Header.Set("Accept", "application/json")
	var data []struct {
		DisplayName string `json:"display_name"`
		Lat         string `json:"lat"`
		Lon         string `json:"lon"`
		Type        string `json:"type"`
		Category    string `json:"category"`
		OSMType     string `json:"osm_type"`
		OSMID       int64  `json:"osm_id"`
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return "", fmt.Errorf("OpenStreetMap Nominatim returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 128<<10)).Decode(&data); err != nil {
		return "", fmt.Errorf("decode OpenStreetMap response: %w", err)
	}
	results := make([]map[string]any, 0, len(data))
	for _, item := range data {
		results = append(results, map[string]any{
			"display_name": item.DisplayName,
			"latitude":     item.Lat,
			"longitude":    item.Lon,
			"category":     item.Category,
			"type":         item.Type,
			"url":          osmObjectURL(item.OSMType, item.OSMID),
		})
	}
	return toolJSON(map[string]any{
		"source":      "OpenStreetMap Nominatim",
		"query":       query,
		"results":     results,
		"attribution": "© OpenStreetMap contributors, ODbL 1.0",
		"policy":      "https://operations.osmfoundation.org/policies/nominatim/",
	})
}

func (c *osmClient) waitForSlot(ctx context.Context) error {
	for {
		c.mu.Lock()
		now := time.Now()
		if !c.next.After(now) {
			c.next = now.Add(time.Second)
			c.mu.Unlock()
			return nil
		}
		wait := time.Until(c.next)
		c.mu.Unlock()
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func osmObjectURL(kind string, id int64) string {
	path := map[string]string{"node": "node", "way": "way", "relation": "relation"}[strings.ToLower(kind)]
	if path == "" || id <= 0 {
		return "https://www.openstreetmap.org/"
	}
	return fmt.Sprintf("https://www.openstreetmap.org/%s/%d", path, id)
}
