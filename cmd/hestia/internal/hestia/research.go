package hestia

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	searchResultLimit = 5
	searchQueryLimit  = 400
	searchBodyLimit   = 1 << 20
)

var (
	wikipediaLanguageCode = regexp.MustCompile(`^[a-z]{2,12}(-[a-z0-9]{1,12}){0,2}$`)
	searchMarkup          = regexp.MustCompile(`<[^<>]*>`)
)

// SearchProvider runs an explicit, read-only search. Results are untrusted
// display data and should never be passed back to the device planner.
type SearchProvider interface {
	Search(ctx context.Context, query string) ([]SearchResult, error)
}

type SearchResult struct {
	Title   string `json:"title"`
	Snippet string `json:"snippet"`
	URL     string `json:"url"`
}

type wikipediaSearch struct {
	base   string
	client *http.Client
}

// NewWikipediaSearch searches a public Wikipedia edition. Empty language uses
// the German edition. Creating the provider makes no request.
func NewWikipediaSearch(language string) (SearchProvider, error) {
	language = strings.ToLower(strings.TrimSpace(language))
	if language == "" {
		language = "de"
	}
	if !wikipediaLanguageCode.MatchString(language) {
		return nil, fmt.Errorf("invalid Wikipedia language code")
	}
	return &wikipediaSearch{
		base: "https://" + language + ".wikipedia.org", client: newSearchClient(),
	}, nil
}

func (w *wikipediaSearch) Search(ctx context.Context, query string) ([]SearchResult, error) {
	query, err := validateSearchQuery(query)
	if err != nil {
		return nil, err
	}
	values := url.Values{"q": {query}, "limit": {strconv.Itoa(searchResultLimit)}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, w.base+"/w/rest.php/v1/search/page?"+values.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("could not prepare Wikipedia search request")
	}
	var data struct {
		Pages []struct {
			Key, Title, Excerpt, Description string
		} `json:"pages"`
	}
	if err := readSearchResponse(w.client, req, &data); err != nil {
		return nil, fmt.Errorf("Wikipedia: %w", err)
	}
	results := make([]SearchResult, 0, min(searchResultLimit, len(data.Pages)))
	for _, page := range data.Pages {
		title := searchText(page.Title, 180)
		if title == "" || page.Key == "" || page.Key == "." || page.Key == ".." || len(page.Key) > 500 {
			continue
		}
		snippet := searchText(page.Excerpt, 400)
		if snippet == "" {
			snippet = searchText(page.Description, 400)
		}
		results = append(results, SearchResult{Title: title, Snippet: snippet, URL: w.base + "/wiki/" + url.PathEscape(page.Key)})
		if len(results) == searchResultLimit {
			break
		}
	}
	return results, nil
}

type searxSearch struct {
	endpoint string
	client   *http.Client
}

// NewWebSearch uses a configured SearXNG endpoint such as
// http://127.0.0.1:8888/search. Remote endpoints must use HTTPS.
func NewWebSearch(endpoint string) (SearchProvider, error) {
	endpoint = strings.TrimSpace(endpoint)
	u, err := url.Parse(endpoint)
	if err != nil || u.Hostname() == "" || u.User != nil || u.ForceQuery || u.RawQuery != "" || strings.Contains(endpoint, "#") {
		return nil, fmt.Errorf("web search requires a SearXNG URL without credentials, query, or fragment")
	}
	loopbackIP, _ := netip.ParseAddr(u.Hostname())
	if u.Scheme != "https" && (u.Scheme != "http" || (!strings.EqualFold(u.Hostname(), "localhost") && !loopbackIP.IsLoopback())) {
		return nil, fmt.Errorf("web search requires HTTPS or a loopback HTTP endpoint")
	}
	if u.Path == "" {
		u.Path = "/search"
	}
	return &searxSearch{endpoint: u.String(), client: newSearchClient()}, nil
}

func (s *searxSearch) Search(ctx context.Context, query string) ([]SearchResult, error) {
	query, err := validateSearchQuery(query)
	if err != nil {
		return nil, err
	}
	form := url.Values{"q": {query}, "format": {"json"}, "categories": {"general"}, "safesearch": {"1"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("could not prepare web search request")
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	var data struct {
		Results []struct {
			Title, Content, URL string
		} `json:"results"`
	}
	if err := readSearchResponse(s.client, req, &data); err != nil {
		return nil, fmt.Errorf("SearXNG: %w", err)
	}
	results := make([]SearchResult, 0, min(searchResultLimit, len(data.Results)))
	seen := make(map[string]bool, searchResultLimit)
	for _, row := range data.Results {
		link := searchLink(row.URL)
		title := searchText(row.Title, 180)
		if link == "" || title == "" || seen[link] {
			continue
		}
		seen[link] = true
		results = append(results, SearchResult{Title: title, Snippet: searchText(row.Content, 400), URL: link})
		if len(results) == searchResultLimit {
			break
		}
	}
	return results, nil
}

func newSearchClient() *http.Client {
	return &http.Client{
		Timeout: 8 * time.Second,
		// Use a direct connection and a fresh client with no cookie jar.
		Transport: &http.Transport{IdleConnTimeout: 30 * time.Second, MaxIdleConns: 2, ForceAttemptHTTP2: true},
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func readSearchResponse(client *http.Client, req *http.Request, dest any) error {
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "Hestia/1.0 (local assistant; https://github.com/SimonWaldherr/GopherLLM)")
	resp, err := client.Do(req)
	if err != nil {
		if err := req.Context().Err(); err != nil {
			return err
		}
		// Transport errors can contain the request URL and query.
		return fmt.Errorf("search provider could not be reached")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("search provider returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, searchBodyLimit+1))
	if err != nil {
		if err := req.Context().Err(); err != nil {
			return err
		}
		return fmt.Errorf("search response could not be read")
	}
	if len(body) > searchBodyLimit {
		return fmt.Errorf("search response is too large")
	}
	if err := json.Unmarshal(body, dest); err != nil {
		return fmt.Errorf("search provider returned invalid JSON")
	}
	return nil
}

func validateSearchQuery(query string) (string, error) {
	query = strings.TrimSpace(query)
	if query == "" || len(query) > searchQueryLimit*utf8.UTFMax || utf8.RuneCountInString(query) > searchQueryLimit || !utf8.ValidString(query) {
		return "", fmt.Errorf("search query must contain 1–%d characters", searchQueryLimit)
	}
	return query, nil
}

func searchLink(raw string) string {
	if len(raw) > 2048 {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" || u.User != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return ""
	}
	return u.String()
}

func searchText(markup string, limit int) string {
	plain := html.UnescapeString(searchMarkup.ReplaceAllString(markup, ""))
	plain = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			return ' '
		}
		return r
	}, plain)
	plain = strings.Join(strings.Fields(plain), " ")
	for i := range plain {
		if limit == 0 {
			return plain[:i]
		}
		limit--
	}
	return plain
}
