package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// ragFetchTimeout bounds the whole POST /rag/fetch round trip: connect, TLS,
// headers, and body. A knowledge-base import is a one-off operator action,
// not a latency-sensitive request path, so this is generous compared to the
// timeouts elsewhere in this package's HTTP clients.
const ragFetchTimeout = 15 * time.Second

// maxRAGFetchBytes caps how much of a fetched URL's body is read. This is an
// extracted-text budget (a web page, an internal wiki article), not a
// document-download limit — well under rag.IngestOptions' own 4 MiB default
// for a local file, since a page pulled over the network deserves a tighter
// leash.
const maxRAGFetchBytes = 2 << 20

// ragFetchClientFunc is a package-level indirection to ragFetchClient so
// tests can substitute an unguarded client pointed at an httptest.Server —
// which listens on loopback, exactly what the guard below exists to refuse —
// while still exercising POST /rag/fetch's content-extraction and indexing
// logic end to end. Production code never reassigns this.
var ragFetchClientFunc = ragFetchClient

// ragFetchClient returns an http.Client hardened against server-side request
// forgery: POST /rag/fetch sends a request to a URL supplied in a JSON body,
// which — unlike the Wikimedia/OpenStreetMap tools' fixed, hardcoded hosts —
// is exactly the shape of input SSRF protection exists for. Two measures:
//
//  1. DialContext resolves the hostname itself and refuses to connect to any
//     resulting address that is loopback, link-local (including the
//     169.254.169.254 cloud metadata endpoint), private, unspecified, or
//     multicast, then dials that already-checked address directly rather
//     than letting the transport re-resolve the hostname a second time —
//     closing the DNS-rebinding gap where the check and the connect see two
//     different answers for the same name.
//  2. CheckRedirect refuses every redirect outright, rather than trying to
//     re-apply the same guard to each hop. A legitimate redirect can still be
//     imported by resolving it and calling POST /rag/fetch again with the
//     final URL.
func ragFetchClient() *http.Client {
	dialer := &net.Dialer{Timeout: 5 * time.Second}
	return &http.Client{
		Timeout: ragFetchTimeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("the response redirected; POST /rag/fetch with the final URL instead")
		},
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, port, err := net.SplitHostPort(addr)
				if err != nil {
					return nil, err
				}
				ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
				if err != nil {
					return nil, err
				}
				for _, ip := range ips {
					if isDisallowedRAGFetchIP(ip) {
						continue
					}
					return dialer.DialContext(ctx, network, net.JoinHostPort(ip.String(), port))
				}
				return nil, errors.New("no permitted address for this host (it resolves only to a private, loopback, or link-local address)")
			},
		},
	}
}

// isDisallowedRAGFetchIP reports whether ip is inside a range POST /rag/fetch
// must never connect to: the machine itself, its private network, or an
// address-autoconfiguration/metadata range. See ragFetchClient's doc comment.
func isDisallowedRAGFetchIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast()
}

// fetchContentAllowed classifies a fetched response's Content-Type for POST
// /rag/fetch: ok reports whether this package can extract text from it at
// all (rejecting images, PDFs, and other binary formats outright rather than
// indexing their raw bytes as garbled "text" — the same reasoning behind
// rag.DefaultTextExtension for uploads); isHTML reports whether the body
// needs tag-stripping before it is usable as retrieval text. A missing
// Content-Type is treated as plain text rather than rejected, since some
// servers (particularly ones serving a bare text file) omit it.
func fetchContentAllowed(contentType string) (isHTML, ok bool) {
	ct := strings.ToLower(contentType)
	if i := strings.IndexByte(ct, ';'); i >= 0 {
		ct = ct[:i]
	}
	ct = strings.TrimSpace(ct)
	switch {
	case ct == "", strings.HasPrefix(ct, "text/plain"), strings.HasPrefix(ct, "text/markdown"),
		ct == "application/json", strings.HasPrefix(ct, "text/xml"), strings.HasPrefix(ct, "application/xml"):
		return false, true
	case strings.HasPrefix(ct, "text/html"), ct == "application/xhtml+xml":
		return true, true
	default:
		return false, false
	}
}

// wikipediaArticleEndpoint recognizes an ordinary Wikipedia article URL and
// turns it into a MediaWiki API request for the complete article as plain
// text. Importing the rendered page directly would also index navigation,
// edit controls, footers, and other chrome; prop=extracts&explaintext keeps
// the knowledge base focused on the article itself and redirects=1 follows
// Wikipedia's own article redirects without enabling arbitrary HTTP
// redirects in ragFetchClient.
func wikipediaArticleEndpoint(target *url.URL) (string, bool) {
	if target == nil || !strings.HasPrefix(target.EscapedPath(), "/wiki/") {
		return "", false
	}
	host := strings.ToLower(target.Hostname())
	parts := strings.Split(host, ".")
	isDesktop := len(parts) == 3 && parts[1] == "wikipedia" && parts[2] == "org"
	isMobile := len(parts) == 4 && parts[1] == "m" && parts[2] == "wikipedia" && parts[3] == "org"
	if (!isDesktop && !isMobile) || parts[0] == "www" || parts[0] == "" {
		return "", false
	}
	// The action API lives on the desktop hostname even when the pasted link
	// came from Wikipedia's mobile site.
	host = parts[0] + ".wikipedia.org"
	title, err := url.PathUnescape(strings.TrimPrefix(target.EscapedPath(), "/wiki/"))
	if err != nil || strings.TrimSpace(title) == "" {
		return "", false
	}
	endpoint := &url.URL{Scheme: "https", Host: host, Path: "/w/api.php"}
	q := endpoint.Query()
	q.Set("action", "query")
	q.Set("format", "json")
	q.Set("formatversion", "2")
	q.Set("prop", "extracts")
	q.Set("explaintext", "1")
	q.Set("redirects", "1")
	q.Set("titles", strings.ReplaceAll(title, "_", " "))
	endpoint.RawQuery = q.Encode()
	return endpoint.String(), true
}

func decodeWikipediaArticle(data []byte) (title, text string, err error) {
	var payload struct {
		Query struct {
			Pages []struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
				Missing bool   `json:"missing"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return "", "", fmt.Errorf("decode Wikipedia article: %w", err)
	}
	if len(payload.Query.Pages) == 0 || payload.Query.Pages[0].Missing {
		return "", "", errors.New("Wikipedia article was not found")
	}
	page := payload.Query.Pages[0]
	text = strings.TrimSpace(page.Extract)
	if text == "" {
		return "", "", errors.New("Wikipedia article had no extractable text")
	}
	return strings.TrimSpace(page.Title), text, nil
}

var (
	ragHTMLIgnored = regexp.MustCompile(`(?is)<(?:script|style|noscript|svg|nav|footer|header)\b[^>]*>.*?</(?:script|style|noscript|svg|nav|footer|header)\s*>`)
	ragHTMLBreaks  = regexp.MustCompile(`(?is)</?(?:article|aside|blockquote|br|dd|div|dl|dt|figcaption|figure|h[1-6]|hr|li|main|ol|p|pre|section|table|td|th|tr|ul)\b[^>]*>`)
	ragHTMLTags    = regexp.MustCompile(`(?s)<[^>]*>`)
	ragHTMLTitle   = regexp.MustCompile(`(?is)<title\b[^>]*>(.*?)</title\s*>`)
)

// extractRAGHTML converts an ordinary web page into retrieval-friendly text.
// It is intentionally conservative rather than a full browser DOM: active
// and navigational regions are removed, structural tags become line breaks,
// entities are decoded, and repeated whitespace is collapsed. Wikipedia is
// handled separately above through MediaWiki's plaintext API.
func extractRAGHTML(data []byte) (title, text string) {
	raw := string(data)
	if match := ragHTMLTitle.FindStringSubmatch(raw); len(match) == 2 {
		title = collapseRAGText(ragHTMLTags.ReplaceAllString(match[1], " "))
		title = strings.ReplaceAll(title, "\n", " ")
	}
	raw = ragHTMLIgnored.ReplaceAllString(raw, "\n")
	raw = ragHTMLBreaks.ReplaceAllString(raw, "\n")
	raw = ragHTMLTags.ReplaceAllString(raw, " ")
	return html.UnescapeString(strings.TrimSpace(title)), collapseRAGText(raw)
}

func collapseRAGText(value string) string {
	value = html.UnescapeString(value)
	lines := strings.Split(value, "\n")
	out := make([]string, 0, len(lines))
	blank := false
	for _, line := range lines {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			if len(out) > 0 && !blank {
				out = append(out, "")
				blank = true
			}
			continue
		}
		out = append(out, line)
		blank = false
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

func validRAGText(data []byte) bool {
	return utf8.Valid(data) && !strings.ContainsRune(string(data), '\x00')
}

// readRAGFetchBody applies the network import's size and UTF-8 policy in one
// place for both generic pages and Wikipedia API responses.
func readRAGFetchBody(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, maxRAGFetchBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxRAGFetchBytes {
		return nil, fmt.Errorf("fetched content exceeds the size limit")
	}
	if !validRAGText(data) {
		return nil, fmt.Errorf("fetched content is not valid UTF-8 text")
	}
	return data, nil
}
