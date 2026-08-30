package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/SimonWaldherr/GopherLLM/rag"
)

// maxRAGUploadFileBytes caps one uploaded file, matching rag.IngestOptions'
// own default MaxFileBytes. Upload additionally accepts HTML because this
// route can strip its markup before indexing, unlike AddFS's raw-text default.
const maxRAGUploadFileBytes = 4 << 20

// maxRAGUploadTotalBytes caps one multipart request: several files at once,
// not an unbounded bulk-import channel — that is what --rag-docs and
// POST /rag/reload are for.
const maxRAGUploadTotalBytes = 20 << 20

const (
	maxRAGFetchSources      = 20
	maxRAGFetchRequestBytes = 64 << 10
)

// registerRAGRoutes registers the management, search, and ingestion
// endpoints for the server-side knowledge base a search_documents tool (see
// agenticToolsFor in server.go) searches during chat:
//
//   - GET  /rag/status    — document/chunk counts and which capabilities are active
//   - GET/POST/DELETE /rag/documents — list, add (one or many, pasted as JSON), or remove
//   - POST /rag/upload    — add one or more uploaded files
//   - POST /rag/fetch     — add or update documents by fetching up to 20 URLs
//   - POST /rag/reload    — re-scan HandlerOptions.RAGDocsDir for new or changed files
//   - POST /rag/search    — preview a search without a chat turn
//
// sem is the same request-concurrency limiter every inference route uses.
// Status, List, and Delete stay unwrapped — they never touch a model, so
// queuing them behind in-flight generations would only add latency for no
// benefit (see rag.Index's package doc on the brute-force scan's cost). Every
// route that can invoke an Embedder (Add, Search, and everything that adds
// documents) is wrapped, since once HandlerOptions.RAGEmbedModelPath is
// configured those calls run model inference and should compete for the same
// budget as a chat completion rather than bypass it.
func registerRAGRoutes(mux *http.ServeMux, state *ragState, sem chan struct{}) {
	mux.HandleFunc("/rag/status", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, map[string]any{
			"documents":           state.index.Docs(),
			"chunks":              state.index.Len(),
			"vector_search":       state.index.HasEmbedder(),
			"docs_dir_configured": state.docsDir != "",
			"persistent":          state.snapshotPath != "",
		})
	})

	addDocuments := withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		addRAGDocuments(w, req, state)
	})
	mux.HandleFunc("/rag/documents", func(w http.ResponseWriter, req *http.Request) {
		switch req.Method {
		case http.MethodGet:
			list := state.index.List()
			docs := make([]map[string]any, len(list))
			for i, d := range list {
				docs[i] = docInfoJSON(d)
			}
			writeJSON(w, map[string]any{"documents": docs})

		case http.MethodPost:
			addDocuments(w, req)

		case http.MethodDelete:
			var body struct {
				ID string `json:"id"`
			}
			if req.Body != nil {
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil && err != io.EOF {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
			}
			body.ID = strings.TrimSpace(body.ID)
			if body.ID == "" {
				http.Error(w, "id must not be empty", http.StatusBadRequest)
				return
			}
			if state.index.Remove(body.ID) == 0 {
				http.Error(w, "document not found", http.StatusNotFound)
				return
			}
			state.save()
			w.WriteHeader(http.StatusNoContent)

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/rag/upload", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		uploadRAGDocuments(w, req, state)
	}))

	mux.HandleFunc("/rag/fetch", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		fetchRAGDocument(w, req, state)
	}))

	mux.HandleFunc("/rag/reload", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if state.docsDir == "" {
			http.Error(w, "no --rag-docs directory is configured", http.StatusConflict)
			return
		}
		added, skipped, err := state.seedFromDir(req.Context(), state.docsDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		state.save()
		writeJSON(w, map[string]any{
			"added": added, "skipped": skipped,
			"documents": state.index.Docs(), "chunks": state.index.Len(),
		})
	}))

	mux.HandleFunc("/rag/search", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Query string `json:"query"`
			TopK  int    `json:"top_k"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		body.Query = strings.TrimSpace(body.Query)
		if body.Query == "" {
			http.Error(w, "query must not be empty", http.StatusBadRequest)
			return
		}
		q := rag.Query{TopK: body.TopK, Lexical: rag.GuessLexicalMode(body.Query)}
		hits, err := state.index.Search(req.Context(), body.Query, q)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out := make([]map[string]any, len(hits))
		for i, h := range hits {
			out[i] = hitJSON(h)
		}
		writeJSON(w, map[string]any{"hits": out})
	}))
}

// ragDocumentInput is one document as accepted by POST /rag/documents, either
// as the request body's top-level fields (a single document) or as one entry
// of its "documents" array (a batch).
type ragDocumentInput struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// addRAGDocuments implements POST /rag/documents. A "documents" array takes a
// batch through one rag.Index.Add call — and so one Embedder.EmbedBatch call
// across every document's chunks, when an embedding model is configured,
// rather than one round trip per document — while the top-level id/title/text
// shorthand keeps the single-document case a one-line request body. The two
// forms are mutually exclusive: a non-empty "documents" array is used
// exactly as given and the top-level fields are ignored, rather than treating
// the shorthand as an implicit extra entry.
func addRAGDocuments(w http.ResponseWriter, req *http.Request, state *ragState) {
	var body struct {
		ragDocumentInput
		Documents []ragDocumentInput `json:"documents"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	inputs := body.Documents
	batch := len(inputs) > 0
	if !batch {
		inputs = []ragDocumentInput{body.ragDocumentInput}
	}

	docs := make([]rag.Doc, 0, len(inputs))
	now := time.Now()
	for i, in := range inputs {
		text := strings.TrimSpace(in.Text)
		if text == "" {
			label := "text"
			if batch {
				label = fmt.Sprintf("documents[%d].text", i)
			}
			http.Error(w, label+" must not be empty", http.StatusBadRequest)
			return
		}
		id := strings.TrimSpace(in.ID)
		if id == "" {
			id = newDocID()
		}
		title := strings.TrimSpace(in.Title)
		if title == "" {
			title = id
		}
		docs = append(docs, rag.Doc{ID: id, Title: title, Text: text, Time: now})
	}

	if err := state.index.Add(req.Context(), docs...); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	state.save()

	if !batch {
		writeJSON(w, docInfoJSON(docByID(state.index, docs[0].ID)))
		return
	}
	out := make([]map[string]any, len(docs))
	for i, d := range docs {
		out[i] = docInfoJSON(docByID(state.index, d.ID))
	}
	writeJSON(w, map[string]any{"documents": out})
}

// uploadRAGDocuments implements POST /rag/upload: a multipart/form-data
// request carrying one or more files under any field name(s), each indexed
// as its own Doc. A file whose extension is not in rag.DefaultTextExtension's
// allowlist, or that exceeds maxRAGUploadFileBytes, is skipped rather than
// failing the whole request — the same "skip, don't fail the batch" policy
// rag.Index.AddFS applies to a directory walk, so one oversized or binary
// file among several does not block the rest.
func uploadRAGDocuments(w http.ResponseWriter, req *http.Request, state *ragState) {
	req.Body = http.MaxBytesReader(w, req.Body, maxRAGUploadTotalBytes)
	if err := req.ParseMultipartForm(maxRAGUploadTotalBytes); err != nil {
		http.Error(w, "invalid upload: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer req.MultipartForm.RemoveAll()

	var headers []*multipart.FileHeader
	for _, fs := range req.MultipartForm.File {
		headers = append(headers, fs...)
	}
	if len(headers) == 0 {
		http.Error(w, "no files were uploaded", http.StatusBadRequest)
		return
	}

	docs := make([]rag.Doc, 0, len(headers))
	var skipped []string
	now := time.Now()
	for _, fh := range headers {
		name := filepath.Base(fh.Filename)
		isHTML := strings.EqualFold(filepath.Ext(name), ".html") || strings.EqualFold(filepath.Ext(name), ".htm")
		switch {
		case !rag.DefaultTextExtension(name) && !isHTML:
			skipped = append(skipped, name+": unsupported file type")
			continue
		case fh.Size > maxRAGUploadFileBytes:
			skipped = append(skipped, name+": exceeds the per-file size limit")
			continue
		}
		data, err := readUploadedFile(fh)
		if err != nil {
			skipped = append(skipped, name+": "+err.Error())
			continue
		}
		if !validRAGText(data) {
			skipped = append(skipped, name+": file is empty or not valid UTF-8 text")
			continue
		}
		title, text := name, strings.TrimSpace(string(data))
		if isHTML {
			extractedTitle, extractedText := extractRAGHTML(data)
			text = extractedText
			if extractedTitle != "" {
				title = extractedTitle
			}
		}
		if text == "" {
			skipped = append(skipped, name+": file is empty or had no extractable text")
			continue
		}
		docs = append(docs, rag.Doc{ID: newDocID(), Title: title, Path: name, Text: text, Time: now, Kind: "upload"})
	}
	if len(docs) == 0 {
		http.Error(w, "no uploaded file could be indexed: "+strings.Join(skipped, "; "), http.StatusBadRequest)
		return
	}

	if err := state.index.Add(req.Context(), docs...); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	state.save()

	out := make([]map[string]any, len(docs))
	for i, d := range docs {
		out[i] = docInfoJSON(docByID(state.index, d.ID))
	}
	writeJSON(w, map[string]any{"documents": out, "skipped": skipped})
}

func readUploadedFile(fh *multipart.FileHeader) ([]byte, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxRAGUploadFileBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxRAGUploadFileBytes {
		return nil, fmt.Errorf("exceeds the per-file size limit")
	}
	return data, nil
}

type ragFetchInput struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	URL   string `json:"url"`
}

type ragFetchError struct {
	status int
	err    error
}

func (e *ragFetchError) Error() string { return e.err.Error() }

func newRAGFetchError(status int, format string, args ...any) error {
	return &ragFetchError{status: status, err: fmt.Errorf(format, args...)}
}

func ragFetchErrorStatus(err error) int {
	if typed, ok := err.(*ragFetchError); ok {
		return typed.status
	}
	return http.StatusInternalServerError
}

// fetchRAGDocument implements POST /rag/fetch. The legacy top-level
// id/title/url fields import one source; "sources" imports up to 20 in one
// request and adds every successful result through a single Index.Add call,
// which also means a single embedding batch. Failed entries in a batch are
// reported without discarding successful imports.
func fetchRAGDocument(w http.ResponseWriter, req *http.Request, state *ragState) {
	req.Body = http.MaxBytesReader(w, req.Body, maxRAGFetchRequestBytes)
	var body struct {
		ragFetchInput
		Sources []ragFetchInput `json:"sources"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	inputs := body.Sources
	batch := len(inputs) > 0
	if !batch {
		inputs = []ragFetchInput{body.ragFetchInput}
	}
	if len(inputs) > maxRAGFetchSources {
		http.Error(w, fmt.Sprintf("at most %d sources may be imported at once", maxRAGFetchSources), http.StatusBadRequest)
		return
	}

	docs := make([]rag.Doc, 0, len(inputs))
	updated := make([]bool, 0, len(inputs))
	var skipped []map[string]string
	seen := make(map[string]bool, len(inputs))
	client := ragFetchClientFunc()
	for _, input := range inputs {
		doc, err := fetchRAGSource(req.Context(), client, input)
		if err != nil {
			if !batch {
				http.Error(w, err.Error(), ragFetchErrorStatus(err))
				return
			}
			skipped = append(skipped, map[string]string{"url": strings.TrimSpace(input.URL), "error": err.Error()})
			continue
		}
		if seen[doc.ID] {
			skipped = append(skipped, map[string]string{"url": doc.Path, "error": "duplicate source in this request"})
			continue
		}
		seen[doc.ID] = true
		docs = append(docs, doc)
		updated = append(updated, docExists(state.index, doc.ID))
	}
	if len(docs) == 0 {
		http.Error(w, "no web source could be imported", http.StatusUnprocessableEntity)
		return
	}
	if err := state.index.Add(req.Context(), docs...); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	state.save()

	out := make([]map[string]any, len(docs))
	for i, doc := range docs {
		out[i] = docInfoJSON(docByID(state.index, doc.ID))
		out[i]["updated"] = updated[i]
	}
	if !batch {
		writeJSON(w, out[0])
		return
	}
	writeJSON(w, map[string]any{"documents": out, "skipped": skipped})
}

// fetchRAGSource retrieves one public URL through the SSRF-guarded client and
// converts it into a Doc. Default IDs are stable hashes of the normalized
// source URL, so importing the same source again updates its chunks instead
// of silently creating a duplicate. A caller can still provide an explicit
// ID, which the UI uses for the Refresh action.
func fetchRAGSource(ctx context.Context, client *http.Client, input ragFetchInput) (rag.Doc, error) {
	rawURL := strings.TrimSpace(input.URL)
	if rawURL == "" {
		return rag.Doc{}, newRAGFetchError(http.StatusBadRequest, "url must not be empty")
	}
	target, err := url.Parse(rawURL)
	if err != nil {
		return rag.Doc{}, newRAGFetchError(http.StatusBadRequest, "url must be an absolute http or https URL")
	}
	target.Scheme = strings.ToLower(target.Scheme)
	if target.Host == "" || (target.Scheme != "http" && target.Scheme != "https") {
		return rag.Doc{}, newRAGFetchError(http.StatusBadRequest, "url must be an absolute http or https URL")
	}
	hostname := strings.ToLower(target.Hostname())
	port := target.Port()
	if (target.Scheme == "http" && port == "80") || (target.Scheme == "https" && port == "443") {
		port = ""
	}
	if port != "" {
		target.Host = net.JoinHostPort(hostname, port)
	} else if strings.Contains(hostname, ":") {
		target.Host = "[" + hostname + "]"
	} else {
		target.Host = hostname
	}
	target.Fragment = ""
	target.RawFragment = ""
	target.RawQuery = target.Query().Encode()
	sourceURL := target.String()

	fetchURL := sourceURL
	isWikipedia := false
	if endpoint, ok := wikipediaArticleEndpoint(target); ok {
		fetchURL, isWikipedia = endpoint, true
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, fetchURL, nil)
	if err != nil {
		return rag.Doc{}, newRAGFetchError(http.StatusBadRequest, "%v", err)
	}
	httpReq.Header.Set("User-Agent", "GopherLLM-RAG-Fetch/1.0 (+document import)")
	if isWikipedia {
		httpReq.Header.Set("Accept", "application/json")
	} else {
		httpReq.Header.Set("Accept", "text/plain, text/markdown, text/html, application/json, text/xml, application/xml")
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return rag.Doc{}, newRAGFetchError(http.StatusBadGateway, "fetch: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return rag.Doc{}, newRAGFetchError(http.StatusBadGateway, "fetch: upstream returned %s", resp.Status)
	}
	data, err := readRAGFetchBody(resp.Body)
	if err != nil {
		status := http.StatusBadGateway
		if strings.Contains(err.Error(), "size limit") {
			status = http.StatusRequestEntityTooLarge
		} else if strings.Contains(err.Error(), "UTF-8") {
			status = http.StatusUnsupportedMediaType
		}
		return rag.Doc{}, newRAGFetchError(status, "fetch: %v", err)
	}
	title := strings.TrimSpace(input.Title)
	text := ""
	kind := "web"
	if isWikipedia {
		wikiTitle, wikiText, decodeErr := decodeWikipediaArticle(data)
		if decodeErr != nil {
			return rag.Doc{}, newRAGFetchError(http.StatusUnprocessableEntity, "%v", decodeErr)
		}
		text, kind = wikiText, "wikipedia"
		if title == "" {
			title = wikiTitle
		}
	} else {
		isHTML, ok := fetchContentAllowed(resp.Header.Get("Content-Type"))
		if !ok {
			return rag.Doc{}, newRAGFetchError(http.StatusUnsupportedMediaType, "unsupported content type: %s", resp.Header.Get("Content-Type"))
		}
		text = string(data)
		if isHTML {
			extractedTitle, extractedText := extractRAGHTML(data)
			text = extractedText
			if title == "" {
				title = extractedTitle
			}
		}
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return rag.Doc{}, newRAGFetchError(http.StatusUnprocessableEntity, "fetched content had no extractable text")
	}

	id := strings.TrimSpace(input.ID)
	if id == "" {
		digest := sha256.Sum256([]byte(sourceURL))
		id = fmt.Sprintf("url-%x", digest[:12])
	}
	if title == "" {
		title = sourceURL
	}
	return rag.Doc{ID: id, Title: title, Path: sourceURL, Text: text, Time: time.Now(), Kind: kind}, nil
}

// docByID looks up one Doc's summary by ID, for reporting back the document a
// mutation just indexed. rag.Index exposes no direct by-ID lookup (see its
// package doc on the exported surface staying small), so this scans List() —
// fine at the size a listing UI already renders in full.
func docByID(ix *rag.Index, id string) rag.DocInfo {
	for _, d := range ix.List() {
		if d.ID == id {
			return d
		}
	}
	return rag.DocInfo{ID: id}
}

func docExists(ix *rag.Index, id string) bool {
	for _, doc := range ix.List() {
		if doc.ID == id {
			return true
		}
	}
	return false
}

func docInfoJSON(d rag.DocInfo) map[string]any {
	m := map[string]any{"id": d.ID, "title": d.Title, "path": d.Path, "kind": d.Kind, "chunks": d.Chunks}
	if !d.Time.IsZero() {
		m["time"] = d.Time.Format(time.RFC3339)
	}
	return m
}

func hitJSON(h rag.Hit) map[string]any {
	return map[string]any{
		"doc_id": h.Chunk.DocID, "chunk_index": h.Chunk.Index,
		"title": h.Title, "path": h.Path, "kind": h.Kind,
		"score": h.Score, "excerpt": h.Text,
	}
}
