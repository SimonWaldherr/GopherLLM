package server

import (
	"context"
	"net/http"
	"time"
)

// RequestObservation contains transport metrics only, never request/response
// payloads or authorization headers. Duration includes admission and writing.
// Status is the committed HTTP status; an SSE stream may carry a later error.
type RequestObservation struct {
	Endpoint  string
	Status    int
	Duration  time.Duration
	Bytes     int64
	Cancelled bool
}
type observedWriter struct {
	http.ResponseWriter
	status int
	bytes  int64
}

func (w *observedWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
		w.ResponseWriter.WriteHeader(code)
	}
}
func (w *observedWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(200)
	}
	n, err := w.ResponseWriter.Write(b)
	w.bytes += int64(n)
	return n, err
}

type observedFlusher struct{ *observedWriter }

func (w *observedFlusher) Flush() {
	if w.status == 0 {
		w.WriteHeader(200)
	}
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}
func (w *observedWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }
func observeRequests(next http.Handler, observe func(RequestObservation)) http.Handler {
	if observe == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		out := &observedWriter{ResponseWriter: w}
		defer func() {
			status := out.status
			if status == 0 {
				status = 200
			}
			endpoint := r.URL.Path
			switch endpoint {
			case "/v1/systemone/csv", "/v1/systemone", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/models", "/generate":
			default:
				endpoint = "other"
			}
			observe(RequestObservation{endpoint, status, time.Since(start), out.bytes, r.Context().Err() == context.Canceled || r.Context().Err() == context.DeadlineExceeded})
		}()
		if _, ok := w.(http.Flusher); ok {
			next.ServeHTTP(&observedFlusher{out}, r)
		} else {
			next.ServeHTTP(out, r)
		}
	})
}

func withInferenceDeadline(next http.Handler, timeout time.Duration) http.Handler {
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/systemone/csv", "/v1/systemone", "/generate", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/generate", "/api/chat", "/api/embed", "/api/embeddings":
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			defer cancel()
			r = r.WithContext(ctx)
		}
		next.ServeHTTP(w, r)
	})
}
