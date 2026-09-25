package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const maxDecisionCSVUpload = 64 << 20
const maxDecisionCSVOutput = 128 << 20

var errDecisionCSVOutputLimit = errors.New("CSV output exceeds 128 MiB")

type decisionCSVWriter struct {
	w         io.Writer
	remaining int64
}

func (w *decisionCSVWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, errDecisionCSVOutputLimit
	}
	n, e := w.w.Write(p)
	w.remaining -= int64(n)
	return n, e
}
func registerDecisionCSVRoutes(mux *http.ServeMux, sem chan struct{}, opts HandlerOptions) {
	mux.HandleFunc("/classify", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			http.Error(w, "use GET", 405)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		io.WriteString(w, decisionCSVPage)
	})
	mux.HandleFunc("/v1/systemone/csv", withLimit(sem, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			writeAPIError(w, 405, "invalid_request", "", "use POST")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxDecisionCSVUpload)
		// A small RAM budget spills the file to disk. Always clean it up, including
		// partially parsed forms. No user-supplied filename is used as a local path.
		defer func() {
			if r.MultipartForm != nil {
				r.MultipartForm.RemoveAll()
			}
		}()
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			csvUploadError(w, err)
			return
		}
		form := r.MultipartForm
		if len(form.File["file"]) != 1 || len(form.File) != 1 || len(form.Value["options"]) != 1 || len(form.Value) != 1 {
			writeAPIError(w, 400, "invalid_request", "", "provide exactly one file and one options JSON field")
			return
		}
		raw := form.Value["options"][0]
		if len(raw) > 256<<10 {
			writeAPIError(w, 413, "invalid_request", "options", "options exceed 256 KiB")
			return
		}
		var csvOpts gopherllm.DecisionCSVOptions
		dec := json.NewDecoder(bytes.NewBufferString(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&csvOpts); err != nil {
			writeAPIError(w, 400, "invalid_request", "options", err.Error())
			return
		}
		var extra any
		if err := dec.Decode(&extra); err != io.EOF {
			writeAPIError(w, 400, "invalid_request", "options", "expected one options object")
			return
		}
		if err := csvOpts.Validate(); err != nil {
			writeAPIError(w, 422, "invalid_request", "options", err.Error())
			return
		}
		file, err := form.File["file"][0].Open()
		if err != nil {
			inferenceAPIError(w, err)
			return
		}
		defer file.Close()
		// Spool output so a malformed late record/inference error never becomes an
		// apparently successful, truncated CSV download. Inference remains row-wise.
		result, err := os.CreateTemp("", "gopherllm-decisions-*.csv")
		if err != nil {
			inferenceAPIError(w, err)
			return
		}
		defer func() { result.Close(); os.Remove(result.Name()) }()
		bounded := &decisionCSVWriter{w: result, remaining: maxDecisionCSVOutput}
		stats, err := opts.DecisionModel.ClassifyCSV(r.Context(), file, bounded, csvOpts)
		if err != nil {
			switch {
			case errors.Is(err, errDecisionCSVOutputLimit):
				writeAPIError(w, 413, "invalid_request", "", err.Error())
			case errors.Is(err, gopherllm.ErrInvalidDecision):
				writeAPIError(w, 422, "invalid_request", "", err.Error())
			default:
				inferenceAPIError(w, err)
			}
			return
		}
		if _, err = result.Seek(0, io.SeekStart); err != nil {
			inferenceAPIError(w, err)
			return
		}
		if err = r.Context().Err(); err != nil {
			inferenceAPIError(w, err)
			return
		}
		w.Header().Set("Content-Type", "text/csv; charset=utf-8")
		w.Header().Set("Content-Disposition", `attachment; filename="classified.csv"`)
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Processed-Rows", strconv.Itoa(stats.Rows))
		w.Header().Set("Content-Length", strconv.FormatInt(maxDecisionCSVOutput-bounded.remaining, 10))
		if _, err = io.Copy(w, result); err != nil && opts.LogWriter != nil {
			fmt.Fprintf(opts.LogWriter, "CSV response write: %v\n", err)
		}
	}))
}
func csvUploadError(w http.ResponseWriter, err error) {
	var tooLarge *http.MaxBytesError
	status := 400
	if errors.As(err, &tooLarge) || errors.Is(err, multipart.ErrMessageTooLarge) {
		status = 413
	}
	writeAPIError(w, status, "invalid_request", "file", err.Error())
}
