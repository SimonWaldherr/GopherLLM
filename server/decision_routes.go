package server

import (
	"errors"
	"net/http"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Classification shares the existing deployment, deadline, observation and
// admission middleware. Only an explicitly preloaded model is exposed; request
// model strings never open arbitrary paths or initiate network access.
func registerDecisionRoutes(mux *http.ServeMux, opts HandlerOptions) {
	if opts.DecisionModel == nil || opts.DeploymentMode == DeploymentBrowser {
		return
	}
	sem := make(chan struct{}, max(1, opts.MaxConcurrentRequests))
	registerDecisionCSVRoutes(mux, sem, opts)
	modelID := opts.DecisionModelID
	if modelID == "" {
		modelID = "laya-rl-agent"
	}
	mux.HandleFunc("/v1/systemone", withLimit(sem, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			writeAPIError(w, 405, "invalid_request", "", "use POST")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		req, err := gopherllm.DecodeDecisionRequest(r.Body)
		if err != nil {
			var tooLarge *http.MaxBytesError
			status := 400
			if errors.As(err, &tooLarge) {
				status = 413
			}
			writeAPIError(w, status, "invalid_request", "", err.Error())
			return
		}
		if req.Model != "" && req.Model != modelID {
			writeAPIError(w, 404, "model_not_found", "model", "requested decision model is not loaded")
			return
		}
		res, err := opts.DecisionModel.Predict(r.Context(), req)
		if err != nil {
			if errors.Is(err, gopherllm.ErrInvalidDecision) {
				writeAPIError(w, 422, "invalid_request", "", err.Error())
			} else {
				inferenceAPIError(w, err)
			}
			return
		}
		res.Model = modelID
		writeJSON(w, res)
	}))
}
