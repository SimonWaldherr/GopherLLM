// Complaints Assistant is a deliberately small local demo server. It keeps the
// browser same-origin and calls GopherLLM as an in-process Go package.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unicode/utf8"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const maxRequestBytes = 24 << 10
const maxDetailsRunes = 4000

type assistRequest struct {
	Order struct {
		Number       string `json:"number"`
		Product      string `json:"product"`
		PurchasedAt  string `json:"purchased_at"`
		Price        string `json:"price"`
		WarrantyEnds string `json:"warranty_ends"`
	} `json:"order"`
	IssueType string `json:"issue_type"`
	Details   string `json:"details"`
}

// assistResponse mirrors the JSON shape supportSystemPrompt asks the model
// for. Decoding into it (instead of forwarding the model's raw text) means a
// malformed reply fails clearly on the server, not as an opaque parse error
// in the browser.
type assistResponse struct {
	Classification      string   `json:"classification"`
	Priority            string   `json:"priority"`
	RequiresHumanReview bool     `json:"requires_human_review"`
	RecommendedActions  []string `json:"recommended_actions"`
	CustomerReply       string   `json:"customer_reply"`
	SafetyNotice        string   `json:"safety_notice"`
}

// stripCodeFence removes one leading/trailing ``` fence, in case the model
// wraps its JSON despite supportSystemPrompt asking it not to.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			s = s[i+1:]
		} else {
			s = strings.TrimPrefix(s, "```")
		}
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8090", "demo listen address")
	modelPath := flag.String("model", "", "path to local GGUF model (required)")
	flag.Parse()
	if strings.TrimSpace(*modelPath) == "" {
		log.Fatal("-model /path/to/model.gguf is required")
	}
	model, err := gopherllm.Open(context.Background(), *modelPath, gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatalf("load local model: %v", err)
	}
	defer model.Close()

	// Serve the static files next to this source file rather than relative
	// to the process's working directory, so `go run .` from inside this
	// directory works the same as `go run ./examples/complaints-assistant`
	// from the repo root.
	_, thisFile, _, _ := runtime.Caller(0)
	staticDir := filepath.Dir(thisFile)

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(staticDir)))
	mux.HandleFunc("/api/assist", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "POST required", http.StatusMethodNotAllowed)
			return
		}
		defer r.Body.Close()
		var in assistRequest
		decoder := json.NewDecoder(io.LimitReader(r.Body, maxRequestBytes))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&in); err != nil {
			http.Error(w, "invalid demo request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(in.Order.Number) == "" || strings.TrimSpace(in.Details) == "" {
			http.Error(w, "choose an order and describe the issue", http.StatusBadRequest)
			return
		}
		if utf8.RuneCountInString(in.Details) > maxDetailsRunes {
			http.Error(w, "issue description is too long", http.StatusBadRequest)
			return
		}

		prompt := fmt.Sprintf("Order: %s\nProduct: %s\nPurchase date: %s\nPrice: %s\nWarranty until: %s\nIssue category: %s\nCustomer description: %s", in.Order.Number, in.Order.Product, in.Order.PurchasedAt, in.Order.Price, in.Order.WarrantyEnds, in.IssueType, in.Details)
		generated, err := model.Generate(r.Context(), prompt,
			gopherllm.WithSystemPrompt(supportSystemPrompt),
			gopherllm.WithMaxTokens(420),
			gopherllm.WithTemperature(0.2),
		)
		if err != nil {
			http.Error(w, "local GopherLLM generation failed: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		if generated.FinishReason == "length" {
			http.Error(w, "the model's draft was cut off before it finished; try a shorter issue description", http.StatusServiceUnavailable)
			return
		}
		var out assistResponse
		if err := json.Unmarshal([]byte(stripCodeFence(generated.Text)), &out); err != nil {
			http.Error(w, "the model's draft was not valid JSON: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(out)
	})

	log.Printf("Complaints Assistant demo: http://%s (in-process model: %s)", *addr, *modelPath)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

const supportSystemPrompt = `You are a careful customer-support intake assistant. Return ONLY valid JSON with this exact shape:
{"classification":"short category","priority":"normal|needs_human_review|safety_review","requires_human_review":true,"recommended_actions":["short action"],"customer_reply":"short friendly reply in the customer's language","safety_notice":"string or empty"}

Do not promise refunds, legal outcomes, repairs, replacements, or delivery dates. Ask for missing facts only when necessary. Suggest low-risk first steps such as restart, another compatible cable, or another compatible charger. If the description mentions burning smell, smoke, sparks, swelling, leaking battery, unusual heat while charging, electric shock, or fire: set priority to safety_review, require human review, give no risky diagnostic steps, and state that the device should not continue charging or be used. Tell the customer to follow the manufacturer's safety guidance and local emergency procedure if there is immediate danger. This is a draft for a human support agent, not an automated decision.`
