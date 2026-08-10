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
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const maxRequestBytes = 24 << 10

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

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir("examples/complaints-assistant")))
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
		if len(in.Details) > 4000 {
			http.Error(w, "issue description is too long", http.StatusBadRequest)
			return
		}

		prompt := fmt.Sprintf("Order: %s\nProduct: %s\nPurchase date: %s\nPrice: %s\nWarranty until: %s\nIssue category: %s\nCustomer description: %s", in.Order.Number, in.Order.Product, in.Order.PurchasedAt, in.Order.Price, in.Order.WarrantyEnds, in.IssueType, in.Details)
		response, err := model.Generate(r.Context(), prompt,
			gopherllm.WithSystemPrompt(supportSystemPrompt),
			gopherllm.WithMaxTokens(420),
			gopherllm.WithTemperature(0.2),
		)
		if err != nil {
			http.Error(w, "local GopherLLM generation failed: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(map[string]string{"draft": response.Text})
	})

	log.Printf("Complaints Assistant demo: http://%s (in-process model: %s)", *addr, *modelPath)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

const supportSystemPrompt = `You are a careful customer-support intake assistant. Return ONLY valid JSON with this exact shape:
{"classification":"short category","priority":"normal|needs_human_review|safety_review","requires_human_review":true,"recommended_actions":["short action"],"customer_reply":"short friendly reply in the customer's language","safety_notice":"string or empty"}

Do not promise refunds, legal outcomes, repairs, replacements, or delivery dates. Ask for missing facts only when necessary. Suggest low-risk first steps such as restart, another compatible cable, or another compatible charger. If the description mentions burning smell, smoke, sparks, swelling, leaking battery, unusual heat while charging, electric shock, or fire: set priority to safety_review, require human review, give no risky diagnostic steps, and state that the device should not continue charging or be used. Tell the customer to follow the manufacturer's safety guidance and local emergency procedure if there is immediate danger. This is a draft for a human support agent, not an automated decision.`
