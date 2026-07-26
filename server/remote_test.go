package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNormalizeRemoteBaseURL(t *testing.T) {
	for _, test := range []struct {
		in, want string
	}{
		{"http://127.0.0.1:11434", "http://127.0.0.1:11434/v1"},
		{"http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080/v1"},
		{"https://api.example.test/v1", "https://api.example.test/v1"},
	} {
		got, err := normalizeRemoteBaseURL(test.in)
		if err != nil || got != test.want {
			t.Fatalf("normalizeRemoteBaseURL(%q) = %q, %v; want %q", test.in, got, err, test.want)
		}
	}
	for _, input := range []string{"ftp://example.test", "https://user@example.test", "https://example.test/?x=1"} {
		if _, err := normalizeRemoteBaseURL(input); err == nil {
			t.Fatalf("normalizeRemoteBaseURL(%q) succeeded", input)
		}
	}
}

func TestRemoteChatProxyWorksWithoutLocalModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %q", req.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["model"] != "remote-test" {
			t.Fatalf("model = %#v", body["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"remote answer"}}]}`))
	}))
	defer upstream.Close()

	server := httptest.NewServer(NewHandler(nil, HandlerOptions{}))
	defer server.Close()
	config := `{"base_url":"` + upstream.URL + `","model":"remote-test"}`
	response, err := http.Post(server.URL+"/remote", "application/json", strings.NewReader(config))
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK {
		t.Fatalf("configure status = %d", response.StatusCode)
	}
	response.Body.Close()

	response, err = http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("proxy status = %d", response.StatusCode)
	}
}
