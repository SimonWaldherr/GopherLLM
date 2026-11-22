package adapters

import (
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
)

type homeAssistantTransport func(*http.Request) (*http.Response, error)

func (f homeAssistantTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func homeAssistantResponse(code int, body string) *http.Response {
	return &http.Response{StatusCode: code, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

func testHomeAssistant(t *testing.T) *HomeAssistant {
	t.Helper()
	a, err := NewHomeAssistant("http://127.0.0.1:8123/", "test-secret", []Entity{{ID: "light.floor_lamp", Name: "Stehlampe", Room: "Wohnzimmer"}})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestHomeAssistantTargetsOneLightAndReadsItsState(t *testing.T) {
	a := testHomeAssistant(t)
	want := []struct {
		method, path string
		brightness   float64
	}{
		{http.MethodGet, "/api/states/light.floor_lamp", 0},
		{http.MethodPost, "/api/services/light/turn_on", 0},
		{http.MethodPost, "/api/services/light/turn_on", 30.5},
		{http.MethodPost, "/api/services/light/turn_off", 0},
		{http.MethodPost, "/api/services/light/turn_off", 0},
	}
	i := 0
	a.client.Transport = homeAssistantTransport(func(req *http.Request) (*http.Response, error) {
		if i >= len(want) {
			t.Fatal("unexpected network request")
		}
		check := want[i]
		i++
		if req.Method != check.method || req.URL.Host != "127.0.0.1:8123" || req.URL.Path != check.path || req.URL.RawQuery != "" || req.Header.Get("Authorization") != "Bearer test-secret" {
			t.Fatalf("unexpected request %s %s", req.Method, req.URL)
		}
		if req.Method == http.MethodGet {
			return homeAssistantResponse(200, `{"entity_id":"light.floor_lamp","state":"on","attributes":{"brightness":128}}`), nil
		}
		var body map[string]any
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["entity_id"] != "light.floor_lamp" || (check.brightness == 0 && len(body) != 1) ||
			(check.brightness != 0 && (len(body) != 2 || body["brightness_pct"] != check.brightness)) {
			t.Fatalf("unexpected service payload: %v", body)
		}
		return homeAssistantResponse(200, `[]`), nil
	})
	ctx := t.Context()
	entities, err := a.Inventory(ctx)
	if err != nil || len(entities) != 1 || entities[0].Kind != "light" {
		t.Fatalf("Inventory() = %+v, %v", entities, err)
	}
	entities[0].ID = "light.other"
	st, err := a.State(ctx, "light.floor_lamp")
	if err != nil || st["power"] != true || st["brightness"] != 128.0*100/255 {
		t.Fatalf("State() = %+v, %v", st, err)
	}
	for _, action := range []struct {
		name, key string
		value     any
	}{
		{"set_power", "power", true},
		{"set_brightness", "brightness", 30.5},
		{"set_brightness", "brightness", 0},
		{"set_power", "power", false},
	} {
		if err := a.Apply(ctx, "light.floor_lamp", action.name, map[string]any{action.key: action.value}); err != nil {
			t.Fatal(err)
		}
	}
	if i != len(want) {
		t.Fatalf("made %d requests, want %d", i, len(want))
	}
}

func TestHomeAssistantRejectsInvalidConfigAndActions(t *testing.T) {
	light := Entity{ID: "light.floor_lamp", Name: "Stehlampe", Room: "Wohnzimmer"}
	for _, endpoint := range []string{"", "http://homeassistant.local", "https://user:pass@example.com", "https://example.com/api", "https://example.com/?token=secret"} {
		if _, err := NewHomeAssistant(endpoint, "secret", []Entity{light}); err == nil {
			t.Errorf("accepted invalid origin %q", endpoint)
		}
	}
	for _, lights := range [][]Entity{nil, {light, light}, {{ID: "switch.floor_lamp", Name: "Lampe", Room: "Raum"}}, {{ID: "light.empty"}}} {
		if _, err := NewHomeAssistant("https://homeassistant.local:8123", "secret", lights); err == nil {
			t.Errorf("accepted invalid lights %+v", lights)
		}
	}
	if _, err := NewHomeAssistant("https://homeassistant.local:8123", "secret\nsecret", []Entity{light}); err == nil || strings.Contains(err.Error(), "secret") {
		t.Fatalf("invalid token error = %v", err)
	}
	a := testHomeAssistant(t)
	if a.client.Transport.(*http.Transport).Proxy != nil || a.client.Jar != nil {
		t.Fatal("Home Assistant client inherited proxy settings or cookies")
	}
	a.client.Transport = homeAssistantTransport(func(*http.Request) (*http.Response, error) {
		t.Fatal("invalid request reached Home Assistant")
		return nil, nil
	})
	if _, err := a.State(t.Context(), "light.other"); err == nil {
		t.Fatal("read an unconfigured entity")
	}
	for _, attempt := range []struct {
		id, action string
		args       map[string]any
	}{
		{"light.other", "set_power", map[string]any{"power": true}},
		{"light.floor_lamp", "toggle", nil},
		{"light.floor_lamp", "set_power", map[string]any{"power": "true"}},
		{"light.floor_lamp", "set_brightness", map[string]any{"brightness": math.NaN()}},
		{"light.floor_lamp", "set_brightness", map[string]any{"brightness": 101}},
	} {
		if err := a.Apply(t.Context(), attempt.id, attempt.action, attempt.args); err == nil {
			t.Errorf("invalid action accepted: %+v", attempt)
		}
	}
}

func TestHomeAssistantDoesNotForwardTokenOrUncertainResponse(t *testing.T) {
	a := testHomeAssistant(t)
	const id = "light.floor_lamp"
	args := map[string]any{"power": true}
	a.client.Transport = homeAssistantTransport(func(req *http.Request) (*http.Response, error) {
		if req.URL.Host != "127.0.0.1:8123" {
			t.Fatal("redirect was followed")
		}
		response := homeAssistantResponse(http.StatusTemporaryRedirect, "test-secret")
		response.Header.Set("Location", "https://example.com/elsewhere")
		return response, nil
	})
	if err := a.Apply(t.Context(), id, "set_power", args); err == nil || strings.Contains(err.Error(), "test-secret") {
		t.Fatalf("redirect error = %v", err)
	}
	a.client.Transport = homeAssistantTransport(func(*http.Request) (*http.Response, error) {
		return homeAssistantResponse(http.StatusBadGateway, "test-secret"), nil
	})
	if err := a.Apply(t.Context(), id, "set_power", args); !errors.Is(err, ErrApplyOutcomeUnknown) || strings.Contains(err.Error(), "test-secret") {
		t.Fatalf("uncertain write error = %v", err)
	}
	for _, body := range []string{
		`{"entity_id":"light.floor_lamp","state":"unavailable"}`,
		`{"entity_id":"light.other","state":"on"}`,
		strings.Repeat(" ", homeAssistantBodyLimit+1),
	} {
		a.client.Transport = homeAssistantTransport(func(*http.Request) (*http.Response, error) {
			return homeAssistantResponse(http.StatusOK, body), nil
		})
		if _, err := a.State(t.Context(), id); err == nil {
			t.Fatal("accepted invalid readback")
		}
	}
}
