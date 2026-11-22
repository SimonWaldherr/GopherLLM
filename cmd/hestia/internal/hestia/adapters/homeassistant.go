package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/netip"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const homeAssistantBodyLimit = 1 << 20

var (
	homeAssistantLightID = regexp.MustCompile(`^light\.[a-z0-9_]{1,128}$`)
	homeAssistantToken   = regexp.MustCompile(`^[A-Za-z0-9._~+/-]+={0,2}$`)

	// ErrApplyOutcomeUnknown is returned if a write may have reached the
	// device but the server's confirmation was lost. Check its reported
	// state rather than repeating the command.
	ErrApplyOutcomeUnknown = errors.New("device write outcome unknown")
)

// HomeAssistant exposes only the configured light entities. Device labels
// and the Home Assistant endpoint are chosen by the local operator.
type HomeAssistant struct {
	endpoint string
	token    string
	client   *http.Client
	lights   []Entity
	allowed  map[string]bool
}

// NewHomeAssistant prepares a readback and action client for the configured
// Home Assistant origin. Constructing it makes no network requests.
func NewHomeAssistant(endpoint, token string, lights []Entity) (*HomeAssistant, error) {
	endpoint = strings.TrimSpace(endpoint)
	u, err := url.Parse(endpoint)
	if err != nil || u.Hostname() == "" || u.User != nil || u.ForceQuery || u.RawQuery != "" || strings.Contains(endpoint, "#") || (u.Path != "" && u.Path != "/") {
		return nil, fmt.Errorf("Home Assistant URL must be an origin without credentials, path, query or fragment")
	}
	loopbackIP, _ := netip.ParseAddr(u.Hostname())
	if u.Scheme != "https" && (u.Scheme != "http" || (!strings.EqualFold(u.Hostname(), "localhost") && !loopbackIP.IsLoopback())) {
		return nil, fmt.Errorf("Home Assistant requires HTTPS or a loopback HTTP endpoint")
	}
	token = strings.TrimSpace(token)
	if len(token) > 4096 || !homeAssistantToken.MatchString(token) {
		return nil, fmt.Errorf("Home Assistant requires a valid bearer token")
	}
	if len(lights) == 0 {
		return nil, fmt.Errorf("Home Assistant requires at least one explicitly configured light")
	}
	adapter := &HomeAssistant{
		endpoint: u.Scheme + "://" + u.Host,
		token:    token,
		client: &http.Client{
			Timeout:   5 * time.Second,
			Transport: &http.Transport{IdleConnTimeout: 30 * time.Second, MaxIdleConns: 2, ForceAttemptHTTP2: true},
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		lights:  make([]Entity, 0, len(lights)),
		allowed: make(map[string]bool, len(lights)),
	}
	for i, device := range lights {
		device.Name = strings.TrimSpace(device.Name)
		device.Room = strings.TrimSpace(device.Room)
		if !homeAssistantLightID.MatchString(device.ID) || (device.Kind != "" && device.Kind != "light") {
			return nil, fmt.Errorf("Home Assistant light %d needs a light.<entity_id>", i+1)
		}
		if !validDeviceLabel(device.Name) || !validDeviceLabel(device.Room) {
			return nil, fmt.Errorf("Home Assistant light %d needs a valid name and room", i+1)
		}
		if adapter.allowed[device.ID] {
			return nil, fmt.Errorf("Home Assistant light %d duplicates an entity", i+1)
		}
		device.Kind = "light"
		adapter.lights = append(adapter.lights, device)
		adapter.allowed[device.ID] = true
	}
	return adapter, nil
}

func validDeviceLabel(label string) bool {
	return label != "" && utf8.ValidString(label) && utf8.RuneCountInString(label) <= 120 && strings.IndexFunc(label, unicode.IsControl) == -1
}

// Inventory returns the configured allowlist without probing the instance.
func (h *HomeAssistant) Inventory(ctx context.Context) ([]Entity, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return slices.Clone(h.lights), nil
}

// State retrieves the current power and, for dimmable lights, the reported
// brightness. An unavailable or unknown state cannot confirm an action.
func (h *HomeAssistant) State(ctx context.Context, entityID string) (State, error) {
	if !h.allowed[entityID] {
		return nil, fmt.Errorf("Home Assistant light is not allowlisted")
	}
	var payload struct {
		EntityID   string `json:"entity_id"`
		State      string `json:"state"`
		Attributes struct {
			Brightness *int `json:"brightness"`
		} `json:"attributes"`
	}
	if err := h.request(ctx, http.MethodGet, "/api/states/"+entityID, nil, &payload); err != nil {
		return nil, err
	}
	if payload.EntityID != entityID || (payload.State != "on" && payload.State != "off") {
		return nil, fmt.Errorf("Home Assistant did not report a usable state for this light")
	}
	state := State{"power": payload.State == "on"}
	if payload.State == "off" {
		state["brightness"] = float64(0)
	} else if b := payload.Attributes.Brightness; b != nil {
		if *b < 0 || *b > 255 {
			return nil, fmt.Errorf("Home Assistant reported an invalid brightness")
		}
		state["brightness"] = float64(*b) * 100 / 255
	}
	return state, nil
}

// Apply issues exactly one absolute on, off, or brightness command.
func (h *HomeAssistant) Apply(ctx context.Context, entityID, action string, args map[string]any) error {
	if !h.allowed[entityID] {
		return fmt.Errorf("Home Assistant light is not allowlisted")
	}
	payload := struct {
		EntityID   string   `json:"entity_id"`
		Brightness *float64 `json:"brightness_pct,omitempty"`
	}{EntityID: entityID}
	service := "turn_off"
	switch action {
	case "set_power":
		power, ok := args["power"].(bool)
		if !ok {
			return fmt.Errorf("Home Assistant power must be a boolean")
		}
		if power {
			service = "turn_on"
		}
	case "set_brightness":
		brightness, ok := numericArg(args["brightness"])
		if !ok || math.IsNaN(brightness) || brightness < 0 || brightness > 100 {
			return fmt.Errorf("Home Assistant brightness must be between 0 and 100")
		}
		if brightness > 0 {
			service = "turn_on"
			payload.Brightness = &brightness
		}
	default:
		return fmt.Errorf("unsupported Home Assistant light action")
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("could not encode Home Assistant light action")
	}
	return h.request(ctx, http.MethodPost, "/api/services/light/"+service, bytes.NewReader(body), nil)
}

func (h *HomeAssistant) request(ctx context.Context, method, path string, body io.Reader, result any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, method, h.endpoint+path, body)
	if err != nil {
		return fmt.Errorf("could not prepare Home Assistant request")
	}
	req.Header.Set("Authorization", "Bearer "+h.token)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := h.client.Do(req)
	if err != nil {
		if method == http.MethodPost {
			return fmt.Errorf("Home Assistant request did not complete: %w", ErrApplyOutcomeUnknown)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		return fmt.Errorf("Home Assistant could not be reached")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		if method == http.MethodPost && (resp.StatusCode >= 500 || resp.StatusCode == http.StatusRequestTimeout) {
			return fmt.Errorf("Home Assistant returned HTTP %d: %w", resp.StatusCode, ErrApplyOutcomeUnknown)
		}
		return fmt.Errorf("Home Assistant returned HTTP %d", resp.StatusCode)
	}
	if result == nil {
		// The action response lists other changed entities. Readback uses
		// only the selected light, so no data from this body is retained.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, homeAssistantBodyLimit))
		return nil
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, homeAssistantBodyLimit+1))
	if err != nil || len(data) > homeAssistantBodyLimit || json.Unmarshal(data, result) != nil {
		return fmt.Errorf("could not read Home Assistant state response")
	}
	return nil
}
