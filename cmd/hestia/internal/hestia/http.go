package hestia

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/SimonWaldherr/GopherLLM/voiceweb"
)

// NewHandler builds Hestia's own http.Handler (CONCEPT.md section 5: "Die
// neue Anwendung bekommt einen eigenen http.Handler. Sie bindet nicht
// pauschal den bestehenden Chatserver samt dessen Verwaltungsrouten ein.").
// Covers the text-command path from section 15 step 1, plus the voice path
// from step 3 (POST .../audio, .../finish) using the short HTTP-PCM-request
// transport section 6 names as the first transport, not a WebSocket.
// The server treats local users as trusted and rejects cross-origin access.
func NewHandler(app *App) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/capabilities", func(w http.ResponseWriter, r *http.Request) {
		backend := ""
		if app.Speech != nil {
			backend = app.Speech.Backend()
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"speech_enabled": app.Speech != nil, "speech_backend": backend, "listen_word": app.ListenWord,
			"wiki_enabled": app.Wikipedia != nil, "web_enabled": app.WebSearch != nil, "chat_enabled": app.Chat != nil,
		})
	})

	mux.HandleFunc("POST /api/conversations", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"conversation_id": newID("conv")})
	})

	mux.HandleFunc("POST /api/conversations/{id}/turns", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Text string `json:"text"`
		}
		if !decodeJSON(w, r, &body) {
			return
		}
		if len(body.Text) > 4096 {
			writeError(w, http.StatusRequestEntityTooLarge, jsonErr("hestia: text exceeds 4096 bytes"))
			return
		}
		t, err := app.ProcessText(r.Context(), r.PathValue("id"), body.Text)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	mux.HandleFunc("POST /api/turns/{id}/confirm", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			DeviceID string `json:"device_id"`
		}
		if err := decodeJSONOptional(w, r, &body); err != nil {
			writeRequestError(w, err)
			return
		}
		t, err := app.Confirm(r.Context(), r.PathValue("id"), body.DeviceID)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	mux.HandleFunc("GET /api/turns/{id}", func(w http.ResponseWriter, r *http.Request) {
		t, ok := app.Turn(r.PathValue("id"))
		if !ok {
			writeError(w, http.StatusNotFound, errNotFound)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	mux.HandleFunc("GET /api/history", func(w http.ResponseWriter, r *http.Request) {
		conv := r.URL.Query().Get("conversation_id")
		if conv == "" {
			writeError(w, http.StatusBadRequest, errMissingConversationID)
			return
		}
		writeJSON(w, http.StatusOK, app.Journal.ByConversation(conv))
	})

	mux.HandleFunc("GET /api/devices", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, app.Inventory.Devices)
	})

	mux.HandleFunc("POST /api/conversations/{id}/turns/voice", func(w http.ResponseWriter, r *http.Request) {
		t, err := app.StartVoiceTurn(r.Context(), r.PathValue("id"))
		if err != nil {
			writeSpeechError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})
	mux.HandleFunc("POST /api/conversations/{id}/turns/listen", func(w http.ResponseWriter, r *http.Request) {
		pcm, ok := readPCM(w, r, 8*16000*2)
		if !ok {
			return
		}
		result, err := app.Listen(r.Context(), r.PathValue("id"), pcm)
		if err != nil {
			writeSpeechError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, result)
	})

	// Sequential little-endian 16-bit mono PCM16 at 16kHz, raw in the body
	// (CONCEPT.md section 6: "kurze HTTP-PCM-Requests"). One turn's chunks
	// must arrive from one client in order; this milestone does not yet
	// implement the sequence-number/gap-reporting protocol section 6
	// describes for reconnects.
	mux.HandleFunc("POST /api/turns/{id}/audio", func(w http.ResponseWriter, r *http.Request) {
		pcm, ok := readPCM(w, r, MaxVoicePCMBytes)
		if !ok {
			return
		}
		t, err := app.PushVoiceAudio(r.Context(), r.PathValue("id"), pcm)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	mux.HandleFunc("POST /api/turns/{id}/finish", func(w http.ResponseWriter, r *http.Request) {
		t, err := app.FinishVoiceTurn(r.Context(), r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	mux.HandleFunc("POST /api/turns/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {
		t, err := app.CancelVoiceTurn(r.PathValue("id"))
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		writeJSON(w, http.StatusOK, t)
	})

	// Shared with GopherLLM's own chat web UI -- see the voiceweb package
	// doc comment for why only this generic capture worklet is factored
	// out, not the rest of either UI's audio handling.
	mux.HandleFunc("GET /audio-worklet.js", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		io.WriteString(w, voiceweb.AudioWorkletJS)
	})

	mux.Handle("GET /", http.FileServer(http.FS(webFS)))

	return localOnly(mux)
}

var (
	errNotFound              = jsonErr("not found")
	errMissingConversationID = jsonErr("conversation_id is required")
)

type jsonErr string

func (e jsonErr) Error() string { return string(e) }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

// writeSpeechError maps the speech-host sentinel errors to the status
// codes the web UI needs to tell "no voice model configured" apart from
// "someone else is already recording" (CONCEPT.md section 3's transparent
// capacity limit, section 12's microphone status states).
func writeSpeechError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrSpeechNotConfigured), errors.Is(err, errListenDisabled):
		writeError(w, http.StatusServiceUnavailable, err)
	case errors.Is(err, ErrVoiceSessionBusy):
		writeError(w, http.StatusConflict, err)
	default:
		writeError(w, http.StatusInternalServerError, err)
	}
}

func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := readJSON(w, r, v); err != nil {
		writeRequestError(w, err)
		return false
	}
	return true
}

// decodeJSONOptional tolerates an empty body (confirming a group turn
// sends no device_id).
func decodeJSONOptional(w http.ResponseWriter, r *http.Request, v any) error {
	if r.ContentLength == 0 {
		return nil
	}
	return readJSON(w, r, v)
}

func readJSON(w http.ResponseWriter, r *http.Request, v any) error {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	switch err := dec.Decode(new(any)); err {
	case io.EOF:
		return nil
	case nil:
		return jsonErr("hestia: expected a single JSON object")
	default:
		return err
	}
}

func readPCM(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, bool) {
	pcm, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBytes))
	if err != nil {
		writeRequestError(w, err)
		return nil, false
	}
	if len(pcm) == 0 || len(pcm)%2 != 0 {
		writeError(w, http.StatusBadRequest, jsonErr("hestia: expected non-empty mono PCM16 audio"))
		return nil, false
	}
	return pcm, true
}

func writeRequestError(w http.ResponseWriter, err error) {
	var sizeErr *http.MaxBytesError
	if errors.As(err, &sizeErr) {
		writeError(w, http.StatusRequestEntityTooLarge, sizeErr)
		return
	}
	writeError(w, http.StatusBadRequest, err)
}
