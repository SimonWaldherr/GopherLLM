package hestia

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
)

// ErrOutcomeUnknown marks a turn whose device action may or may not have
// taken effect (CONCEPT.md section 7: "outcome_unknown"). Callers must
// offer a state check, not a blind retry.
var ErrOutcomeUnknown = errors.New("hestia: action outcome unknown")

// App wires the planner, resolver, executor, inventory, journal and
// knowledge base into the text- and voice-command paths (CONCEPT.md
// section 15, steps 1, 3 and 4): "Sprachauftrag steuert den gleichen
// Executor wie Texteingabe, einschließlich sichtbarer
// Transkriptrevisionen."
type App struct {
	Journal   *Journal
	Adapter   adapters.Adapter
	Inventory Inventory
	Speech    *SpeechHost    // nil if no voice model is configured
	Knowledge *KnowledgeBase // nil if no knowledge base is configured

	mu    sync.Mutex
	turns map[string]*Turn
	voice map[string]*VoiceSession // turnID -> in-progress recording
}

// NewApp constructs an App. inventory is the full server-side device set;
// callers are responsible for narrowing it per requesting person before
// it reaches here, once identity/permissions exist (CONCEPT.md section 7).
// speech and knowledge may be nil; their respective endpoints then report
// unavailable while the text/device-action path keeps working (CONCEPT.md
// section 3).
func NewApp(journal *Journal, adapter adapters.Adapter, inventory Inventory, speech *SpeechHost, knowledge *KnowledgeBase) *App {
	return &App{
		Journal: journal, Adapter: adapter, Inventory: inventory, Speech: speech, Knowledge: knowledge,
		turns: make(map[string]*Turn), voice: make(map[string]*VoiceSession),
	}
}

func newID(prefix string) string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

func (a *App) Turn(turnID string) (*Turn, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	t, ok := a.turns[turnID]
	return t, ok
}

func (a *App) journal(conversationID, turnID, typ string, payload map[string]any) {
	_, _ = a.Journal.Append(Event{
		SchemaVersion: 1, EventID: newID("evt"),
		ConversationID: conversationID, TurnID: turnID, Type: typ, Payload: payload,
	})
}

// ProcessText starts a new turn from finalized text input (CONCEPT.md
// section 2's text-equivalent path). Voice turns reach the same
// interpretation/resolution/execution logic through FinishVoiceTurn below,
// once their transcript is finalized.
func (a *App) ProcessText(ctx context.Context, conversationID, text string) (*Turn, error) {
	turnID := newID("turn")
	t := &Turn{ID: turnID, ConversationID: conversationID, Status: TurnCreated, InputText: text}
	a.journal(conversationID, turnID, "turn.created", map[string]any{"input_text": text})

	a.interpretAndAct(ctx, t)

	a.mu.Lock()
	a.turns[turnID] = t
	a.mu.Unlock()
	return t, nil
}

// interpretAndAct runs the planner/resolver/executor chain against
// t.InputText, mutating t in place. It is the single path both ProcessText
// and FinishVoiceTurn funnel through, satisfying CONCEPT.md section 15
// step 3's "Sprachauftrag steuert den gleichen Executor wie Texteingabe".
func (a *App) interpretAndAct(ctx context.Context, t *Turn) {
	t.Status = TurnInterpreting
	proposal := ParseTextProposal(t.InputText)
	t.Proposal = &proposal
	a.journal(t.ConversationID, t.ID, "plan.proposed", map[string]any{
		"intent": string(proposal.Intent), "action": proposal.Action,
		"target_phrase": proposal.TargetPhrase, "room_phrase": proposal.RoomPhrase, "scope": proposal.Scope,
	})

	switch proposal.Intent {
	case IntentCancel:
		t.Status = TurnCancelled
		t.Result = "Auftrag abgebrochen."
		a.journal(t.ConversationID, t.ID, "turn.cancelled", nil)
	case IntentConversation:
		t.Status = TurnCompleted
		t.Result = "Das ist eine Frage oder Erklärung, keine Geräteaktion."
		a.journal(t.ConversationID, t.ID, "turn.completed", map[string]any{"kind": "conversation"})
	case IntentKnowledgeQuery:
		a.answerKnowledgeQuery(ctx, t, proposal.TargetPhrase)
	case IntentDeviceAction:
		a.resolveAndAct(ctx, t, proposal)
	default:
		t.Status = TurnFailed
		t.Result = fmt.Sprintf("Unbekannter Intent %q.", proposal.Intent)
		a.journal(t.ConversationID, t.ID, "turn.failed", map[string]any{"reason": "unknown_intent"})
	}
}

// StartVoiceTurn creates a turn in the "capturing" state and reserves the
// speech host's one recording slot (CONCEPT.md section 3's capacity
// limit -- surfaced to the caller as ErrVoxtralBusy when another recording
// is already in progress).
func (a *App) StartVoiceTurn(ctx context.Context, conversationID string) (*Turn, error) {
	if a.Speech == nil {
		return nil, ErrSpeechNotConfigured
	}
	vs, err := a.Speech.StartVoiceSession(ctx)
	if err != nil {
		return nil, err
	}
	turnID := newID("turn")
	t := &Turn{ID: turnID, ConversationID: conversationID, Status: TurnCapturing}
	a.journal(conversationID, turnID, "turn.created", map[string]any{"kind": "voice"})
	a.journal(conversationID, turnID, "audio.recording_started", nil)

	a.mu.Lock()
	a.turns[turnID] = t
	a.voice[turnID] = vs
	a.mu.Unlock()
	return t, nil
}

// PushVoiceAudio feeds one chunk of 16kHz mono PCM16 to a capturing turn
// and updates its live (partial, never acted-on) transcript.
func (a *App) PushVoiceAudio(ctx context.Context, turnID string, pcm []byte) (*Turn, error) {
	t, vs, err := a.voiceTurn(turnID)
	if err != nil {
		return nil, err
	}
	text, err := vs.PushPCM16(ctx, pcm)
	if err != nil {
		return nil, fmt.Errorf("hestia: pushing audio: %w", err)
	}
	t.InputText = text
	return t, nil
}

// FinishVoiceTurn flushes the recording to a final transcript revision,
// releases the recording slot, and runs the same interpret/resolve/execute
// chain ProcessText uses (CONCEPT.md section 6: "Stoppen bedeutet zunächst
// Audio-Flush, anschließend Interpretation.").
func (a *App) FinishVoiceTurn(ctx context.Context, turnID string) (*Turn, error) {
	t, vs, err := a.voiceTurn(turnID)
	if err != nil {
		return nil, err
	}
	text, err := vs.Finish(ctx)
	a.mu.Lock()
	delete(a.voice, turnID)
	a.mu.Unlock()
	if err != nil {
		t.Status = TurnFailed
		t.Result = "Sprachaufnahme fehlgeschlagen: " + err.Error()
		a.journal(t.ConversationID, t.ID, "turn.failed", map[string]any{"reason": err.Error()})
		return t, nil
	}
	t.InputText = text
	t.Status = TurnFinalizing
	a.journal(t.ConversationID, t.ID, "audio.transcript_finalized", map[string]any{"input_text": text})

	a.interpretAndAct(ctx, t)
	return t, nil
}

// CancelVoiceTurn aborts a capturing/finalizing recording without
// interpreting it (CONCEPT.md section 13: POST /api/turns/{id}/cancel).
func (a *App) CancelVoiceTurn(turnID string) (*Turn, error) {
	t, vs, err := a.voiceTurn(turnID)
	if err != nil {
		return nil, err
	}
	if abortErr := vs.Abort(); abortErr != nil {
		return nil, fmt.Errorf("hestia: cancelling recording: %w", abortErr)
	}
	a.mu.Lock()
	delete(a.voice, turnID)
	a.mu.Unlock()
	t.Status = TurnCancelled
	t.Result = "Aufnahme abgebrochen."
	a.journal(t.ConversationID, t.ID, "turn.cancelled", map[string]any{"kind": "voice"})
	return t, nil
}

func (a *App) voiceTurn(turnID string) (*Turn, *VoiceSession, error) {
	a.mu.Lock()
	t, ok := a.turns[turnID]
	vs, vsOK := a.voice[turnID]
	a.mu.Unlock()
	if !ok || !vsOK {
		return nil, nil, fmt.Errorf("hestia: %q is not an in-progress voice turn", turnID)
	}
	return t, vs, nil
}

// answerKnowledgeQuery runs a HestiaRAG search and journals the result
// (CONCEPT.md section 15 step 4). It never touches a device: retrieval and
// device control are separate intents with separate code paths, so
// document content can never authorize a device action (CONCEPT.md
// section 10: "Dokumentinhalt kann keine Geräteaktion autorisieren").
func (a *App) answerKnowledgeQuery(ctx context.Context, t *Turn, query string) {
	if a.Knowledge == nil {
		t.Status = TurnCompleted
		t.Result = "Keine Wissensdatenbank konfiguriert."
		a.journal(t.ConversationID, t.ID, "turn.completed", map[string]any{"kind": "knowledge_query_unconfigured"})
		return
	}
	hits, err := a.Knowledge.Search(ctx, query, 3)
	if err != nil {
		t.Status = TurnFailed
		t.Result = "Dokumentensuche fehlgeschlagen: " + err.Error()
		a.journal(t.ConversationID, t.ID, "turn.failed", map[string]any{"reason": err.Error()})
		return
	}
	t.Status = TurnCompleted
	if len(hits) == 0 {
		t.Result = "Dazu wurde nichts in den Dokumenten gefunden."
		a.journal(t.ConversationID, t.ID, "turn.completed", map[string]any{"kind": "knowledge_query", "hit_count": 0})
		return
	}
	var sb strings.Builder
	sources := make([]string, 0, len(hits))
	for i, h := range hits {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		sb.WriteString(h.Content)
		sources = append(sources, h.Title)
	}
	t.Result = sb.String()
	a.journal(t.ConversationID, t.ID, "turn.completed", map[string]any{"kind": "knowledge_query", "hit_count": len(hits), "sources": sources})
}

func (a *App) resolveAndAct(ctx context.Context, t *Turn, proposal Proposal) {
	result, err := Resolve(a.Inventory, proposal)
	if err != nil {
		t.Status = TurnFailed
		t.Result = err.Error()
		a.journal(t.ConversationID, t.ID, "turn.failed", map[string]any{"reason": err.Error()})
		return
	}
	switch result.Outcome {
	case ResolveNone:
		t.Status = TurnCompleted
		t.Result = "Kein passendes Gerät gefunden."
		a.journal(t.ConversationID, t.ID, "resolve.none", map[string]any{"target_phrase": proposal.TargetPhrase, "room_phrase": proposal.RoomPhrase})
	case ResolveAmbiguous:
		t.Candidates = result.Candidates
		t.Status = TurnClarifying
		t.Result = "Mehrdeutig: bitte ein Gerät auswählen."
		a.journal(t.ConversationID, t.ID, "clarification.requested", map[string]any{"candidate_count": len(result.Candidates)})
	case ResolveGroup:
		t.Candidates = result.Candidates
		t.Status = TurnClarifying
		t.Result = fmt.Sprintf("Gruppenaktion für %d Geräte: bitte bestätigen.", len(result.Candidates))
		a.journal(t.ConversationID, t.ID, "group.confirmation_requested", map[string]any{"candidate_count": len(result.Candidates)})
	case ResolveSingle:
		device := result.Candidates[0]
		t.ResolvedTarget = &device
		a.journal(t.ConversationID, t.ID, "resolve.single", map[string]any{"device_id": device.ID})
		a.execute(ctx, t, []Device{device}, proposal.Action, proposal.Arguments)
	}
}

// Confirm proceeds a clarifying turn: deviceID selects one candidate from
// an ambiguous single-target turn, or "" confirms an already-resolved
// group turn's full candidate list (CONCEPT.md section 7, rule 4).
func (a *App) Confirm(ctx context.Context, turnID, deviceID string) (*Turn, error) {
	a.mu.Lock()
	t, ok := a.turns[turnID]
	a.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("hestia: unknown turn %q", turnID)
	}
	if t.Status != TurnClarifying {
		return nil, fmt.Errorf("hestia: turn %q is not awaiting clarification (status %s)", turnID, t.Status)
	}
	if t.Proposal == nil {
		return nil, fmt.Errorf("hestia: turn %q has no proposal", turnID)
	}

	if t.Proposal.Scope == "group" {
		a.journal(t.ConversationID, t.ID, "group.confirmation_received", map[string]any{"candidate_count": len(t.Candidates)})
		a.execute(ctx, t, t.Candidates, t.Proposal.Action, t.Proposal.Arguments)
		return t, nil
	}

	var chosen *Device
	for i := range t.Candidates {
		if t.Candidates[i].ID == deviceID {
			chosen = &t.Candidates[i]
			break
		}
	}
	if chosen == nil {
		return nil, fmt.Errorf("hestia: %q is not one of this turn's candidates", deviceID)
	}
	t.ResolvedTarget = chosen
	a.journal(t.ConversationID, t.ID, "clarification.answered", map[string]any{"device_id": chosen.ID})
	a.execute(ctx, t, []Device{*chosen}, t.Proposal.Action, t.Proposal.Arguments)
	return t, nil
}

func (a *App) execute(ctx context.Context, t *Turn, targets []Device, action string, args map[string]any) {
	t.Status = TurnExecuting
	a.journal(t.ConversationID, t.ID, "execution.started", map[string]any{"action": action, "target_count": len(targets)})

	if len(targets) == 1 {
		state, err := ExecuteSingle(ctx, a.Adapter, targets[0], action, args)
		if err != nil {
			if errors.Is(err, ErrOutcomeUnknown) {
				t.Status = TurnOutcomeUnknown
				t.Result = "Status nach Ausführung unklar: " + err.Error()
				a.journal(t.ConversationID, t.ID, "execution.outcome_unknown", map[string]any{"device_id": targets[0].ID, "error": err.Error()})
				return
			}
			t.Status = TurnFailed
			t.Result = err.Error()
			a.journal(t.ConversationID, t.ID, "execution.failed", map[string]any{"device_id": targets[0].ID, "error": err.Error()})
			return
		}
		t.Status = TurnCompleted
		t.Result = fmt.Sprintf("%s: %v", targets[0].Name, state)
		a.journal(t.ConversationID, t.ID, "device.state_observed", map[string]any{"device_id": targets[0].ID, "state": state, "source": "device_readback"})
		return
	}

	results := ExecuteGroup(ctx, a.Adapter, targets, action, args)
	failures := 0
	for id, err := range results {
		if err != nil {
			failures++
			a.journal(t.ConversationID, t.ID, "execution.failed", map[string]any{"device_id": id, "error": err.Error()})
		} else {
			a.journal(t.ConversationID, t.ID, "device.state_observed", map[string]any{"device_id": id, "source": "device_readback"})
		}
	}
	if failures == 0 {
		t.Status = TurnCompleted
		t.Result = fmt.Sprintf("%d Geräte erfolgreich geschaltet.", len(targets))
	} else if failures < len(targets) {
		t.Status = TurnOutcomeUnknown
		t.Result = fmt.Sprintf("%d von %d Geräten fehlgeschlagen.", failures, len(targets))
	} else {
		t.Status = TurnFailed
		t.Result = "Alle Geräte fehlgeschlagen."
	}
}
