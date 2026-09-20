package hestia

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
)

func testApp(t *testing.T) *App {
	t.Helper()
	dir := t.TempDir()
	journal, err := OpenJournal(filepath.Join(dir, "journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { journal.Close() })

	sim := adapters.NewSimulator(
		adapters.Entity{ID: "light_living_floor", Name: "Stehlampe", Room: "Wohnzimmer", Kind: "light"},
		adapters.Entity{ID: "light_living_ceiling", Name: "Deckenlampe", Room: "Wohnzimmer", Kind: "light"},
		adapters.Entity{ID: "light_kitchen", Name: "Lampe", Room: "Küche", Kind: "light"},
	)
	inv := Inventory{Devices: []Device{
		{ID: "light_living_floor", Name: "Stehlampe", Room: "Wohnzimmer", Kind: "light"},
		{ID: "light_living_ceiling", Name: "Deckenlampe", Room: "Wohnzimmer", Kind: "light"},
		{ID: "light_kitchen", Name: "Lampe", Room: "Küche", Kind: "light"},
	}}
	return NewApp(journal, sim, inv, nil, nil)
}

// TestSingleUnambiguousLampSwitchesDirectly is CONCEPT.md section 2's
// worked example: exactly one matching device runs immediately.
func TestSingleUnambiguousLampSwitchesDirectly(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Schalte die Stehlampe im Wohnzimmer ein.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted {
		t.Fatalf("status = %s, want completed (result: %s)", turn.Status, turn.Result)
	}
	if turn.ResolvedTarget == nil || turn.ResolvedTarget.ID != "light_living_floor" {
		t.Fatalf("resolved target = %+v, want light_living_floor", turn.ResolvedTarget)
	}
	state, err := app.Adapter.State(context.Background(), "light_living_floor")
	if err != nil {
		t.Fatal(err)
	}
	if state["power"] != true {
		t.Fatalf("device state = %+v, want power=true", state)
	}
}

// TestAmbiguousRoomLampAsksNeverActs is CONCEPT.md section 3's acceptance
// row: "die Lampe" matching two Wohnzimmer lamps must ask, not guess, and
// must switch zero devices until answered.
func TestAmbiguousRoomLampAsksNeverActs(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Mach die Lampe im Wohnzimmer an.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnClarifying {
		t.Fatalf("status = %s, want clarifying (result: %s)", turn.Status, turn.Result)
	}
	if len(turn.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2", len(turn.Candidates))
	}
	for _, id := range []string{"light_living_floor", "light_living_ceiling", "light_kitchen"} {
		st, err := app.Adapter.State(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if st["power"] != false {
			t.Fatalf("device %s power = %v before clarification answered, want false (no side effect yet)", id, st["power"])
		}
	}

	answered, err := app.Confirm(context.Background(), turn.ID, "light_living_ceiling")
	if err != nil {
		t.Fatal(err)
	}
	if answered.Status != TurnCompleted {
		t.Fatalf("status after clarification = %s, want completed (result: %s)", answered.Status, answered.Result)
	}
	st, _ := app.Adapter.State(context.Background(), "light_living_ceiling")
	if st["power"] != true {
		t.Fatalf("Deckenlampe power = %v, want true", st["power"])
	}
	st, _ = app.Adapter.State(context.Background(), "light_living_floor")
	if st["power"] != false {
		t.Fatalf("Stehlampe power = %v, want unchanged false", st["power"])
	}
}

// TestOnlyStehlampeAmongSimilarNamesResolvesExactly checks the acceptance
// row "Nur die Stehlampe, zusätzliche ähnliche Namen": an exact-name
// command must not be defeated by an unrelated device merely sharing a
// room, and must not match on partial/fuzzy similarity.
func TestOnlyStehlampeAmongSimilarNamesResolvesExactly(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Schalte die Stehlampe im Wohnzimmer ein.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted || turn.ResolvedTarget == nil || turn.ResolvedTarget.ID != "light_living_floor" {
		t.Fatalf("turn = %+v", turn)
	}
}

// TestAllLivingRoomLampsRequiresExplicitConfirmation is the "Alle Lampen im
// Wohnzimmer" acceptance row: a resolved group list is never executed
// without confirmation, and a confirmation switches only that room's
// devices.
func TestAllLivingRoomLampsRequiresExplicitConfirmation(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Alle Lampen im Wohnzimmer an.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnClarifying {
		t.Fatalf("status = %s, want clarifying awaiting confirmation (result: %s)", turn.Status, turn.Result)
	}
	if len(turn.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2 (only Wohnzimmer)", len(turn.Candidates))
	}
	st, _ := app.Adapter.State(context.Background(), "light_kitchen")
	if st["power"] != false {
		t.Fatalf("Küche lamp power = %v before/without confirmation, want untouched false", st["power"])
	}

	done, err := app.Confirm(context.Background(), turn.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	if done.Status != TurnCompleted {
		t.Fatalf("status after confirm = %s, want completed (result: %s)", done.Status, done.Result)
	}
	for _, id := range []string{"light_living_floor", "light_living_ceiling"} {
		st, _ := app.Adapter.State(context.Background(), id)
		if st["power"] != true {
			t.Fatalf("device %s power = %v, want true", id, st["power"])
		}
	}
	st, _ = app.Adapter.State(context.Background(), "light_kitchen")
	if st["power"] != false {
		t.Fatalf("Küche lamp power = %v after Wohnzimmer group action, want still false", st["power"])
	}
}

// TestNegationProducesNoWriteCall is the "Schalte ... nein, doch nicht"
// acceptance row: a self-correction must never reach device_action.
func TestNegationProducesNoWriteCall(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Schalte die Stehlampe ein... nein, doch nicht.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCancelled {
		t.Fatalf("status = %s, want cancelled (result: %s)", turn.Status, turn.Result)
	}
	st, _ := app.Adapter.State(context.Background(), "light_living_floor")
	if st["power"] != false {
		t.Fatalf("Stehlampe power = %v, want untouched false", st["power"])
	}
}

// TestExplanationQuestionDoesNotAct is the "Wie würde ich die Lampe
// einschalten?" acceptance row: a question about how to do something must
// explain, not act.
func TestExplanationQuestionDoesNotAct(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Wie würde ich die Lampe einschalten?")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted || turn.ResolvedTarget != nil {
		t.Fatalf("turn = %+v, want completed conversation turn with no resolved target", turn)
	}
	st, _ := app.Adapter.State(context.Background(), "light_living_floor")
	if st["power"] != false {
		t.Fatalf("Stehlampe power = %v, want untouched false", st["power"])
	}
}

// TestUnknownDeviceNameResolvesToNoneNotGuess ensures a nonexistent device
// name is reported, not silently matched to something else.
func TestUnknownDeviceNameResolvesToNoneNotGuess(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Schalte die Badlampe im Bad ein.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted || turn.ResolvedTarget != nil {
		t.Fatalf("turn = %+v, want completed with no device matched", turn)
	}
}

// TestJournalSurvivesRestart is section 15 step 1's own acceptance
// criterion: "ein nachvollziehbarer Auftrag nach Neustart weiterhin
// lesbar" -- reopening the journal file must replay prior events.
func TestJournalSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")

	j1, err := OpenJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := j1.Append(Event{ConversationID: "conv1", TurnID: "turn1", Type: "turn.created"}); err != nil {
		t.Fatal(err)
	}
	if err := j1.Close(); err != nil {
		t.Fatal(err)
	}

	j2, err := OpenJournal(path)
	if err != nil {
		t.Fatal(err)
	}
	defer j2.Close()
	events := j2.ByTurn("turn1")
	if len(events) != 1 || events[0].Type != "turn.created" {
		t.Fatalf("events after reopen = %+v, want one turn.created event", events)
	}

	// A second journal instance must continue sequence numbers rather than
	// restart at 1 and collide with replayed events.
	e, err := j2.Append(Event{ConversationID: "conv1", TurnID: "turn1", Type: "turn.completed"})
	if err != nil {
		t.Fatal(err)
	}
	if e.Sequence != 2 {
		t.Fatalf("sequence after reopen+append = %d, want 2", e.Sequence)
	}
}

func TestConfirmRejectsUnknownTurn(t *testing.T) {
	app := testApp(t)
	if _, err := app.Confirm(context.Background(), "turn_doesnotexist", ""); err == nil {
		t.Fatal("expected an error for an unknown turn ID")
	}
}

func TestConfirmRejectsNonClarifyingTurn(t *testing.T) {
	app := testApp(t)
	turn, err := app.ProcessText(context.Background(), "conv1", "Schalte die Stehlampe im Wohnzimmer ein.")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted {
		t.Fatalf("precondition: status = %s, want completed", turn.Status)
	}
	if _, err := app.Confirm(context.Background(), turn.ID, "light_living_floor"); err == nil {
		t.Fatal("expected an error confirming an already-completed turn")
	}
}

func TestSimulatorRejectsImplicitGlobalTarget(t *testing.T) {
	sim := adapters.NewSimulator(adapters.Entity{ID: "a", Name: "A", Room: "R", Kind: "light"})
	if err := sim.Apply(context.Background(), "", "set_power", map[string]any{"power": true}); err == nil {
		t.Fatal("expected an error for an empty entity ID")
	}
}

func TestLoadConfigRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"listen":"127.0.0.1:9","data_dir":"x","unknown_field":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Fatal("expected an error for an unknown config field")
	}
}

func TestStartVoiceTurnWithoutSpeechHostErrors(t *testing.T) {
	app := testApp(t) // testApp passes speech=nil
	if _, err := app.StartVoiceTurn(context.Background(), "conv1"); !errors.Is(err, ErrSpeechNotConfigured) {
		t.Fatalf("err = %v, want ErrSpeechNotConfigured", err)
	}
}

func TestVoiceOperationsOnUnknownTurnError(t *testing.T) {
	app := testApp(t)
	if _, err := app.PushVoiceAudio(context.Background(), "turn_doesnotexist", []byte{0, 0}); err == nil {
		t.Fatal("expected an error pushing audio to an unknown turn")
	}
	if _, err := app.FinishVoiceTurn(context.Background(), "turn_doesnotexist"); err == nil {
		t.Fatal("expected an error finishing an unknown turn")
	}
	if _, err := app.CancelVoiceTurn("turn_doesnotexist"); err == nil {
		t.Fatal("expected an error cancelling an unknown turn")
	}
}

func TestVoiceTurnEndpointReturns503WithoutSpeechHost(t *testing.T) {
	app := testApp(t)
	srv := httptest.NewServer(NewHandler(app))
	defer srv.Close()

	convResp, err := http.Post(srv.URL+"/api/conversations", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	var conv struct {
		ConversationID string `json:"conversation_id"`
	}
	if err := json.NewDecoder(convResp.Body).Decode(&conv); err != nil {
		t.Fatal(err)
	}
	convResp.Body.Close()

	r, err := http.Post(srv.URL+"/api/conversations/"+conv.ConversationID+"/turns/voice", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", r.StatusCode, http.StatusServiceUnavailable)
	}
}
