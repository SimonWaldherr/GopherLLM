package hestia

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// Event is one journal entry (CONCEPT.md section 8). This is a deliberately
// small placeholder for the eventual tinySQL `eventlog`-backed journal: an
// append-only JSONL file, fsynced per append, replayed into memory at
// startup. It gives the durability property the roadmap's first milestone
// needs ("ein nachvollziehbarer Auftrag nach Neustart weiterhin lesbar")
// without requiring the not-yet-available tinySQL integration.
type Event struct {
	SchemaVersion  int            `json:"schema_version"`
	EventID        string         `json:"event_id"`
	ConversationID string         `json:"conversation_id"`
	TurnID         string         `json:"turn_id"`
	Sequence       int            `json:"sequence"`
	At             time.Time      `json:"at"`
	Type           string         `json:"type"`
	Payload        map[string]any `json:"payload,omitempty"`
}

// Journal appends events durably and serves them back in order. Safe for
// concurrent use.
type Journal struct {
	mu     sync.Mutex
	f      *os.File
	enc    *json.Encoder
	events []Event
	seq    int
}

// OpenJournal opens (creating if needed) a JSONL journal file at path and
// replays its existing contents into memory.
func OpenJournal(path string) (*Journal, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening journal: %w", err)
	}
	j := &Journal{f: f, enc: json.NewEncoder(f)}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	// Reading from the same *os.File we're appending to would race the
	// current offset; the file was just opened read-write, so seek is safe
	// before any appends have happened.
	for sc.Scan() {
		var e Event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			continue // tolerate a truncated last line from a prior crash
		}
		j.events = append(j.events, e)
		if e.Sequence > j.seq {
			j.seq = e.Sequence
		}
	}
	if err := sc.Err(); err != nil {
		f.Close()
		return nil, fmt.Errorf("reading journal: %w", err)
	}
	return j, nil
}

// Append durably records one event, assigning it the next sequence number.
func (j *Journal) Append(e Event) (Event, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.seq++
	e.Sequence = j.seq
	if e.At.IsZero() {
		e.At = time.Now().UTC()
	}
	if err := j.enc.Encode(e); err != nil {
		return Event{}, fmt.Errorf("appending event: %w", err)
	}
	if err := j.f.Sync(); err != nil {
		return Event{}, fmt.Errorf("syncing journal: %w", err)
	}
	j.events = append(j.events, e)
	return e, nil
}

// ByConversation returns a conversation's events in append order.
func (j *Journal) ByConversation(conversationID string) []Event {
	j.mu.Lock()
	defer j.mu.Unlock()
	var out []Event
	for _, e := range j.events {
		if e.ConversationID == conversationID {
			out = append(out, e)
		}
	}
	return out
}

// ByTurn returns one turn's events in append order.
func (j *Journal) ByTurn(turnID string) []Event {
	j.mu.Lock()
	defer j.mu.Unlock()
	var out []Event
	for _, e := range j.events {
		if e.TurnID == turnID {
			out = append(out, e)
		}
	}
	return out
}

func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.f.Close()
}
