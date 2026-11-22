package hestia

import (
	"context"
	"fmt"
	"strings"
)

func (a *App) answerExternalSearch(ctx context.Context, t *Turn, provider SearchProvider, name, query string) {
	t.Status = TurnCompleted
	if provider == nil {
		t.Result = name + " ist deaktiviert."
		return
	}
	if query == "" {
		t.Result = "Bitte nenne einen Suchbegriff für " + name + "."
		return
	}
	hits, err := provider.Search(ctx, query)
	if err != nil {
		t.Status = TurnFailed
		t.Result = name + " konnte gerade keine Ergebnisse abrufen."
		a.journal(t.ConversationID, t.ID, "turn.failed", map[string]any{"reason": "external_search_failed"})
		return
	}
	t.Sources = hits
	if len(hits) == 0 {
		t.Result = name + " hat dazu nichts gefunden."
	} else {
		t.Result = fmt.Sprintf("%s fand %d Treffer. %s", name, len(hits), hits[0].Title)
		if snippet := strings.TrimSpace(hits[0].Snippet); snippet != "" {
			t.Result += ". " + snippet
		}
	}
	a.journal(t.ConversationID, t.ID, "turn.completed", map[string]any{"kind": string(t.Proposal.Intent), "hit_count": len(hits)})
}
