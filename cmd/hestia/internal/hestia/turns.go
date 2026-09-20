package hestia

// TurnStatus is one of the explicit states from CONCEPT.md section 6:
// created -> capturing -> finalizing -> interpreting -> clarifying/ready ->
// executing/retrieving -> completed, plus the terminal states cancelled,
// failed and outcome_unknown.
type TurnStatus string

const (
	TurnCreated        TurnStatus = "created"
	TurnCapturing      TurnStatus = "capturing"  // voice only: recording in progress
	TurnFinalizing     TurnStatus = "finalizing" // voice only: flushing the decoder
	TurnInterpreting   TurnStatus = "interpreting"
	TurnClarifying     TurnStatus = "clarifying"
	TurnReady          TurnStatus = "ready"
	TurnExecuting      TurnStatus = "executing"
	TurnCompleted      TurnStatus = "completed"
	TurnCancelled      TurnStatus = "cancelled"
	TurnFailed         TurnStatus = "failed"
	TurnOutcomeUnknown TurnStatus = "outcome_unknown"
)

// Intent is one of the allowed proposal intents (CONCEPT.md section 7).
type Intent string

const (
	IntentDeviceAction          Intent = "device_action"
	IntentKnowledgeQuery        Intent = "knowledge_query"
	IntentConversation          Intent = "conversation"
	IntentClarificationResponse Intent = "clarification_response"
	IntentCancel                Intent = "cancel"
)

// Proposal is the model's structured suggestion for one transcript
// revision (CONCEPT.md section 7): closed schema, never executed as-is.
type Proposal struct {
	Intent       Intent
	Action       string // e.g. "set_power", only meaningful for IntentDeviceAction
	TargetPhrase string
	RoomPhrase   string
	Scope        string // "single" or "group"
	Arguments    map[string]any
}

// Turn is one conversational order and its accumulated state.
type Turn struct {
	ID             string
	ConversationID string
	Status         TurnStatus
	InputText      string
	Proposal       *Proposal
	Candidates     []Device // resolver output awaiting clarification, if any
	ResolvedTarget *Device  // single confirmed target, once known
	Result         string   // human-readable outcome, set on completion
}
