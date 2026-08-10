# Ticket Triage — local JSONL batch example

Runs a local GGUF model directly from Go against fictional support tickets and
writes JSONL results to stdout. No model server, database, or external API is
involved.

```sh
go run ./examples/ticket-triage -model /path/to/model.gguf \
  > triaged-tickets.jsonl
```

Use `-input your-tickets.jsonl` for another JSONL file. Each row needs `id`,
`subject`, and `body`. The model output is deliberately a human-review draft;
it may classify and suggest low-risk first steps, but it never authorises a
refund, repair, or safety decision. Do not feed real customer data into this
example before adding access control, retention/deletion policy, logging,
redaction, and domain-reviewed escalation rules.
