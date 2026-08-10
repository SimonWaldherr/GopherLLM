# Complaints Assistant — local GopherLLM demo

This is a fictional, local customer-support and complaints workflow. A customer
chooses one of three fake orders, describes the problem, and receives a
structured **draft** from a local GopherLLM server. It is intentionally small:
there is no database, payment data, real order lookup, automatic refund, or
automated decision.

> **Demo only.** A human support agent must verify facts, warranty, policy, and
> safety implications before sending any reply or taking action.

## Run it

Run the demo with the GGUF model that it should load directly:

```sh
go run ./examples/complaints-assistant -model /path/to/model.gguf
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090). The demo process opens
the model once with `gopherllm.Open`, then calls `model.Generate` directly for
each form submission. The browser only talks to the demo's small local handler;
there is no OpenAI-compatible API, no GopherLLM inference server, no proxy to
another process, and no CORS configuration involved.

## What to try

- Select the speaker and describe that it will not turn on: the draft should
  ask only necessary follow-up questions and recommend low-risk checks such as
  a restart or compatible cable/charger.
- Select the charger and enter “it smells burnt and becomes very hot”: the
  draft must become `safety_review`, require human review, and avoid risky
  troubleshooting or charging advice.
- Enter a damaged-on-arrival report: the model may classify and formulate a
  friendly reply, but must not promise a refund or replacement.

## Local-data boundary

The selected fake order and typed description stay in this local demo process
and its in-process GopherLLM model. The demo has no cloud API configuration. Do
not use it with real customer data until access,
retention, legal basis, and privacy requirements have been reviewed.

## What a product still needs

- Authenticated customer and agent roles; actual order-system integration;
  consent/notice, retention and deletion rules; audit trail and redaction.
- A policy engine owned by the business for warranty, returns, refunds, and
  escalation—not an LLM deciding outcomes.
- Human approval before every external customer message or operational action.
- Security review, rate limits, input/output handling, observability, tests
  with representative languages and accessibility review.
- Domain-reviewed safety wording and emergency escalation procedures for
  electrical, battery, fire, medical, and other hazardous reports.
