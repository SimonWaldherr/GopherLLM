# Pool Lifeguard Assist — local MVP

> **Demo and operator-assistance proof of concept — not a drowning-detection
> system, not a safety product, and not a replacement for lifeguards, rescue
> procedures, or an operator's legal duties. Do not use it as the sole or
> decisive basis for an emergency response.**

This example uses GopherLLM's local live-vision UI as an additional set of
eyes for an *already supervised* pool or spa. It is deliberately designed to
tell an on-site operator **where to look and why**, not to diagnose an
emergency or trigger a rescue on its own.

## Why local inference

With a local GopherLLM server or browser/WebGPU runtime, camera images can be
processed in the pool, spa, or operator's browser. They do not need to be sent
to an external AI API. That supports data minimisation and removes avoidable
serial dependencies such as the ISP, DNS, API authentication, cloud capacity,
and a particular external model provider.

Local processing does **not** remove the fundamental dependencies of a camera
assist system: camera, local network path, compute host, power, loaded model,
and the operator. The MVP therefore shows camera/model/inference health rather
than silently pretending that a stopped feed is still being observed.

## What the MVP does

Open the GopherLLM chat UI, choose **Camera → Start live camera** (or screen
capture), then use the controls in the live overlay:

For scripted or batch testing outside the browser UI, `main.go` in this
directory sends one or more already-captured frames through the same
evidence-only prompt pattern:

```sh
go run ./examples/pool-lifeguard-assist \
  -model /path/to/text-model.gguf \
  -mmproj /path/to/mmproj.gguf \
  -zone "Pool A" \
  -frames ./frame1.jpg,./frame2.jpg
```

It is a local CLI companion for testing prompts and frames, not a
replacement for the live UI's health indicators or armed sound/notify
controls described below.

| Control | Meaning |
| --- | --- |
| **Zone / camera** | An operator-owned label such as `Pool A`, `diving well`, or `camera east`. It is supplied to the model so a useful response can name the location; it does not draw or enforce a physical zone. |
| **Current frame** | Sends one freshly captured frame for low-latency scene description. |
| **Change · 5 sec** | Keeps one real media sample per second, up to five samples, and sends a timestamped collage when enough samples exist. |
| **Timeline · 10 sec** | Keeps ten one-second samples, then chooses six representative samples for the collage so tiles remain readable. |
| **Camera / Model / Inference** | A visible, client-side health indication. It is an operational hint, not a certified monitoring or availability guarantee. |
| **Sound / Notify / Mark** | Explicitly armed local browser actions. They are never enabled merely because the model produces alarming language. |

The temporal profiles sample the actual camera or screen stream independently
of how long the model needs to answer. They are not five or ten consecutive
25-fps frames, and they are not a cache of previously sent AI requests. A
collage provides limited narrative context; it cannot reliably determine motion,
medical condition, or an emergency.

### Safe operating pattern

1. Keep a qualified lifeguard and the facility's normal rescue procedures in
   place. The tool is secondary attention support only.
2. Label each camera view consistently with the physical pool plan.
3. Start in **shadow mode**: no sound/notifications; record whether a human
   considered each hint useful, irrelevant, or impossible to assess.
4. After a site-specific review, enable at most a local “please check” sound
   with a cooldown. Never wire this demo directly to public alarms, doors,
   emergency services, or staffing reductions.
5. Treat `camera stopped`, `model not loaded`, a long inference delay, or a
   stale last-success indication as an unavailable assistant—not as evidence
   that the pool is safe.

A good prompt asks for visible evidence and uncertainty, for example:

> Watch `Pool A` for smoke, crowding, a person in visible distress, or an
> obstructed view. State only what is visible in one sentence. If unclear,
> say so and ask the operator to check the zone.

Do not ask the model to decide that a person is drowning. Prefer a human-action
prompt such as “please check Pool A” with an attached timestamped context.

## Research and regulatory notes

These notes are a product-design starting point, **not legal advice**. A real
operator needs site-specific review with its management, qualified pool-safety
expert, insurer, and data-protection officer.

### Supervision in Germany and Bavaria

- The industry guideline [DGfdB R 94.05](https://www.dgfdb.de/unsere-themen/richtlinie-dgfdb-r-9405)
  treats safe public-pool operation as the combination of qualified personnel,
  organisation, facility, and documentation—not as a camera feature.
- In Bavaria, [Art. 27 BayLStVG](https://www.gesetze-bayern.de/Content/Document/BayLStVG-27?view=Print)
  permits municipalities to issue safety and supervision rules for bathing
  facilities; local rules and the operator's concept must therefore be checked.
- The [OLG München decision 1 U 7114/20](https://www.gesetze-bayern.de/Content/Document/Y-300-Z-BECKRS-B-2022-N-363?view=Print)
  describes regular supervisory observation and suitable viewpoints, rather
  than continuous observation of every swimmer. This MVP does not change that
  duty.

### Privacy and AI

- For a municipal operator, [Art. 24 BayDSG](https://www.gesetze-bayern.de/Content/Document/BayDSG-24)
  covers video monitoring: necessity, visible notice, purpose limitation,
  involvement of the data-protection officer before deployment, and a general
  maximum two-month retention rule for stored data subject to exceptions.
- The product default should be: water surfaces only; no changing rooms,
  showers, or sanitary areas; no facial recognition, biometric templates, or
  person identification; RAM-only short ring buffer; and no cloud upload.
- The EU AI Act is product regulation whose classification depends on the
  actual function, claims, and integration. Do not market this MVP as a
  certified life-saving or autonomous decision system. See the [official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?eliuri=eli%3Areg%3A2024%3A1689%3Aoj&locale=de).

### Why temporal confirmation matters

Video-analytics practice separates detection, tracking over time, and an
operator-facing decision. NVIDIA's [tracking documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
describes persistent tracking across frame batches; pool-detection research
also combines detection, tracking, and repeated observations to reduce false
alarms. [Example study](https://repository.li.mahidol.ac.th/items/e95d30ce-f7f0-4b4c-a8a2-73882d421368)

The MVP intentionally does not claim to provide either specialised detection
or tracking. Its collages are a human-readable second look only. NIST evaluates
video activity systems using missed detections and fixed false-alarm rates,
which is why a headline accuracy number alone would be insufficient for a real
safety product. [NIST ActEV](https://actev.nist.gov/srl)

### Existing specialist products

Commercial systems already target this domain with specialised hardware,
calibration, computer vision, operational processes, and support. Examples
include [Poseidon](https://poseidon-tech.com/de/),
[AngelEye LifeGuard](https://angeleye.tech/en/en-lifeguard/), and
[Lynxight](https://www.lynxight.com/). Their performance and marketing claims
are their own and need independent, site-specific verification. GopherLLM's
distinct role is a self-hosted, general local assistant—not a drop-in,
validated replacement for those specialised systems.

## What a finished product would still require

This is intentionally a long list. None of these items should be waved away by
calling the software “AI”.

### Safety engineering and validation

- A formal hazard analysis, safety case, intended-use statement, and clear
  residual-risk documentation.
- Site-specific camera design: occlusion, glare, water depth, lighting,
  crowding, blind spots, camera failure, and coverage tests.
- A labelled, consented evaluation dataset covering the actual pool conditions;
  independent testing of missed-event rate, false alerts per operating hour,
  alert latency, and performance by lighting/crowding conditions.
- Specialised local perception and multi-object tracking before the LLM;
  temporal confirmation, zone rules, confidence thresholds, cooldowns, and a
  defined human-review workflow.
- Regular drills with lifeguards, a documented escalation protocol, and no
  automatic staffing reduction or emergency escalation based solely on AI.

### Reliability and operations

- RTSP/ONVIF camera ingestion, per-camera heartbeats, frame-age monitoring,
  local watchdog/restart, structured logs, and alerting when the assistant is
  degraded.
- Explicit resilience design for local power/network/host failures, ideally
  including a UPS where appropriate; health checks must be independent from the
  model's own answer text.
- An event store with operator confirmation, access controls, audit trail,
  retention/deletion jobs, backups, and secure updates.
- A calibration UI that draws persistent physical zones, not merely the MVP's
  text label; it must be restricted to authorised administrators.

### Governance, privacy, and procurement

- A documented purpose, legal basis, data-protection impact assessment where
  required, signage, deletion concept, role-based access, processor contracts
  where applicable, and a documented policy for recorded incidents.
- Legal and insurance review of claims, contracts, and operator procedures.
- AI-literacy/training material for operators and an explicit process for model
  updates, rollback, incident reporting, and periodic re-validation.

## Development roadmap

1. Run this local MVP in shadow mode at a consenting test site and collect only
   operator judgements, not claims of safety performance.
2. Build camera health, persistent zone calibration, event review, and a
   privacy-preserving audit store.
3. Add a specialised local perception/tracking component and measure it on
   site-specific data before enabling any operator sound cue.
4. Commission independent safety, privacy, and legal review before describing
   the result as anything beyond an operator-assistance prototype.
