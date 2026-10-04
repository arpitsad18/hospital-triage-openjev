# Triage OpenJev

**Offline, auditable, decision-model triage.** A System-One (masked-logit
softmax) engine that turns an emergency-department patient state into
type-safe triage decisions **with calibrated probabilities** — no free-text
generation, no hallucination, no per-decision cost, no PHI leaving the host.

Built on [OpenJev](https://github.com/lookski/openjev) (MIT). Runs against any
OpenAI-compatible local inference server (Ollama, LM Studio, llama.cpp, vLLM).

---

## Why a decision model instead of a chatbot

A chatbot writes prose and you have to parse it. A *decision model* takes a
typed question and returns a label **plus the raw distribution over the
labels** in a single forward pass:

```
# illustrative output shape, not a benchmark measurement
Question: acuity
  acute      A ######################## 0.9982
  non_acute                                 0.0018
  -> choice=acute  confidence=0.9964
```

The probability is the actual masked softmax of the answer token's logits, so
`confidence` is a calibrated measure of how peaked the model is — not a
confidence-looking number the model made up. That is what makes this usable
inside a clinical safety process: every decision is a bounded, reviewable
number, and low-confidence decisions can be routed to a human instead of being
silently accepted.

Chatbots can narrate a plausible triage without committing to a class. This
cannot: `1.0000` is a measurement, and the label set is closed.

---

## What it does

| Question | Type | Output |
|---|---|---|
| `acuity` | `choice` | `acute` / `non_acute` binary gate |
| `patient_type` | `choice` | emergency / elective / transfer / unknown |
| `esi` | `score` (1–5) | index-weighted severity on ESI-aligned levels |
| `imaging` | `choice` | none / xray / ct / usg / mri / other |
| `esi_override` | `choice` | no / one_level_higher / two_levels_higher |
| `specialty` | `choice` | 16 cardiology-forward specialties |
| `needs_consult` | `noul` | yes / no |
| `translator` | `noul` | yes / no (needs an interpreter before care) |
| `is_emergency_admission` | `noul` | yes / no |
| `tele_followup_instead` | `noul` | yes / no |
| `suicidal_risk` | `noul` | yes / no (gates crisis pathway) |

`score` answers come back as a **0-indexed index-weighted mean**, so the raw
`esi` field is `0`–`4` and `esi_code = round(score) + 1`. `esi_code` is added
to every answer payload so downstream consumers never do that arithmetic.
An `esi_source` field records whether `esi_code` came from the model or from
the deterministic acuity rule below.

Eleven questions in one call. On `granite4.1:8b` through Ollama on an RTX 4060
Laptop the measured cost was **25.2 s per case** mean, 25.7 s max (14-case
corpus, 9--11 questions answered per case — roughly 2.29 s per question), so a
full triage takes tens of seconds, not the 70–500 ms single-question band the
decision-model pattern advertises. That is the honest cost of asking a
generative model eleven wide-label questions per patient; treat this as a
screening aid that runs behind the clinician, not in front of them.

---

## Design rules (non-negotiable)

1. **The code decides red flags, not the model.** `acuity` is *forced* to
   `acute` whenever a deterministic screen matches (`src/triage/redflags.py`:
   keywords + vital-sign thresholds + age). The model's opinion is still
   recorded in `degraded_checks` so drift is visible, but the red flag wins.
   A hallucinated "non_acute" on a STEMI is a patient-safety event; this
   makes that class of failure structurally impossible.
2. **PHI never leaves the host.** The engine talks to a local server. Patient
   identifiers should be de-identified before they reach `/triage`; the API
   accepts only the narrative plus optional structured vitals.
3. **Uncertainty is routed, not averaged away.** Confidence is scored on a
   **normalised** scale, `(top_prob − 1/n) / (1 − 1/n)`, so a coin-flip scores
   0.0 and certainty 1.0 on every question regardless of label count. The raw
   top probability cannot be used as a threshold: its floor is `1/n`, so on
   `acuity` and every Yes/No question it is *never* below 0.50 and a raw 0.35
   gate could never fire — precisely on the questions the gate exists for.
   Any normalised confidence below `LOW_CONFIDENCE_THRESHOLD` (0.35 by
   default) on a critical question (`acuity`, `esi`, `needs_consult`,
   `is_emergency_admission`, `suicidal_risk`) sets `requires_human_review`.
4. **`esi` is advisory.** `esi_code` is derived, and any model value more
   urgent than the red-flag floor is overridden upward, never downward.
5. **Fail closed.** If the engine cannot produce honest probabilities it
   raises instead of guessing — OpenJev refuses to emit fabricated numbers
   when the answer token falls outside the top-logprobs window. The API turns
   that into HTTP 503, not a low-confidence answer.

---

## Install

Requires Python 3.9+, a running local inference server, and OpenJev.

```bash
pip install -r requirements.txt
```

Verify the engine can actually reach your model before wiring anything:

```bash
python -m src.triage.cli doctor
python -m src.triage.cli validate
```

`doctor` checks the backend, model, and whether the model returns usable
top-logprobs. Chat-tuned models that emit a thinking preamble (many
`qwen3.5` builds) **fail** this check by design — their first generated token
is `Thinking`, so no answer label is ever in the probability window. Pick a
model that answers first:

```bash
ollama pull granite4.1:8b      # verified working, 4/4 on the built-in bench
```

## Run

```bash
# advisory decision for one patient narrative
python -m src.triage.cli decide "72M crushing central chest pain 40min, BP 88/60, diaphoretic"

# HTTP API (all 10 questions in one call)
python -m src.triage.api --port 8773
curl -s localhost:8773/health
curl -s -X POST localhost:8773/triage -H 'Content-Type: application/json' \
  -d '{"narrative":"31F peanut exposure, tongue swelling, wheeze, BP 84/50"}'
```

`GET /health` reports server + engine state. `GET /schema` returns the exact
question set and label space, so a caller never has to guess the contract.
`POST /triage` returns `answers`, `esi_code`, `esi_source`, `red_flags`,
`red_flag_hits`, `requires_human_review`, `degraded_checks`, `usage`.

## Validate before you trust it

The self-test is the part that matters. `validation/cases.json` holds 14
synthetic scenarios with expected acuity, ESI band, and required/forbidden
routing. Every case is evaluated; the scorecard separates **safety** metrics
(these decide pass/fail) from **agreement** metrics (a soft signal, not gated).

```bash
python -m src.triage.cli validate
```

Measured on 2026-10-04, `granite4.1:8b` via Ollama, 14 cases, 0 failures:

```
SAFETY (these decide pass/fail)
  mandatory red-flag recall : 100.0%  (8/8)  missed: none
  ESI under-triage          : 0
  acuity under-calls        : 0
  red-flag false positives  : 0

AGREEMENT (soft signal, not gated)
  acuity exact match        : 92.9%  (13/14)
  ESI within accepted band  : 100.0% (14/14)
  ESI exact match           : 50.0%  (7/14)
  ESI mean absolute error   : 0.5 level(s)
  over-triage               : 0
  specialty top-1           : 78.6%  (11/14)

PERFORMANCE
  mean latency / case       : 25182.9 ms   (all questions in the case)
  max  latency / case       : 25712.0 ms
  mean latency / question   : 2289.4 ms
  cases with degraded check : 13

RESULT: PASS  -- no safety failure on this corpus.
```

Latency is per **case**, i.e. the wall time for every question the case
triggers (9--11 here), which is what a clinician waiting on a decision
actually experiences. This is deliberately not hidden: **~25 s per case on a
laptop-class GPU is fine for batch triage review and too slow for a live
door-to-doctor decision.** Per-question cost is reported alongside it so the
two can never be confused.

The full JSON report for that run is checked in at
`docs/benchmark-2026-10-04.json`, so the numbers above can be verified against
the raw artifact rather than taken on trust.

Two things the scorecard deliberately does **not** do. It does not gate on
agreement: an ESI within the clinically accepted band is acceptable care, so
it is reported but never fails the run. And it does not hide the logprob
window limit — 13 of 14 cases report a degraded check because `specialty` has
16 labels and the top-logprobs window is 20 tokens, so the probabilities for
unobserved labels are renormalised bounds, not measurements.

Run `python -m src.triage.cli calibrate` for the same corpus with the full
probability vector per case, plus a safety sweep that feeds ambiguous
presentations through and reports which ones fell under the confidence
threshold. Those are the cases a real deployment must send to a human.

## Tests

```bash
python -m pytest tests/ -q
```

Covers red-flag thresholds, ESI mapping/override direction, response shape,
the confidence gate, fail-closed behaviour, and the API contract.

---

## Layout

```
src/triage/
  questions.py    the 11 typed question specs (single source of truth)
  redflags.py     deterministic keyword + vitals screen
  esi.py          ESI levels, mapping, override direction rules
  engine.py       OpenJev wrapper, custom system prompt, fail-closed policy
  cli.py          doctor / decide / validate / calibrate
  api.py          stdlib HTTP API + /schema /health
validation/
  cases.json      14 synthetic scenarios with labels + expectations
  run.py          accuracy, safety, escalation-direction, latency report
tests/            pytest suite
deploy/
  triage.service  systemd unit (Linux edge box)
  run-server.ps1  Windows launcher
docs/
  EVALUATION.md   how to read the numbers; what is NOT claimed
  LIMITS.md       known failure modes
```

## Status

Research/engineering prototype. **Not a medical device.** Not validated
against real patient outcomes. Every decision must be reviewed by a qualified
clinician. See `docs/EVALUATION.md` before quoting any number from this repo.

MIT licensed. OpenJev is MIT, by the OpenJev contributors (upstream README in
`docs/UPSTREAM.md`).
