# Evaluation: how to read our numbers, and what is not claimed

This document exists because the scorecard in the README is easy to
over-read. Read this before quoting any number from this repository.

**Status: research/engineering prototype. Not a medical device. Not validated
against real patient outcomes.**

## The corpus

`validation/cases.json` — **14 synthetic scenarios**, hand-written. Each case
carries an expected acuity, an ESI band (a range, not a single value), and
expected/forbidden routing.

14 cases is enough to catch gross regressions. It is **not** enough to
estimate anything with a meaningful confidence interval. A single case is
7.1% of the corpus; the difference between "92.9%" and "100%" is one patient.

Do not quote these numbers as if they came from a clinical study.

## What the scorecard measures

The report separates two kinds of metric, and only one of them can fail a run.

### Safety (decides pass/fail)

| Metric | Meaning |
|---|---|
| mandatory red-flag recall | of the cases where a red flag **must** fire, how many did the deterministic screen catch |
| ESI under-triage | cases where the applied ESI was **more urgent** than clinically acceptable — i.e. the patient was under-triaged |
| acuity under-calls | cases where the system said non-acute but the reference said acute |
| red-flag false positives | red flags fired where none was expected |

These gate `pass`. A safety failure means the run fails regardless of how good
the agreement numbers look.

### Agreement (soft signal, never gated)

| Metric | Meaning |
|---|---|
| acuity exact match | exact binary agreement |
| ESI within accepted band | applied ESI inside the published clinically acceptable range |
| ESI exact match | applied ESI equal to the reference level exactly |
| ESI mean absolute error | mean `abs(applied − reference)` in ESI levels |
| over-triage | applied ESI more urgent than the reference |
| specialty top-1 | exact speciality routing agreement |

An ESI inside the accepted band is **acceptable care**. That is why the band
rate is reported and the exact rate is not gated: a model that consistently
chooses the top of a band is providing acceptable care and should not be
failed for it.

## Latest measured run

`docs/benchmark-2026-10-04.json` holds the raw JSON report. Environment:
`granite4.1:8b` served by Ollama on an RTX 4060 Laptop.

```
cases             : 14 evaluated, 0 failed

SAFETY
  mandatory red-flag recall : 100.0%  (8/8)
  ESI under-triage          : 0
  acuity under-calls        : 0
  red-flag false positives  : 0

AGREEMENT
  acuity exact match        : 92.9%  (13/14)
  ESI within accepted band  : 100.0% (14/14)
  ESI exact match           : 50.0%  (7/14)
  ESI mean absolute error   : 0.5 level(s)
  over-triage               : 0
  specialty top-1           : 78.6%  (11/14)

CONFIDENCE (normalised scale, gate 0.35)
  cases below threshold     : 1  (ankle-sprain)
  mean conf | ESI correct   : 0.7118
  mean conf | ESI wrong     : 0.6773

PERFORMANCE
  mean latency / case       : 25182.9 ms   (all questions in the case)
  max  latency / case       : 25712.0 ms
  mean latency / question   : 2289.4 ms
  cases with degraded check : 13
```

## The three things not to over-read

### 1. Perfect safety on 14 synthetic cases is not evidence of safety

Red-flag recall of 100% means the deterministic screen caught all 8 mandatory
red flags *in this corpus*. The screen is keyword- and threshold-based
(see `src/triage/redflags.py`); it will miss atypical presentations that a
clinician would catch. That failure mode is invisible to this corpus.

### 2. Confidence does not separate correct from incorrect on this corpus

Mean normalised confidence when ESI was correct was **0.7118**; when ESI was
wrong, **0.6773** — a gap of 0.03. A second run of the same corpus gave 0.6959
vs 0.6754, a gap of 0.02. The sign is consistent across both, but the gap is
about the size of the run-to-run wobble — and it reverses if you read the raw,
un-normalised top probabilities instead. So it carries no usable signal; do not
read it as a correctness predictor (see `LIMITS.md`).

Both runs are checked in and comparable field-by-field:
`benchmark-2026-10-04.json` (run 1) and `benchmark-2026-10-04-run2.json`
(run 2). Every gated axis — red-flag recall, under/over-triage, acuity,
in-band rate, specialty, the single below-threshold case — is identical
across the two; only the confidence means and the wall-clock latency differ.

This is the single most important honest finding here. It means **confidence
cannot be used as a standalone selector for "trust this answer"** on this
corpus: a wrong ESI was, on average, almost exactly as confident as a right
one. The low-confidence gate is still correct to have (it is a fail-safe, and
it did catch `ankle-sprain` at 0.3224), but it is not a substitute for
clinical review, and the fraction of errors it would catch is unmeasured.

### 3. `confidence` is only a measurement inside the logprob window

13 of 14 cases carry a degraded check. The cause is structural: `specialty`
has 16 labels but the top-logprobs window is 20 tokens, so the probability
mass for unobserved labels is a **renormalised bound, not a measurement**.

In other words: for wide questions, the reported probabilities for labels
outside the window are not observations. They are constrained to sum to 1 by
construction. Any number derived from them (including a confidence) inherits
that weakness.

## Reproducing

```bash
python -m src.triage.cli doctor          # confirm the model can answer at all
python -m src.triage.cli validate --json # the report above
python -m src.triage.cli calibrate       # full probability vectors per case
```

The engine fails closed: if the answer token falls outside the probability
window, OpenJev refuses to emit a number and the engine records a failed
question rather than guessing. A high `cases with degraded check` count is
therefore expected and is the system working as designed, not a defect.

## Escalation behaviour is deliberately conservative

In the measured run, **9 of 14 cases set `requires_human_review`**:

- **8** were escalated because a deterministic red flag fired. Rule 1 in the
  README forces acuity to `acute` and flags for review whenever the screen
  matches — regardless of how confident the model is.
- **1** (`ankle-sprain`) was escalated by the confidence gate at a normalised
  confidence of 0.3224 (raw top probability 0.6612 — which would never have
  tripped a raw 0.35 threshold).

A >50% escalation rate sounds high, but the corpus is deliberately weighted
toward emergencies. It is the correct behaviour for a screening aid: escalate
broadly, and let the clinician do the narrowing.
