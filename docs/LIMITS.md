# Known failure modes and limits

Read alongside `docs/EVALUATION.md`. This file lists what this system gets
wrong, so nobody has to discover it in a deployment.

## Structural limits

### The logprob window truncates wide questions

`specialty` has 16 labels. The top-logprobs window is 20 tokens. When the
answer token is not in that window, the probability mass for unobserved labels
is a **renormalised bound, not a measurement**.

Consequence: for wide questions, the reported distribution is partly
constructed rather than observed. In the measured run, 13 of 14 cases carried
this degraded check — essentially every case, because `specialty` is asked
every time.

Mitigation in place: the tail-bounding logic caps the answer window on wide
label sets and records a degraded check naming the observed mass. The engine
never silently treats a bound as a measurement.

### Chat-tuned models fail by design

A model that emits a thinking preamble places no answer label in the
probability window, so OpenJev refuses to answer. Many `qwen3.5` builds do
this. `python -m src.triage.cli doctor` exists to separate "model is
unsuitable" from "install is broken" before anyone wires this into a workflow.

**Pick a model that answers first.** `granite4.1:8b` is the verified working
choice here.

### Raw top probability is meaningless across questions of different width

The raw top probability of a closed-set softmax has a floor of `1/n`:

| labels | floor of raw top prob |
|---|---|
| 2 (acuity, every Yes/No question) | 0.50 |
| 5 (esi) | 0.20 |
| 16 (specialty) | 0.0625 |

A model that is maximally uncertain on `acuity` still reports 0.50. So a raw
0.35 threshold is unreachable on every binary question, and trivially
satisfied on `specialty`.

**This was a live defect.** The review gate used to compare the raw top
probability, which made it dead code on `acuity` and all three binary Noul
questions — 4 of the 5 `CRITICAL_QUESTIONS` the gate exists to protect.

**Fix:** the gate now compares a normalised confidence,
`(top_prob − 1/n) / (1 − 1/n)`, which maps a coin-flip to 0.0 and certainty to
1.0 on any label count. The same formula is used in the scorecard (there is
one implementation, in `src/triage/engine.py`, that the harness imports) so
the reported numbers describe the system that actually runs.

### Confidence is weak as a correctness signal

On the measured corpus, mean normalised confidence was 0.7118 when ESI was
correct and 0.6773 when it was wrong — a separation of ~0.03. **A wrong answer
is about as confident as a right one.**

Do not build on the assumption that "high confidence ⇒ correct". The gate is a
fail-safe for the clearly-undecided case, not an accuracy filter.

The gap is small and its **sign depends on the scale you read it on**. On the
normalised scale the gate uses, two runs of the same corpus gave +0.03 (0.7118
correct vs 0.6773 wrong) and +0.02 (0.6959 vs 0.6754) — correct consistently
higher, but by a margin (~0.02) no larger than the run-to-run wobble (~0.015).
On the **raw** top-probability scale the *same two runs* read −0.0003 (0.7695
vs 0.7698) and −0.01 (0.7567 vs 0.7685): the sign reverses. The raw scale
compresses every question toward its `1/n` floor, and that compression is what
flips it — which is itself the argument for normalising. Either way the
magnitude is far too small to select on. Greedy decoding (temperature 0.0) is
not bit-deterministic on GPU, so the low-order probability decimals move
between runs — and the model's own ESI answer moved too: on two cases here it
returned 2 where the first run returned 1. Only the floor+override kept the
*applied* label identical, which is the point: what reproduced exactly was the
gated output, not the model.

## Failure modes seen in measurement

### ESI exact match is much worse than ESI band match

50.0% exact vs 100.0% within band. The system is systematically conservative:
the deterministic red-flag floor pushes applied ESI to 1 in every red-flag
case, while the model's own ESI for those cases was 3. Both are inside the
accepted band, so this is not a safety failure — but it means the *precise*
ESI you see is often the floor, not the model's judgement. Check `esi_source`
on the response to know which it was.

### Specialty confusion is a specificity problem, not a random error

When `internal_medicine` was expected, the model produced `nephrology`,
`pulmonology`, and `orthopaedics`. It is choosing the organ-specific
subspeciality over the general one. Whether that is wrong depends on the
deployment: for a routing aid in a hospital with subspeciality clinics on
site, "nephrology" may be a *better* answer than "internal medicine", even
though the corpus scores it as a miss.

The 16-label specialty set is cardiology-forward and closed. A presentation
outside it is forced into the nearest available label with no signal that the
fit was poor.

### Red flags are synthetic-corpus perfect

100% recall on 8 mandatory red flags in this corpus says the keyword and
vital-sign screen works on the phrasings it was written against. It says
nothing about atypical phrasing, negation, or a narrative that omits the
decisive detail. The screen is in `src/triage/redflags.py` and is meant to be
read and extended by a clinician, not trusted blind.

## What is not claimed

- **Not a medical device.** No regulatory clearance of any kind.
- **Not validated against patient outcomes.** Every number here comes from
  synthetic cases written by the authors.
- **Not a mis-triage rate.** A 14-case corpus cannot produce one.
- **Not clinically reviewed.** No emergency physician has signed off on the
  corpus, the thresholds, or the label sets.
- **Not safe to deploy unattended.** Every decision must be reviewed by a
  qualified clinician. `requires_human_review` is a flag, not a workflow.

## If you extend this

1. Add cases to `validation/cases.json` before trusting any rate. One case
   moves the percentages by ~7 points.
2. Keep safety metrics gating and agreement metrics advisory. Do not let a
   routing improvement hide a red-flag regression.
3. If you change the confidence formula, change it in
   `src/triage/engine.py` only — the harness imports it deliberately so the
   scorecard cannot drift from the runtime.
4. Have a clinician review any threshold you change. `TRIAGE_LOW_CONF` and the
   ESI floor are patient-safety parameters.
