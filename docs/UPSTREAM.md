# Upstream

This project is built on **OpenJev**, MIT licensed, by the OpenJev
contributors — <https://github.com/lookski/openjev>.

OpenJev is **not vendored** in this repository. It is a normal dependency
(`requirements.txt`: `openjev>=0.1.0`) installed from PyPI, and the verified
working version on the development host is **0.1.0**.

## What we rely on

| OpenJev capability | Where we use it |
|---|---|
| `openjev.easy.make_engine` | construct the decision engine (`src/triage/engine.py`) |
| `openjev.types.Question` | typed question specs (`src/triage/questions.py`) |
| masked-logit softmax over a closed label set | every question's probability vector |
| top-logprobs window | the probability distribution the engine reads |
| refusal to fabricate probabilities outside the window | our fail-closed guarantee |

## What we changed on top

Everything else is ours:

- **The question set.** 11 typed question specs with a cardiology-forward
  16-label specialty space (`src/triage/questions.py`).
- **Deterministic red-flag screening** over narrative keywords, vital-sign
  thresholds, and age (`src/triage/redflags.py`) — runs before the model and
  can force acuity.
- **ESI mapping and one-directional override** (`src/triage/esi.py`).
- **The confidence policy.** Normalised confidence
  `(top_prob − 1/n) / (1 − 1/n)` and the human-review gate
  (`src/triage/engine.py`). This is not an OpenJev feature; it is a safety
  layer we wrote because the raw top probability cannot serve as a threshold
  across questions of different label count. See `docs/LIMITS.md`.
- **The validation harness and corpus** (`validation/`).
- **The CLI and HTTP API** (`src/triage/cli.py`, `src/triage/api.py`).

## Upstream licence

OpenJev is MIT licensed. Per its terms, the upstream copyright notice and
licence text should be retained in distributions of this work. This repository
does not restate the upstream licence text; read it at the upstream URL above.

## If you upgrade OpenJev

`make_engine`, `Question`, and the window/refusal behaviour are the surface we
depend on. Before upgrading:

```bash
python -m src.triage.cli doctor     # backend reachable AND model answers
python -m pytest tests/ -q          # 141 tests
python -m src.triage.cli validate   # safety metrics must still pass
```

If `doctor` starts reporting that no answer token is in the probability
window, the window behaviour changed and the tail-bounding logic in
`src/triage/engine.py` needs review.
