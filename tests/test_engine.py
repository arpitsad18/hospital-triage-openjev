#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engine: response-shape adaptation, fail-closed behaviour, tail bounding.

Nothing here touches the network: the OpenJev engine is replaced with a stub
and ``urlopen`` is monkeypatched, so the probability math is checked exactly.
"""

import json
import math
import sys
import types

import pytest

from src.triage import engine as engine_mod
from src.triage import questions as Q


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal duck-typed stand-in for an OpenJev engine."""

    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls = 0
        self.model = "fake-model"
        self.system_prompt = "sys"
        self.base_url = "http://127.0.0.1:9"

    def answer_one(self, question, state):
        self.calls += 1
        if self.error is not None:
            err = self.error.pop(0) if isinstance(self.error, list) else self.error
            if err is not None:
                raise err
        for qid, q in BUILD_CACHE.items():
            if q is question:
                return self.results[qid]
        raise AssertionError("unknown question handed to the stub")


BUILD_CACHE = {}


def make_engine(results=None, error=None, question_ids=None):
    """A TriageEngine wired to a stub backend, with no network construction."""
    eng = engine_mod.TriageEngine.__new__(engine_mod.TriageEngine)
    eng.questions = Q.build_questions(question_ids)
    eng.question_ids = list(eng.questions)
    eng.low_confidence = engine_mod.LOW_CONFIDENCE_THRESHOLD
    eng.system_prompt = "sys"
    eng.backend = "stub"
    # ``triage()`` reports the model in its usage rollup, so the stub has to
    # carry one; ``__init__`` normally sets it and is bypassed here.
    eng.model = "stub-model"
    BUILD_CACHE.clear()
    BUILD_CACHE.update(eng.questions)
    eng._engine = FakeEngine(results, error)
    return eng


def _silence_sleep(monkeypatch):
    monkeypatch.setattr(engine_mod, "time",
                        types.SimpleNamespace(time=__import__("time").time,
                                              sleep=lambda *_: None))


# ---------------------------------------------------------------------------
# choice shape
# ---------------------------------------------------------------------------

def test_choice_answer_maps_labels_in_criteria_order():
    labels = list(Q.SPECIALTIES)
    eng = make_engine({"specialty": {
        "choice": "cardiology",
        "probabilities": {l: (0.6 if l == "cardiology" else 0.4 / (len(labels) - 1))
                          for l in labels},
        "confidence": 0.6}})
    out = eng._answer_one("specialty", eng.questions["specialty"], "chest pain")
    assert out["label"] == "cardiology"
    assert out["labels"] == labels
    assert len(out["probs"]) == len(labels)
    assert out["probs"][0] == pytest.approx(0.6)
    assert sum(out["probs"]) == pytest.approx(1.0)
    assert out["observed_mass"] is None
    assert out["latency_ms"] >= 0


def test_choice_missing_a_label_fails_closed():
    """A truncated probability map must raise, never be silently zero-filled."""
    eng = make_engine({"specialty": {
        "choice": "cardiology",
        "probabilities": {"cardiology": 1.0},
        "confidence": 1.0}})
    with pytest.raises(RuntimeError) as e:
        eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert "missing labels" in str(e.value)


def test_choice_without_a_probability_map_fails_closed():
    eng = make_engine({"specialty": {
        "choice": "cardiology", "probabilities": None, "confidence": 1.0}})
    with pytest.raises(RuntimeError) as e:
        eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert "probability map" in str(e.value)


def test_choice_confidence_is_taken_from_the_engine_not_recomputed():
    labels = list(Q.SPECIALTIES)
    eng = make_engine({"specialty": {
        "choice": "cardiology",
        "probabilities": {l: 1.0 / len(labels) for l in labels},
        "confidence": 0.42}})
    out = eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert out["confidence"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# score shape
# ---------------------------------------------------------------------------

def test_score_answer_exposes_indexed_levels():
    levels = list(Q.build_questions(["esi"])["esi"].criteria)
    eng = make_engine({"esi": {
        "score": 0.0,
        "probabilities": {str(i): (1.0 if i == 0 else 0.0)
                          for i in range(len(levels))},
        "confidence": 1.0}})
    out = eng._answer_one("esi", eng.questions["esi"], "collapsed")
    assert out["label"] == 0.0
    assert out["labels"] == [str(i) for i in range(len(levels))]


def test_score_missing_a_level_fails_closed():
    eng = make_engine({"esi": {
        "score": 0.0, "probabilities": {"0": 1.0}, "confidence": 1.0}})
    with pytest.raises(RuntimeError) as e:
        eng._answer_one("esi", eng.questions["esi"], "x")
    assert "missing levels" in str(e.value)


# ---------------------------------------------------------------------------
# noul shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_yes,expected", [
    (0.9, "Yes"), (0.5, "Yes"), (0.49, "No"), (0.0, "No"), (1.0, "Yes"),
])
def test_noul_thresholds_at_one_half(p_yes, expected):
    eng = make_engine({"needs_consult": {"noul": p_yes}})
    out = eng._answer_one("needs_consult", eng.questions["needs_consult"], "x")
    assert out["label"] == expected
    assert out["labels"] == ["Yes", "No"]
    assert sum(out["probs"]) == pytest.approx(1.0)
    assert out["confidence"] == pytest.approx(max(p_yes, 1.0 - p_yes))


def test_noul_probability_is_clamped_to_the_unit_interval():
    eng = make_engine({"needs_consult": {"noul": 4.2}})
    out = eng._answer_one("needs_consult", eng.questions["needs_consult"], "x")
    assert out["probs"][0] == pytest.approx(1.0)
    assert out["label"] == "Yes"


def test_noul_without_a_value_fails_closed():
    eng = make_engine({"needs_consult": {"noul": None}})
    with pytest.raises(RuntimeError) as e:
        eng._answer_one("needs_consult", eng.questions["needs_consult"], "x")
    assert "no 'noul' value" in str(e.value)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

def test_transient_timeout_is_retried_once(monkeypatch):
    _silence_sleep(monkeypatch)
    labels = list(Q.SPECIALTIES)
    eng = make_engine(
        {"specialty": {"choice": "cardiology",
                       "probabilities": {l: 1.0 / len(labels) for l in labels},
                       "confidence": 1.0 / len(labels)}},
        error=[TimeoutError("cold model"), None])
    out = eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert out["label"] == "cardiology"
    assert eng._engine.calls == 2


def test_a_persistent_transport_error_still_fails_closed(monkeypatch):
    _silence_sleep(monkeypatch)
    eng = make_engine({}, error=[TimeoutError("down"), TimeoutError("down")])
    with pytest.raises(TimeoutError):
        eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert eng._engine.calls == 2


def test_an_unrelated_runtime_error_is_not_swallowed(monkeypatch):
    """Only the wide-label-set case may be rescued; everything else raises."""
    _silence_sleep(monkeypatch)
    eng = make_engine({}, error=RuntimeError("model returned garbage"))
    with pytest.raises(RuntimeError) as e:
        eng._answer_one("specialty", eng.questions["specialty"], "x")
    assert "garbage" in str(e.value)


def test_wide_label_error_is_routed_to_tail_bounding(monkeypatch, ):
    """The logprobs-window RuntimeError must trigger the honest tail path."""
    _silence_sleep(monkeypatch)
    labels = list(Q.SPECIALTIES)
    eng = make_engine(
        {}, error=RuntimeError("label set is wider than top-logprobs window"))
    sentinel = {"choice": "cardiology",
                "probabilities": {l: 1.0 / len(labels) for l in labels},
                "confidence": 1.0}
    seen = {}

    def fake_tail(engine, question, state):
        seen["engine"] = engine
        return sentinel, 0.97

    monkeypatch.setattr(engine_mod, "_choice_probs_with_tail", fake_tail)
    out = eng._answer_one("specialty", eng.questions["specialty"], "chest pain")
    assert out["label"] == "cardiology"
    assert out["observed_mass"] == pytest.approx(0.97)
    assert seen["engine"] is eng._engine


def test_score_question_is_never_routed_to_tail_bounding(monkeypatch):
    """A wide *score* question has no tail escape hatch — it must raise."""
    _silence_sleep(monkeypatch)
    eng = make_engine({}, error=RuntimeError("top-logprobs window too small"))
    called = []
    monkeypatch.setattr(engine_mod, "_choice_probs_with_tail",
                        lambda *a: called.append(1))
    with pytest.raises(RuntimeError):
        eng._answer_one("esi", eng.questions["esi"], "x")
    assert called == []


# ---------------------------------------------------------------------------
# tail bounding: the logprob window itself
# ---------------------------------------------------------------------------

def _wide_question(n=24):
    """A synthetic choice question with more labels than the logprob window.

    The tail-bounding path only engages above the window, and today's built-in
    label spaces (16 specialties, 6 modalities) sit below it — so exercise the
    path with a question that actually crosses the line.
    """
    from openjev.types import Choice
    return Choice(instructions="Pick the best fit.",
                  criteria={"label_%02d" % i: "criterion %d" % i
                            for i in range(n)})


def _fake_urlopen(payload, monkeypatch):
    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured = {}

    def fake(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(json.dumps(payload).encode("utf-8"))

    import urllib.request as _u
    monkeypatch.setattr(_u, "urlopen", fake)
    return captured


def _window(chosen, logprobs):
    return {"choices": [{"logprobs": {"content": [{
        "token": chosen,
        "top_logprobs": [{"token": t, "logprob": lp}
                         for t, lp in logprobs.items()]}]}}]}


def _stub_engine(base_url="http://127.0.0.1:9"):
    return types.SimpleNamespace(model="m", system_prompt="s", base_url=base_url)


def test_tail_bounding_converts_the_letter_to_a_label_name(monkeypatch):
    """Regression: the winner used to be returned as 'A' instead of a label."""
    from openjev.core import LETTERS
    question = _wide_question()
    labels = list(question.criteria)
    letters = LETTERS[:len(labels)]
    payload = _window(letters[0], {letters[0]: math.log(0.90),
                                   letters[1]: math.log(0.08)})
    captured = _fake_urlopen(payload, monkeypatch)

    out, bound = engine_mod._choice_probs_with_tail(
        _stub_engine(), question, "chest pain radiating to the arm")

    assert out["choice"] == labels[0]
    assert set(out["probabilities"]) == set(labels)
    assert bound == pytest.approx(0.98)
    assert captured["body"]["top_logprobs"] == 20
    assert captured["body"]["max_tokens"] == 1
    assert captured["body"]["temperature"] == 0.0


def test_tail_bounding_normalises_the_observed_mass(monkeypatch):
    from openjev.core import LETTERS
    question = _wide_question()
    letters = LETTERS[:len(list(question.criteria))]
    payload = _window(letters[0], {letters[0]: math.log(0.7),
                                   letters[1]: math.log(0.28)})
    _fake_urlopen(payload, monkeypatch)
    out, bound = engine_mod._choice_probs_with_tail(_stub_engine(), question, "x")
    # probabilities are renormalised over the observed mass ...
    assert sum(out["probabilities"].values()) == pytest.approx(1.0)
    assert out["probabilities"][list(question.criteria)[0]] == pytest.approx(0.7 / 0.98)
    # ... while the bound reports the raw mass actually seen
    assert bound == pytest.approx(0.98)


def test_tail_bounding_refuses_when_too_little_mass_is_observed(monkeypatch):
    """Below the floor the winner is not determinable — raise, never guess."""
    from openjev.core import LETTERS
    question = _wide_question()
    letters = LETTERS[:len(list(question.criteria))]
    payload = _window(letters[0], {letters[0]: math.log(0.2),
                                   letters[1]: math.log(0.1)})
    _fake_urlopen(payload, monkeypatch)
    with pytest.raises(RuntimeError) as e:
        engine_mod._choice_probs_with_tail(_stub_engine(), question, "x")
    assert "20-token window" in str(e.value)


def test_tail_bounding_rejects_a_non_letter_answer(monkeypatch):
    from openjev.core import LETTERS
    question = _wide_question()
    letters = LETTERS[:len(list(question.criteria))]
    payload = _window("1", {letters[0]: math.log(0.9)})
    _fake_urlopen(payload, monkeypatch)
    with pytest.raises(RuntimeError) as e:
        engine_mod._choice_probs_with_tail(_stub_engine(), question, "x")
    assert "not one of the" in str(e.value)


def test_tail_bounding_reports_an_identified_nothing(monkeypatch):
    question = _wide_question()
    payload = _window("Z", {"Z": math.log(0.9)})
    _fake_urlopen(payload, monkeypatch)
    with pytest.raises(RuntimeError) as e:
        engine_mod._choice_probs_with_tail(_stub_engine(), question, "x")
    assert "identified no label" in str(e.value)


# ---------------------------------------------------------------------------
# the confidence gate: normalised scale, and the dead-gate regression
# ---------------------------------------------------------------------------

def _probs_over(labels, winner, p_winner):
    """A probability map over ``labels`` giving ``winner`` ``p_winner``."""
    rest = (1.0 - p_winner) / (len(labels) - 1.0)
    return {l: (p_winner if l == winner else rest) for l in labels}


def test_normalised_confidence_maps_uniform_to_zero_and_certainty_to_one():
    assert engine_mod.normalised_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert engine_mod.normalised_confidence([1.0, 0.0]) == pytest.approx(1.0)
    # 0.6 is 0.2 of the way from a coin flip to certainty on two labels ...
    assert engine_mod.normalised_confidence([0.6, 0.4]) == pytest.approx(0.2)
    # ... but 0.6 on five labels is 0.5 of the way. One threshold, same meaning.
    assert engine_mod.normalised_confidence(
        [0.6, 0.1, 0.1, 0.1, 0.1]) == pytest.approx(0.5)


def test_normalised_confidence_is_none_without_a_distribution():
    """"No evidence" must stay distinguishable from "genuinely undecided"."""
    assert engine_mod.normalised_confidence(None) is None
    assert engine_mod.normalised_confidence([]) is None
    assert engine_mod.normalised_confidence([1.0]) is None


def test_a_coin_flip_acuity_call_escalates_to_human_review():
    """Regression: the gate could never fire on a two-label question.

    ``acuity`` is binary, so its raw top probability can never fall below 0.50
    -- a 0.35 raw threshold was unreachable, and a 0.51 acuity call (a coin
    flip, on the most consequential question in the system) sailed through
    unflagged. On the normalised scale it is 0.02 and escalates.
    """
    labels = list(Q.build_questions(["acuity"])["acuity"].criteria)
    probs = _probs_over(labels, labels[0], 0.51)
    eng = make_engine({"acuity": {"choice": labels[0], "probabilities": probs,
                                  "confidence": 0.51}},
                      question_ids=["acuity"])
    out = eng.triage("stable patient, no red flags")

    # the old raw gate passed this straight through ...
    assert 0.51 > engine_mod.LOW_CONFIDENCE_THRESHOLD
    # ... the normalised gate does not.
    assert out.confidences["acuity"] == pytest.approx(0.02, abs=1e-3)
    assert out.requires_human_review is True
    assert any("acuity" in r for r in out.review_reasons)
    assert out.answers["acuity"] in labels


def test_a_certain_acuity_call_does_not_escalate():
    labels = list(Q.build_questions(["acuity"])["acuity"].criteria)
    probs = _probs_over(labels, labels[0], 1.0)
    eng = make_engine({"acuity": {"choice": labels[0], "probabilities": probs,
                                  "confidence": 1.0}},
                      question_ids=["acuity"])
    out = eng.triage("stable patient")
    assert out.confidences["acuity"] == pytest.approx(1.0)
    assert out.requires_human_review is False


def test_a_soft_non_critical_answer_is_recorded_but_not_escalated():
    """The gate applies to every question; escalation only to critical ones."""
    eng = make_engine({"tele_followup_instead": {"noul": 0.51}},
                      question_ids=["tele_followup_instead"])
    out = eng.triage("stable patient")
    assert out.confidences["tele_followup_instead"] == pytest.approx(0.02, abs=1e-3)
    assert any("tele_followup_instead" in r for r in out.review_reasons)
    assert out.requires_human_review is False


def test_the_gate_reads_the_normalised_scale_in_review_reasons():
    """An operator must be able to see both numbers behind an escalation."""
    labels = list(Q.build_questions(["acuity"])["acuity"].criteria)
    eng = make_engine({"acuity": {"choice": labels[0],
                                  "probabilities": _probs_over(labels, labels[0], 0.51),
                                  "confidence": 0.51}},
                      question_ids=["acuity"])
    out = eng.triage("stable patient")
    reason = [r for r in out.review_reasons if r.startswith("acuity")][0]
    assert "normalised" in reason and "0.51" in reason
