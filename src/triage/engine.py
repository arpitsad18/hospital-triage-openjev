#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenJev-backed triage engine: the model call, wrapped in safety policy.

Three things this module owns:

1. **The forward pass.** A thin wrapper over ``openjev.easy.make_engine``
   (any OpenAI-compatible local endpoint) that asks the ten typed questions
   and returns label + raw probability vector per question.
2. **Fail-closed policy.** OpenJev raises when the answer token is not in the
   returned top-logprobs window rather than inventing a probability. That
   exception is *not* swallowed here: a question that cannot be answered
   honestly becomes a recorded failure, and the caller decides. Silently
   downgrading to a guess is the one behaviour that is never acceptable.
3. **Safety policy that does not consult the model.** Red flags are screened
   deterministically first; acuity is forced when they fire; ESI is clamped
   upward to the floor; low confidence is flagged for human review.

The engine is deliberately synchronous and stateless per call — an ED gate
should not accumulate hidden state between patients.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openjev.easy import make_engine
from openjev.types import Question

from . import questions as Q
from . import redflags as RF
from .esi import resolve as esi_resolve

# Confidence below which a decision is not allowed to stand on its own.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("TRIAGE_LOW_CONF", "0.35"))

# Question ids whose confidence is individually gated (silent low confidence on
# one of these is what turns into a mis-triage; the rest are documentation).
CRITICAL_QUESTIONS = ("acuity", "esi", "needs_consult",
                      "is_emergency_admission", "suicidal_risk")


def _token_usage(result: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Return a provider-neutral token usage shape from an engine response.

    Engines are not required to report usage (OpenJev's typed answer shape
    does not), so missing values remain ``None`` rather than being guessed.
    This keeps callers independent of a particular backend's response shape.
    """
    usage = result.get("usage") or result.get("token_usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "prompt": int(prompt) if prompt is not None else None,
        "completion": int(completion) if completion is not None else None,
        "total": int(total) if total is not None else None,
    }


def normalised_confidence(probs) -> Optional[float]:
    """Top probability rescaled so a coin-flip scores 0.0 and certainty 1.0.

    The raw top probability is unusable as a *threshold* because its floor is
    ``1/n``: it can never fall below 0.50 on a two-label question, so a raw
    0.35 gate is dead code on ``acuity`` and on every Yes/No question -- which
    is precisely the set of critical questions the gate exists for. Rescaling
    by the uniform prior makes one threshold mean the same thing on a 2-label
    and a 16-label question.

    Returns None when there is no distribution to judge, so a caller can tell
    "no evidence" apart from "genuinely undecided".
    """
    if not probs or len(probs) < 2:
        return None
    uniform = 1.0 / len(probs)
    return (max(probs) - uniform) / (1.0 - uniform)


# Ollama caps ``top_logprobs`` at 20, so a choice question with more labels
# than that can never have every label inside the window. Rather than refuse
# (and rather than silently renormalise without saying so), treat the mass of
# the window itself as a *lower bound* on the winner's probability and expose
# ``prob_floor`` so a caller can gate on how much of the distribution was
# actually observed. Above this floor there is no ambiguity left to resolve.
CHOICE_TAIL_MAX_FLOOR = 0.95


def _choice_probs_with_tail(
    engine: Any, question: Question, state: str
):
    """Answer a wide ``choice`` question, bounding the unobserved tail.

    Re-sends the exact OpenJev payload and reads the top-logprobs window
    directly. Returns ``(render_answer_shape, bound)`` where ``bound`` is the
    observed probability mass (>= the true winner probability). Raises when the
    winner is not identified or the mass is too small to be conclusive.
    """
    import json as _json
    import urllib.request as _urllib

    from openjev.core import LETTERS, build_user_text

    labels = list(question.criteria.keys())
    letters = LETTERS[: len(labels)]
    payload = {
        "model": engine.model,
        "messages": [
            {"role": "system", "content": engine.system_prompt},
            {"role": "user", "content": build_user_text(state, question)},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    req = _urllib.Request(
        engine.base_url + "/chat/completions",
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _urllib.urlopen(req, timeout=180) as resp:
        data = _json.loads(resp.read().decode("utf-8"))

    info = data["choices"][0]["logprobs"]["content"][0]
    window = {e["token"]: float(e["logprob"]) for e in info["top_logprobs"]}
    import math as _math

    probs: List[float] = []
    for letter in letters:
        lp = window.get(letter)
        probs.append(_math.exp(lp) if lp is not None else 0.0)
    total = sum(probs)
    if total <= 0.0:
        raise RuntimeError(
            "logprobs window identified no label for '%s'" % question.instructions[:60])
    probs = [p / total for p in probs]

    winner = info.get("token")
    if winner not in letters:
        raise RuntimeError(
            "model answered %r, not one of the %d labels; cannot score it"
            % (winner, len(labels)))
    if total < CHOICE_TAIL_MAX_FLOOR:
        raise RuntimeError(
            "only %.3f of the probability mass fell inside the 20-token window; "
            "the winner is not determinable at this floor" % total)
    # ``winner`` is the *letter*; callers want the label name, exactly as
    # openjev's own render_answer returns it for in-window questions.
    return {"choice": labels[letters.index(winner)],
            "probabilities": dict(zip(labels, probs)),
            "confidence": 1.0}, total


@dataclass
class TriageResult:
    """One complete triage decision with provenance and safety metadata."""

    answers: Dict[str, Any] = field(default_factory=dict)
    probs: Dict[str, List[float]] = field(default_factory=dict)
    # Normalised confidence per question (coin-flip = 0.0, certain = 1.0). This
    # is the value the review gate compares against, exposed so an operator can
    # see *why* something was escalated rather than just that it was.
    confidences: Dict[str, float] = field(default_factory=dict)
    esi_code: int = 5
    esi_source: str = "model"
    esi_detail: Dict[str, Any] = field(default_factory=dict)
    red_flags: bool = False
    red_flag_hits: List[str] = field(default_factory=list)
    red_flag_severity: str = "none"
    red_flag_explanation: str = "no deterministic red flag matched"
    requires_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    degraded_checks: List[str] = field(default_factory=list)
    failed_questions: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answers": self.answers,
            "probabilities": self.probs,
            "confidences": self.confidences,
            "esi_code": self.esi_code,
            "esi_source": self.esi_source,
            "esi_detail": self.esi_detail,
            "red_flags": self.red_flags,
            "red_flag_hits": self.red_flag_hits,
            "red_flag_severity": self.red_flag_severity,
            "red_flag_explanation": self.red_flag_explanation,
            "requires_human_review": self.requires_human_review,
            "review_reasons": self.review_reasons,
            "degraded_checks": self.degraded_checks,
            "failed_questions": self.failed_questions,
            "provenance": self.provenance,
            "usage": self.usage,
        }


class TriageEngine:
    """Wrap an OpenJev decision engine with ED triage policy."""
    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        question_ids: Optional[List[str]] = None,
        system_prompt: str = Q.DEFAULT_SYSTEM_PROMPT,
        low_confidence: float = LOW_CONFIDENCE_THRESHOLD,
    ):
        self.backend = backend
        self.model = model
        self.low_confidence = low_confidence
        self.system_prompt = system_prompt
        self.questions: Dict[str, Question] = Q.build_questions(question_ids)
        self.question_ids = list(self.questions.keys())

        kwargs: Dict[str, Any] = {"backend": backend}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._engine = make_engine(**kwargs)
        # OpenJev engines share the prompt hook; set it when supported.
        if hasattr(self._engine, "system_prompt"):
            self._engine.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # model call
    # ------------------------------------------------------------------

    def _qid(self, question: Question) -> str:
        """Reverse lookup of a Question's id, for error messages that name it."""
        for qid, q in self.questions.items():
            if q is question:
                return qid
        return getattr(question, "name", None) or "<unnamed>"

    def _answer_one(self, qid: str, question: Question, state: str) -> Dict[str, Any]:
        """One typed question -> one label + probability vector.

        Returns a dict with ``label``, ``confidence``, ``probs``, ``labels``,
        ``latency_ms``. Raises whatever OpenJev raises (fail-closed).
        """
        t0 = time.time()
        bound = None
        attempts = 1
        try:
            result = self._engine.answer_one(question, state)
        except RuntimeError as exc:
            # A label set wider than the logprob window is answerable honestly
            # by bounding the unseen mass; anything else must still fail closed.
            if question.type_name == "choice" and "top-logprobs" in str(exc):
                result, bound = _choice_probs_with_tail(
                    self._engine, question, state if isinstance(state, str) else str(state))
            else:
                raise
        except (TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            # One retry: a cold local model or a momentarily busy server is a
            # transport hiccup, not an unanswerable question.
            attempts = 2
            time.sleep(2.0)
            result = self._engine.answer_one(question, state)
        latency = (time.time() - t0) * 1000.0

        # OpenJev's ``render_answer`` shape (openjev/types.py) is:
        #   choice -> {choice, probabilities: {label: p}, confidence}
        #   score  -> {score, probabilities: {"0".."N-1": p}, confidence}
        #   noul   -> {noul: p_yes}                     <- no vector, no confidence
        # Everything below is that shape adapted to list form; a mismatch here
        # is an install/version problem and must raise, never guess.
        if question.type_name == "choice":
            labels = list(question.criteria.keys())
            label = result["choice"]
            pvec = result.get("probabilities")
            if not isinstance(pvec, dict):
                raise RuntimeError(
                    "choice answer for '%s' has no probability map (got %r); "
                    "openjev version mismatch" % (self._qid(question), type(pvec).__name__))
            missing = [l for l in labels if l not in pvec]
            if missing:
                raise RuntimeError(
                    "choice answer for '%s' is missing labels %s"
                    % (self._qid(question), missing))
            probs = [float(pvec[l]) for l in labels]
            conf = float(result["confidence"])

        elif question.type_name == "score":
            levels = list(question.criteria)
            labels = [str(i) for i in range(len(levels))]
            pvec = result.get("probabilities")
            if not isinstance(pvec, dict):
                raise RuntimeError(
                    "score answer for '%s' has no probability map (got %r); "
                    "openjev version mismatch" % (self._qid(question), type(pvec).__name__))
            missing = [l for l in labels if l not in pvec]
            if missing:
                raise RuntimeError(
                    "score answer for '%s' is missing levels %s"
                    % (self._qid(question), missing))
            probs = [float(pvec[l]) for l in labels]
            label = float(result["score"])
            conf = float(result["confidence"])

        else:  # noul: a single P(Yes) scalar; the vector is implied
            p_yes = result.get("noul")
            if p_yes is None:
                raise RuntimeError(
                    "noul answer for '%s' has no 'noul' value" % self._qid(question))
            p_yes = min(1.0, max(0.0, float(p_yes)))
            labels = ["Yes", "No"]
            probs = [p_yes, 1.0 - p_yes]
            label = "Yes" if p_yes >= 0.5 else "No"
            conf = max(p_yes, 1.0 - p_yes)

        return {
            "label": label,
            "confidence": conf,
            "probs": probs,
            "labels": labels,
            "latency_ms": latency,
            "observed_mass": bound,
            "metadata": {
                "provenance": {
                    "engine": self.backend,
                    "model": self.model,
                    "question_id": qid,
                },
                "attempts": attempts,
                "latency_ms": round(latency, 1),
                "tokens": _token_usage(result),
            },
        }

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def triage(
        self,
        narrative: str,
        vitals: Optional[Dict[str, float]] = None,
        fail_open: bool = False,
    ) -> TriageResult:
        """Full triage pass for one patient.

        Parameters
        ----------
        narrative : str
            De-identified clinical state.
        vitals : dict, optional
            Structured vitals (``sbp``, ``hr``, ``rr``, ``spo2``, ``temp``,
            ``gcs``, ``age``). Used by the deterministic screen.
        fail_open : bool
            When True, a question that OpenJev refuses to answer is recorded in
            ``degraded_checks`` instead of aborting the whole pass. The default
            is **False** — an unanswerable critical question raises, because a
            triage decision assembled from missing pieces is worse than no
            decision. ``fail_open`` is for batch evaluation runs only.
        """
        out = TriageResult()
        out.provenance = {
            "engine": self.backend,
            "model": self.model,
            "question_ids": list(self.question_ids),
        }

        # 1. deterministic screen first — it must not depend on the model
        flags = RF.scan(narrative, vitals)
        out.red_flags = flags.acute
        out.red_flag_hits = flags.hits
        out.red_flag_severity = flags.severity
        out.red_flag_explanation = flags.explanation

        # 2. model pass, one question at a time
        raw: Dict[str, Dict[str, Any]] = {}
        for qid in self.question_ids:
            question = self.questions[qid]
            try:
                raw[qid] = self._answer_one(qid, question, narrative)
            except Exception as exc:
                detail = "model could not answer '%s': %s: %s" % (
                    qid, type(exc).__name__, str(exc)[:160])
                out.degraded_checks.append(detail)
                out.failed_questions.append(qid)
                if not fail_open:
                    raise RuntimeError(detail) from exc
                continue

        for qid, r in raw.items():
            out.answers[qid] = r["label"]
            out.probs[qid] = r["probs"]
            if r.get("observed_mass") is not None:
                # Wide choice question: the vector covers a bounded window, so
                # the winning probability is a floor, not an exact value.
                out.degraded_checks.append(
                    "%s: %.3f of the probability mass was inside the 20-token "
                    "logprob window; probabilities for the unobserved labels are "
                    "renormalised bounds, not measurements"
                    % (qid, r["observed_mass"]))

        # 3. acuity is forced by the screen, model opinion recorded
        model_acuity = out.answers.get("acuity")
        if flags.acute and model_acuity != "acute":
            out.answers["acuity"] = "acute"
            if model_acuity is not None:
                out.degraded_checks.append(
                    "model acuity '%s' overridden to 'acute' by deterministic "
                    "red flags (%s)" % (model_acuity, ", ".join(flags.hits[:4])))
        if flags.acute:
            out.requires_human_review = True
            out.review_reasons.append(
                "deterministic red flag: " + flags.explanation[:200])

        # 4. ESI: floor, override direction, provenance
        model_score = raw.get("esi", {}).get("label")
        override_label = out.answers.get("esi_override")
        res = esi_resolve(model_score=model_score,
                          override_label=override_label,
                          redflags=flags)
        out.esi_code = res.code
        out.esi_source = res.source
        out.esi_detail = res.to_dict()
        if res.model_code is not None and res.code < res.model_code:
            out.degraded_checks.append(
                "model ESI %d less urgent than the applied ESI %d (%s)"
                % (res.model_code, res.code, res.source))

        # 5. confidence gate
        #
        # Gated on the NORMALISED confidence, and the reason is a real bug this
        # gate used to have: ``confidence`` here is the raw top probability,
        # whose floor is 1/n. On a two-label question that means it can never
        # be below 0.50, so comparing it to a 0.35 threshold was dead code for
        # ``acuity`` and for every Yes/No question -- i.e. for four of the five
        # CRITICAL_QUESTIONS, acuity included. Only the five-level ``esi``
        # could ever trip it. Rescaling by the uniform prior makes the
        # threshold mean the same thing on every question.
        for qid in self.question_ids:
            r = raw.get(qid)
            if not r:
                continue
            norm = normalised_confidence(r.get("probs"))
            if norm is not None:
                out.confidences[qid] = round(norm, 4)
            if norm is not None and norm < self.low_confidence:
                out.review_reasons.append(
                    "%s confidence %.3f (raw %.3f < %.2f normalised, label=%s)"
                    % (qid, norm, r["confidence"], self.low_confidence,
                       r["label"]))
                if qid in CRITICAL_QUESTIONS:
                    out.requires_human_review = True

        # 6. dignity / safety checks that must not depend on the model
        if out.answers.get("suicidal_risk") == "Yes":
            out.requires_human_review = True
            out.review_reasons.append("suicidal-risk screen positive")
        if out.answers.get("translator") == "Yes":
            out.review_reasons.append("interpreter required before care")

        # 7. usage rollup
        lat = [r["latency_ms"] for r in raw.values()]
        token_rows = [r["metadata"]["tokens"] for r in raw.values()]
        token_totals = {
            key: (sum(row[key] for row in token_rows
                      if row[key] is not None) if any(row[key] is not None for row in token_rows)
                  else None)
            for key in ("prompt", "completion", "total")
        }
        out.usage = {
            "questions_asked": len(raw),
            "questions_failed": len(out.failed_questions),
            "latency_ms_total": round(sum(lat), 1),
            "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
            "backend": self.backend,
            "model": self.model,
            "tokens": token_totals,
            "attempts": sum(r["metadata"]["attempts"] for r in raw.values()),
            "results": {qid: r["metadata"] for qid, r in raw.items()},
        }
        return out

    # convenience pass-throughs for the CLI
    def answer_one(self, qid: str, state: str) -> Dict[str, Any]:
        if qid not in self.questions:
            raise KeyError("unknown question id: %s" % qid)
        return self._answer_one(qid, self.questions[qid], state)
