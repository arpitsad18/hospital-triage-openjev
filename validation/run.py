#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The validation runner: drive the engine over a labelled corpus and score it.

Design rules
------------
* **Nothing is guessed.** A case the engine refuses to answer is reported as a
  failure, not skipped quietly. Skipping is how a broken engine looks healthy.
* **Accuracy is not the gate.** Under-triage, missed mandatory red flags and
  acuity under-calls are the gate; they are reported first and they decide
  ``pass``.
* **The corpus is allowed to be small.** A 14-case corpus cannot estimate a
  real mis-triage rate; it can only catch gross regressions. The scorecard says
  so rather than implying statistical confidence it does not have.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

# default corpus lives next to this module
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CASES = os.path.join(_HERE, "cases.json")

LOW_CONF = float(os.environ.get("TRIAGE_LOW_CONF", "0.35"))

try:  # the engine is the single source of truth for the default threshold
    from src.triage.engine import LOW_CONFIDENCE_THRESHOLD as _ENGINE_DEFAULT
    _DEFAULT_LOW_CONF = _ENGINE_DEFAULT
except Exception:  # pragma: no cover - exercised only without openjev
    try:
        from ..src.triage.engine import LOW_CONFIDENCE_THRESHOLD as _ENGINE_DEFAULT
        _DEFAULT_LOW_CONF = _ENGINE_DEFAULT
    except Exception:
        _DEFAULT_LOW_CONF = 0.35
# The env var still wins (it is what the engine itself reads), but the fallback
# default is the engine's constant rather than a second hard-coded 0.35 that
# could drift away from it.
LOW_CONF = float(os.environ.get("TRIAGE_LOW_CONF", _DEFAULT_LOW_CONF))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or DEFAULT_CASES
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data["cases"] if isinstance(data, dict) else data
    if not cases:
        raise ValueError("no cases in %s" % path)
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError("duplicate case ids: %s" % sorted(dupes))
    for c in cases:
        if "narrative" not in c or "esi_code_expected" not in c:
            raise ValueError("case %r missing narrative/esi_code_expected"
                             % c.get("id"))
    return cases


def _top_confidence(probs: Optional[List[float]]) -> Optional[float]:
    return max(probs) if probs else None


def _norm_confidence(probs: Optional[List[float]]) -> Optional[float]:
    """The engine's own normalised-confidence function, reused rather than
    reimplemented.

    The gate must be scored on exactly the scale the engine gates on; a second
    copy here could silently drift out of step with it and the scorecard would
    then be describing a system nobody runs. The import is deferred and guarded
    only so the runner can still be imported in an environment without OpenJev
    installed (e.g. docs builds).
    """
    try:
        from src.triage.engine import normalised_confidence
    except Exception:  # pragma: no cover - exercised only without openjev
        try:
            from ..src.triage.engine import normalised_confidence
        except Exception:
            return _norm_confidence_fallback(probs)
    return normalised_confidence(probs or [])


def _norm_confidence_fallback(
    probs: Optional[List[float]]
) -> Optional[float]:
    """Local copy of the rescaling, used only when the engine cannot import."""
    if not probs or len(probs) < 2:
        return None
    uniform = 1.0 / len(probs)
    return (max(probs) - uniform) / (1.0 - uniform)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(
    engine_factory: Callable[[], Any],
    cases_path: Optional[str] = None,
    fail_open: bool = True,
) -> Dict[str, Any]:
    """Run every case and return a report dict (also the JSON output)."""
    cases = load_cases(cases_path)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    degraded: Dict[str, List[str]] = {}

    # metric accumulators
    acuity_correct = 0
    acuity_under: List[str] = []
    esi_in_range = 0
    esi_exact = 0
    esi_under: List[str] = []
    esi_over: List[str] = []
    esi_abs_err: List[int] = []
    rf_required: List[str] = []
    rf_caught: List[str] = []
    rf_missed: List[str] = []
    rf_false_pos: List[str] = []
    specialty_correct = 0
    confusion: Dict[str, Dict[str, int]] = {}
    low_conf_cases: List[str] = []
    per_q_conf: Dict[str, List[float]] = {}
    per_q_conf_norm: Dict[str, List[float]] = {}
    latencies: List[float] = []          # per-QUESTION mean latency
    latencies_case: List[float] = []     # per-CASE total latency
    conf_correct: List[float] = []
    conf_wrong: List[float] = []
    conf_correct_n: List[float] = []
    conf_wrong_n: List[float] = []

    for case in cases:
        cid = case["id"]
        engine = engine_factory()
        try:
            res = engine.triage(
                case["narrative"],
                vitals=case.get("vitals"),
                fail_open=fail_open,
            )
            data = res.to_dict()
        except Exception as exc:  # engine refused the case outright
            failures.append({"id": cid, "error": "%s: %s"
                             % (type(exc).__name__, str(exc)[:300])})
            continue

        if data.get("failed_questions"):
            # A case with unanswered questions is NOT evaluated: its metrics
            # would be computed over an incomplete answer set, which flatters
            # the numbers. It is counted as failed and blocks the gate instead.
            failures.append({"id": cid, "error": "unanswered questions: %s"
                             % ",".join(data["failed_questions"])})
            if data.get("degraded_checks"):
                degraded[cid] = data["degraded_checks"]
            continue
        if data.get("degraded_checks"):
            degraded[cid] = data["degraded_checks"]

        answers = data["answers"]
        probs = data["probabilities"]
        got_esi = int(data["esi_code"])
        exp_esi = int(case["esi_code_expected"])
        lo, hi = (case.get("esi_range") or [exp_esi, exp_esi])[:2]

        # --- acuity ---
        exp_acuity = case.get("expected_acuity")
        got_acuity = answers.get("acuity")
        if exp_acuity:
            if got_acuity == exp_acuity:
                acuity_correct += 1
            if exp_acuity == "acute" and got_acuity != "acute":
                acuity_under.append(cid)

        # --- esi ---
        if lo <= got_esi <= hi:
            esi_in_range += 1
        if got_esi == exp_esi:
            esi_exact += 1
        if got_esi > hi:
            # Outside the band on the dangerous side. The band IS the
            # tolerance: a value inside it is acceptable care, and calling it
            # a safety failure would make the gate contradict its own corpus.
            esi_under.append(cid)
        elif got_esi < lo:
            esi_over.append(cid)
        esi_abs_err.append(abs(got_esi - exp_esi))

        # --- deterministic red flags ---
        got_rf = bool(data.get("red_flags"))
        if case.get("red_flag_required"):
            rf_required.append(cid)
            if got_rf:
                rf_caught.append(cid)
            else:
                rf_missed.append(cid)
        elif got_rf and not case.get("expected_red_flag"):
            rf_false_pos.append(cid)

        # --- specialty routing ---
        exp_spec = case.get("expected_specialty")
        got_spec = answers.get("specialty")
        if exp_spec:
            if got_spec is not None and got_spec == exp_spec:
                specialty_correct += 1
            else:
                # An unanswered specialty is recorded as "(unanswered)" rather
                # than a bare None, so the confusion table stays sortable and
                # the miss is visible rather than crashing the renderer.
                got_spec_key = got_spec if got_spec is not None else "(unanswered)"
                confusion.setdefault(exp_spec, {})
                confusion[exp_spec][got_spec_key] = \
                    confusion[exp_spec].get(got_spec_key, 0) + 1

        # --- confidence ---
        case_confs = []
        case_confs_n = []
        for qid, vec in (probs or {}).items():
            c = _top_confidence(vec)
            if c is None:
                continue
            per_q_conf.setdefault(qid, []).append(c)
            case_confs.append(c)
            cn = _norm_confidence(vec)
            if cn is not None:
                per_q_conf_norm.setdefault(qid, []).append(cn)
                case_confs_n.append(cn)
        min_conf = min(case_confs) if case_confs else None
        min_conf_n = min(case_confs_n) if case_confs_n else None
        # The gate is applied to the NORMALISED signal: a raw threshold would
        # be meaningless across a 2-label and a 16-label question.
        if min_conf_n is not None and min_conf_n < LOW_CONF:
            low_conf_cases.append(cid)

        # calibration signal: confidence of a correct vs incorrect ESI call
        if min_conf is not None:
            (conf_correct if got_esi == exp_esi else conf_wrong).append(min_conf)
        if min_conf_n is not None:
            (conf_correct_n if got_esi == exp_esi
             else conf_wrong_n).append(min_conf_n)

        usage = data.get("usage") or {}
        # ``latency_ms_mean`` is the mean over this case's QUESTIONS, not the
        # case duration. Averaging it into a "per case" figure understates
        # case latency by roughly the number of questions (~11x here). The
        # case duration is ``latency_ms_total``.
        if usage.get("latency_ms_mean"):
            latencies.append(usage["latency_ms_mean"])
        if usage.get("latency_ms_total"):
            latencies_case.append(usage["latency_ms_total"])

        rows.append({
            "id": cid,
            "acuity_expected": exp_acuity,
            "acuity_got": got_acuity,
            "esi_expected": exp_esi,
            "esi_range": [lo, hi],
            "esi_got": got_esi,
            "esi_source": data.get("esi_source"),
            "red_flag_expected": bool(case.get("expected_red_flag")),
            "red_flag_got": got_rf,
            "specialty_expected": exp_spec,
            "specialty_got": got_spec,
            "min_confidence": None if min_conf is None else round(min_conf, 4),
            "min_confidence_norm": (None if min_conf_n is None
                                    else round(min_conf_n, 4)),
            "requires_human_review": bool(data.get("requires_human_review")),
            "degraded": data.get("degraded_checks") or [],
        })

    n_total = len(cases)
    n_ok = len(rows)

    def frac(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    report: Dict[str, Any] = {
        "corpus": cases_path or DEFAULT_CASES,
        "cases_total": n_total,
        "cases_evaluated": n_ok,
        "cases_failed": len(failures),
        "failures": failures,
        "acuity": {
            "accuracy": frac(acuity_correct, n_ok),
            "correct": acuity_correct,
            "under_calls": acuity_under,
        },
        "esi": {
            "in_range_rate": frac(esi_in_range, n_ok),
            "in_range": esi_in_range,
            "exact_rate": frac(esi_exact, n_ok),
            "exact": esi_exact,
            "mean_abs_error": (round(sum(esi_abs_err) / len(esi_abs_err), 3)
                               if esi_abs_err else None),
            "under_triage": esi_under,
            "over_triage": esi_over,
        },
        "red_flags": {
            "required": len(rf_required),
            "caught": len(rf_caught),
            "recall": frac(len(rf_caught), len(rf_required)),
            "missed": rf_missed,
            "false_positives": rf_false_pos,
        },
        "specialty": {
            "accuracy": frac(specialty_correct, n_ok),
            "correct": specialty_correct,
            "confusion": confusion,
        },
        "confidence": {
            "low_confidence_threshold": LOW_CONF,
            "cases_below_threshold": low_conf_cases,
            "scale": "normalised: (top_prob - uniform) / (1 - uniform)",
            "mean_confidence_when_esi_correct": (
                round(sum(conf_correct) / len(conf_correct), 4)
                if conf_correct else None),
            "mean_confidence_when_esi_wrong": (
                round(sum(conf_wrong) / len(conf_wrong), 4)
                if conf_wrong else None),
            "mean_confidence_norm_when_esi_correct": (
                round(sum(conf_correct_n) / len(conf_correct_n), 4)
                if conf_correct_n else None),
            "mean_confidence_norm_when_esi_wrong": (
                round(sum(conf_wrong_n) / len(conf_wrong_n), 4)
                if conf_wrong_n else None),
            "per_question_mean": {
                q: round(sum(v) / len(v), 4) for q, v in sorted(per_q_conf.items())
                if v
            },
            "per_question_mean_norm": {
                q: round(sum(v) / len(v), 4)
                for q, v in sorted(per_q_conf_norm.items()) if v
            },
        },
        "latency": {
            # Per-CASE (wall time for all questions in a case) -- what a user
            # waiting on a decision experiences.
            "mean_ms_per_case": (round(sum(latencies_case) / len(latencies_case), 1)
                                 if latencies_case else None),
            "max_ms_per_case": (round(max(latencies_case), 1)
                                if latencies_case else None),
            # Per-QUESTION mean latency, averaged across cases. Reported
            # separately so the two cannot be confused again.
            "mean_ms_per_question": (round(sum(latencies) / len(latencies), 1)
                                     if latencies else None),
        },
        "degraded": degraded,
        "rows": rows,
    }

    # The gate. Only safety failures block; accuracy is reported, not gated.
    blocking = []
    if failures:
        blocking.append("%d case(s) could not be evaluated" % len(failures))
    if rf_missed:
        blocking.append("missed mandatory red flag: %s" % ", ".join(rf_missed))
    if esi_under:
        blocking.append("ESI under-triage: %s" % ", ".join(esi_under))
    if acuity_under:
        blocking.append("acuity under-call: %s" % ", ".join(acuity_under))
    report["blocking"] = blocking
    report["pass"] = not blocking

    return report


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


def _ids(xs: List[str]) -> str:
    return ", ".join(xs) if xs else "none"


def print_scorecard(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("TRIAGE VALIDATION SCORECARD")
    print("=" * 72)
    print("corpus            : %s" % report["corpus"])
    print("cases             : %d evaluated, %d failed, %d total"
          % (report["cases_evaluated"], report["cases_failed"],
             report["cases_total"]))
    if report["failures"]:
        print()
        print("UNEVALUATED CASES (these block the gate -- never skipped silently)")
        for f in report["failures"]:
            print("  ! %-24s %s" % (f["id"], f["error"]))
    print()

    rf = report["red_flags"]
    print("SAFETY (these decide pass/fail)")
    print("  mandatory red-flag recall : %s  (%d/%d)  missed: %s"
          % (_pct(rf["recall"]), rf["caught"], rf["required"], _ids(rf["missed"])))
    print("  ESI under-triage          : %d  %s"
          % (len(report["esi"]["under_triage"]), _ids(report["esi"]["under_triage"])))
    print("  acuity under-calls        : %d  %s"
          % (len(report["acuity"]["under_calls"]), _ids(report["acuity"]["under_calls"])))
    print("  red-flag false positives  : %d  %s"
          % (len(rf["false_positives"]), _ids(rf["false_positives"])))
    print()

    ac = report["acuity"]
    es = report["esi"]
    sp = report["specialty"]
    print("AGREEMENT (soft signal, not gated)")
    print("  acuity exact match        : %s  (%d/%d)"
          % (_pct(ac["accuracy"]), ac["correct"], report["cases_evaluated"]))
    print("  ESI within accepted band  : %s  (%d/%d)"
          % (_pct(es["in_range_rate"]), es["in_range"], report["cases_evaluated"]))
    print("  ESI exact match           : %s  (%d/%d)"
          % (_pct(es["exact_rate"]), es["exact"], report["cases_evaluated"]))
    print("  ESI mean absolute error   : %s level(s)" % es["mean_abs_error"])
    print("  over-triage               : %d  %s"
          % (len(es["over_triage"]), _ids(es["over_triage"])))
    print("  specialty top-1           : %s  (%d/%d)"
          % (_pct(sp["accuracy"]), sp["correct"], report["cases_evaluated"]))
    if sp["confusion"]:
        print("  routing confusion         :")
        for exp, got in sorted(sp["confusion"].items()):
            pair = ", ".join("%s x%d" % (k, v)
                             for k, v in sorted(got.items(),
                                                key=lambda kv: str(kv[0])))
            print("      expected %-18s -> %s" % (exp, pair))
    print()

    cf = report["confidence"]
    print("CONFIDENCE")
    print("  gate threshold            : %.2f" % cf["low_confidence_threshold"])
    print("  cases below threshold     : %d  %s"
          % (len(cf["cases_below_threshold"]), _ids(cf["cases_below_threshold"])))
    print("  mean conf | ESI correct   : %s  (normalised)"
          % cf["mean_confidence_norm_when_esi_correct"])
    print("  mean conf | ESI wrong     : %s  (normalised)"
          % cf["mean_confidence_norm_when_esi_wrong"])
    print()
    lat = report["latency"]
    print("PERFORMANCE")
    print("  mean latency / case       : %s ms   (all questions in the case)"
          % lat["mean_ms_per_case"])
    print("  max  latency / case       : %s ms" % lat["max_ms_per_case"])
    print("  mean latency / question   : %s ms" % lat.get("mean_ms_per_question"))
    if report["degraded"]:
        print("  cases with degraded check : %d" % len(report["degraded"]))
    print()

    print("-" * 72)
    if report["pass"]:
        print("RESULT: PASS  -- no safety failure on this corpus.")
    else:
        print("RESULT: FAIL")
        for b in report["blocking"]:
            print("  ! %s" % b)
    print("-" * 72)
    print("This corpus is synthetic and small. It can catch gross regressions;")
    print("it cannot estimate a real-world mis-triage rate. Do not quote these")
    print("numbers as if they came from a clinical study.")
    print("=" * 72)


def print_calibration(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("TRIAGE CALIBRATION / LOW-CONFIDENCE SAFETY SWEEP")
    print("=" * 72)
    print("corpus : %s" % report["corpus"])
    print("cases  : %d evaluated, %d failed" % (report["cases_evaluated"],
                                               report["cases_failed"]))
    print()

    print("PER-QUESTION MEAN CONFIDENCE  (top probability, normalised so")
    print("  uniform = 0.00 and certain = 1.00 -- comparable across questions)")
    per_q = (report.get("confidence") or {}).get("per_question_mean_norm") or {}
    if not per_q:
        print("  (no probability vectors returned)")
    for qid, m in per_q.items():
        bar = "#" * int(round(max(0.0, min(1.0, m)) * 30))
        print("  %-24s %.3f  %s" % (qid, m, bar))
    print()

    thr = (report.get("confidence") or {}).get("low_confidence_threshold")
    low = (report.get("confidence") or {}).get("cases_below_threshold") or []
    blocked_hits = [c for c in low if c in set(report["esi"]["under_triage"])
                    or c in set(report["red_flags"]["missed"])]
    print("LOW-CONFIDENCE SWEEP  (threshold %.2f, normalised scale)" % thr)
    print("  cases that would be gated for human review : %d  %s"
          % (len(low), _ids(low)))
    print("  of those, real safety failures             : %d  %s"
          % (len(blocked_hits), _ids(blocked_hits)))
    if report["esi"]["under_triage"] and not blocked_hits:
        print("  NOTE: a safety failure occurred with NO low-confidence signal --")
        print("        the gate would not have caught it. Investigate before use.")
    print()

    print("PER-CASE DETAIL  (confidence = lowest top-probability across questions)")
    print("  %-24s %-4s %-9s %-5s %-5s %-6s" %
          ("case", "ESI", "range", "flags", "conf", "review"))
    for row in report["rows"]:
        lo, hi = row["esi_range"]
        print("  %-24s %-4d %-9s %-5s %-5s %-6s"
              % (row["id"], row["esi_got"], "%d-%d" % (lo, hi),
                 "yes" if row["red_flag_got"] else "-",
                 "-" if row["min_confidence"] is None
                 else "%.3f" % row["min_confidence"],
                 "YES" if row["requires_human_review"] else "-"))
    print("=" * 72)
