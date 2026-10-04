#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The validation harness itself: the gate must fail on safety, never on style.

These tests feed the harness hand-built cases and a fake engine, so the
scorecard arithmetic and the pass/fail gate are checked exactly — not
eyeballed from a run against a live model.
"""

import json

import pytest

from validation import run as V


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _case(cid, esi=3, esi_range=None, acuity="non_acute", rf_required=False,
          rf_expected=False, specialty="internal_medicine"):
    return {
        "id": cid,
        "narrative": "synthetic narrative for %s" % cid,
        "vitals": {"age": 40, "sbp": 120, "hr": 80},
        "esi_code_expected": esi,
        "esi_range": esi_range or [esi, esi],
        "expected_acuity": acuity,
        "expected_red_flag": rf_expected,
        "red_flag_required": rf_required,
        "expected_specialty": specialty,
    }


def _write_cases(tmp_path, cases, name="cases.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return str(p)


class FakeResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class FakeEngine:
    """Returns a canned payload per case id; raises when told to."""

    def __init__(self, payloads, errors=None, record=None):
        self.payloads = payloads
        self.errors = errors or {}
        self.record = record

    def triage(self, narrative, vitals=None, fail_open=True):
        cid = narrative.split()[-1]
        if self.record is not None:
            self.record.append({"id": cid, "vitals": vitals,
                                "fail_open": fail_open})
        if cid in self.errors:
            raise self.errors[cid]
        return FakeResult(self.payloads[cid])


def _payload(esi=3, acuity="non_acute", red_flags=None, specialty="internal_medicine",
             probs=None, failed=None, degraded=None, review=False,
             esi_source="model"):
    answers = {"acuity": acuity, "specialty": specialty}
    return {
        "esi_code": esi,
        "esi_source": esi_source,
        "answers": answers,
        "probabilities": probs if probs is not None else {
            "acuity": [0.9, 0.1], "specialty": [0.8]},
        "red_flags": red_flags,
        "failed_questions": failed or [],
        "degraded_checks": degraded or [],
        "requires_human_review": review,
        "usage": {"latency_ms_mean": 1234.0, "latency_ms_total": 13574.0},
    }


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

def test_loader_rejects_duplicate_ids(tmp_path):
    path = _write_cases(tmp_path, [_case("a"), _case("a")])
    with pytest.raises(ValueError) as e:
        V.load_cases(path)
    assert "duplicate" in str(e.value)


def test_loader_rejects_a_case_without_a_label(tmp_path):
    bad = _case("a")
    del bad["esi_code_expected"]
    path = _write_cases(tmp_path, [bad])
    with pytest.raises(ValueError):
        V.load_cases(path)


def test_loader_rejects_an_empty_corpus(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        V.load_cases(str(path))


def test_the_shipped_corpus_loads_and_is_unique():
    cases = V.load_cases()
    assert len(cases) >= 10
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    required = [c["id"] for c in cases if c.get("red_flag_required")]
    assert required, "corpus must contain cases that gate on red-flag recall"


def test_the_shipped_corpus_bands_contain_the_reference_level():
    for c in V.load_cases():
        lo, hi = c["esi_range"]
        assert lo <= c["esi_code_expected"] <= hi, \
            "band excluding the reference level in %s" % c["id"]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def test_a_clean_run_passes(tmp_path):
    cases = [_case("a"), _case("b")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3), "b": _payload(esi=3)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is True
    assert rep["blocking"] == []
    assert rep["cases_evaluated"] == 2


def test_latency_per_case_uses_case_total_not_question_mean(tmp_path):
    """``latency_ms_mean`` in engine usage is the mean over a case's
    QUESTIONS; the case duration is ``latency_ms_total``. Averaging the
    former into a 'per case' figure understates latency by roughly the
    question count (~11x in the real corpus)."""
    path = _write_cases(tmp_path, [_case("a")])
    eng = FakeEngine({"a": _payload()})
    rep = V.evaluate(lambda: eng, path)
    assert rep["latency"]["mean_ms_per_case"] == 13574.0
    assert rep["latency"]["max_ms_per_case"] == 13574.0
    assert rep["latency"]["mean_ms_per_question"] == 1234.0


def test_a_missed_mandatory_red_flag_blocks(tmp_path):
    cases = [_case("a", rf_required=True, rf_expected=True)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, red_flags=None)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is False
    assert rep["red_flags"]["missed"] == ["a"]
    assert any("red flag" in b for b in rep["blocking"])


def test_a_caught_mandatory_red_flag_clears_it(tmp_path):
    cases = [_case("a", rf_required=True, rf_expected=True)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, red_flags={"hits": ["acs_chest_pain"]})})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is True
    assert rep["red_flags"]["recall"] == 1.0


def test_esi_under_triage_blocks(tmp_path):
    cases = [_case("a", esi=1, esi_range=[1, 2])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is False
    assert rep["esi"]["under_triage"] == ["a"]
    assert any("under-triage" in b for b in rep["blocking"])


def test_esi_over_triage_is_reported_but_not_gated(tmp_path):
    """Over-triage wastes resources; it does not harm the patient, so it is
    a soft signal rather than a blocking failure."""
    cases = [_case("a", esi=4, esi_range=[3, 4])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=2)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["esi"]["over_triage"] == ["a"]
    assert rep["pass"] is True


def test_esi_one_level_below_a_band_edge_blocks(tmp_path):
    """The band's upper edge is the tolerance; past it is under-triage."""
    cases = [_case("a", esi=2, esi_range=[2, 3])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=4)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["esi"]["under_triage"] == ["a"]
    assert rep["pass"] is False


def test_an_answer_at_the_top_of_the_band_is_acceptable(tmp_path):
    """Regression: a value inside the published band used to be reported as
    under-triage whenever it sat above the reference level, which made the
    gate contradict the corpus it was scoring."""
    cases = [_case("a", esi=1, esi_range=[1, 2])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=2)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["esi"]["in_range"] == 1
    assert rep["esi"]["under_triage"] == []
    assert rep["pass"] is True
    assert rep["esi"]["exact_rate"] == 0.0, "reference match is still reported"


def test_the_band_floor_still_tolerates_being_inside_it(tmp_path):
    cases = [_case("a", esi=2, esi_range=[2, 3])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3)})
    rep = V.evaluate(lambda: eng, path)
    assert rep["esi"]["in_range"] == 1
    assert rep["esi"]["under_triage"] == []
    assert rep["pass"] is True


def test_acuity_under_call_blocks(tmp_path):
    cases = [_case("a", acuity="acute")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, acuity="non_acute")})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is False
    assert rep["acuity"]["under_calls"] == ["a"]


def test_a_false_positive_red_flag_is_reported_not_gated(tmp_path):
    cases = [_case("a", rf_expected=False)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, red_flags={"hits": ["sepsis"]})})
    rep = V.evaluate(lambda: eng, path)
    assert rep["red_flags"]["false_positives"] == ["a"]
    assert rep["pass"] is True


# ---------------------------------------------------------------------------
# an unevaluated case is a failure, never a skip
# ---------------------------------------------------------------------------

def test_an_engine_exception_blocks_and_is_recorded(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({}, errors={"a": RuntimeError("model offline")})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is False
    assert rep["cases_evaluated"] == 0
    assert rep["failures"][0]["id"] == "a"
    assert "model offline" in rep["failures"][0]["error"]


def test_a_case_with_unanswered_questions_is_not_evaluated(tmp_path):
    """Scoring a partial answer set would flatter every metric."""
    cases = [_case("a", esi=1, rf_required=True)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=1, failed=["specialty"],
                                    degraded=["specialty"])})
    rep = V.evaluate(lambda: eng, path)
    assert rep["cases_evaluated"] == 0
    assert rep["pass"] is False
    assert "unanswered" in rep["failures"][0]["error"]
    # the degraded check is still surfaced for diagnosis
    assert rep["degraded"]["a"] == ["specialty"]


def test_metrics_are_computed_over_evaluated_cases_only(tmp_path):
    cases = [_case("a"), _case("b")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3), "b": _payload(esi=3)},
                     errors={})
    eng.errors = {"b": RuntimeError("boom")}
    rep = V.evaluate(lambda: eng, path)
    assert rep["cases_total"] == 2
    assert rep["cases_evaluated"] == 1
    assert rep["esi"]["in_range_rate"] == 1.0


def test_a_failed_and_a_passing_case_both_block_on_the_failure(tmp_path):
    cases = [_case("a"), _case("b")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3), "b": _payload(esi=3)},
                     errors={"b": RuntimeError("boom")})
    rep = V.evaluate(lambda: eng, path)
    assert rep["pass"] is False
    assert rep["cases_evaluated"] == 1


# ---------------------------------------------------------------------------
# a fresh engine per case
# ---------------------------------------------------------------------------

def test_a_new_engine_is_built_for_every_case(tmp_path):
    """Per-case isolation: one case must not inherit another's state."""
    cases = [_case("a"), _case("b"), _case("c")]
    path = _write_cases(tmp_path, cases)
    built = []

    def factory():
        eng = FakeEngine({"a": _payload(esi=3), "b": _payload(esi=3),
                          "c": _payload(esi=3)})
        built.append(eng)
        return eng

    rep = V.evaluate(factory, path)
    assert len(built) == 3
    assert len({id(e) for e in built}) == 3
    assert rep["cases_evaluated"] == 3


def test_fail_open_is_forwarded_to_the_engine(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    record = []
    eng = FakeEngine({"a": _payload(esi=3)}, record=record)
    V.evaluate(lambda: eng, path, fail_open=False)
    assert record[0]["fail_open"] is False
    assert record[0]["vitals"] == {"age": 40, "sbp": 120, "hr": 80}


# ---------------------------------------------------------------------------
# confidence bookkeeping
# ---------------------------------------------------------------------------

def test_min_confidence_is_the_lowest_top_probability_across_questions(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, probs={"acuity": [0.8, 0.2],
                                                 "specialty": [0.55, 0.45]})})
    rep = V.evaluate(lambda: eng, path)
    assert rep["rows"][0]["min_confidence"] == pytest.approx(0.55)
    assert rep["confidence"]["per_question_mean"] == {
        "acuity": 0.8, "specialty": 0.55}


def test_a_case_below_the_threshold_is_flagged_for_review(tmp_path):
    """The threshold lives on the normalised scale, so a 2-label question
    needs a very lopsided distribution to clear it."""
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    # raw 0.4/0.6 -> top 0.6 -> normalised (0.6-0.5)/0.5 = 0.2 < 0.35
    eng = FakeEngine({"a": _payload(esi=3, probs={"acuity": [0.4, 0.6]})})
    rep = V.evaluate(lambda: eng, path)
    assert rep["confidence"]["cases_below_threshold"] == ["a"]
    assert rep["rows"][0]["min_confidence_norm"] == pytest.approx(0.2)


def test_the_normalised_scale_ranks_the_opposite_way_to_the_raw_one(tmp_path):
    """The reason the scale exists: raw top-probability and 'how decided is
    this' disagree, and they disagree in the dangerous direction.

      narrow question, raw 0.55  -> normalised 0.10  (a shrug)
      wide  question, raw 0.40  -> normalised 0.36  (a clear call)

    A raw 0.35 threshold would gate the *wide* case and wave through the
    narrow one -- exactly backwards.
    """
    cases = [_case("narrow"), _case("wide")]
    path = _write_cases(tmp_path, cases)
    wide = [0.40] + [0.6 / 15] * 15
    eng = FakeEngine({
        "narrow": _payload(esi=3, probs={"acuity": [0.45, 0.55]}),
        "wide": _payload(esi=3, probs={"specialty": wide}),
    })
    rep = V.evaluate(lambda: eng, path)
    by_id = {r["id"]: r for r in rep["rows"]}
    assert by_id["narrow"]["min_confidence"] > by_id["wide"]["min_confidence"]
    assert by_id["narrow"]["min_confidence_norm"] < \
        by_id["wide"]["min_confidence_norm"]
    assert rep["confidence"]["cases_below_threshold"] == ["narrow"]


def test_a_near_uniform_wide_question_is_gated(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3, probs={"specialty": [1.0 / 16] * 16})})
    rep = V.evaluate(lambda: eng, path)
    assert rep["confidence"]["cases_below_threshold"] == ["a"]
    assert rep["rows"][0]["min_confidence_norm"] == pytest.approx(0.0)


def test_confidence_when_wrong_is_separated_from_confidence_when_right(tmp_path):
    cases = [_case("a", esi=3), _case("b", esi=3)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({
        "a": _payload(esi=3, probs={"acuity": [0.95, 0.05]}),
        "b": _payload(esi=3, probs={"acuity": [0.51, 0.49]}),
    })
    rep = V.evaluate(lambda: eng, path)
    assert rep["confidence"]["mean_confidence_when_esi_correct"] == pytest.approx(
        (0.95 + 0.51) / 2)
    assert rep["confidence"]["mean_confidence_when_esi_wrong"] is None


def test_missing_probabilities_do_not_crash_the_report(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    payload = _payload(esi=3)
    payload["probabilities"] = None
    eng = FakeEngine({"a": payload})
    rep = V.evaluate(lambda: eng, path)
    assert rep["rows"][0]["min_confidence"] is None
    assert rep["confidence"]["per_question_mean"] == {}
    assert rep["pass"] is True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_scorecard_and_calibration_render_without_error(tmp_path, capsys):
    cases = [_case("a", esi=1, esi_range=[1, 2], acuity="acute",
                   rf_required=True, rf_expected=True)]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=2, acuity="acute",
                                    red_flags={"hits": ["acs_chest_pain"]},
                                    specialty="cardiology",
                                    probs={"acuity": [0.9, 0.1]})})
    rep = V.evaluate(lambda: eng, path)
    V.print_scorecard(rep)
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "synthetic" in out.lower(), "the caveat must always print"
    V.print_calibration(rep)
    cal = capsys.readouterr().out
    assert "LOW-CONFIDENCE SWEEP" in cal


def test_failing_scorecard_says_fail_and_lists_the_reason(tmp_path, capsys):
    cases = [_case("a", esi=1, esi_range=[1, 2])]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=4)})
    rep = V.evaluate(lambda: eng, path)
    V.print_scorecard(rep)
    out = capsys.readouterr().out
    assert "RESULT: FAIL" in out
    assert "under-triage" in out


def test_report_is_json_serialisable(tmp_path):
    """The report is written to disk by the CLI; it must survive json.dumps."""
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3)})
    rep = V.evaluate(lambda: eng, path)
    blob = json.dumps(rep)
    assert json.loads(blob)["cases_total"] == 1


def test_report_never_carries_the_raw_narrative(tmp_path):
    cases = [_case("a")]
    path = _write_cases(tmp_path, cases)
    eng = FakeEngine({"a": _payload(esi=3)})
    rep = V.evaluate(lambda: eng, path)
    assert "synthetic narrative" not in json.dumps(rep)
