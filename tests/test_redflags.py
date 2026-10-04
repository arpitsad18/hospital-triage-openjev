#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic screen: keyword recall, negation guard, vitals, floor."""

import pytest

from src.triage import redflags as rf


# ---------------------------------------------------------------------------
# keyword recall — the cases that must never be missed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_hit", [
    ("Sudden crushing central chest pain radiating to the left arm",
     "acs_chest_pain"),
    ("Facial droop and slurred speech, onset 30 minutes ago", "stroke"),
    ("Anaphylaxis after a bee sting, tongue swelling", "anaphylaxis"),
    ("Massive haematemesis, vomiting blood, clammy", "haemorrhage"),
    ("Tearing chest pain radiating to the back", "aortic_dissection"),
    ("Cardiac arrest, CPR in progress", "cardiac_arrest"),
    ("GCS 9, unresponsive, not rousable", "altered_consciousness"),
    ("Fever with confusion and rigors, suspect sepsis", "sepsis"),
    ("Vomiting blood and melaena with dizziness", "haemorrhage"),
    ("Penetrating wound to the abdomen after a stabbing", "major_trauma"),
])
def test_critical_keywords_fire(text, expected_hit):
    res = rf.scan_text(text)
    assert expected_hit in res.hits
    assert res.acute is True


def test_benign_narrative_is_quiet():
    res = rf.scan_text("Mild sore throat for two days, no fever, eating and "
                       "drinking normally.")
    assert res.hits == []
    assert res.acute is False
    assert rf.esi_floor(res) == 5


# ---------------------------------------------------------------------------
# negation guard — a denied finding must not raise severity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Ankle sprain, no deformity, no open fracture, able to weight bear.",
    "Denies chest pain, no shortness of breath, no palpitations.",
    "Headache, no neck stiffness, no photophobia, no rash.",
    "No evidence of bleeding, no melaena, no haematemesis.",
    "No focal weakness, no slurred speech, no facial droop.",
])
def test_negated_findings_do_not_fire(text):
    res = rf.scan_text(text)
    assert res.hits == [], "false positive on negated narrative: %r" % res.hits
    assert res.acute is False
    assert rf.esi_floor(res) == 5


def test_negation_is_recorded_for_provenance():
    """A denied finding that does match a pattern must be logged as negated."""
    res = rf.scan_text("No evidence of bleeding, no melaena, no haematemesis.")
    assert res.hits == []
    assert any("haemorrhage" in n for n in res.negated)


def test_negation_does_not_mask_a_later_real_finding():
    """A negator far earlier must not suppress a genuine later match."""
    text = ("No chest pain at rest initially, but now heavy crushing chest "
            "pain radiating to the jaw with sweating.")
    res = rf.scan_text(text)
    assert "acs_chest_pain" in res.hits
    assert res.acute is True


def test_patterns_that_begin_with_a_negator_still_match():
    """"no pulse" / "not breathing" are the findings, not negations of them."""
    res = rf.scan_text("Collapsed, no pulse, not breathing, CPR ongoing")
    assert "cardiac_arrest" in res.hits
    assert res.acute is True


# ---------------------------------------------------------------------------
# vitals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vitals,key", [
    ({"sbp": 78}, "sbp"),
    ({"sbp": 230}, "sbp"),
    ({"hr": 145}, "hr"),
    ({"hr": 32}, "hr"),
    ({"rr": 36}, "rr"),
    ({"spo2": 86}, "spo2"),
    ({"temp": 39.8}, "temp"),
    ({"temp": 34.2}, "temp"),
    ({"gcs": 11}, "gcs"),
])
def test_abnormal_vital_fires(vitals, key):
    res = rf.scan_vitals(vitals)
    assert any(h.startswith(key + ":") for h in res.vital_hits)
    assert res.acute is True


def test_normal_vitals_are_quiet():
    res = rf.scan_vitals({"sbp": 122, "hr": 78, "rr": 16, "spo2": 98,
                          "temp": 36.8, "gcs": 15, "age": 41})
    assert res.vital_hits == []
    assert res.acute is False


def test_vital_thresholds_are_boundaries_not_estimates():
    """Exactly at the cut-off must not fire; one step past must."""
    assert rf.scan_vitals({"sbp": 90}).vital_hits == []
    assert rf.scan_vitals({"sbp": 89}).vital_hits
    assert rf.scan_vitals({"hr": 130}).vital_hits == []
    assert rf.scan_vitals({"hr": 131}).vital_hits
    assert rf.scan_vitals({"spo2": 92}).vital_hits == []
    assert rf.scan_vitals({"spo2": 91}).vital_hits


def test_missing_or_junk_vitals_do_not_crash():
    assert rf.scan_vitals(None).vital_hits == []
    assert rf.scan_vitals({}).vital_hits == []
    assert rf.scan_vitals({"sbp": "not a number"}).vital_hits == []


def test_vital_reason_interpolates_the_observed_value():
    """Regression: a printf template built with the threshold swallows it."""
    res = rf.scan_vitals({"sbp": 78})
    assert "78" in " ".join(res.reasons)


# ---------------------------------------------------------------------------
# combinations
# ---------------------------------------------------------------------------

def test_sepsis_screen_combination():
    res = rf.scan(narrative="Admitted with fever and tachycardia",
                  vitals={"temp": 38.6, "hr": 118})
    assert "sepsis_screen" in res.hits
    assert res.severity == "critical"


def test_shock_screen_combination():
    res = rf.scan(narrative="Feels unwell", vitals={"sbp": 88, "hr": 121})
    assert "shock_screen" in res.hits


def test_paediatric_and_elderly_fever_combinations():
    assert "infant_fever" in rf.scan(
        vitals={"age": 0.5, "temp": 38.4}).hits
    assert "elderly_fever" in rf.scan(
        vitals={"age": 82, "temp": 38.2}).hits


def test_combinations_need_both_arms():
    """Fever alone, or tachycardia alone, is not the screen firing."""
    assert "sepsis_screen" not in rf.scan(vitals={"temp": 38.6}).hits
    assert "sepsis_screen" not in rf.scan(vitals={"hr": 118}).hits


def test_scan_is_a_pure_function():
    """Same input, same output — no hidden state between calls."""
    a = rf.scan("Chest pain radiating to the arm", {"sbp": 85})
    b = rf.scan("Chest pain radiating to the arm", {"sbp": 85})
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("severity,floor", [
    ("critical", 1), ("high", 2), ("moderate", 3), ("none", 5),
])
def test_esi_floor_mapping(severity, floor):
    assert rf.esi_floor(rf.RedFlagResult(acute=False, severity=severity)) == floor


def test_floor_is_never_more_permissive_than_the_finding():
    """The floor can only ever be at least as urgent as the severity implies."""
    for severity in ("critical", "high", "moderate", "none"):
        res = rf.RedFlagResult(acute=severity in ("critical", "high"),
                               severity=severity)
        assert 1 <= rf.esi_floor(res) <= 5


def test_result_serialises_all_provenance_fields():
    res = rf.scan("Denies chest pain", {"spo2": 88})
    d = res.to_dict()
    assert set(d) == {"acute", "severity", "hits", "vital_hits",
                      "explanation", "negated"}
