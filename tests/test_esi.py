#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESI resolution: the model may only ever move the code toward urgency."""

from src.triage import esi as E
from src.triage.redflags import RedFlagResult


def _floor(severity):
    return RedFlagResult(acute=severity in ("critical", "high"), severity=severity)


# ---------------------------------------------------------------------------
# the direction rule — the whole point of this module
# ---------------------------------------------------------------------------

def test_model_cannot_undercut_a_critical_floor():
    res = E.resolve(model_score=4.0, redflags=_floor("critical"))
    assert res.code == 1
    assert res.source == "red_flag_floor"
    assert res.floor == 1
    assert res.model_code == 5


def test_model_cannot_undercut_a_high_floor():
    res = E.resolve(model_score=4.0, redflags=_floor("high"))
    assert res.code == 2


def test_model_more_urgent_than_the_floor_is_kept():
    """Escalation is always allowed — only de-escalation is blocked."""
    res = E.resolve(model_score=0.0, redflags=_floor("high"))
    assert res.code == 1
    assert res.source == "model"


def test_floor_alone_is_used_when_the_model_says_nothing():
    res = E.resolve(model_score=None, redflags=_floor("high"))
    assert res.code == 2
    assert res.source == "red_flag_floor"
    assert res.model_code is None


def test_no_floor_and_no_model_defaults_to_least_urgent():
    res = E.resolve(model_score=None, redflags=None)
    assert res.code == 5
    assert res.floor == 5


def test_benign_presentation_keeps_the_model_level():
    res = E.resolve(model_score=3.0, redflags=_floor("none"))
    assert res.code == 4
    assert res.source == "model"


# ---------------------------------------------------------------------------
# override direction
# ---------------------------------------------------------------------------

def test_override_raises_urgency():
    res = E.resolve(model_score=2.0, override_label="one_level_higher",
                    redflags=_floor("none"))
    assert res.code == 3 - 1
    assert res.override_applied == 1
    assert res.source == "override"


def test_two_level_override_is_supported():
    res = E.resolve(model_score=3.0, override_label="two_levels_higher",
                    redflags=_floor("none"))
    assert res.code == 2
    assert res.override_applied == 2


def test_override_cannot_exceed_esi_one():
    res = E.resolve(model_score=0.0, override_label="two_levels_higher",
                    redflags=_floor("none"))
    assert res.code == 1


def test_unknown_override_label_is_ignored_not_guessed():
    res = E.resolve(model_score=1.0, override_label="three_levels_higher",
                    redflags=_floor("none"))
    assert res.override_applied == 0
    assert res.code == 2


def test_override_never_loosens_a_critical_floor():
    res = E.resolve(model_score=4.0, override_label="one_level_higher",
                    redflags=_floor("critical"))
    assert res.code == 1


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------

def test_code_is_always_a_valid_esi_code():
    for score in (None, 0.0, 1.5, 2.4, 3.99, 4.0):
        for severity in ("none", "moderate", "high", "critical"):
            res = E.resolve(model_score=score, redflags=_floor(severity))
            assert res.code in E.ESI_CODES


def test_clamp():
    assert E.clamp(-3) == 1
    assert E.clamp(0) == 1
    assert E.clamp(9) == 5


def test_worse_of_picks_the_more_urgent_code():
    assert E.worse_of(1, 4) == 1
    assert E.worse_of(4, 2) == 2
    assert E.worse_of(3, 3) == 3


def test_notes_explain_every_step():
    res = E.resolve(model_score=4.0, override_label="one_level_higher",
                    redflags=_floor("critical"))
    joined = " | ".join(res.notes)
    assert "model proposed ESI" in joined
    assert "floor" in joined


def test_to_dict_carries_full_provenance():
    res = E.resolve(model_score=4.0, redflags=_floor("high"))
    d = res.to_dict()
    assert d["esi_code"] == 2
    assert d["esi_model_code"] == 5
    assert d["esi_floor"] == 2
    assert d["esi_source"] == "red_flag_floor"
    assert isinstance(d["esi_notes"], list)


# ---------------------------------------------------------------------------
# escalation is monotone: adding information can never make a patient safer
# ---------------------------------------------------------------------------

def test_adding_a_red_flag_never_deescalates():
    for score in (0.0, 1.0, 2.0, 3.0, 4.0):
        none = E.resolve(model_score=score, redflags=_floor("none")).code
        crit = E.resolve(model_score=score, redflags=_floor("critical")).code
        assert crit <= none
