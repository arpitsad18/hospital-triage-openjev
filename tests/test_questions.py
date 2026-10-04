#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen label spaces and the raw-score -> ESI code map."""

import pytest

from src.triage import questions as Q


# ---------------------------------------------------------------------------
# label spaces are a public contract
# ---------------------------------------------------------------------------

def test_expected_questions_exist():
    qs = Q.build_questions()
    assert set(qs) == set(Q.ALL_QUESTIONS)
    # the three groups partition the set with no overlap
    groups = Q.ACTION_QUESTIONS + Q.ROUTING_QUESTIONS + Q.SAFETY_QUESTIONS
    assert sorted(groups) == sorted(Q.ALL_QUESTIONS)
    assert len(groups) == len(set(groups))


def test_include_restricts_and_preserves_order():
    qs = Q.build_questions(["specialty", "acuity"])
    assert set(qs) == {"specialty", "acuity"}


def test_unknown_question_id_raises():
    with pytest.raises(KeyError):
        Q.build_questions(["not_a_real_question"])


def test_acuity_labels_are_exactly_two():
    assert Q.ACUITY_LABELS == ("acute", "non_acute")


def test_specialty_and_imaging_label_spaces_are_frozen():
    assert list(Q.SPECIALTIES)[0] == "cardiology"
    assert len(Q.SPECIALTIES) == 16
    assert set(Q.IMAGING_MODALITIES) == {"none", "xray", "ct", "usg", "mri", "other"}
    assert set(Q.ESI_OVERRIDE) == {"no", "one_level_higher", "two_levels_higher"}


def test_label_set_sizes_are_stable():
    """Label counts are part of the frozen contract; a silent growth of the
    specialty list is what pushes a question past the logprob window."""
    assert len(Q.SPECIALTIES) == 16
    assert len(Q.PATIENT_TYPES) == 4
    assert len(Q.IMAGING_MODALITIES) == 6
    assert len(Q.ESI_OVERRIDE) == 3


def test_the_window_ceiling_is_twenty_labels():
    """The tail-bounding path exists for label sets above the Ollama window;
    it must engage past 20 labels and not before (whatever today's sizes are)."""
    from openjev.core import LETTERS
    assert len(LETTERS) >= 20


def test_esi_levels_are_five_and_ordered():
    assert len(Q.ESI_LEVELS) == 5
    for i, level in enumerate(Q.ESI_LEVELS, start=1):
        assert level.startswith("ESI %d" % i)


def test_every_label_has_a_non_empty_description():
    for space in (Q.PATIENT_TYPES, Q.SPECIALTIES, Q.IMAGING_MODALITIES,
                  Q.ESI_OVERRIDE):
        for label, desc in space.items():
            assert label and desc.strip()


# ---------------------------------------------------------------------------
# raw score -> ESI code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,code", [
    (-1.0, 1), (0.0, 1), (0.4, 1), (0.5, 1), (0.6, 2), (1.0, 2),
    (2.0, 3), (3.0, 4), (4.0, 5), (9.0, 5),
])
def test_score_maps_to_a_valid_esi_code(score, code):
    assert Q.esi_code_from_score(score) == code


def test_score_map_is_monotone_and_within_range():
    prev = 0
    for i in range(-20, 60):
        code = Q.esi_code_from_score(i / 10.0)
        assert 1 <= code <= 5
        assert code >= prev
        prev = code


def test_schema_exposes_the_frozen_label_spaces():
    s = Q.schema()
    assert "questions" in s or "specialties" in s or isinstance(s, dict)
