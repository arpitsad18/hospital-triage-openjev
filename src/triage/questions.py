#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The typed question set for ED triage — single source of truth.

Every question here is a System-One question: a closed label space answered in
one forward pass with a raw probability vector. Nothing in this module calls a
model; it only declares the contract, so the API, the CLI and the validation
runner can never drift apart.

Design notes
------------
* Label *names* (dict keys / list order) are part of the public API. Renaming
  one breaks stored answers, so treat them as frozen.
* ``esi`` is a ``score`` question. OpenJev returns an index-weighted mean with
  a 0-based index, so the raw score runs 0..4 and the human-facing ESI code is
  ``round(score) + 1``. :func:`esi_code_from_score` is the only place that
  arithmetic lives.
* Questions are grouped into ACTIONS / ROUTING / SAFETY so a caller can run a
  cheap subset (e.g. acuity only) without paying for ten forward passes.
"""

from __future__ import annotations

from typing import Dict, List

from openjev.types import Choice, Noul, Question, Score

# ---------------------------------------------------------------------------
# label spaces (frozen public API)
# ---------------------------------------------------------------------------

ACUITY_LABELS = ("acute", "non_acute")

PATIENT_TYPES = {
    "emergency": "Genuine emergency presentation needing immediate ED care",
    "elective": "Planned, scheduled or follow-up care",
    "transfer": "Arriving from or being referred by another facility",
    "unknown": "No information in the state to determine this",
}

SPECIALTIES = {
    "cardiology": "Heart, chest pain, arrhythmia, heart failure, blood pressure",
    "neurology": "Stroke, seizure, headache, weakness, confusion",
    "orthopaedics": "Fracture, joint, limb injury, back pain with trauma",
    "gastroenterology": "Abdominal pain, vomiting, bleeding, jaundice",
    "pulmonology": "Breathing difficulty, asthma, cough, pneumonia",
    "nephrology": "Kidney injury, dialysis, urinary, electrolyte",
    "endocrinology": "Diabetes, thyroid, adrenal, metabolic",
    "general_surgery": "Acute abdomen, appendicitis, hernia, wounds needing theatre",
    "obstetrics_gynaecology": "Pregnancy, labour, vaginal bleeding, pelvic pain",
    "paediatrics": "Patient is a child and the above do not clearly dominate",
    "psychiatry": "Self-harm, agitation, psychosis, substance withdrawal",
    "oncology": "Known or suspected malignancy, chemotherapy complications",
    "ent": "Ear, nose, throat, airway-adjacent complaints",
    "ophthalmology": "Eye injury, sudden vision loss, red painful eye",
    "dermatology": "Rash, skin infection, allergic skin reaction",
    "internal_medicine": "Undifferentiated adult medical problem",
}

IMAGING_MODALITIES = {
    "none": "No imaging is indicated from this state",
    "xray": "Plain radiograph",
    "ct": "Computed tomography",
    "usg": "Ultrasound / sonography",
    "mri": "Magnetic resonance imaging",
    "other": "Angiography, echocardiography, endoscopy or another study",
}

ESI_OVERRIDE = {
    "no": "Keep the ESI level implied by the presentation",
    "one_level_higher": "Deterioration risk warrants one level more urgent",
    "two_levels_higher": "Extremely high risk; two levels more urgent",
}

ESI_LEVELS = [
    "ESI 1 - resuscitation: immediately life-threatening, needs immediate intervention",
    "ESI 2 - emergent: high risk, should not wait, needs rapid assessment",
    "ESI 3 - urgent: stable but needs multiple resources, can wait briefly",
    "ESI 4 - less urgent: one resource needed, can wait longer",
    "ESI 5 - non-urgent: no resources, can be seen in routine order or clinic",
]


def esi_code_from_score(score: float) -> int:
    """Map a raw 0-based ``esi`` index-weighted score to an ESI code 1..5.

    ``weighted_score`` returns ``sum(i * p_i)`` over five levels, so the value
    is a real number in ``[0, 4]``. Rounding to the nearest level and shifting
    to 1-based gives the human ESI code.
    """
    return max(1, min(5, int(round(float(score))) + 1))


# ---------------------------------------------------------------------------
# question builders
# ---------------------------------------------------------------------------

def build_questions(
    include: List[str] | None = None,
) -> Dict[str, Question]:
    """Return the triage question set, optionally restricted to ``include``.

    Parameters
    ----------
    include : list of question ids, optional
        ``None`` or empty returns all ten. Unknown ids raise ``KeyError``.

    Returns
    -------
    dict
        Mapping ``qid -> Question``, in a stable presentation order.
    """
    questions: Dict[str, Question] = {
        "acuity": Choice(
            instructions=(
                "Triage acuity gate. Is this patient ACUTE or NON-ACUTE? "
                "ACUTE means time-critical: an immediate threat to life, limb "
                "or function that needs emergency resuscitation or rapid "
                "intervention right now. NON-ACUTE means the patient is "
                "physiologically stable and safe in a routine queue."
            ),
            criteria={
                "acute": "Time-critical; immediate emergency intervention required",
                "non_acute": "Stable and safe to wait in the routine queue",
            },
        ),
        "patient_type": Choice(
            instructions=(
                "What kind of presentation is this? Choose the single best fit."
            ),
            criteria=PATIENT_TYPES,
        ),
        "esi": Score(
            instructions=(
                "Assign an Emergency Severity Index level from 1 (most urgent) "
                "to 5 (least urgent), based on threat to life and resources "
                "expected to be needed."
            ),
            criteria=ESI_LEVELS,
        ),
        "imaging": Choice(
            instructions=(
                "What is the single most appropriate imaging modality for this "
                "presentation? Choose 'none' if no imaging is indicated."
            ),
            criteria=IMAGING_MODALITIES,
        ),
        "esi_override": Choice(
            instructions=(
                "Some presentations carry a risk of rapid deterioration that "
                "the level alone understates. Does this patient need their "
                "level raised above the plain presentation?"
            ),
            criteria=ESI_OVERRIDE,
        ),
        "specialty": Choice(
            instructions=(
                "Which single specialty should primarily receive this patient?"
            ),
            criteria=SPECIALTIES,
        ),
        "needs_consult": Noul(
            instructions=(
                "Does this patient need a specialist consultation during this "
                "ED encounter, beyond the assessing emergency clinician?"
            ),
            criteria="Yes if a second specialty must be physically involved now.",
        ),
        "translator": Noul(
            instructions=(
                "Is there any indication that this patient needs an interpreter "
                "or communication assistance before care can proceed?"
            ),
            criteria="Yes only if the state indicates a language barrier.",
        ),
        "is_emergency_admission": Noul(
            instructions=(
                "Will this patient require hospital admission through the "
                "emergency pathway rather than discharge or routine clinic?"
            ),
            criteria="Yes if inpatient emergency care is required.",
        ),
        "tele_followup_instead": Noul(
            instructions=(
                "Could this problem be managed by teleconsultation or a clinic "
                "review instead of an in-person emergency visit?"
            ),
            criteria="Yes only if the presentation is clearly low-acuity.",
        ),
        "suicidal_risk": Noul(
            instructions=(
                "Does this state indicate active suicidal ideation, "
                "self-harm intent, or a deliberate overdose?"
            ),
            criteria="Yes only on affirmative evidence in the state.",
        ),
    }

    if include:
        missing = [q for q in include if q not in questions]
        if missing:
            raise KeyError("unknown question id(s): %s" % ", ".join(missing))
        questions = {q: questions[q] for q in include}
    return questions


# Convenience groupings for callers that want a cheap subset.
ACTION_QUESTIONS = ["acuity", "esi", "esi_override", "imaging"]
ROUTING_QUESTIONS = ["patient_type", "specialty", "needs_consult",
                     "is_emergency_admission", "tele_followup_instead"]
SAFETY_QUESTIONS = ["translator", "suicidal_risk"]
ALL_QUESTIONS = ACTION_QUESTIONS + ROUTING_QUESTIONS + SAFETY_QUESTIONS

DEFAULT_SYSTEM_PROMPT = (
    "You are an emergency department triage decision engine. "
    "You do not write prose, explanations, or reasoning. "
    "You read a patient state and answer with exactly the single label "
    "character or word that the question asks for."
)


def schema() -> dict:
    """Machine-readable description of the question contract (for /schema)."""
    out = {}
    for qid, q in build_questions().items():
        entry = {"type": q.type_name, "instructions": q.instructions}
        if q.type_name == "choice":
            entry["labels"] = list(q.criteria.keys())
            entry["label_descriptions"] = dict(q.criteria)
        elif q.type_name == "score":
            entry["levels"] = list(q.criteria)
            entry["level_indices"] = list(range(len(q.criteria)))
            entry["note"] = "raw score is 0-indexed; esi_code = round(score) + 1"
        else:
            entry["labels"] = ["Yes", "No"]
        out[qid] = entry
    return out
