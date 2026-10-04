#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic red-flag screen — the safety floor under the model.

Why this exists
---------------
A masked-softmax decision model is a *statistical* judgement. On a red-flag
presentation (STEMI, stroke, anaphylaxis, sepsis) a single wrong label is a
patient-safety event, and no amount of calibration makes that acceptable in a
gate that can silently downgrade a sick patient. So acuity is not delegated:
this module hard-codes the recognition of time-critical physiology and keyword
patterns, and the engine forces ``acute`` whenever anything here fires.

The model still answers, and its answer is compared against the screen. A
disagreement is surfaced as a *degraded check* — evidence the model drifted,
logged for review, never used to soften the decision.

Everything in here is a pure function of the input string plus optional
structured vitals. No randomness, no network, no model.

Reference framing: keyword lists follow common ED triage mnemonics; numeric
thresholds are the widely used adult cut-offs (they are deliberately
conservative — a false "acute" costs an extra review, a false "non_acute"
costs a patient).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# thresholds (adult; conservative)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "sbp_low": 90,        # systolic hypotension
    "sbp_high": 220,      # hypertensive emergency range
    "hr_low": 40,
    "hr_high": 130,
    "rr_high": 30,
    "spo2_low": 92,       # room air
    "temp_high": 39.0,
    "temp_low": 35.0,
    "gcs_low": 13,
    "age_neonate": 0.1,   # under ~1 month
    "age_elderly": 75,
}

# ---------------------------------------------------------------------------
# keyword patterns: (name, severity, regex)
#   severity "critical" -> always forces acute
#   severity "high"     -> forces acute
#   severity "moderate" -> recorded, forces acute only with a supporting vital
# ---------------------------------------------------------------------------

_PATTERNS: List[tuple] = [
    # --- airway / breathing -------------------------------------------------
    ("airway_compromise", "critical",
     r"\b(unable to (speak|breathe)|stridor|gurgling|airway obstruct|"
     r"choking|tracheal deviation|facial (burns|swelling)|angioedema)\b"),
    ("respiratory_distress", "high",
     r"\b(severe (breathlessness|dyspnoea|dyspnea|shortness of breath)|"
     r"cannot (speak|complete a sentence)|gasping|using accessory muscles|"
     r"silent chest|cyanosis|cyanotic|saturations?( of)? (8|7|6|5|4|3|2|1)\d ?%?)\b"),
    ("anaphylaxis", "critical",
     r"\b(anaphyla|peanut|nut (exposure|allergy)|bee ?sting|wasp ?sting|"
     r"tongue (swelling|swollen)|lip swelling|throat (closing|tight)|"
     r"widespread (urticaria|hives)|epipen|adrenaline auto|allergic reaction)\b"),

    # --- circulation --------------------------------------------------------
    ("cardiac_arrest", "critical",
     r"\b(cardiac arrest|arrest|no pulse|unresponsive( and)? not breathing|"
     r"cpr (in progress|ongoing)|collapsed and (unresponsive|not breathing)|"
     r"asystole|pulseless|vt arrest|v ?fib)\b"),
    ("acs_chest_pain", "high",
     r"\b(crushing|heavy|pressure|squeez|tightness|elephant).{0,30}"
     r"(chest|substernal|retrosternal)|"
     r"chest pain.{0,60}(radiat|left arm|jaw|sweat|diaphor|crush)|"
     r"(radiat|left arm).{0,30}chest|"
     r"\b(stemi|nstemi|acute coronary|myocardial infarct|heart attack|"
     r"unstable angina)\b"),
    ("stroke", "critical",
     r"\b(stroke|cva|facial droop|face droop|arm weakness.{0,30}speech|"
     r"speech (slurred|difficulty)|slurred speech|hemiplegia|hemiparesis|"
     r"hemi[ ]?paresis|sudden (weakness|numbness).{0,30}(one side|arm|leg|face)|"
     r"thunderclap headache|sudden vision loss|worst headache of (my|his|her) life|"
     r"thrombolysis|thrombectomy)\b"),
    ("shock_hypoperfusion", "critical",
     r"\b(shock|hypoperfus|hypotens|clammy|cold and sweaty|mottled|"
     r"capillary refill|altered (mental status|consciousness) with hypotension|"
     r"septic)\b"),
    ("haemorrhage", "critical",
     r"\b(massive (bleeding|haemorrhage|hemorrhage)|haemorrhag|hemorrhag|"
     r"vomiting blood|haematemesis|hematemesis|coughing (up )?blood|"
     r"haemoptysis|hemoptysis|melaena|melena|rectal bleeding with (dizz|faint|collapse)|"
     r"postpartum (haemorrhage|hemorrhage)|vaginal bleeding in pregnancy|"
     r"bleeding (that will not stop|did not stop|soaking|pouring)|"
     r"head injury.{0,40}(blood thinner|warfarin|anticoagul)|"
     r"anticoagul.{0,40}head injury)\b"),

    # --- neurology / consciousness ------------------------------------------
    ("altered_consciousness", "critical",
     r"\b(unconscious|unresponsive|coma|comatose|gcs ?(of )?([0-9]|1[0-2])\b|"
     r"not (rousable|waking|responding)|no response to|"
     r"altered (mental status|consciousness)|confusion.{0,30}acute|"
     r"acute confusion|disorient|drowsy and confused|"
     r"convuls|seizure|fit(s)? (now|ongoing|in progress)|status epilepticus)\b"),
    ("meningitis_signs", "high",
     r"\b(neck stiffness|neck rigidity|photophobia with fever|"
     r"non[- ]blanching rash|purpuric rash|petechial rash)\b"),

    # --- metabolic / sepsis -------------------------------------------------
    ("sepsis", "high",
     r"\b(sepsis|septic (shock|patient)|suspect sepsis|fever with "
     r"(confusion|hypotens|rigors)|rigors|"
     r"(fever|temperature).{0,40}(confusion|hypotens|tachycard|breathless))\b"),
    ("dka_metabolic", "high",
     r"\b(diabetic ketoacidosis|dka|ketoacidosis|"
     r"(very )?(high|raised) (blood )?(sugar|glucose) with (vomit|drowsy|confus|"
     r"breath|kussmaul)|kussmaul|decompensated diabetes)\b"),

    # --- obstetric ----------------------------------------------------------
    ("obstetric_emergency", "critical",
     r"\b(eclampsia|pre[- ]?eclampsia with|severe pre[- ]?eclampsia|"
     r"cord prolapse|shoulder dystocia|placental abruption|placenta praevia.{0,30}bleed|"
     r"(pregnant|pregnancy|gravid).{0,40}(bleeding|bleed|fitting|seizure|"
     r"severe headache|blurred vision|no fetal movement)|"
     r"\b(term|40 weeks|39 weeks|38 weeks|37 weeks).{0,40}(labour|labour pain|contractions)|"
     r"imminent (delivery|birth)|crowning|in (established )?labour)\b"),

    # --- trauma -------------------------------------------------------------
    ("major_trauma", "critical",
     r"\b(major trauma|serious trauma|polytrauma|rta|road traffic|"
     r"high[- ]speed|fall from (height|a height|> ?3|over ?3)|"
     r"fell (from|off) (a )?(roof|ladder|balcony|building)|"
     r"stab(bed|bing)? (wound|injury)|gunshot|penetrating (wound|injury|trauma)|"
     r"(deformity|open fracture|bone (visible|through skin))|"
     r"head injury with (vomit|loss of consciousness|worsening)|"
     r"(loss of consciousness|knocked out) after (a )?(fall|crash|blow|hit))\b"),
    ("burns", "high",
     r"\b(burn|burns|scald|scalded|flame|chemical burn|electrical burn)\b.{0,80}"
     r"\b(face|neck|airway|hands|perineum|circumferential|"
     r"large|extensive|[2-9]\d ?%|deep|full thickness|smoke inhalation)\b"),
    ("bites_rabies", "moderate",
     r"\b(dog bite|animal bite|snake bite|snakebite|scorpion|rabies|"
     r"monkey bite|cat bite|human bite)\b"),
    ("poisoning_overdose", "high",
     r"\b(overdose|poisoning|ingested|swallowed|took (too many|a lot of)|"
     r"organophosphate|pesticide|paracetamol overdose|opioid overdose|"
     r"intentional (overdose|ingestion)|self[- ]?poisoning)\b"),
    ("self_harm", "high",
     r"\b(suicid|self[- ]?harm|attempted suicide|wants to die|end (my|his|her) life|"
     r"cut (my|his|her) (wrist|arm)|hanging|jumped from|"
     r"not want to (live|be here)|killed (myself|himself|herself))\b"),

    # --- severe pain / abdominal -------------------------------------------
    ("acute_abdomen", "high",
     r"\b(rigid abdomen|board[- ]?like|guarding|rebound tenderness|"
     r"severe abdominal pain with (vomit|fever|sweat|syncope)|"
     r"(testicular|ovarian) torsion|testicular pain with swelling|"
     r"strangulated hernia|bowel obstruction|"
     r"acute abdomen|peritonitis)\b"),
    ("aortic_dissection", "critical",
     r"\b(tearing (chest|back) pain|aortic dissection|dissecting aneurysm|"
     r"ripping (chest|back) pain|back pain radiating to (legs|leg) with "
     r"(pulse|blood pressure) difference|unequal (arm )?(pulses|blood pressure))\b"),
    ("pe_pulmonary_embolism", "high",
     r"\b(pulmonary embol|pe\b.{0,30}(breathless|shortness of breath|chest pain)|"
     r"pleuritic chest pain with (breathless|calves|immobil)|"
     r"dvt with (breathless|shortness of breath)|"
     r"recent (surgery|flight|immobilisation).{0,40}(breathless|collapse))\b"),
    ("eye_vision_threat", "high",
     r"\b(sudden vision loss|loss of vision|chemical (in|to the) eye|"
     r"penetrating eye|ocular foreign body|acute glaucoma|"
     r"red painful eye with (vision|halo)|retinal detachment|"
     r"central retinal|sudden blindness)\b"),

    # --- paediatric ---------------------------------------------------------
    ("paediatric_red_flag", "high",
     r"\b(?:(neonate|newborn|infant).{0,60}(fever|not feeding|floppy|lethargic|"
     r"grunting|apnoea|apnea|jaundice|bulging fontanelle|rash)|"
     r"(floppy|floppiness).{0,30}(baby|infant|child)|"
     r"child.{0,30}(not drinking|not passing urine|drowsy|unresponsive|"
     r"grunting|blue lips)|"
     r"baby.{0,30}(very hot|cold|blue|grey|floppy|unresponsive))\b"),
]

_COMPILED = [(name, sev, re.compile(pat, re.IGNORECASE)) for name, sev, pat in _PATTERNS]

# ---------------------------------------------------------------------------
# negation guard
# ---------------------------------------------------------------------------
# Triage narratives document what is ABSENT as carefully as what is present:
# "no deformity", "denies chest pain", "no evidence of bleeding". A bare
# keyword scan reads those as findings and fires a red flag on a well patient,
# which is the failure the ankle-sprain case shows. A negator immediately
# before the match means the finding is *denied*.
#
# The cue must sit directly in front of the match (at most a couple of filler
# words between), and only the text BEFORE the match is inspected -- patterns
# that legitimately begin with a negation ("no pulse", "not breathing") match
# from their own first character, so their own negator is never seen here.
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|denies|denied|denying|negative\s+for|nil|"
    r"free\s+of|absent|unremarkable|no\s+evidence\s+of|no\s+signs?\s+of|"
    r"no\s+history\s+of|no\s+obvious|rules?\s+out)\b"
    r"(?:\s+(?:evidence|signs?|obvious|definite|clinical|any|of|a|an|the|"
    r"features?|findings?)){0,3}\s*$",
    re.IGNORECASE,
)

# Look-back window. Long enough for "there is no evidence of X", short enough
# that an unrelated earlier clause cannot suppress a later real finding.
_NEG_WINDOW = 32


def _is_negated(text: str, start: int) -> bool:
    """True when the match at ``start`` is preceded by a negation cue."""
    lo = max(0, start - _NEG_WINDOW)
    return bool(_NEGATION.search(text[lo:start]))


@dataclass
class RedFlagResult:
    """Outcome of the deterministic screen."""

    acute: bool
    severity: str = "none"            # none | moderate | high | critical
    hits: List[str] = field(default_factory=list)
    vital_hits: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    negated: List[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        if not self.hits and not self.vital_hits:
            return "no deterministic red flag matched"
        return "; ".join(self.reasons)

    def to_dict(self) -> dict:
        return {
            "acute": self.acute,
            "severity": self.severity,
            "hits": list(self.hits),
            "vital_hits": list(self.vital_hits),
            "explanation": self.explanation,
            "negated": list(self.negated),
        }


_SEV_ORDER = {"none": 0, "moderate": 1, "high": 2, "critical": 3}


def _worse(a: str, b: str) -> str:
    return a if _SEV_ORDER[a] >= _SEV_ORDER[b] else b


def scan_text(text: str) -> RedFlagResult:
    """Match the keyword patterns against a free-text narrative.

    A match that is explicitly negated ("no deformity", "denies chest pain")
    is recorded in ``negated`` and does **not** raise severity: the narrative
    states the finding is absent, and firing on it is a false alarm on a
    patient who is, by the text, well.
    """
    res = RedFlagResult(acute=False)
    if not text:
        return res
    for name, sev, rx in _COMPILED:
        # A pattern can match leftmost on a *negated* mention and swallow a
        # genuine later finding ("no chest pain ... but now crushing chest pain
        # radiating to the jaw"), which would silently under-triage. So walk
        # every candidate: skip past each negated sighting and fire on the
        # first occurrence that is not negated.
        pos = 0
        neg_snippet = None
        fired = None
        while pos <= len(text):
            m = rx.search(text, pos)
            if not m:
                break
            if not _is_negated(text, m.start()):
                fired = m
                break
            if neg_snippet is None:
                neg_snippet = m.group(0).strip()[:60]
            # Resume just past the negation cue (and always past this match's
            # start, so the walk is strictly monotonic and cannot loop).
            lo = max(0, m.start() - _NEG_WINDOW)
            cue = _NEGATION.search(text[lo:m.start()])
            pos = max(lo + (cue.end() if cue else 0), m.start() + 1)
        if fired is None:
            if neg_snippet is not None:
                res.negated.append("%s (%s): '%s'" % (name, sev, neg_snippet))
            continue
        res.hits.append(name)
        res.severity = _worse(res.severity, sev)
        res.reasons.append("%s (%s): '%s'"
                           % (name, sev, fired.group(0).strip()[:60]))
    if res.severity in ("high", "critical"):
        res.acute = True
    return res


def scan_vitals(vitals: Optional[Dict[str, float]]) -> RedFlagResult:
    """Apply numeric thresholds to structured vitals (any key may be absent)."""
    res = RedFlagResult(acute=False)
    if not vitals:
        return res

    def num(*keys):
        for k in keys:
            v = vitals.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    # Each ``fmt`` is a CALLABLE taking the observed value and returning text.
    # Plain `"..." % threshold` templates cannot work here: substituting the
    # threshold at build time consumes the printf slot, so the later
    # ``fmt % v`` either raises or drops the number. A callable keeps both.
    def _f(tmpl, *pre):
        return lambda v: tmpl % ((v,) + pre)

    checks = [
        ("sbp", lambda v: v < THRESHOLDS["sbp_low"],
         _f("systolic BP %.0f mmHg < %d (hypotension)", THRESHOLDS["sbp_low"])),
        ("sbp", lambda v: v > THRESHOLDS["sbp_high"],
         _f("systolic BP %.0f mmHg > %d (hypertensive emergency)",
            THRESHOLDS["sbp_high"])),
        ("hr", lambda v: v > THRESHOLDS["hr_high"],
         _f("heart rate %.0f > %d (marked tachycardia)", THRESHOLDS["hr_high"])),
        ("hr", lambda v: v < THRESHOLDS["hr_low"],
         _f("heart rate %.0f < %d (bradycardia)", THRESHOLDS["hr_low"])),
        ("rr", lambda v: v > THRESHOLDS["rr_high"],
         _f("respiratory rate %.0f > %d", THRESHOLDS["rr_high"])),
        ("spo2", lambda v: v < THRESHOLDS["spo2_low"],
         _f("SpO2 %.0f%% < %d%% (hypoxaemia)", THRESHOLDS["spo2_low"])),
        ("temp", lambda v: v >= THRESHOLDS["temp_high"],
         _f("temperature %.1f >= %.1f (high fever)", THRESHOLDS["temp_high"])),
        ("temp", lambda v: v <= THRESHOLDS["temp_low"],
         _f("temperature %.1f <= %.1f (hypothermia)", THRESHOLDS["temp_low"])),
        ("gcs", lambda v: v < THRESHOLDS["gcs_low"],
         _f("GCS %.0f < %d (reduced consciousness)", THRESHOLDS["gcs_low"])),
        ("age", lambda v: v < THRESHOLDS["age_neonate"],
         _f("age %.2f (neonate, under %.2f)", THRESHOLDS["age_neonate"])),
    ]

    val = num("sbp", "systolic", "bp_systolic", "systolic_bp")
    for key, test, fmt in checks:
        v = num(key, {"sbp": "systolic"}.get(key, key))
        if v is None:
            continue
        try:
            fired = test(v)
        except Exception:
            continue
        if fired:
            res.vital_hits.append(key + ":" + ("%.4g" % v))
            try:
                res.reasons.append(fmt(v) if callable(fmt) else fmt % v)
            except Exception:
                res.reasons.append("%s abnormal (%s)" % (key, "%.4g" % v))
    if res.vital_hits:
        res.severity = _worse(res.severity, "high")
        res.acute = True
    if val is not None and val < THRESHOLDS["sbp_low"]:
        res.acute = True
    return res


def scan(
    narrative: Optional[str] = None,
    vitals: Optional[Dict[str, float]] = None,
) -> RedFlagResult:
    """Full screen: narrative keywords + numeric vitals + combinations.

    The combination rules matter — each is individually survivable but
    together they define shock, sepsis and ACS in ways a single threshold
    misses.
    """
    text_res = scan_text(narrative or "")
    vital_res = scan_vitals(vitals)

    hits = list(dict.fromkeys(text_res.hits + vital_res.hits))
    vit = list(dict.fromkeys(vital_res.vital_hits))
    negated = list(dict.fromkeys(text_res.negated))
    reasons = list(text_res.reasons) + [r for r in vital_res.reasons
                                       if r not in text_res.reasons]
    sev = _worse(text_res.severity, vital_res.severity)
    acute = text_res.acute or vital_res.acute

    if vitals:
        def num(*keys):
            for k in keys:
                v = vitals.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return None

        sbp, hr, rr, spo2, temp, age = (
            num("sbp", "systolic", "bp_systolic"), num("hr", "pulse", "heart_rate"),
            num("rr", "respiratory_rate"), num("spo2", "o2sat", "sat"),
            num("temp", "temperature"), num("age"),
        )
        combos = [
            # sepsis screen: SIRS-like, requires fever/hypothermia + tachycardia
            (temp is not None and temp >= 38.0 and hr is not None and hr > 110,
             "sepsis_screen", "fever >=38.0 with HR >110 (sepsis screen positive)"),
            (temp is not None and temp <= 35.5 and hr is not None and hr > 100,
             "sepsis_screen_hypothermia",
             "hypothermia <=35.5 with HR >100 (sepsis screen positive)"),
            (sbp is not None and sbp < 100 and hr is not None and hr > 110,
             "shock_screen", "SBP <100 with HR >110 (shock physiology)"),
            (spo2 is not None and spo2 < 94 and rr is not None and rr > 24,
             "respiratory_failure_screen",
             "SpO2 <94% with RR >24 (impending respiratory failure)"),
            (age is not None and age >= 75 and temp is not None and temp >= 38.0,
             "elderly_fever", "age >=75 with fever (low reserve; ESI 2)"),
            (age is not None and age < 1 and temp is not None and temp >= 38.0,
             "infant_fever", "infant under 1 year with fever >=38.0 (ESI 1-2)"),
        ]
        for fired, name, why in combos:
            if fired:
                hits.append(name)
                reasons.append(why)
                sev = _worse(sev, "critical")
                acute = True

    hits = list(dict.fromkeys(hits))
    return RedFlagResult(acute=acute, severity=sev, hits=hits,
                         vital_hits=vit, reasons=reasons, negated=negated)


def esi_floor(result: RedFlagResult) -> int:
    """Most-urgent ESI code that the screen is willing to allow (lower = worse).

    This is the floor the model cannot undercut: final ESI is
    ``min(model_esi_code, esi_floor(result))``.
    """
    if result.severity == "critical":
        return 1
    if result.severity == "high":
        return 2
    if result.severity == "moderate":
        return 3
    return 5
