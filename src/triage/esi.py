#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESI mapping and override *direction* rules.

The single most dangerous thing a language model can do in triage is label a
sick patient as less urgent than they are. So the direction of every override
is enforced in code:

* the red-flag screen sets a **floor** (most urgent permitted code);
* the model may ask for a level to be raised (1 or 2 levels more urgent);
* the model's own ``esi`` estimate is only ever allowed to move the code
  **toward** urgency, never away from it;
* nothing may move the code below the floor.

``min()`` on integer ESI codes (where 1 is most urgent) is exactly that rule,
so the whole function is a handful of clamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .redflags import RedFlagResult

OVERRIDE_DELTA = {
    "no": 0,
    "one_level_higher": 1,
    "two_levels_higher": 2,
}

ESI_CODES = (1, 2, 3, 4, 5)


@dataclass
class ESIResolution:
    """Final ESI decision plus the provenance of every step."""

    code: int
    source: str            # "model" | "red_flag_floor" | "override" | "floor+override"
    model_code: Optional[int] = None
    floor: int = 5
    override_applied: int = 0
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict:
        return {
            "esi_code": self.code,
            "esi_source": self.source,
            "esi_model_code": self.model_code,
            "esi_floor": self.floor,
            "esi_override_levels": self.override_applied,
            "esi_notes": list(self.notes),
        }


def clamp(code: int) -> int:
    return max(1, min(5, int(code)))


def resolve(
    model_score: Optional[float] = None,
    override_label: Optional[str] = None,
    redflags: Optional[RedFlagResult] = None,
    esi_code_from_score=None,
) -> ESIResolution:
    """Combine model ESI, model override request and the red-flag floor.

    Parameters
    ----------
    model_score : float or None
        Raw 0-based ``esi`` score from the model.
    override_label : str or None
        One of ``OVERRIDE_DELTA``; unknown values are ignored (treated ``no``).
    redflags : RedFlagResult or None
        Deterministic screen result; supplies the floor.
    esi_code_from_score : callable, optional
        Injected mapper (keeps this module import-light and testable).

    Returns
    -------
    ESIResolution
    """
    if esi_code_from_score is None:
        from .questions import esi_code_from_score as _m
        esi_code_from_score = _m

    notes = []
    model_code = esi_code_from_score(model_score) if model_score is not None else None
    if model_code is not None:
        notes.append("model proposed ESI %d from raw score %.3f"
                     % (model_code, float(model_score)))

    floor = 5
    if redflags is not None:
        from .redflags import esi_floor
        floor = esi_floor(redflags)
        if floor < 5:
            notes.append("red-flag screen severity '%s' sets floor ESI %d"
                         % (redflags.severity, floor))

    code = model_code if model_code is not None else floor
    if model_code is None:
        notes.append("no model ESI available; using red-flag floor")

    delta = OVERRIDE_DELTA.get(override_label or "no", 0)
    if delta:
        raised = clamp(code) - delta
        if raised < code:
            notes.append("override '%s' raises ESI %d -> %d"
                         % (override_label, code, raised))
        else:
            notes.append("override '%s' ignored: already at ESI 1"
                         % override_label)
        code = clamp(raised)

    if code > floor:
        notes.append("model/override value ESI %d less urgent than floor ESI %d; "
                     "rising to the floor" % (code, floor))
        code = floor

    # provenance
    if delta and floor < 5 and model_code is not None:
        source = "floor+override"
    elif delta:
        source = "override"
    elif floor < 5 and model_code is not None and floor < model_code:
        source = "red_flag_floor"
    elif floor < 5 and model_code is None:
        source = "red_flag_floor"
    else:
        source = "model"

    return ESIResolution(code=clamp(code), source=source, model_code=model_code,
                         floor=floor, override_applied=delta, notes=notes)


def worse_of(a: int, b: int) -> int:
    """Most urgent (lowest) of two ESI codes."""
    return min(clamp(a), clamp(b))
