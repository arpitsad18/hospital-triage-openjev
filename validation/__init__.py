#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation package: honest scoring of the triage engine against labelled cases.

There is no hidden test set here and no tuned threshold that makes the numbers
look good. The corpus in ``cases.json`` is synthetic and every metric printed
is computed from the engine's own outputs. The scorecard is allowed to fail.

The headline metric is **under-triage**, not accuracy. A triage system that
labels a septic patient as "less urgent" has done the one thing it exists to
prevent, and no amount of correct calls elsewhere compensates for it. So
``pass`` requires zero under-triage, zero missed mandatory red flags and zero
acuity under-calls -- accuracy above that is a soft signal, not a gate.
"""

from . import run

__all__ = ["run"]
