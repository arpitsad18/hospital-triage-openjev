#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line interface: ``python -m src.triage.cli <command>``.

Commands
--------
doctor     check backend reachability and whether the model can answer at all
decide     one-off advisory triage for a narrative on the command line
validate   run the labelled validation corpus and print the honest scorecard
calibrate  same corpus, full probability vectors + low-confidence safety sweep
schema     print the question contract as JSON

Why ``doctor`` exists
---------------------
A model that emits a thinking preamble never places an answer token in the
top-logprobs window, so OpenJev refuses to answer. That failure looks like a
broken install but is really a model-choice problem. ``doctor`` separates the
two before anyone wires the engine into a workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# allow `python -m src.triage.cli` and `python src/triage/cli.py` alike
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from src.triage import questions as Q
    from src.triage.engine import TriageEngine
    from src.triage.engine import LOW_CONFIDENCE_THRESHOLD
    from src.triage import redflags as RF
else:
    from . import questions as Q
    from .engine import TriageEngine
    from .engine import LOW_CONFIDENCE_THRESHOLD
    from . import redflags as RF

DEFAULT_BACKEND = os.environ.get("TRIAGE_BACKEND", "ollama")
DEFAULT_MODEL = os.environ.get("TRIAGE_MODEL", "granite4.1:8b")


def _effective_low_conf() -> float:
    """The confidence threshold actually in force.

    The engine reads ``TRIAGE_LOW_CONF`` and falls back to its own constant.
    Reporting anything else here (a second hard-coded 0.35) risks the schema
    advertising a threshold the engine does not use.
    """
    return float(os.environ.get("TRIAGE_LOW_CONF", LOW_CONFIDENCE_THRESHOLD))


def _make_engine(args) -> TriageEngine:
    ids = None
    if getattr(args, "questions", None):
        ids = [q.strip() for q in args.questions.split(",") if q.strip()]
    return TriageEngine(
        backend=getattr(args, "backend", DEFAULT_BACKEND),
        model=getattr(args, "model", DEFAULT_MODEL),
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        question_ids=ids,
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    print("triage doctor")
    print("  backend : %s" % args.backend)
    print("  model   : %s" % args.model)
    print()

    engine = _make_engine(args)
    state = ("55M, sudden severe central chest pain for 30 minutes, "
             "sweating, feels like pressure, pain radiating to the left arm, "
             "BP 92/60, HR 112.")
    probe = [("acuity", "acute"),
             ("needs_consult", "Yes"),
             ("esi", 0)]

    print("  probing the model with a textbook ACS presentation...")
    failures = 0
    for qid, expect_desc in probe:
        try:
            r = engine.answer_one(qid, state)
            probs = ", ".join("%.3f" % p for p in r["probs"][:6])
            print("    %-14s -> %-12s conf=%.3f  p=[%s]"
                  % (qid, str(r["label"]), r["confidence"], probs))
        except Exception as exc:
            failures += 1
            print("    %-14s -> FAILED: %s: %s"
                  % (qid, type(exc).__name__, str(exc)[:120]))

    print()
    if failures:
        print("  RESULT: %d/%d probes failed." % (failures, len(probe)))
        print("  Most common cause: the model writes a thinking preamble, so no")
        print("  answer token lands in the top-logprobs window. Pick a model that")
        print("  answers first, e.g. `ollama pull granite4.1:8b`.")
        return 1
    print("  RESULT: model answers with usable probabilities. OK")
    print("  Run `validate` next to see agreement on labelled cases.")
    return 0


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

def cmd_decide(args) -> int:
    engine = _make_engine(args)
    vitals = json.loads(args.vitals) if args.vitals else None
    try:
        res = engine.triage(args.narrative, vitals=vitals, fail_open=args.fail_open)
    except Exception as exc:
        print("triage failed (fail-closed): %s" % exc, file=sys.stderr)
        print("The engine will not emit a decision it cannot back with "
              "probabilities. Use --fail-open for evaluation runs only.",
              file=sys.stderr)
        return 2

    data = res.to_dict()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    print("=" * 68)
    print("TRIAGE DECISION  (advisory — clinician review required)")
    print("=" * 68)
    print("ESI %d  (source: %s)" % (res.esi_code, res.esi_source))
    if res.red_flags:
        print("RED FLAGS: %s  [severity %s]"
              % (", ".join(res.red_flag_hits[:6]), res.red_flag_severity))
        print("  %s" % res.red_flag_explanation[:200])
    else:
        print("RED FLAGS: none matched by the deterministic screen")
    if res.requires_human_review:
        print("HUMAN REVIEW REQUIRED")
        for r in res.review_reasons[:5]:
            print("  - %s" % r)
    print("-" * 68)
    for qid in engine.question_ids:
        label = data["answers"].get(qid)
        probs = data["probabilities"].get(qid) or []
        bar = ""
        if probs:
            top = max(probs)
            bar = "%s p=%.3f" % ("#" * int(round(top * 20)).ljust(20, "."), top)
        print("  %-24s %-14s %s" % (qid, str(label), bar))
    print("-" * 68)
    print("latency: %.0f ms total, %.0f ms mean" %
          (res.usage.get("latency_ms_total") or 0,
           res.usage.get("latency_ms_mean") or 0))
    if res.degraded_checks:
        print("degraded checks:")
        for d in res.degraded_checks:
            print("  ! %s" % d)
    return 0


# ---------------------------------------------------------------------------
# validate / calibrate
# ---------------------------------------------------------------------------

def _load_runner():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from validation import run as runner
    return runner


def cmd_validate(args) -> int:
    runner = _load_runner()
    report = runner.evaluate(
        engine_factory=lambda: _make_engine(args),
        cases_path=args.cases,
        fail_open=True,
    )
    runner.print_scorecard(report)
    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


def cmd_calibrate(args) -> int:
    runner = _load_runner()
    report = runner.evaluate(
        engine_factory=lambda: _make_engine(args),
        cases_path=args.cases,
        fail_open=True,
    )
    runner.print_calibration(report)
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


def cmd_schema(args) -> int:
    print(json.dumps({
        "questions": Q.schema(),
        "low_confidence_threshold": _effective_low_conf(),
        "esi_note": "esi_code 1 = most urgent; model values can only be raised "
                    "toward urgency, never lowered below the red-flag floor",
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage",
        description="Offline, auditable ED triage on OpenJev decision models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--backend", default=DEFAULT_BACKEND,
                   help="openjev backend: ollama|lmstudio|llamacpp|vllm|openai|"
                        "openrouter|local (default %s)" % DEFAULT_BACKEND)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model name served by the backend (default %s)" % DEFAULT_MODEL)
    p.add_argument("--base-url", default=None, help="custom OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None, help="bearer token if the endpoint needs one")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    sub = p.add_subparsers(dest="command")

    def _json(subp):
        # Accept --json after the subcommand as well as before it; argparse
        # would otherwise reject the trailing form with a usage error.
        subp.add_argument("--json", action="store_true",
                          default=argparse.SUPPRESS,
                          help="machine-readable output")
        return subp

    d = _json(sub.add_parser("doctor", help="check the backend and model are usable"))
    d.add_argument("--questions", default=None, help="comma-separated question ids")
    d.set_defaults(func=cmd_doctor)

    dec = _json(sub.add_parser("decide", help="advisory triage for one narrative"))
    dec.add_argument("narrative", help="de-identified clinical state")
    dec.add_argument("--vitals", default=None,
                     help='JSON vitals, e.g. \'{"sbp":88,"hr":118,"spo2":94}\'')
    dec.add_argument("--questions", default=None, help="comma-separated question ids")
    dec.add_argument("--fail-open", action="store_true",
                     help="record unanswerable questions instead of aborting")
    dec.set_defaults(func=cmd_decide)

    v = _json(sub.add_parser("validate", help="scorecard vs the labelled corpus"))
    v.add_argument("--cases", default=None, help="path to cases.json")
    v.add_argument("--questions", default=None, help="comma-separated question ids")
    v.set_defaults(func=cmd_validate)

    c = _json(sub.add_parser("calibrate", help="full probability vectors + safety sweep"))
    c.add_argument("--cases", default=None, help="path to cases.json")
    c.add_argument("--questions", default=None, help="comma-separated question ids")
    c.set_defaults(func=cmd_calibrate)

    s = _json(sub.add_parser("schema", help="print the question contract as JSON"))
    s.set_defaults(func=cmd_schema)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.json = bool(getattr(args, "json", False))
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
