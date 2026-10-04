#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stdlib HTTP API around the triage engine.

    GET  /health    liveness + engine/model identity
    GET  /schema    the exact question contract (labels, types, threshold)
    POST /triage    {"narrative": str, "vitals": {...}} -> full decision

Deliberately stdlib-only (``http.server``) so the service has no dependency
beyond OpenJev itself and can run on a locked-down hospital edge box. It is
single-process, threaded, and stateless per request.

Health and schema are cheap and safe; ``/triage`` is the only endpoint that
touches a model.

Status codes
------------
200  decision produced
400  malformed JSON or missing/empty ``narrative``
422  the model could not answer (fail-closed) — OpenJev refused to fabricate
     probabilities, usually a thinking-preamble model. Response body carries
     ``failed_questions`` and ``degraded_checks``.
503  engine could not be constructed (backend unreachable at startup)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from src.triage import questions as Q
    from src.triage.engine import TriageEngine, LOW_CONFIDENCE_THRESHOLD
else:
    from . import questions as Q
    from .engine import TriageEngine, LOW_CONFIDENCE_THRESHOLD

_ENGINE: Optional[TriageEngine] = None
_ARGS: Dict[str, Any] = {}
STARTED_AT = time.time()


def get_engine() -> TriageEngine:
    global _ENGINE
    if _ENGINE is None:
        raise RuntimeError("engine not initialised")
    return _ENGINE


class Handler(BaseHTTPRequestHandler):
    server_version = "triage-openjev/0.1"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # keep logs terse
        sys.stderr.write("[triage-api] %s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------
    # routes
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            engine = get_engine()
            self._send(200, {
                "status": "ok",
                "uptime_s": round(time.time() - STARTED_AT, 1),
                "backend": engine.backend,
                "model": engine.model,
                "questions": engine.question_ids,
                "low_confidence_threshold": engine.low_confidence,
            })
        elif self.path == "/schema":
            self._send(200, {
                "questions": Q.schema(),
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "endpoint": "POST /triage",
                "request": {"narrative": "string (required)",
                            "vitals": {"sbp": "number", "hr": "number",
                                       "rr": "number", "spo2": "number",
                                       "temp": "number", "gcs": "number",
                                       "age": "number"}},
                "esi_note": "esi_code 1 = most urgent; the red-flag floor can "
                            "only raise urgency",
            })
        else:
            self._send(404, {"error": "not found", "routes": ["/health", "/schema",
                                                              "POST /triage"]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/triage":
            self._send(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as exc:
            self._send(400, {"error": "invalid JSON body: %s" % exc})
            return

        narrative = (payload.get("narrative") or "").strip()
        if not narrative:
            self._send(400, {"error": "'narrative' is required and must be non-empty"})
            return
        vitals = payload.get("vitals")
        if vitals is not None and not isinstance(vitals, dict):
            self._send(400, {"error": "'vitals' must be an object when present"})
            return

        try:
            engine = get_engine()
        except RuntimeError as exc:
            self._send(503, {"error": str(exc)})
            return

        try:
            res = engine.triage(narrative, vitals=vitals, fail_open=False)
        except Exception as exc:
            message = str(exc)
            failed = []
            for qid in Q.ALL_QUESTIONS:
                if "'%s'" % qid in message:
                    failed.append(qid)
            self._send(422, {
                "error": "model could not produce honest probabilities",
                "detail": message[:400],
                "failed_questions": failed,
                "action": "check GET /health, then run `python -m src.triage.cli "
                          "doctor`. A thinking-preamble model cannot be used here.",
            })
            return

        body = res.to_dict()
        body["disclaimer"] = ("Advisory decision aid for research use. Not a "
                              "medical device. Clinician review required.")
        self._send(200, body)


def main(argv: Optional[list] = None) -> int:
    global _ENGINE
    p = argparse.ArgumentParser(description="Triage HTTP API over the OpenJev engine")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1; keep it loopback unless "
                        "the network is trusted — this endpoint handles PHI)")
    p.add_argument("--port", type=int, default=8773)
    p.add_argument("--backend", default=os.environ.get("TRIAGE_BACKEND", "ollama"))
    p.add_argument("--model", default=os.environ.get("TRIAGE_MODEL", "granite4.1:8b"))
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--questions", default=None,
                   help="comma-separated subset of question ids")
    p.add_argument("--preload", action="store_true",
                   help="construct the engine now instead of on first request")
    args = p.parse_args(argv)

    ids = [q.strip() for q in args.questions.split(",")] if args.questions else None
    _ARGS.update(vars(args))

    if args.preload:
        _ENGINE = TriageEngine(backend=args.backend, model=args.model,
                               base_url=args.base_url, api_key=args.api_key,
                               question_ids=ids)
    else:
        _ENGINE = TriageEngine(backend=args.backend, model=args.model,
                               base_url=args.base_url, api_key=args.api_key,
                               question_ids=ids)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("triage-api listening on http://%s:%d  (backend=%s model=%s)"
          % (args.host, args.port, args.backend, args.model))
    print("  POST /triage   GET /health   GET /schema")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
