#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`doctor` must diagnose the *actual* failure, not the most common one.

A real run reported "the model writes a thinking preamble" for every failure,
including the case where no backend was running at all -- which sent the
operator to swap models when the fix was `ollama serve`. These tests pin the
classifier to the transport faults that misled it, and pin the remedy so a
transport fault never recommends a different model.
"""

import socket
import urllib.error

import pytest

from src.triage import engine
from src.triage.engine import (FAILURE_CONTRACT, FAILURE_DEGENERATE,
                               FAILURE_REJECTED, FAILURE_UNREACHABLE,
                               classify_failure, failure_remedy)


# --- the three faults that were previously all reported as "thinking" -------

def test_connection_refused_is_unreachable_not_a_model_problem():
    """`ollama serve` not running: urllib wraps this in URLError."""
    exc = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    assert classify_failure(exc) == FAILURE_UNREACHABLE


def test_bare_connection_refused_is_unreachable():
    assert classify_failure(ConnectionRefusedError(
        61, "Connection refused")) == FAILURE_UNREACHABLE


def test_timeout_is_unreachable():
    assert classify_failure(TimeoutError("timed out")) == FAILURE_UNREACHABLE
    assert classify_failure(socket.timeout()) == FAILURE_UNREACHABLE


def test_dns_failure_is_unreachable():
    assert classify_failure(socket.gaierror(-2, "Name or service not known")) \
        == FAILURE_UNREACHABLE


def test_http_error_is_rejected_and_wins_over_oserror():
    """An HTTPError is also an OSError; it must not be misread as unreachable."""
    exc = urllib.error.HTTPError(
        "http://x/v1/chat/completions", 404, "Not Found", {}, None)  # type: ignore[arg-type]
    assert isinstance(exc, OSError)
    assert classify_failure(exc) == FAILURE_REJECTED


# --- genuine model-quality faults keep their own class ----------------------

def test_no_label_mass_is_degenerate():
    """This is the real 'model thought instead of answering' case."""
    exc = RuntimeError(
        "logprobs window identified no label for 'Choose the acuity'")
    assert classify_failure(exc) == FAILURE_DEGENERATE


def test_winner_outside_window_is_degenerate():
    exc = RuntimeError(
        "only 0.041 of the probability mass fell inside the 20-token window; "
        "the winner is not determinable at this floor")
    assert classify_failure(exc) == FAILURE_DEGENERATE


def test_unexpected_response_shape_is_contract():
    exc = RuntimeError("model answered 'Maybe', not one of the 3 labels")
    assert classify_failure(exc) == FAILURE_CONTRACT
    assert classify_failure(KeyError("choices")) == FAILURE_CONTRACT


# --- the remedy must match the diagnosis -----------------------------------

def test_unreachable_remedy_never_blames_the_model():
    msg = failure_remedy(FAILURE_UNREACHABLE, backend="ollama",
                         base_url="http://127.0.0.1:11434/v1")
    assert "ollama serve" in msg
    assert "http://127.0.0.1:11434/v1" in msg
    assert "not a model quality problem" in msg
    assert "granite" not in msg.split("then")[0]  # no swap as the first move


def test_lmstudio_unreachable_remedy_names_lm_studio():
    msg = failure_remedy(FAILURE_UNREACHABLE, backend="lmstudio")
    assert "LM Studio" in msg
    assert "not a model quality problem" in msg


def test_only_degenerate_advises_a_different_model():
    """Model-swap advice is confined to the one class where it applies."""
    degenerate = failure_remedy(FAILURE_DEGENERATE)
    assert "pick a model that answers first" in degenerate
    for other in (FAILURE_UNREACHABLE, FAILURE_REJECTED, FAILURE_CONTRACT):
        assert "pick a model" not in failure_remedy(other)


def test_rejected_remedy_points_at_the_request_not_the_network():
    msg = failure_remedy(FAILURE_REJECTED, base_url="http://h/v1")
    assert "rejected the request" in msg
    assert "--api-key" in msg


# --- every class has a remedy ----------------------------------------------

@pytest.mark.parametrize("kind", [FAILURE_UNREACHABLE, FAILURE_REJECTED,
                                  FAILURE_DEGENERATE, FAILURE_CONTRACT,
                                  engine.FAILURE_QUESTION])
def test_every_failure_class_has_a_nonempty_remedy(kind):
    msg = failure_remedy(kind)
    assert isinstance(msg, str) and msg.strip()
