#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The CLI contract: `--json` must parse both before AND after the subcommand.

A real run failed with ``triage: error: unrecognized arguments: --json`` when
the flag trailed the subcommand (``validate --json``). That form is the one
everyone types, so it is pinned here — one test per subcommand for both
positions, plus the defaults that make ``main`` work.
"""

import argparse

import pytest

from src.triage import cli


SUBCOMMANDS = ["doctor", "validate", "calibrate", "schema"]

# `decide` needs a positional narrative; handled separately below.
POSITIONAL = {"decide": ["fake narrative"]}


@pytest.mark.parametrize("cmd", SUBCOMMANDS + ["decide"])
def test_json_flag_accepted_after_the_subcommand(cmd):
    """`cli <cmd> --json` must not raise SystemExit."""
    argv = [cmd] + POSITIONAL.get(cmd, []) + ["--json"]
    args = cli.build_parser().parse_args(argv)
    assert args.json is True
    assert args.command == cmd


@pytest.mark.parametrize("cmd", SUBCOMMANDS + ["decide"])
def test_json_flag_accepted_before_the_subcommand(cmd):
    """`cli --json <cmd>` (the other order) must also work."""
    argv = ["--json", cmd] + POSITIONAL.get(cmd, [])
    args = cli.build_parser().parse_args(argv)
    assert args.json is True
    assert args.command == cmd


@pytest.mark.parametrize("cmd", SUBCOMMANDS + ["decide"])
def test_json_defaults_to_false_when_omitted(cmd):
    argv = [cmd] + POSITIONAL.get(cmd, [])
    args = cli.build_parser().parse_args(argv)
    assert getattr(args, "json", False) is False


def test_main_without_a_command_prints_help_and_returns_nonzero(capsys):
    rc = cli.main([])
    assert rc == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_backend_and_model_flags_are_global_not_per_subcommand():
    """The backend selection is set once, before the subcommand."""
    args = cli.build_parser().parse_args(
        ["--backend", "ollama", "--model", "granite4.1:8b",
         "--json", "validate"])
    assert args.backend == "ollama"
    assert args.model == "granite4.1:8b"
    assert args.json is True


def test_schema_runs_offline_and_emits_json(capsys):
    """`schema` needs no model, so it is the one command safe to run here."""
    rc = cli.main(["--json", "schema"])
    out = capsys.readouterr().out
    assert rc == 0
    import json
    payload = json.loads(out)
    assert "questions" in payload
    assert "acuity" in payload["questions"]
