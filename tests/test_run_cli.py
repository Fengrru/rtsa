"""Tests for experiments.run (C10): parser and versioned-run helpers."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rtsa.experiments.run as run_module
from rtsa.experiments.run import build_parser, git_commit


def test_parser_exposes_all_commands():
    parser = build_parser()
    subparsers = parser._subparsers._group_actions[0].choices
    assert {"extract", "analyze", "prune", "calibrate", "annotate", "all"} <= set(subparsers)


def test_make_run_dir_creates_timestamped_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "RUNS_ROOT", tmp_path)
    run_dir = run_module.make_run_dir("extract")
    assert run_dir.exists()
    assert run_dir.name.startswith("extract_")


def test_write_manifest_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "RUNS_ROOT", tmp_path)
    run_dir = run_module.make_run_dir("analyze")
    run_module.write_manifest(run_dir, "analyze", {"dataset": "gsm8k"})
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "analyze"
    assert manifest["args"]["dataset"] == "gsm8k"
    assert "python_version" in manifest
    assert "git_commit" in manifest


def test_git_commit_never_raises():
    assert isinstance(git_commit(), str)
