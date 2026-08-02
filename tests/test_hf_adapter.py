"""Tests for utils.hf_adapter (C9) — offline parts (no datasets dependency)."""

import json

from rtsa.utils.hf_adapter import (
    COT_PARSERS, _parse_gsm8k, _parse_math, _parse_plain, _sniff_parser,
    save_hf_traces,
)


def test_parse_gsm8k_splits_answer():
    cot, ans = _parse_gsm8k("Step one. Step two. #### 42")
    assert "Step one." in cot
    assert ans == "42"


def test_parse_gsm8k_no_answer_marker():
    cot, ans = _parse_gsm8k("Just reasoning.")
    assert cot == "Just reasoning."
    assert ans == ""


def test_parse_math_boxed():
    sol = "Compute x. The answer is \\boxed{3}."
    cot, ans = _parse_math(sol)
    assert ans == "3"
    assert "boxed" not in cot


def test_parse_math_answer_is_pattern():
    sol = "Compute x. The answer is 3."
    cot, ans = _parse_math(sol)
    assert ans == "3"   # trailing punctuation stripped by the parser
    assert "Compute x." in cot


def test_parse_plain():
    cot, ans = _parse_plain("Just reasoning.")
    assert cot == "Just reasoning."
    assert ans == ""


def test_sniff_parser():
    assert _sniff_parser("a #### 5") == "gsm8k"
    assert _sniff_parser("a \\boxed{5}") == "math"
    assert _sniff_parser("plain text") == "plain"


def test_parser_registry_kinds():
    assert set(COT_PARSERS) == {"gsm8k", "math", "plain"}


def test_save_hf_traces_roundtrip(tmp_path):
    out = tmp_path / "traces.jsonl"
    records = [{"cot_text": "x", "question_id": "a"}]
    path = save_hf_traces(records, str(out))
    with open(path, encoding="utf-8") as f:
        assert json.loads(f.readline())["question_id"] == "a"
