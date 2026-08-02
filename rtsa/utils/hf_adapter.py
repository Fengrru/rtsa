"""HuggingFace datasets adapter (C9).

A generic bridge between the HuggingFace ``datasets`` library and RTSA's
trace-record format, so experiments can consume *any* CoT-capable HF dataset
without writing a bespoke loader per dataset:

    from rtsa.utils.hf_adapter import load_hf_traces

    traces = load_hf_traces(
        "openai/gsm8k", split="test", cot_field="answer", cot_parser="gsm8k",
    )
    # -> [{"cot_text", "question_id", "model", "domain", "answer",
    #      "answer_correct", "question", "cot_length_tokens", ...}, ...]

Built-in parsers cover the two formats already shipped in this repo
(GSM8K ``####`` split and MATH ``\\boxed{}`` split); ``cot_parser="auto"``
sniffs the data, and a custom ``mapper`` overrides everything. The reverse
direction (RTSA records -> HF Dataset) is provided by :func:`to_hf_dataset`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoT parsers: raw answer/solution text -> (cot_text, final_answer)
# ---------------------------------------------------------------------------

def _parse_gsm8k(answer: str) -> tuple:
    """GSM8K: step-by-step reasoning followed by ``#### <number>``."""
    parts = re.split(r"\s*####\s*", answer)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return answer.strip(), ""


def _parse_math(solution: str) -> tuple:
    """MATH: reasoning followed by ``\\boxed{...}`` (or 'The answer is ...')."""
    boxed = re.search(r"\\boxed\{(.*?)\}", solution, re.DOTALL)
    if boxed:
        cot = re.sub(r"\\(?:boxed|text)\{.*?\}", "", solution[: boxed.start()])
        return re.sub(r"\s+", " ", cot).strip(), boxed.group(1).strip()
    ans = re.search(
        r"(?:the\s+)?answer\s+is\s+[\":]?\s*(.+?)[.\s]*$",
        solution, re.IGNORECASE,
    )
    if ans:
        return solution[: ans.start()].strip(), ans.group(1).strip()
    return solution.strip(), ""


def _parse_plain(text: str) -> tuple:
    """No known structure: the whole field is the CoT, no final answer."""
    return text.strip(), ""


COT_PARSERS: Dict[str, Callable[[str], tuple]] = {
    "gsm8k": _parse_gsm8k,
    "math": _parse_math,
    "plain": _parse_plain,
}


def _sniff_parser(text: str) -> str:
    """Guess the parser from the text itself."""
    if "####" in text:
        return "gsm8k"
    if "\\boxed" in text:
        return "math"
    return "plain"


# ---------------------------------------------------------------------------
# Loading (HF -> RTSA records)
# ---------------------------------------------------------------------------

def _load_hf_dataset(dataset_name: str, config: Optional[str], split: str):
    """Load a HF dataset, retrying with a config when the plain name fails."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install: pip install datasets"
        )
    logger.info("Loading %s (config=%s, split=%s)...", dataset_name, config, split)
    if config is not None:
        return load_dataset(dataset_name, config, split=split)
    try:
        return load_dataset(dataset_name, split=split)
    except Exception:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_name)
        if configs:
            cfg = next((c for c in configs if c != "default"), configs[0])
            logger.warning("Plain load failed; retrying with config '%s'", cfg)
            return load_dataset(dataset_name, cfg, split=split)
        raise


def _default_record(
    item: Dict[str, Any],
    idx: int,
    split: str,
    id_prefix: str,
    cot_field: str,
    question_field: str,
    answer_field: Optional[str],
    parser_name: str,
) -> Optional[Dict[str, Any]]:
    """Convert one HF item to an RTSA trace record (no custom mapper)."""
    raw = str(item.get(cot_field, ""))
    if not raw.strip():
        return None
    parser = (
        COT_PARSERS[parser_name]
        if parser_name in COT_PARSERS
        else COT_PARSERS[_sniff_parser(raw)]
    )
    cot_text, final_answer = parser(raw)
    if len(cot_text.split()) < 5:
        return None
    answer = final_answer
    if not answer and answer_field and item.get(answer_field):
        answer = str(item[answer_field])

    subject = item.get("subject") or item.get("type") or "general"
    return {
        "cot_text": cot_text,
        "question_id": f"{id_prefix}_{split}_{int(idx):05d}",
        "model": "human",
        "domain": str(subject).lower().replace(" ", "_"),
        "answer": answer,
        "answer_correct": bool(item.get("answer_correct", True)),
        "question": str(item.get(question_field, "")),
        "cot_length_tokens": len(cot_text.split()),
        "dataset_source": id_prefix,
    }


def load_hf_traces(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    max_samples: int = 100,
    seed: int = 42,
    cot_field: str = "answer",
    question_field: str = "question",
    answer_field: Optional[str] = None,
    cot_parser: str = "auto",
    mapper: Optional[Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]] = None,
    id_prefix: str = "hf",
) -> List[Dict[str, Any]]:
    """Load traces from an arbitrary HuggingFace dataset.

    Args:
        dataset_name: HF dataset id (e.g. ``"openai/gsm8k"``).
        config: dataset config name; None tries the plain dataset first.
        split: dataset split (``"train"`` / ``"test"`` / ...).
        max_samples: maximum number of traces to return (deterministically
            subsampled with ``seed``).
        cot_field: field holding the raw reasoning text.
        question_field: field holding the question text.
        answer_field: optional field for the final answer when the CoT
            parser cannot extract one.
        cot_parser: one of ``"gsm8k"``, ``"math"``, ``"plain"``, or
            ``"auto"`` (sniff per record; default).
        mapper: optional callable ``(item, idx) -> record | None`` that
            fully overrides the default conversion.
        id_prefix: prefix for generated ``question_id`` values.

    Returns:
        List of RTSA trace records (same schema as the JSONL corpus).
    """
    dataset = _load_hf_dataset(dataset_name, config, split)

    rng = np.random.RandomState(seed)
    n_available = min(len(dataset), max_samples * 2)
    indices = sorted(rng.choice(len(dataset), size=n_available, replace=False))
    indices = indices[:max_samples]

    traces: List[Dict[str, Any]] = []
    for idx in indices:
        item = dict(dataset[int(idx)])
        if mapper is not None:
            record = mapper(item, int(idx))
        else:
            record = _default_record(
                item, int(idx), split, id_prefix, cot_field,
                question_field, answer_field, cot_parser,
            )
        if record is not None:
            traces.append(record)
    logger.info("Loaded %d traces from %s (%s)", len(traces), dataset_name, split)
    return traces


def iter_hf_traces(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    max_samples: int = 100,
    cot_field: str = "answer",
    question_field: str = "question",
    answer_field: Optional[str] = None,
    cot_parser: str = "auto",
    mapper: Optional[Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]] = None,
    id_prefix: str = "hf",
) -> Iterator[Dict[str, Any]]:
    """Streaming variant for large datasets (no random subsampling)."""
    dataset = _load_hf_dataset(dataset_name, config, split)
    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break
        record = (
            mapper(dict(item), idx)
            if mapper is not None
            else _default_record(
                dict(item), idx, split, id_prefix, cot_field,
                question_field, answer_field, cot_parser,
            )
        )
        if record is not None:
            yield record


# ---------------------------------------------------------------------------
# Export (RTSA records -> HF Dataset / JSONL)
# ---------------------------------------------------------------------------

def to_hf_dataset(records: List[Dict[str, Any]]):
    """Convert RTSA trace records into a HuggingFace ``Dataset``."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install: pip install datasets"
        )
    return Dataset.from_list(records)


def save_hf_traces(
    records: List[Dict[str, Any]], output_path: str
) -> str:
    """Persist RTSA trace records as JSONL (same corpus format)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d traces to %s", len(records), path)
    return str(path)
