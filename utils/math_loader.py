"""MATH dataset loader for RTSA.

Loads MATH from HuggingFace (lighteval/MATH), parses the step-by-step
solutions, and outputs CoT traces in RTSA-compatible JSONL format.

MATH dataset structure (lighteval/MATH):
    - problem: The math problem text
    - solution: Step-by-step solution with \boxed{...} wrapping the answer
    - answer: The final answer
    - level: Difficulty level (1-5)
    - type: Subject category (Algebra, Geometry, etc.)

Usage:
    python -m utils.math_loader
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Dataset IDs to try in order of availability
MATH_DATASETS = [
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
]


def _try_load_dataset(split: str):
    """Try to load MATH dataset from available sources.

    Tries multiple dataset sources and configs to maximize coverage.
    Returns a combined dataset with all available subjects.
    """
    from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

    for ds_name in MATH_DATASETS:
        try:
            configs = get_dataset_config_names(ds_name)
            logger.info(f"Available configs for {ds_name}: {configs}")

            # Load each subject config and combine for balanced coverage
            subject_configs = [c for c in configs if c not in ("default",)]
            if subject_configs:
                logger.info(f"Loading {len(subject_configs)} subject configs individually...")
                all_splits = []
                for cfg in subject_configs:
                    try:
                        ds = load_dataset(ds_name, cfg, split=split)
                        if len(ds) > 0:
                            logger.info(f"  {cfg}: {len(ds)} samples")
                            all_splits.append(ds)
                    except Exception as e:
                        logger.warning(f"  {cfg}: failed ({e})")
                        continue

                if all_splits:
                    combined = concatenate_datasets(all_splits)
                    logger.info(f"Combined: {len(combined)} samples across {len(all_splits)} subjects")
                    return combined, ds_name

            # Fallback: try default config
            if "default" in configs:
                logger.info(f"Falling back to {ds_name} with 'default' config...")
                dataset = load_dataset(ds_name, "default", split=split)
                if len(dataset) > 0:
                    logger.info(f"Loaded from {ds_name} (default): {len(dataset)} samples")
                    return dataset, ds_name

            # Final fallback: no config
            logger.info(f"Trying {ds_name} without config...")
            dataset = load_dataset(ds_name, split=split)
            if len(dataset) > 0:
                logger.info(f"Loaded from {ds_name}: {len(dataset)} samples")
                return dataset, ds_name

        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            continue

    raise ImportError(
        f"Could not load MATH dataset from any source. "
        f"Available sources: {MATH_DATASETS}"
    )


def _extract_cot_from_solution(solution: str) -> tuple[str, str]:
    """Extract CoT text and answer from a MATH solution string.

    MATH solutions have the format:
        <step-by-step reasoning>\n\nThe answer is \boxed{<answer>}.

    Returns (cot_text, final_answer).
    """
    # Remove \boxed{...} and extract answer
    boxed_match = re.search(r"\\boxed\{(.*?)\}", solution, re.DOTALL)
    if boxed_match:
        final_answer = boxed_match.group(1).strip()
        # Everything before \boxed is the CoT
        cot_text = solution[:boxed_match.start()].strip()
    else:
        # Try "The answer is ..." pattern
        ans_match = re.search(
            r"(?:the\s+)?answer\s+is\s+[\":]?\s*(.+?)[.\s]*$",
            solution,
            re.IGNORECASE,
        )
        if ans_match:
            final_answer = ans_match.group(1).strip()
            cot_text = solution[:ans_match.start()].strip()
        else:
            # No clear answer boundary: use entire solution as CoT
            cot_text = solution.strip()
            final_answer = ""

    # Clean up LaTeX artifacts in CoT
    cot_text = re.sub(r"\\(?:boxed|text)\{.*?\}", "", cot_text)
    cot_text = re.sub(r"\s+", " ", cot_text).strip()

    return cot_text, final_answer


def load_math_cot(
    split: str = "train",
    max_samples: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load MATH split and extract CoT reasoning traces.

    Each MATH entry has problem, solution, answer, level, type fields.

    Args:
        split: Dataset split ('train' or 'test').
        max_samples: Maximum number of samples to return.
        seed: Random seed for deterministic sampling.

    Returns:
        List of dicts with cot_text, question_id, domain, answer_correct, metadata.
    """
    dataset, source = _try_load_dataset(split)

    # Deterministic sampling
    import numpy as np
    rng = np.random.RandomState(seed)

    # Maximize coverage across subjects and levels
    n_available = min(len(dataset), max_samples * 3)
    indices = rng.choice(len(dataset), size=n_available, replace=False)
    indices = sorted(indices)[:max_samples]

    traces = []
    for idx in indices:
        item = dataset[int(idx)]
        problem: str = item.get("problem", "")
        solution: str = item.get("solution", "")
        answer: str = item.get("answer", "")
        level_raw: str = str(item.get("level", "0"))
        # Handle "Level 1", "Level 2", etc.
        level_match = re.search(r"(\d+)", level_raw)
        level: int = int(level_match.group(1)) if level_match else 0
        subject: str = item.get("type", "unknown")

        # Extract CoT from solution
        cot_text, extracted_answer = _extract_cot_from_solution(solution)

        # Skip if CoT is too short
        if len(cot_text.split()) < 5:
            continue

        # Use provided answer as fallback
        final_answer = extracted_answer or answer

        traces.append({
            "cot_text": cot_text,
            "question_id": f"math_{split}_{int(idx):05d}",
            "model": "human",
            "domain": f"math_{subject.lower().replace(' & ', '_').replace(' ', '_')}",
            "answer": final_answer,
            "answer_correct": True,  # MATH solutions are ground truth
            "level": level,
            "subject": subject,
            "cot_length_tokens": len(cot_text.split()),
            "question": problem,
            "dataset_source": source,
        })

    logger.info(f"Loaded {len(traces)} MATH CoT traces (sampled from {split})")
    return traces


def save_math_traces(
    traces: List[Dict[str, Any]],
    output_path: str = "data/raw_cots/math_100.jsonl",
) -> str:
    """Save MATH traces to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(traces)} traces to {path}")
    return str(path)


def load_saved_traces(
    filepath: str = "data/raw_cots/math_100.jsonl",
) -> List[Dict[str, Any]]:
    """Load previously saved MATH traces."""
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    logger.info(f"Loaded {len(traces)} traces from {filepath}")
    return traces


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load and save MATH CoT traces
    traces = load_math_cot(split="train", max_samples=100)
    save_math_traces(traces)

    # Print summary
    lengths = [t["cot_length_tokens"] for t in traces]
    subjects = {}
    for t in traces:
        subj = t.get("subject", "unknown")
        subjects[subj] = subjects.get(subj, 0) + 1

    print(f"\nMATH Traces Summary:")
    print(f"  Count: {len(traces)}")
    print(f"  Avg length: {sum(lengths) / len(lengths):.0f} tokens")
    print(f"  Min length: {min(lengths)} tokens")
    print(f"  Max length: {max(lengths)} tokens")
    print(f"  Subjects:")
    for subj, count in sorted(subjects.items()):
        print(f"    {subj}: {count}")
    print(f"\nSample trace:")
    print(f"  Question: {traces[0]['question'][:100]}...")
    print(f"  CoT length: {traces[0]['cot_length_tokens']} tokens")
    print(f"  Subject: {traces[0]['subject']}, Level: {traces[0]['level']}")
