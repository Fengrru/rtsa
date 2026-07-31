"""GSM8K dataset loader for RTSA.

Loads GSM8K from HuggingFace, parses the step-by-step reasoning,
and outputs CoT traces in RTSA-compatible JSONL format.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_gsm8k_cot(
    split: str = "train",
    max_samples: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load GSM8K train split and extract CoT reasoning traces.

    Each GSM8K entry has 'question' and 'answer' fields.
    The answer field contains step-by-step CoT followed by '#### <number>'.

    Args:
        split: Dataset split ('train' or 'test').
        max_samples: Maximum number of samples to return.
        seed: Random seed for deterministic sampling.

    Returns:
        List of dicts with cot_text, question_id, domain, answer_correct, metadata.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required. Install: pip install datasets"
        )

    logger.info(f"Loading GSM8K ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    # Deterministic sampling
    import numpy as np
    rng = np.random.RandomState(seed)

    n_available = min(len(dataset), max_samples * 2)
    indices = rng.choice(len(dataset), size=n_available, replace=False)
    indices = sorted(indices)[:max_samples]

    traces = []
    for idx in indices:
        item = dataset[int(idx)]
        question: str = item["question"]
        answer: str = item["answer"]

        # Parse CoT: everything before "####" is the reasoning
        # The answer format is: "Step 1 ... Step 2 ... #### 123"
        cot_match = re.split(r"\s*####\s*", answer)
        if len(cot_match) >= 2:
            cot_text = cot_match[0].strip()
            final_answer = cot_match[1].strip()
        else:
            cot_text = answer.strip()
            final_answer = ""

        # Skip if CoT is too short
        if len(cot_text.split()) < 5:
            continue

        # Extract answer correctness signal from final number
        answer_num = None
        try:
            answer_num = float(final_answer)
        except ValueError:
            # Try to find any number
            nums = re.findall(r"-?\d+\.?\d*", final_answer)
            if nums:
                answer_num = float(nums[-1])

        traces.append({
            "cot_text": cot_text,
            "question_id": f"gsm8k_{split}_{int(idx):05d}",
            "model": "human",
            "domain": "math",
            "answer": final_answer,
            "answer_correct": True,  # GSM8K solutions are ground truth
            "answer_num": answer_num,
            "cot_length_tokens": len(cot_text.split()),
            "question": question,
        })

    logger.info(f"Loaded {len(traces)} GSM8K CoT traces (sampled from {split})")
    return traces


def save_gsm8k_traces(
    traces: List[Dict[str, Any]],
    output_path: str = "data/raw_cots/gsm8k_50.jsonl",
) -> str:
    """Save GSM8K traces to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(traces)} traces to {path}")
    return str(path)


def load_saved_traces(
    filepath: str = "data/raw_cots/gsm8k_50.jsonl",
) -> List[Dict[str, Any]]:
    """Load previously saved GSM8K traces."""
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
    logging.basicConfig(level=logging.INFO)

    # Load and save GSM8K CoT traces
    traces = load_gsm8k_cot(split="train", max_samples=50)
    save_gsm8k_traces(traces)

    # Print summary
    lengths = [t["cot_length_tokens"] for t in traces]
    print(f"\nGSM8K Traces Summary:")
    print(f"  Count: {len(traces)}")
    print(f"  Avg length: {sum(lengths) / len(lengths):.0f} tokens")
    print(f"  Min length: {min(lengths)} tokens")
    print(f"  Max length: {max(lengths)} tokens")
    print(f"\nSample trace:")
    print(f"  Question: {traces[0]['question'][:100]}...")
    print(f"  CoT: {traces[0]['cot_text'][:200]}...")
