"""
Data loading utilities for RTSA experiments (from v3.1).

Provides standardized I/O for:
- CoT trace datasets (JSONL format)
- Extracted graph collections (JSONL format)
- Benchmark datasets (MATH, HumanEval, GSM8K, etc.)
- Motif catalogs (JSON format)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph


# ---------------------------------------------------------------------------
# CoT Trace I/O
# ---------------------------------------------------------------------------

def load_cot_traces(filepath: str) -> List[Dict[str, Any]]:
    """Load CoT traces from a JSONL file.

    Expected format per line:
    {
        "cot_text": "...",
        "model": "qwen2.5-7b",
        "question_id": "math_001",
        "domain": "math",
        "answer": "...",
        "answer_correct": true,
        "cot_length_tokens": 145
    }
    """
    traces = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def save_cot_traces(traces: List[Dict[str, Any]], filepath: str) -> None:
    """Save CoT traces to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def cot_traces_to_canonical(
    traces: List[Dict[str, Any]],
    extractor,
    extractor_id: str = "rbe",
) -> List:
    """Convert CoT trace dicts to canonical graph objects using an extractor.

    Args:
        traces: List of trace dicts from load_cot_traces().
        extractor: An extractor instance with an extract() method.
        extractor_id: Extractor identifier for the schema.

    Returns:
        List of graph objects.
    """
    schemas = []
    for trace in traces:
        schema = extractor.extract(
            cot_text=trace["cot_text"],
            trace_id=trace.get("trace_id", ""),
            model=trace.get("model", "unknown"),
            question_id=trace.get("question_id", "unknown"),
            domain=trace.get("domain", "unknown"),
        )
        schemas.append(schema)
    return schemas


# ---------------------------------------------------------------------------
# Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    traces: List[Dict[str, Any]],
    n_per_domain: Dict[str, int],
    domain_key: str = "domain",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Stratified random sampling of CoT traces by domain.

    Args:
        traces: Full trace list.
        n_per_domain: Dict mapping domain -> number of samples.
        domain_key: Key in trace dict for domain label.
        seed: Random seed.

    Returns:
        Sampled trace list.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    sampled = []

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for t in traces:
        domain = t.get(domain_key, "unknown")
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(t)

    for domain, n in n_per_domain.items():
        available = by_domain.get(domain, [])
        n_sample = min(n, len(available))
        if n_sample > 0:
            indices = rng.choice(len(available), size=n_sample, replace=False)
            sampled.extend([available[i] for i in indices])

    return sampled


# ---------------------------------------------------------------------------
# Graph Collection I/O
# ---------------------------------------------------------------------------

def load_extracted_graphs(filepath: str) -> List:
    """Load extracted RTGs from JSONL."""
    graphs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Support both v3.1 (CanonicalSchema) and v3.2 (ReasoningTraceGraph) formats
                graph_data = data.get("graph", data)
                nodes_data = graph_data.get("nodes", [])
                edges_data = graph_data.get("edges", [])
                graphs.append(ReasoningTraceGraph(
                    trace_id=data.get("trace_id", ""),
                    model=data.get("model", ""),
                    question_id=data.get("question_id", ""),
                    domain=data.get("domain", ""),
                    extractor=data.get("extractor", ""),
                    metadata=data.get("metadata", {}),
                    nodes=[
                        GraphNode(
                            id=n["id"],
                            type=NodeType.from_string(n["type"]),
                            span=tuple(n["span"]) if n.get("span") is not None else (0, 0),
                        )
                        for n in nodes_data
                    ],
                    edges=[tuple(e) for e in edges_data],
                ))
    return graphs


def save_extracted_graphs(graphs: List, filepath: str) -> None:
    """Save extracted RTGs to JSONL."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in graphs:
            if hasattr(g, "to_canonical_dict"):
                f.write(json.dumps(g.to_canonical_dict(), ensure_ascii=False) + "\n")
            elif hasattr(g, "model_dump_json"):
                f.write(g.model_dump_json(exclude_none=True) + "\n")
            else:
                f.write(json.dumps(g, default=str, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Motif Catalog I/O
# ---------------------------------------------------------------------------

def save_motif_catalog(motifs: dict, filepath: str) -> None:
    """Save motif catalog to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for mid, motif in motifs.items():
        serializable[mid] = {
            "motif_id": motif.motif_id,
            "description": motif.description,
            "size": motif.size,
        }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_motif_catalog(filepath: str) -> dict:
    """Load motif catalog from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Benchmark Dataset Loaders
# ---------------------------------------------------------------------------

def load_math_dataset(
    path: str,
    levels: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    """Load MATH dataset problems.

    Args:
        path: Path to MATH dataset directory.
        levels: Optional filter by difficulty level (1–5).

    Returns:
        List of {"problem": "...", "solution": "...", "level": int, "type": str}.
    """
    problems = []
    math_path = Path(path)

    for level_dir in math_path.iterdir():
        if not level_dir.is_dir():
            continue
        level = int(level_dir.name.split()[-1]) if "Level" in level_dir.name else None
        if levels and level not in levels:
            continue

        for type_dir in level_dir.iterdir():
            if not type_dir.is_dir():
                continue
            for prob_file in type_dir.glob("*.json"):
                with open(prob_file, "r") as f:
                    data = json.load(f)
                    data["level"] = level
                    data["type"] = type_dir.name
                    problems.append(data)

    return problems


def load_humaneval_dataset(path: str) -> List[Dict[str, str]]:
    """Load HumanEval dataset.

    Args:
        path: Path to HumanEval.jsonl.

    Returns:
        List of {"task_id": str, "prompt": str, "canonical_solution": str, ...}.
    """
    problems = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def batch_process(
    items: List[Any],
    process_fn,
    batch_size: int = 10,
    desc: str = "Processing",
) -> List[Any]:
    """Process items in batches with progress bar.

    Args:
        items: List of items to process.
        process_fn: Function(item) -> result.
        batch_size: Number of items per batch (for logging).
        desc: Description for progress bar.

    Returns:
        List of results.
    """
    from tqdm import tqdm

    results = []
    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i:i + batch_size]
        for item in batch:
            results.append(process_fn(item))
    return results
