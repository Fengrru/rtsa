"""Structure-Correctness Correlation Analysis.

A2 from the professionalization plan — mirrors the core contribution of
LLM-MindMap (EMNLP 2025): quantify how strongly graph-level structural
properties correlate with reasoning correctness.

For every structural metric (n_nodes, depth, branching, verify_density,
backtrack_rate, entropy, ...) the script computes Spearman's rho against
per-trace correctness labels and reports p-values.

Label sources (in priority order):
1. ``--labels-file <jsonl>`` — explicit per-trace labels
   ``{"question_id": "...", "correct": true|false}``;
2. ``--synthetic`` — generate traces whose redundancy level is *known* to
   correlate with a synthetic correctness signal, used to validate the
   pipeline end-to-end;
3. fallback — ``answer_correct`` from the loaded trace records (note: GSM8K
   human solutions are all ground-truth correct, so the real-data variant
   only produces meaningful correlations once model-generated answers with
   natural error rates are available).

Output: a JSON report under ``experiments/results/`` plus a console table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # for direct imports

from rtsa.core.metrics import compute_graph_features_batch, GraphMetrics
from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.extractors.rule_based import RuleBasedExtractor
from rtsa.utils.data_loader import load_cot_traces

METRIC_NAMES = [
    "n_nodes", "n_edges", "depth", "branching",
    "verify_density", "backtrack_rate", "entropy",
    "avg_degree", "graph_density",
]


def metric_matrix(graphs: List[ReasoningTraceGraph]) -> np.ndarray:
    """N x len(METRIC_NAMES) feature matrix from GraphMetrics records."""
    metrics = compute_graph_features_batch(graphs)
    rows = []
    for m in metrics:
        rows.append([getattr(m, name) for name in METRIC_NAMES])
    return np.asarray(rows, dtype=float)


def load_labels(labels_path: Optional[str]) -> Dict[str, bool]:
    """Load ``{question_id: correct}`` from a JSONL labels file."""
    if not labels_path:
        return {}
    labels: Dict[str, bool] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record.get("question_id") or record.get("trace_id")
            if qid is not None:
                labels[str(qid)] = bool(record.get("correct", True))
    return labels


def synthetic_graphs(n: int = 60, seed: int = 42) -> List[ReasoningTraceGraph]:
    """Generate traces whose redundancy level drives a synthetic correctness
    label: verbose, verify-heavy traces are flagged incorrect."""
    rng = np.random.RandomState(seed)
    graphs = []
    for i in range(n):
        redundancy = rng.uniform(0.0, 1.0)
        base = max(3, int(4 + 8 * redundancy))
        # More redundancy -> more Verify nodes sprinkled through the chain.
        n_verify = int(base * redundancy)
        nodes = []
        edges = []
        prev = 1
        for k in range(base):
            ntype = "Verify" if k >= base - n_verify else "Transform"
            nodes.append({"id": k + 1, "type": ntype, "text": f"step {k}"})
            if k > 0:
                edges.append((k, k + 1))
        graphs.append(ReasoningTraceGraph(
            trace_id=f"syn_{i:03d}",
            nodes=[GraphNode(
                id=nd["id"],
                type=NodeType.from_string(nd["type"]),
                text=nd["text"],
            ) for nd in nodes],
            edges=edges,
            domain="synthetic",
            metadata={"correct": redundancy < 0.5},
        ))
    return graphs


def run(
    graphs: List[ReasoningTraceGraph],
    labels: Dict[str, bool],
) -> Dict[str, object]:
    """Compute Spearman correlations between metrics and correctness."""
    mat = metric_matrix(graphs)
    y = np.asarray([labels.get(g.trace_id, True) for g in graphs], dtype=float)

    n_variant = float(np.std(y))
    if n_variant == 0.0:
        raise ValueError(
            "All labels are identical (e.g. GSM8K human solutions are all "
            "correct). Provide --labels-file with model-generated answers, "
            "or use --synthetic to validate the pipeline."
        )

    correlations = []
    for i, name in enumerate(METRIC_NAMES):
        rho, p = spearmanr(mat[:, i], y)
        correlations.append({
            "metric": name,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        })
    correlations.sort(key=lambda c: abs(c["spearman_rho"]), reverse=True)
    return {
        "n_graphs": len(graphs),
        "n_incorrect": int(np.sum(y == 0.0)),
        "correlations": correlations,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    parser.add_argument("--max-traces", type=int, default=50)
    parser.add_argument("--labels-file", type=str, default=None,
                        help="JSONL with {question_id, correct} per trace")
    parser.add_argument("--synthetic", action="store_true",
                        help="Validate pipeline on synthetic correlated data")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: experiments/results/)")
    args = parser.parse_args(argv)

    if args.synthetic:
        graphs = synthetic_graphs()
        labels = {g.trace_id: g.metadata.get("correct", True) for g in graphs}
        print(f"Synthetic validation set: {len(graphs)} traces "
              f"({sum(1 for v in labels.values() if not v)} incorrect)")
    else:
        traces = load_cot_traces(f"data/raw_cots/{args.dataset}_50.jsonl"
                                 if args.dataset == "gsm8k"
                                 else f"data/raw_cots/{args.dataset}_100.jsonl")
        traces = traces[: args.max_traces]
        extractor = RuleBasedExtractor()
        graphs = []
        for i, t in enumerate(traces):
            g = extractor.extract(
                t.get("cot_text", ""),
                trace_id=t.get("question_id", f"t{i}"),
                answer=t.get("answer", ""),
            )
            graphs.append(g)
        labels = load_labels(args.labels_file)
        if not labels:
            labels = {t.get("question_id", f"t{i}"): t.get("answer_correct", True)
                      for i, t in enumerate(traces)}

    try:
        report = run(graphs, labels)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"\n  {'Metric':<18} {'Spearman rho':>12} {'p-value':>10}  Sig")
    print(f"  {'-' * 52}")
    for c in report["correlations"]:
        sig = "*" if c["significant"] else " "
        rho, p = c["spearman_rho"], c["p_value"]
        rho_s = f"{rho:>12.3f}" if np.isfinite(rho) else f"{'n/a':>12}"
        p_s = f"{p:>10.4f}" if np.isfinite(p) else f"{'n/a':>10}"
        print(f"  {c['metric']:<18} {rho_s} {p_s}  {sig}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        results_dir / f"correlation_{'synthetic' if args.synthetic else args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({**report, "labels_source": "synthetic" if args.synthetic else "file/trace"}, f, indent=2)
    print(f"\n  Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
