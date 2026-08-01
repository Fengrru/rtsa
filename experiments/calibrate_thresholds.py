"""Threshold Calibration for the Pruning Pipeline (A1).

Codifies the manual "annotate -> tune threshold" loop into a script:

1. Load per-trace redundancy annotations (JSONL: ``{"question_id", "cot_text",
   "redundant": [node ids]}``), or generate synthetic annotated graphs whose
   ground truth is known by construction (``--synthetic``);
2. Grid-scan each ``PruneConfig`` threshold independently (coordinate
   descent: one parameter moves while the others stay frozen), measuring
   node-level precision / recall / F1 against the annotations;
3. Report the optimal value per parameter plus a ready-to-paste
   ``PruneConfig`` fragment.

The rationale mirrors CRV (arXiv 2510.09312): structural error signatures
are dataset- and domain-dependent, so thresholds should be calibrated on
the target distribution instead of being hard-coded.

Example:
    python -m experiments.calibrate_thresholds --synthetic
    python -m experiments.calibrate_thresholds --annotations-file data/my_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # for direct imports

from analysis.prune import PruneConfig, RedundancyAnalyzer
from core.types import GraphNode, NodeType, ReasoningTraceGraph
from extractors.rule_based import RuleBasedExtractor

# ---------------------------------------------------------------------------
# Scan grid: one parameter moves, the rest stay frozen (coordinate descent)
# ---------------------------------------------------------------------------

PARAM_GRID: Dict[str, List[Any]] = {
    "verify_density_high": [0.30, 0.35, 0.40, 0.45, 0.50],
    "verify_late_stage_ratio": [0.50, 0.60, 0.70],
    "branch_utilization_min": [0.40, 0.50, 0.60],
    "max_consecutive_transforms": [2, 3, 4],
    "max_depth_to_nodes_ratio": [0.20, 0.25, 0.30],
    "entropy_low_threshold": [0.40, 0.50, 0.60],
    "min_confidence_threshold": [0.40, 0.50, 0.60],
}

METRIC_NAMES = ["precision", "recall", "f1"]


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> Tuple[List[ReasoningTraceGraph], List[Set[int]]]:
    """Load JSONL annotations; graphs are extracted with the RuleBasedExtractor.

    Each record: ``{"question_id", "cot_text", "redundant": [node ids]}``.
    Node ids refer to the graph produced by the RBE for that trace.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    extractor = RuleBasedExtractor()
    graphs: List[ReasoningTraceGraph] = []
    annotations: List[Set[int]] = []
    for i, rec in enumerate(records):
        g = extractor.extract(
            rec.get("cot_text", ""),
            trace_id=rec.get("question_id", f"t{i}"),
            answer=rec.get("answer", ""),
        )
        graphs.append(g)
        annotations.append(set(rec.get("redundant", [])))
    return graphs, annotations


# ---------------------------------------------------------------------------
# Synthetic annotated graphs (ground truth known by construction)
# ---------------------------------------------------------------------------

def synthetic_annotated_graphs(
    n: int = 40, seed: int = 42
) -> Tuple[List[ReasoningTraceGraph], List[Set[int]]]:
    """Generate traces with injected, known-redundant regions.

    - ~30% get a trailing cluster of 2-3 Verify nodes (late-stage
      verification, annotated redundant);
    - ~30% get a 4-6 node Transform chain appended (nodes beyond the first 3
      are annotated redundant);
    - the rest are clean chains with an empty annotation.
    """
    rng = np.random.RandomState(seed)
    graphs: List[ReasoningTraceGraph] = []
    annotations: List[Set[int]] = []

    for i in range(n):
        nodes: List[GraphNode] = []
        edges: List[Tuple[int, int]] = []
        redundant: Set[int] = set()
        nid = 1

        # Main Transform chain.
        base = int(rng.randint(6, 11))
        prev = 0
        for k in range(base):
            nodes.append(GraphNode(id=nid, type=NodeType.TRANSFORM, text=f"step {nid}"))
            if prev:
                edges.append((prev, nid))
            prev = nid
            nid += 1

        # Trailing Verify cluster (late-stage verification bloat).
        if rng.rand() < 0.30:
            n_verify = int(rng.randint(2, 4))
            for v in range(n_verify):
                nodes.append(GraphNode(id=nid, type=NodeType.VERIFY, text=f"check {nid}"))
                edges.append((prev, nid))
                redundant.add(nid)
                prev = nid
                nid += 1

        # Long Transform chain (first 3 kept as anchor, the rest redundant).
        if rng.rand() < 0.30:
            chain_len = int(rng.randint(4, 7))
            for e in range(chain_len):
                nodes.append(GraphNode(id=nid, type=NodeType.TRANSFORM, text=f"chain {nid}"))
                edges.append((prev, nid))
                if e >= 3:
                    redundant.add(nid)
                prev = nid
                nid += 1

        graphs.append(ReasoningTraceGraph(
            trace_id=f"cal_{i:03d}",
            nodes=nodes,
            edges=edges,
            domain="synthetic",
            metadata={"n_redundant": len(redundant)},
        ))
        annotations.append(redundant)
    return graphs, annotations


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict_redundant(
    analyzer: RedundancyAnalyzer, graph: ReasoningTraceGraph
) -> Set[int]:
    """Nodes the analyzer flags as prunable (delete/merge, above confidence)."""
    report = analyzer.analyze(graph)
    cfg = analyzer.config
    pred: Set[int] = set()
    for r in report.redundancy_regions:
        if (
            r.suggested_action in ("delete", "merge")
            and r.confidence >= cfg.min_confidence_threshold
        ):
            pred.update(r.node_ids)
    return pred


def aggregate(
    preds: List[Set[int]], gts: List[Set[int]]
) -> Dict[str, float]:
    """Global node-level precision / recall / F1 over all graphs."""
    tp = sum(len(p & g) for p, g in zip(preds, gts))
    fp = sum(len(p - g) for p, g in zip(preds, gts))
    fn = sum(len(g - p) for p, g in zip(preds, gts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Coordinate-descent scan
# ---------------------------------------------------------------------------

def run(
    graphs: List[ReasoningTraceGraph],
    annotations: List[Set[int]],
    base_config: Optional[PruneConfig] = None,
    iterations: int = 2,
    metric: str = "f1",
) -> Dict[str, Any]:
    """Scan every parameter in ``PARAM_GRID``, freezing the others.

    Returns the per-iteration history plus the best parameter set found.
    """
    base_config = base_config or PruneConfig()
    best_params: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    for it in range(iterations):
        improved = False
        for param, values in PARAM_GRID.items():
            frozen = replace(base_config, **best_params)
            scan = []
            for v in values:
                cfg = replace(frozen, **{param: v})
                analyzer = RedundancyAnalyzer(config=cfg)
                preds = [
                    predict_redundant(analyzer, g)
                    for g in graphs
                ]
                agg = aggregate(preds, annotations)
                scan.append({"value": v, **agg})
            best = max(scan, key=lambda s: s[metric])
            history.append({
                "iteration": it,
                "param": param,
                "best_value": best["value"],
                "best_metric": {m: best[m] for m in METRIC_NAMES},
                "curve": [
                    {"value": s["value"], **{m: s[m] for m in METRIC_NAMES}}
                    for s in scan
                ],
            })
            if best_params.get(param) != best["value"]:
                best_params[param] = best["value"]
                improved = True
        if not improved and it > 0:
            break

    best_cfg = replace(base_config, **best_params)
    analyzer = RedundancyAnalyzer(config=best_cfg)
    preds = [predict_redundant(analyzer, g) for g in graphs]
    return {
        "n_graphs": len(graphs),
        "n_annotated_redundant": sum(len(g) for g in annotations),
        "iterations": it + 1,
        "metric": metric,
        "best_params": best_params,
        "best_aggregate": aggregate(preds, annotations),
        "history": history,
    }


def format_config_fragment(best_params: Dict[str, Any]) -> str:
    """Render the best parameters as a paste-ready PruneConfig fragment."""
    lines = ["# Copy into PruneConfig(domain_overrides={...}) or a YAML config:"]
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-file", type=str, default=None,
                        help="JSONL with {question_id, cot_text, redundant}")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic annotated graphs (no data needed)")
    parser.add_argument("--n-synthetic", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=2,
                        help="Coordinate-descent rounds")
    parser.add_argument("--metric", choices=METRIC_NAMES, default="f1")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: experiments/results/)")
    args = parser.parse_args(argv)

    if args.annotations_file:
        graphs, annotations = load_annotations(args.annotations_file)
        source = "annotations"
    elif args.synthetic:
        graphs, annotations = synthetic_annotated_graphs(n=args.n_synthetic)
        source = "synthetic"
        print(f"Synthetic calibration set: {len(graphs)} traces, "
              f"{sum(len(a) for a in annotations)} annotated redundant nodes")
    else:
        parser.error("provide --annotations-file or --synthetic")

    report = run(graphs, annotations, iterations=args.iterations, metric=args.metric)

    print(f"\n  Calibration on {source} data ({args.metric} objective)")
    print(f"  {'Parameter':<28} {'Best value':>12}  P/R/F1")
    print(f"  {'-' * 62}")
    for h in report["history"]:
        if h["iteration"] != report["iterations"] - 1:
            continue  # only the final descent round
        bm = h["best_metric"]
        print(f"  {h['param']:<28} {str(h['best_value']):>12}  "
              f"{bm['precision']:.2f}/{bm['recall']:.2f}/{bm['f1']:.2f}")
    agg = report["best_aggregate"]
    print(f"\n  Best aggregate: precision={agg['precision']:.3f} "
          f"recall={agg['recall']:.3f} f1={agg['f1']:.3f}")
    print(f"\n{format_config_fragment(report['best_params'])}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        results_dir / f"calibration_{source}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
