"""
End-to-End Pruning Utility Experiment (E2E-PUE)

Validates that RTSA pruning reduces reasoning-trace size while preserving
structural integrity.

Datasets:
    - synthetic  : 50 traces with *controlled* redundancy (best for
                   demonstrating prune compression & fidelity)
    - gsm8k      : 50 real GSM8K human CoT traces (tests robustness on
                   naturally compact reasoning)
    - mixed      : 25 synthetic + 25 gsm8k (balance of both)

Conditions per trace:
    - Control      : no pruning
    - Structural   : heuristic-only pruning (default PruneConfig)
    - Signal       : signal-enhanced pruning (calibration + PRM)

Metrics:
    - Node / edge / token compression ratios
    - NGS validation pass rate (pre- and post-prune)
    - Structural integrity score
    - Plan-redundancy score delta (reasonplan insight)
    - Redundancy-type distribution
    - Per-condition region counts

Usage:
    python -m experiments.end_to_end_prune --dataset synthetic --n 50
    python -m experiments.end_to_end_prune --dataset gsm8k --n 50
    python -m experiments.end_to_end_prune --dataset mixed --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rtsa.analysis.benchmark import ExtractorBenchmark
from rtsa.analysis.prune import PruneConfig, RedundancyAnalyzer
from rtsa.core.ngs_validator import NGSValidator
from rtsa.core.types import ReasoningTraceGraph
from rtsa.extractors import RuleBasedExtractor
from rtsa.utils.gsm8k_loader import load_saved_traces


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    trace_id: str
    dataset_source: str
    original_nodes: int
    original_edges: int
    original_tokens_est: int
    original_ngs_pass: bool
    original_plan_redundancy: float

    structural_nodes: int = 0
    structural_edges: int = 0
    structural_tokens_est: int = 0
    structural_ngs_pass: bool = False
    structural_integrity: float = 0.0
    structural_savings: int = 0
    structural_regions: int = 0
    structural_plan_redundancy: float = 0.0

    signal_nodes: int = 0
    signal_edges: int = 0
    signal_tokens_est: int = 0
    signal_ngs_pass: bool = False
    signal_integrity: float = 0.0
    signal_savings: int = 0
    signal_regions: int = 0
    signal_plan_redundancy: float = 0.0


@dataclass
class AggregateStats:
    n_traces: int
    n_skipped: int
    dataset: str

    structural_node_compression_mean: float
    structural_edge_compression_mean: float
    structural_token_compression_mean: float
    structural_ngs_pass_rate: float
    structural_mean_integrity: float
    structural_mean_savings: float
    structural_mean_regions: float
    structural_plan_redundancy_delta: float

    signal_node_compression_mean: float
    signal_edge_compression_mean: float
    signal_token_compression_mean: float
    signal_ngs_pass_rate: float
    signal_mean_integrity: float
    signal_mean_savings: float
    signal_mean_regions: float
    signal_plan_redundancy_delta: float

    region_type_counts: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(graph: ReasoningTraceGraph | None, avg_per_node: int = 35) -> int:
    if graph is None:
        return 0
    return len(graph.nodes) * avg_per_node


def _load_dataset(dataset: str, n: int) -> List[Dict[str, Any]]:
    """Load traces from the chosen dataset."""
    if dataset == "synthetic":
        from rtsa.experiments.synthetic_redundant_cots import generate_synthetic_cots
        return generate_synthetic_cots(n=n, seed=42)

    if dataset == "gsm8k":
        path = _PROJECT_ROOT / "data" / "raw_cots" / "gsm8k_50.jsonl"
        traces = load_saved_traces(str(path))
        for t in traces:
            t["dataset_source"] = "gsm8k"
        return traces[:n]

    if dataset == "mixed":
        from rtsa.experiments.synthetic_redundant_cots import generate_synthetic_cots
        syn = generate_synthetic_cots(n=n // 2 + n % 2, seed=42)
        for t in syn:
            t["dataset_source"] = "synthetic"
        path = _PROJECT_ROOT / "data" / "raw_cots" / "gsm8k_50.jsonl"
        real = load_saved_traces(str(path))[: n // 2]
        for t in real:
            t["dataset_source"] = "gsm8k"
        return syn + real

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Experiment engine
# ---------------------------------------------------------------------------

def run_experiment(
    traces: List[Dict[str, Any]],
    dataset_name: str,
    avg_tokens_per_node: int = 35,
) -> Dict[str, Any]:
    start = time.time()

    extractor = RuleBasedExtractor()
    ngs = NGSValidator()

    configs = {
        "structural": PruneConfig(avg_tokens_per_node=avg_tokens_per_node),
        "signal": PruneConfig(
            avg_tokens_per_node=avg_tokens_per_node,
            use_calibration_signal=True,
            use_prm_signal=True,
        ),
    }
    analyzers = {k: RedundancyAnalyzer(config=v) for k, v in configs.items()}

    results: List[TraceResult] = []
    skipped = 0
    region_type_counts: Dict[str, int] = {}

    print("\n" + "=" * 70)
    print("E2E Pruning Utility Experiment")
    print("=" * 70)
    print(f"  Dataset: {dataset_name}  |  Traces: {len(traces)}  |  Token est: {avg_tokens_per_node}/node\n")

    for i, trace in enumerate(traces):
        trace_id = trace.get("question_id", f"trace_{i}")
        source = trace.get("dataset_source", dataset_name)
        cot_text = trace.get("cot_text", "")

        graph = extractor.extract(cot_text, trace_id=trace_id)
        if len(graph.nodes) == 0:
            skipped += 1
            continue

        orig_nodes = len(graph.nodes)
        orig_edges = len(graph.edges)
        orig_tokens = estimate_tokens(graph, avg_tokens_per_node)
        orig_ngs_pass, _ = ngs.validate(graph)
        orig_plan_red = ExtractorBenchmark._compute_plan_redundancy([graph])

        tr = TraceResult(
            trace_id=trace_id,
            dataset_source=source,
            original_nodes=orig_nodes,
            original_edges=orig_edges,
            original_tokens_est=orig_tokens,
            original_ngs_pass=orig_ngs_pass,
            original_plan_redundancy=orig_plan_red,
        )

        for cond_name in ("structural", "signal"):
            report = analyzers[cond_name].analyze(graph, apply_pruning=True)
            pruned = report.pruned_graph

            if pruned:
                nodes_after = len(pruned.nodes)
                edges_after = len(pruned.edges)
                tokens_after = estimate_tokens(pruned, avg_tokens_per_node)
                ngs_pass_after, _ = ngs.validate(pruned)
                plan_red_after = ExtractorBenchmark._compute_plan_redundancy([pruned])
            else:
                nodes_after = orig_nodes
                edges_after = orig_edges
                tokens_after = orig_tokens
                ngs_pass_after = orig_ngs_pass
                plan_red_after = orig_plan_red

            actual_savings = max(0, orig_tokens - tokens_after)
            if cond_name == "structural":
                tr.structural_nodes = nodes_after
                tr.structural_edges = edges_after
                tr.structural_tokens_est = tokens_after
                tr.structural_ngs_pass = ngs_pass_after
                tr.structural_integrity = report.structural_integrity_score
                tr.structural_savings = actual_savings
                tr.structural_regions = len(report.redundancy_regions)
                tr.structural_plan_redundancy = plan_red_after
            else:
                tr.signal_nodes = nodes_after
                tr.signal_edges = edges_after
                tr.signal_tokens_est = tokens_after
                tr.signal_ngs_pass = ngs_pass_after
                tr.signal_integrity = report.structural_integrity_score
                tr.signal_savings = actual_savings
                tr.signal_regions = len(report.redundancy_regions)
                tr.signal_plan_redundancy = plan_red_after

            for r in report.redundancy_regions:
                region_type_counts[r.region_type] = region_type_counts.get(r.region_type, 0) + 1

        results.append(tr)

        if (i + 1) % 10 == 0 or (i + 1) == len(traces):
            print(f"  Processed {i + 1:>3}/{len(traces)}  |  Valid: {len(results)}  |  Skipped: {skipped}")

    if not results:
        raise RuntimeError("No valid graphs extracted. Cannot compute statistics.")

    def _mean(vals: List[float]) -> float:
        return float(np.mean(vals))

    def _comp(orig: int, after: int) -> float:
        return (orig - after) / orig if orig else 0.0

    stats = AggregateStats(
        n_traces=len(results),
        n_skipped=skipped,
        dataset=dataset_name,
        structural_node_compression_mean=_mean([_comp(r.original_nodes, r.structural_nodes) for r in results]),
        structural_edge_compression_mean=_mean([_comp(r.original_edges, r.structural_edges) for r in results]),
        structural_token_compression_mean=_mean([_comp(r.original_tokens_est, r.structural_tokens_est) for r in results]),
        structural_ngs_pass_rate=_mean([1.0 if r.structural_ngs_pass else 0.0 for r in results]),
        structural_mean_integrity=_mean([r.structural_integrity for r in results]),
        structural_mean_savings=_mean([r.structural_savings for r in results]),
        structural_mean_regions=_mean([r.structural_regions for r in results]),
        structural_plan_redundancy_delta=_mean([r.original_plan_redundancy - r.structural_plan_redundancy for r in results]),
        signal_node_compression_mean=_mean([_comp(r.original_nodes, r.signal_nodes) for r in results]),
        signal_edge_compression_mean=_mean([_comp(r.original_edges, r.signal_edges) for r in results]),
        signal_token_compression_mean=_mean([_comp(r.original_tokens_est, r.signal_tokens_est) for r in results]),
        signal_ngs_pass_rate=_mean([1.0 if r.signal_ngs_pass else 0.0 for r in results]),
        signal_mean_integrity=_mean([r.signal_integrity for r in results]),
        signal_mean_savings=_mean([r.signal_savings for r in results]),
        signal_mean_regions=_mean([r.signal_regions for r in results]),
        signal_plan_redundancy_delta=_mean([r.original_plan_redundancy - r.signal_plan_redundancy for r in results]),
        region_type_counts=region_type_counts,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        runtime_seconds=time.time() - start,
    )

    return {"aggregate": asdict(stats), "traces": [asdict(r) for r in results]}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(data: Dict[str, Any]) -> None:
    agg = data["aggregate"]
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  Dataset           : {agg['dataset']}")
    print(f"  Traces analyzed   : {agg['n_traces']}  (skipped empty: {agg['n_skipped']})")
    print(f"  Runtime           : {agg['runtime_seconds']:.1f}s")
    print()

    print("  ┌─ Structural Pruning (heuristic-only) ─────────────────────────────┐")
    print(f"  │  Node compression   : {agg['structural_node_compression_mean']:.1%}")
    print(f"  │  Edge compression   : {agg['structural_edge_compression_mean']:.1%}")
    print(f"  │  Token compression  : {agg['structural_token_compression_mean']:.1%}")
    print(f"  │  NGS pass rate      : {agg['structural_ngs_pass_rate']:.1%}")
    print(f"  │  Integrity score    : {agg['structural_mean_integrity']:.3f}")
    print(f"  │  Avg token savings  : {agg['structural_mean_savings']:.0f}")
    print(f"  │  Avg regions found  : {agg['structural_mean_regions']:.1f}")
    print(f"  │  Plan redundancy Δ  : {agg['structural_plan_redundancy_delta']:+.3f}")
    print("  └───────────────────────────────────────────────────────────────────┘")
    print()

    print("  ┌─ Signal-Enhanced Pruning (calibration + PRM) ─────────────────────┐")
    print(f"  │  Node compression   : {agg['signal_node_compression_mean']:.1%}")
    print(f"  │  Edge compression   : {agg['signal_edge_compression_mean']:.1%}")
    print(f"  │  Token compression  : {agg['signal_token_compression_mean']:.1%}")
    print(f"  │  NGS pass rate      : {agg['signal_ngs_pass_rate']:.1%}")
    print(f"  │  Integrity score    : {agg['signal_mean_integrity']:.3f}")
    print(f"  │  Avg token savings  : {agg['signal_mean_savings']:.0f}")
    print(f"  │  Avg regions found  : {agg['signal_mean_regions']:.1f}")
    print(f"  │  Plan redundancy Δ  : {agg['signal_plan_redundancy_delta']:+.3f}")
    print("  └───────────────────────────────────────────────────────────────────┘")
    print()

    if agg["region_type_counts"]:
        print("  Redundancy-type distribution (merged):")
        for rt, cnt in sorted(agg["region_type_counts"].items(), key=lambda x: -x[1]):
            print(f"    {rt:<30} : {cnt:>3}")
    print()

    print("  ── Interpretation ──")
    if agg["structural_ngs_pass_rate"] >= 0.95 and agg["signal_ngs_pass_rate"] >= 0.95:
        print("  Both strategies maintain NGS validity in >=95% of traces.")
    else:
        print("  WARNING: NGS pass rate < 95% — pruning may damage structure.")

    if agg["signal_node_compression_mean"] > agg["structural_node_compression_mean"]:
        print("  Signal-enhanced achieves HIGHER compression.")
    elif agg["signal_node_compression_mean"] < agg["structural_node_compression_mean"]:
        print("  Signal-enhanced is MORE CONSERVATIVE (fewer false positives).")
    else:
        print("  Compression rates are equivalent between strategies.")

    print(f"  Signal-enhanced saves ~{agg['signal_mean_savings']:.0f} tokens/trace on average.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RTSA End-to-End Pruning Utility Experiment")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "gsm8k", "mixed"],
                        help="Dataset to use")
    parser.add_argument("--n", type=int, default=50, help="Number of traces")
    parser.add_argument("--avg-tokens-per-node", type=int, default=35,
                        help="Token estimation factor")
    parser.add_argument("--output", type=str, default="experiments/results/e2e_prune.json",
                        help="JSON output path")
    args = parser.parse_args()

    traces = _load_dataset(args.dataset, args.n)
    if not traces:
        print(f"ERROR: No traces loaded for dataset '{args.dataset}'")
        sys.exit(1)

    data = run_experiment(traces, dataset_name=args.dataset, avg_tokens_per_node=args.avg_tokens_per_node)
    print_report(data)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
