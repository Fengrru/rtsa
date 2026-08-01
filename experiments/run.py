"""Unified experiment entrypoint (C10).

Wraps the standalone experiment scripts behind one CLI with versioned
result directories:

    python -m experiments.run extract   --dataset gsm8k --max-traces 50
    python -m experiments.run analyze   --dataset gsm8k
    python -m experiments.run prune     --dataset synthetic --n 50
    python -m experiments.run calibrate --synthetic
    python -m experiments.run annotate  --traces-file data/raw_cots/gsm8k_50.jsonl
    python -m experiments.run all       --dataset gsm8k

Every run lands in ``experiments/results/runs/<command>_<timestamp>/`` with
a ``manifest.json`` recording the git commit, python version, arguments and
timestamps — making experiment outputs reproducible and comparable across
runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # for direct imports

from core.metrics import compute_graph_features_batch
from core.types import ReasoningTraceGraph
from extractors.rule_based import RuleBasedExtractor
from utils.data_loader import load_cot_traces

RUNS_ROOT = _PROJECT_ROOT / "experiments" / "results" / "runs"

LOCAL_DATASETS = {
    "gsm8k": "data/raw_cots/gsm8k_50.jsonl",
    "math": "data/raw_cots/math_100.jsonl",
}


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def git_commit() -> str:
    """Short HEAD hash, or 'unknown' outside a git work tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(tag: str) -> Path:
    """Create a timestamped, versioned output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, command: str, args: Dict[str, Any],
                   extra: Optional[Dict[str, Any]] = None) -> Path:
    """Persist run provenance (git hash, python, args, timestamps)."""
    manifest = {
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": sys.version.split()[0],
        "args": {k: str(v) for k, v in args.items()},
    }
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_traces_for(args, max_traces: int) -> List[Dict[str, Any]]:
    """Load traces from a local corpus or a HuggingFace dataset id."""
    if getattr(args, "hf_dataset", None):
        from utils.hf_adapter import load_hf_traces
        return load_hf_traces(
            args.hf_dataset,
            config=getattr(args, "hf_config", None),
            split=getattr(args, "split", "train"),
            max_samples=max_traces,
            seed=args.seed,
        )
    path = LOCAL_DATASETS.get(args.dataset, LOCAL_DATASETS["gsm8k"])
    return load_cot_traces(str(_PROJECT_ROOT / path))[:max_traces]


def extract_graphs(traces: List[Dict[str, Any]],
                   max_traces: int) -> List[ReasoningTraceGraph]:
    """Batch-extract graphs with the rule-based extractor."""
    extractor = RuleBasedExtractor()
    graphs = []
    for i, t in enumerate(traces[:max_traces]):
        graphs.append(extractor.extract(
            t.get("cot_text", ""),
            trace_id=t.get("question_id", f"t{i}"),
            answer=t.get("answer", ""),
        ))
    return graphs


def save_graphs(graphs: List[ReasoningTraceGraph], path: Path) -> None:
    """Serialize graphs to a plain JSON array (no pickling)."""
    payload = []
    for g in graphs:
        payload.append({
            "trace_id": g.trace_id,
            "domain": g.domain,
            "nodes": [
                {"id": n.id, "type": n.type.value, "text": n.text}
                for n in g.nodes
            ],
            "edges": [[u, v] for u, v in g.edges],
        })
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _run_extract(args, run_dir: Path) -> int:
    traces = load_traces_for(args, max_traces=args.max_traces)
    graphs = extract_graphs(traces, args.max_traces)
    save_graphs(graphs, run_dir / "graphs.json")

    types = {}
    for g in graphs:
        for n in g.nodes:
            types[n.type.value] = types.get(n.type.value, 0) + 1
    print(f"Extracted {len(graphs)} graphs -> {run_dir / 'graphs.json'}")
    print(f"  node types: {types}")
    write_manifest(run_dir, "extract", vars(args),
                   extra={"n_graphs": len(graphs)})
    return 0


def _run_analyze(args, run_dir: Path) -> int:
    traces = load_traces_for(args, max_traces=args.max_traces)
    graphs = extract_graphs(traces, args.max_traces)
    metrics = compute_graph_features_batch(graphs)

    names = list(metrics[0].__dataclass_fields__) if metrics else []
    summary = {}
    for name in names:
        vals = []
        for m in metrics:
            try:
                vals.append(float(getattr(m, name)))
            except (TypeError, ValueError):
                continue  # non-numeric field (e.g. trace_id)
        if not vals:
            continue
        summary[name] = {
            "mean": float(sum(vals) / len(vals)),
            "std": _std(vals),
            "min": min(vals),
            "max": max(vals),
        }

    report = {"n_traces": len(graphs), "metric_summary": summary}
    if args.labels_file:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from experiments.correlation_analysis import run as corr_run
        from experiments.correlation_analysis import load_labels
        labels = load_labels(args.labels_file)
        labels = labels or {
            t.get("question_id", f"t{i}"): t.get("answer_correct", True)
            for i, t in enumerate(traces)
        }
        try:
            report["correlation"] = corr_run(graphs, labels)
        except ValueError as exc:
            print(f"  (correlation skipped: {exc})")

    out = run_dir / "analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Analyzed {len(graphs)} traces -> {out}")
    write_manifest(run_dir, "analyze", vars(args),
                   extra={"n_traces": len(graphs)})
    return 0


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _run_prune(args, run_dir: Path) -> int:
    from experiments.end_to_end_prune import run_experiment, print_report
    from experiments.synthetic_redundant_cots import generate_synthetic_cots

    if args.dataset == "synthetic":
        traces = generate_synthetic_cots(
            n=getattr(args, "n", None) or getattr(args, "max_traces", 50),
            seed=args.seed,
        )
    else:
        traces = load_traces_for(
            args,
            max_traces=(getattr(args, "n", None)
                        or getattr(args, "max_traces", 50)),
        )
    if not traces:
        print("ERROR: no traces loaded")
        return 1
    data = run_experiment(traces, dataset_name=args.dataset,
                          avg_tokens_per_node=args.avg_tokens_per_node)
    print_report(data)
    out = run_dir / "prune.json"
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Pruning results -> {out}")
    write_manifest(run_dir, "prune", vars(args), extra={"n_traces": len(traces)})
    return 0


def _run_calibrate(args, run_dir: Path) -> int:
    from experiments.calibrate_thresholds import run as cal_run
    from experiments.calibrate_thresholds import (
        load_annotations, synthetic_annotated_graphs,
    )

    if args.annotations_file:
        graphs, annotations = load_annotations(args.annotations_file)
        source = "annotations"
    else:
        graphs, annotations = synthetic_annotated_graphs(n=args.n_synthetic)
        source = "synthetic"

    report = cal_run(graphs, annotations, iterations=args.iterations,
                     metric=args.metric)
    out = run_dir / "calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration on {source} data (best {args.metric}): "
          f"{report['best_params']}")
    print(f"Calibration report -> {out}")
    write_manifest(run_dir, "calibrate", vars(args),
                   extra={"source": source})
    return 0


def _run_annotate(args, run_dir: Path) -> int:
    from experiments.annotate_steps import annotate_trace
    from core.ngs_validator import NGSValidator

    traces = load_cot_traces(args.traces_file)[: args.max_traces]
    extractor = RuleBasedExtractor()
    validator = NGSValidator()
    records = []
    for i, t in enumerate(traces):
        g = extractor.extract(
            t.get("cot_text", ""),
            trace_id=t.get("question_id", f"t{i}"),
            answer=t.get("answer", ""),
        )
        records.extend(annotate_trace(g, validator))

    from collections import Counter
    counts = Counter(r["is_correct"] for r in records)
    out = run_dir / "step_annotations.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Annotated {len(records)} nodes (correct={counts[True]}, "
          f"error={counts[False]}) -> {out}")
    write_manifest(run_dir, "annotate", vars(args),
                   extra={"n_nodes": len(records)})
    return 0


def _run_correlation(args, run_dir: Path) -> int:
    """Performance-correlation benchmark (structure -> correctness/score)."""
    from analysis.performance_correlation import (
        run_performance_correlation,
        synthetic_performance_graphs,
    )
    from experiments.correlation_analysis import load_labels

    if args.synthetic:
        graphs = synthetic_performance_graphs(n=args.n, seed=args.seed)
        labels = {
            g.trace_id: g.metadata.get("correct", True) for g in graphs
        }
        source = "synthetic"
    else:
        traces = load_traces_for(args, max_traces=args.max_traces)
        graphs = extract_graphs(traces, args.max_traces)
        labels = load_labels(args.labels_file)
        labels = labels or {
            t.get("question_id", f"t{i}"): t.get("answer_correct", True)
            for i, t in enumerate(traces)
        }
        source = args.dataset

    report = run_performance_correlation(
        graphs, labels,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(report.summary())

    (run_dir / "correlation.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "correlation.md").write_text(
        report.to_markdown(), encoding="utf-8")
    print(f"Correlation benchmark ({source}) -> "
          f"{run_dir / 'correlation.json'} + correlation.md")
    write_manifest(run_dir, "correlation", vars(args),
                   extra={"n_graphs": len(graphs), "source": source})
    return 0


def _run_all(args, run_dir: Path) -> int:
    """extract -> analyze -> prune in one shot."""
    tag = f"all_{args.dataset}"
    steps = ["extract", "analyze", "prune"]
    for step in steps:
        step_dir = RUNS_ROOT / f"{tag}_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if step == "prune":
            rc = _run_prune(args, step_dir)
        else:
            rc = _run_extract(args, step_dir) if step == "extract" \
                else _run_analyze(args, step_dir)
        if rc != 0:
            return rc
    write_manifest(run_dir, "all", vars(args), extra={"steps": steps})
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RTSA unified experiment entrypoint (versioned runs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--hf-dataset", type=str, default=None,
                       help="HuggingFace dataset id (overrides --dataset)")
        p.add_argument("--hf-config", type=str, default=None)
        p.add_argument("--split", type=str, default="train")

    p = sub.add_parser("extract", help="extract reasoning graphs")
    add_common(p)
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--max-traces", type=int, default=50)

    p = sub.add_parser("analyze", help="graph metric summary + optional correlation")
    add_common(p)
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--max-traces", type=int, default=50)
    p.add_argument("--labels-file", type=str, default=None,
                   help="JSONL {question_id, correct} for Spearman correlation")

    p = sub.add_parser("prune", help="end-to-end pruning experiment")
    add_common(p)
    p.add_argument("--dataset", choices=["synthetic", "gsm8k", "mixed"],
                   default="synthetic")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--avg-tokens-per-node", type=int, default=35)

    p = sub.add_parser("calibrate", help="threshold calibration scan")
    p.add_argument("--annotations-file", type=str, default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="use synthetic annotated graphs (default when no "
                        "--annotations-file is given)")
    p.add_argument("--n-synthetic", type=int, default=40)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--metric", choices=["precision", "recall", "f1"],
                   default="f1")

    p = sub.add_parser("annotate", help="step-level annotation generation")
    p.add_argument("--traces-file", type=str,
                   default="data/raw_cots/gsm8k_50.jsonl")
    p.add_argument("--max-traces", type=int, default=50)

    p = sub.add_parser("correlation",
                       help="performance-correlation benchmark (structure vs "
                            "correctness)")
    add_common(p)
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--max-traces", type=int, default=50)
    p.add_argument("--labels-file", type=str, default=None,
                   help="JSONL {question_id, correct} per trace")
    p.add_argument("--synthetic", action="store_true",
                   help="validate on synthetic graphs with known signal")
    p.add_argument("--n", type=int, default=60,
                   help="synthetic graph count (--synthetic only)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="FDR significance level")
    p.add_argument("--n-bootstrap", type=int, default=0,
                   help=">0 to compute 95%% bootstrap CI on each rho")

    p = sub.add_parser("all", help="extract + analyze + prune in one run")
    add_common(p)
    p.add_argument("--dataset",
                   choices=["synthetic", "gsm8k", "mixed"],
                   default="gsm8k")
    p.add_argument("--max-traces", type=int, default=50)
    p.add_argument("--n", type=int, default=None,
                   help="alias for --max-traces (prune step)")
    p.add_argument("--avg-tokens-per-node", type=int, default=35)
    p.add_argument("--labels-file", type=str, default=None,
                   help="JSONL {question_id, correct} for correlation")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "extract": _run_extract,
        "analyze": _run_analyze,
        "prune": _run_prune,
        "calibrate": _run_calibrate,
        "annotate": _run_annotate,
        "correlation": _run_correlation,
        "all": _run_all,
    }
    handler = handlers[args.command]
    run_dir = make_run_dir(args.command)
    print(f"Run directory: {run_dir}")
    return handler(args, run_dir)


if __name__ == "__main__":
    sys.exit(main())
