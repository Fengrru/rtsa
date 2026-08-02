"""Comprehensive Data Analysis Pipeline for RTSA.

Loads GSM8K CoT traces, runs all extractors (RBE, SBE, DeepSeek),
computes graph features, motif frequencies, TSI/JSD matrices,
and correlates structure with answer correctness.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # for direct imports

import json
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from rtsa.extractors.rule_based import RuleBasedExtractor
from rtsa.extractors.syntax_based import SyntaxBasedExtractor
from rtsa.extractors.random_baseline import RandomBaselineExtractor
from rtsa.extractors.llm_extractor import create_extractor_e7
from rtsa.extractors.gcp_validator import GCPValidator, GCS_CORPUS_FULL, make_gcp_adapter
from rtsa.extractors.synthetic_validation import SyntheticValidator
from rtsa.core.metrics import (
    compute_graph_features_batch, compute_feature_matrix,
    compute_pairwise_tsi,
)
from rtsa.core.motif_matcher import MotifMatcher
from rtsa.core.types import ReasoningTraceGraph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DATA_FILES = {
    "gsm8k": "data/raw_cots/gsm8k_50.jsonl",
    "math": "data/raw_cots/math_100.jsonl",
}


def load_traces(dataset: str = "gsm8k") -> List[Dict]:
    """Load traces from specified dataset."""
    from rtsa.utils.data_loader import load_cot_traces
    path = DATA_FILES.get(dataset, DATA_FILES["gsm8k"])
    return load_cot_traces(path)


def main(max_traces: int = 50, deepseek_n: int = 10, dataset: str = "gsm8k"):
    """Run full data analysis pipeline."""
    start = time.time()
    out = Path("experiments")
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RTSA Data Analysis Pipeline ({dataset.upper()})")
    print(f"{'='*60}")

    traces = load_traces(dataset)
    traces = traces[:max_traces]
    print(f"\n  Loaded {len(traces)} {dataset.upper()} traces")
    lengths = [t["cot_length_tokens"] for t in traces]
    print(f"  Length range: {min(lengths)} - {max(lengths)} tokens")

    # ── 2. Initialize extractors ────────────────────────────────
    rbe = RuleBasedExtractor()
    sbe = SyntaxBasedExtractor()
    rand = RandomBaselineExtractor()

    # DeepSeek (on subset only)
    import os
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        print("  WARNING: DEEPSEEK_API_KEY not set. Skipping DeepSeek extraction.")
        e7 = None
    else:
        e7 = create_extractor_e7(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")

    results = {}

    # ── 3. RBE Extraction ───────────────────────────────────────
    print(f"\n  [1/5] RBE Extraction on {len(traces)} traces...")
    rbe_graphs = []
    for i, t in enumerate(traces):
        g = rbe.extract(t["cot_text"], trace_id=t.get("question_id", f"trace_{i}"),
                         answer=t.get("answer", ""),
                         answer_correct=t.get("answer_correct", True))
        rbe_graphs.append(g)
    print(f"  RBE: {sum(len(g.nodes) for g in rbe_graphs)} total nodes")

    # ── 4. SBE Extraction ───────────────────────────────────────
    print(f"\n  [2/5] SBE Extraction on {len(traces)} traces...")
    sbe_graphs = []
    for i, t in enumerate(traces):
        g = sbe.extract(t["cot_text"], trace_id=t.get("question_id", f"trace_{i}"))
        sbe_graphs.append(g)
    print(f"  SBE: {sum(len(g.nodes) for g in sbe_graphs)} total nodes")

    # ── 5. DeepSeek Extraction (subset) ────────────────────────
    ds_n = min(deepseek_n, len(traces)) if e7 else 0
    ds_graphs = []
    if e7:
        print(f"\n  [3/5] DeepSeek E7 Extraction on {ds_n} traces...")
        for i in range(ds_n):
            t = traces[i]
            try:
                g = e7.extract(t["cot_text"], trace_id=t.get("question_id", f"ds_{i}"))
            except Exception as ex:
                logger.warning(f"DeepSeek failed on trace {i}: {ex}")
                g = ReasoningTraceGraph(trace_id=f"ds_{i}", extractor="deepseek", nodes=[], edges=[])
            ds_graphs.append(g)
            if (i + 1) % 5 == 0:
                print(f"    DeepSeek progress: {i+1}/{ds_n}")
        print(f"  DeepSeek: {sum(len(g.nodes) for g in ds_graphs)} total nodes")
    else:
        print(f"\n  [3/5] DeepSeek E7 Extraction skipped (no API key)")

    # ── 6. Graph Feature Analysis ──────────────────────────────
    print(f"\n  [4/5] Graph Feature Analysis...")

    for name, graphs in [("RBE", rbe_graphs), ("SBE", sbe_graphs), ("DeepSeek", ds_graphs)]:
        if not graphs:
            continue
        metrics = compute_graph_features_batch(graphs)
        feat_mat = compute_feature_matrix(graphs)

        print(f"\n  --- {name} Features ---")
        print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*52}")
        for idx, metric_name in enumerate(["n_nodes", "n_edges", "depth", "branching",
                                            "verify_density", "entropy"]):
            vals = feat_mat[:, idx]
            print(f"  {metric_name:<20} {np.mean(vals):>8.2f} {np.std(vals):>8.2f} "
                  f"{np.min(vals):>8.2f} {np.max(vals):>8.2f}")

        results[name] = {"n_graphs": len(graphs), "metrics": metrics}

    # ── 7. Motif Frequency Analysis ─────────────────────────────
    print(f"\n  [5/5] Motif Frequency & TSI Analysis...")
    matcher = MotifMatcher()

    for name, graphs in [("RBE", rbe_graphs), ("SBE", sbe_graphs), ("DeepSeek", ds_graphs)]:
        if len(graphs) < 2:
            continue

        # Motif frequency matrix
        freq_mat = matcher.compute_motif_frequency_matrix(graphs)
        motif_ids = sorted(matcher._code_motif_graphs.keys())
        mean_freq = np.mean(freq_mat, axis=0)

        print(f"\n  --- {name} Motif Frequencies ---")
        print(f"  {'Motif':<10} {'Desc':<30} {'Mean Freq':>10}")
        print(f"  {'-'*50}")
        for i, mid in enumerate(motif_ids):
            motif = matcher.preset_motifs.get(mid)
            desc = motif.description if motif else mid
            print(f"  {mid:<10} {desc:<30} {mean_freq[i]:>10.3f}")

        # Pairwise TSI
        nx_graphs = [g.to_networkx() for g in graphs]
        tsi_mat, motif_sim_mat, wl_sim_mat = compute_pairwise_tsi(nx_graphs)

        n = len(tsi_mat)
        triu = np.triu_indices(n, k=1)
        mean_tsi = np.mean(tsi_mat[triu]) if n > 1 else 1.0
        mean_motif_sim = np.mean(motif_sim_mat[triu]) if n > 1 else 1.0
        mean_wl_sim = np.mean(wl_sim_mat[triu]) if n > 1 else 1.0

        # Topology Diversity Index
        tdi = mean_tsi
        tdi_label = "LOW diversity (good)" if tdi < 0.5 else (
            "MODERATE diversity" if tdi < 0.8 else "HIGH diversity (collapsing)"
        )

        print(f"\n  --- {name} Pairwise Similarity ---")
        print(f"  Mean TSI:          {mean_tsi:.4f}  ({tdi_label})")
        print(f"  Mean Motif Sim:    {mean_motif_sim:.4f}")
        print(f"  Mean WL Kernel:    {mean_wl_sim:.4f}")
        print(f"  TDI:               {tdi:.4f}")

        results[name]["tsi_mean"] = float(mean_tsi)
        results[name]["tdi"] = float(tdi)
        results[name]["motif_frequencies"] = {
            mid: float(mean_freq[i]) for i, mid in enumerate(motif_ids)
        }

    # ── 8. Answer Correctness Correlation ──────────────────────
    print(f"\n  --- Answer Correctness Correlation ---")
    if len(traces) >= 10:
        answer_correct = [t.get("answer_correct", True) for t in traces]
        correct_ratio = sum(answer_correct) / len(answer_correct)
        print(f"  Correct answers: {correct_ratio:.1%} ({sum(answer_correct)}/{len(answer_correct)})")

        # RBE feature correlation
        rbe_metrics = compute_graph_features_batch(rbe_graphs)
        correct_indices = [i for i, c in enumerate(answer_correct) if c]
        incorrect_indices = [i for i, c in enumerate(answer_correct) if not c]

        if len(correct_indices) > 0 and len(incorrect_indices) > 0:
            print(f"\n  RBE: Correct vs Incorrect feature comparison")
            feat_names = ["n_nodes", "n_edges", "depth", "branching", "verify_density", "entropy"]
            feat_mat = compute_feature_matrix(rbe_graphs)
            correct_feats = feat_mat[correct_indices]
            incorrect_feats = feat_mat[incorrect_indices]

            print(f"  {'Feature':<20} {'Correct':>10} {'Incorrect':>10} {'Diff':>10}")
            print(f"  {'-'*50}")
            for i, fname in enumerate(feat_names):
                c_mean = np.mean(correct_feats[:, i])
                i_mean = np.mean(incorrect_feats[:, i])
                print(f"  {fname:<20} {c_mean:>10.2f} {i_mean:>10.2f} {c_mean - i_mean:>10.2f}")

    # ── 9. Save Results ─────────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_traces": len(traces),
        "deepseek_n": ds_n,
        "results": {k: v for k, v in results.items() if isinstance(v, dict)},
    }
    report_path = out / "data_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")
    print(f"\n  Total time: {time.time() - start:.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RTSA Data Analysis Pipeline")
    parser.add_argument("--max-traces", type=int, default=50, help="Max traces to process")
    parser.add_argument("--deepseek-n", type=int, default=10, help="DeepSeek traces (subset)")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"],
                        help="Dataset to analyze")
    args = parser.parse_args()

    main(max_traces=args.max_traces, deepseek_n=args.deepseek_n, dataset=args.dataset)
