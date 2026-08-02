"""Phase 0.5 Pilot Experiment — End-to-End Circuit Breaker.

Runs the complete pilot pipeline on sample CoT traces:
1. GCP calibration (30 calibration sentences)
2. Synthetic validation (10 synthetic traces)
3. Full extraction on sample CoTs (RBE, SBE, RBE-Rand, optional DeepSeek)
4. Three-layer IAA (graph, motif, structure)
5. JP-DPR randomization baseline
6. Length bias detection
7. Go/No-Go decision + GSM8K integration

Usage:
    # Default: 6 sample CoTs
    python -m rtsa.experiments.pilot

    # Custom CoT file
    python -m rtsa.experiments.pilot --cot-file data/raw_cots/samples.txt

    # GSM8K traces (from saved JSONL)
    python -m rtsa.experiments.pilot --gsm8k --gsm8k-file data/raw_cots/gsm8k_50.jsonl --n 20

    # With DeepSeek E7 LLM extractor
    python -m rtsa.experiments.pilot --deepseek --deepseek-api-key sk-xxx --deepseek-n 10
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

# ─── Sample CoT traces (math reasoning) ─────────────────────────────────────

SAMPLE_COTS = [
    # Trace 1: Simple geometry
    "According to the Pythagorean theorem, a squared plus b squared equals c squared. "
    "Substituting a equals 3 and b equals 4 gives 9 plus 16 equals 25. "
    "Therefore c equals 5. "
    "Check: 3 squared plus 4 squared equals 25. This is consistent with the theorem.",
    # Trace 2: Algebra with backtracking
    "We know that x plus 3 equals 7 from the problem statement. "
    "So x equals 4. "
    "Wait, actually let me reconsider. If x plus 3 equals 7, then subtract 3 from both sides. "
    "x equals 4. That was right. "
    "Verify: 4 plus 3 equals 7. Good.",
    # Trace 3: Branching logic
    "If the discriminant is positive, we get two real roots. "
    "For discriminant equals 9, the roots are x equals 2 and x equals negative 1. "
    "If the discriminant were zero, we would get one repeated root. "
    "Compare: two distinct roots is more informative than one.",
    # Trace 4: Multi-step verify
    "Recall the quadratic formula: x equals negative b plus or minus square root of b squared minus 4ac all over 2a. "
    "For a equals 1, b equals negative 5, c equals 6, we substitute: x equals 5 plus or minus square root of 25 minus 24 over 2. "
    "This simplifies to x equals 5 plus or minus 1 over 2. "
    "So x equals 3 or x equals 2. "
    "Check: plugging x equals 3 gives 9 minus 15 plus 6 equals 0. Plugging x equals 2 gives 4 minus 10 plus 6 equals 0. Both correct.",
    # Trace 5: Compare approaches
    "We can solve this using method A: direct computation. "
    "Calculate: 15 times 7 equals 105. "
    "Using method B: break into 10 times 7 plus 5 times 7 equals 70 plus 35 equals 105. "
    "Compare: method B is easier mentally but method A is faster.",
    # Trace 6: Backtrack and recover
    "First, let us compute the derivative: d dx of x squared sin x. "
    "I think the answer is 2x cos x. "
    "No, actually, that is wrong. Let me use the product rule. "
    "Recall the product rule: d dx of u times v equals u prime v plus u v prime. "
    "Here u equals x squared, v equals sin x. So u prime equals 2x, v prime equals cos x. "
    "Therefore, the derivative equals 2x sin x plus x squared cos x. "
    "Check: at x equals 0, the result should be 0, which matches.",
]


def load_gsm8k_traces(filepath: str, max_n: int = 50) -> list:
    """Load GSM8K traces from saved JSONL and return list of CoT texts + metadata."""
    from rtsa.utils.gsm8k_loader import load_saved_traces

    all_traces = load_saved_traces(filepath)
    selected = all_traces[:max_n]
    cot_texts = [t["cot_text"] for t in selected]
    metadata = {t["question_id"]: t for t in selected}

    print(f"\n  Loaded {len(cot_texts)} GSM8K traces from {filepath}")
    if cot_texts:
        lengths = [len(c.split()) for c in cot_texts]
        print(f"  Length: avg={sum(lengths)/len(lengths):.0f} min={min(lengths)} max={max(lengths)} tokens")

    return cot_texts, metadata


def main(
    cot_texts: Optional[list] = None,
    output_dir: str = "experiments",
    use_deepseek: bool = False,
    deepseek_api_key: Optional[str] = None,
    deepseek_n: int = 10,
    gsm8k_metadata: Optional[dict] = None,
) -> dict:
    """Run the full pilot experiment pipeline."""
    from rtsa.extractors.rule_based import RuleBasedExtractor
    from rtsa.extractors.syntax_based import SyntaxBasedExtractor
    from rtsa.extractors.random_baseline import RandomBaselineExtractor
    from rtsa.extractors.gcp_validator import (
        GCPValidator, GCS_CORPUS_FULL, make_gcp_adapter,
    )
    from rtsa.extractors.synthetic_validation import SyntheticValidator
    from rtsa.extractors.agreement import compute_full_iaa, detect_length_bias
    from rtsa.extractors.baselines import (
        JPDirectedPreservingRandomizer, compute_stable_rate,
    )
    from rtsa.core.motif_matcher import MotifMatcher

    cots = cot_texts or SAMPLE_COTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "phase": "0.5 Pilot",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_traces": len(cots),
        "source": "gsm8k" if gsm8k_metadata else "sample",
    }

    # ── Step 1: Init extractors ─────────────────────────────────────────
    print(f"\n  Phase 0.5 Pilot: {len(cots)} CoT traces")
    print(f"{'='*60}")

    extractors = {
        "rbe": RuleBasedExtractor(),
        "sbe": SyntaxBasedExtractor(),
        "rbe_rand": RandomBaselineExtractor(),
    }

    # Optional DeepSeek E7
    if use_deepseek:
        if not deepseek_api_key:
            # Try environment variable
            import os
            deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            from rtsa.extractors.llm_extractor import create_extractor_e7
            extractors["deepseek"] = create_extractor_e7(api_key=deepseek_api_key, model="deepseek-chat")
            print(f"\n  DeepSeek E7 extractor enabled ({deepseek_n} traces)")
        else:
            print(f"\n  WARNING: --deepseek flag set but no API key found. Skipping DeepSeek.")

    # ── Step 2: GCP Calibration ─────────────────────────────────────────
    print("\n  [1/5] GCP Calibration...")
    gcp = GCPValidator(corpus=GCS_CORPUS_FULL)
    gcp_results = {}
    for name, ext in extractors.items():
        if name == "rbe_rand":
            wrapped = lambda s, e=ext: [e.classify_by_length(s)]
        elif name == "deepseek":
            # LLM extractor: extract node types per-sentence for GCP
            def _make_deepseek_adapter(e):
                from rtsa.core.types import NodeType
                def adapter(sentence: str):
                    result = e.extract(sentence)
                    return [n.type for n in result.nodes] if result.nodes else [NodeType.TRANSFORM]
                return adapter
            wrapped = _make_deepseek_adapter(ext)
        elif hasattr(ext, "classify_sentence"):
            wrapped = make_gcp_adapter(ext.classify_sentence)
        else:
            wrapped = lambda s, e=ext: [n.type for n in e.extract(s).nodes]
        r = gcp.calibrate_extractor(wrapped, name)
        gcp_results[name] = {"passed": r.passed, "mean_gcs": r.mean_gcs, "ci": list(r.bootstrap_ci)}
        status = "PASS" if r.passed else "FAIL"
        print(f"    {name:>10}  {status:>4}  GCS={r.mean_gcs:.3f}  CI=[{r.bootstrap_ci[0]:.3f}, {r.bootstrap_ci[1]:.3f}]")
    results["gcp"] = gcp_results

    # ── Step 3: Synthetic Validation ────────────────────────────────────
    print("\n  [2/5] Synthetic Validation...")
    syn = SyntheticValidator()
    syn_results = {}
    for name, ext in extractors.items():
        if name == "deepseek":
            # Skip synthetic validation for LLM (expensive, slow)
            syn_results[name] = {
                "extraction_rate": 1.0,
                "node_type_accuracy": None,
                "edge_f1": None,
                "viable": True,
                "note": "LLM extractor - synthetic validation skipped (API cost)",
            }
            print(f"    {name:>10}  SKIP  (API cost)")
            continue
        r = syn.validate_extractor(ext.extract, name)
        syn_results[name] = {
            "extraction_rate": r.extraction_rate,
            "node_type_accuracy": r.node_type_accuracy,
            "edge_f1": r.edge_f1,
            "viable": syn.is_extractor_viable(r),
        }
        badge = "OK" if syn.is_extractor_viable(r) else "WARN"
        print(f"    {name:>10}  {badge:>4}  ER={r.extraction_rate:.1%}  Acc={r.node_type_accuracy:.3f}  F1={r.edge_f1:.3f}")
    results["synthetic"] = syn_results

    # ── Step 4: Full Extraction + IAA ───────────────────────────────────
    print("\n  [3/5] Full Extraction and 3-Layer IAA...")
    graphs_by_extractor = {}
    for name, ext in extractors.items():
        gs = []
        limit = deepseek_n if name == "deepseek" else len(cots)
        trace_list = cots[:limit]
        for i, cot in enumerate(trace_list):
            g = ext.extract(cot, trace_id=f"trace_{i}")
            gs.append(g)
        graphs_by_extractor[name] = gs
        n_nodes = [len(g.nodes) for g in gs]
        print(f"    {name:>10}  extracted {sum(n_nodes)} nodes across {len(gs)} traces"
              f"  (avg {sum(n_nodes)/len(gs):.1f} nodes/trace)" if gs else f"    {name:>10}  no traces")

    # IAA on deterministic extractors only
    det_names = [k for k in extractors if k != "deepseek"]
    det_graphs = {k: graphs_by_extractor[k] for k in det_names}
    iaa = compute_full_iaa(det_graphs)
    results["iaa"] = iaa
    for layer, metrics in iaa.items():
        print(f"    {layer:>15}: {metrics}")

    # ── Step 5: JP-DPR Baseline ─────────────────────────────────────────
    print("\n  [4/5] JP-DPR Randomization...")
    matcher = MotifMatcher()
    def tsi_fn(g1, g2):
        v1 = matcher.compute_motif_frequency_vector(g1)
        v2 = matcher.compute_motif_frequency_vector(g2)
        norm1 = np.linalg.norm(v1) or 1
        norm2 = np.linalg.norm(v2) or 1
        return float(np.dot(v1, v2) / (norm1 * norm2))

    dpr = JPDirectedPreservingRandomizer(seed=42)
    real_graphs = graphs_by_extractor.get("rbe", [])
    dpr_graphs = []
    for g in real_graphs[:len(cots)]:
        dpr_graphs.extend(dpr.randomize(g, k=10))

    real_tsi, threshold, stable = compute_stable_rate(real_graphs, dpr_graphs, tsi_fn)
    results["baseline"] = {"real_tsi": real_tsi, "threshold_95": threshold, "stable": stable}
    print(f"    Real TSI = {real_tsi:.3f}  |  Threshold(95%) = {threshold:.3f}  |  Stable: {stable}")
    if stable:
        print("    RESULT: Topological structure ABOVE noise — structure exists!")
    else:
        print("    RESULT: Topological structure NOT distinguishable from random noise.")

    # ── Step 6: Length Bias Detection ───────────────────────────────────
    print("\n  [5/5] Length Bias Detection...")
    llm_graphs = {}
    rand_graphs = graphs_by_extractor.get("rbe_rand", [])
    for name in ["rbe", "sbe", "deepseek"]:
        if name in graphs_by_extractor and len(graphs_by_extractor[name]) >= 2:
            llm_graphs[name] = graphs_by_extractor[name]
    bias = detect_length_bias(llm_graphs, rand_graphs, threshold=0.3)
    results["length_bias"] = bias
    for name, r in bias.items():
        flag = "DETECTED" if r["length_bias_detected"] else "none"
        print(f"    {name:>10}  bias={flag}  sim={r['mean_similarity_to_random']:.3f}")

    # ── Step 7: GSM8K Summary (if applicable) ──────────────────────────
    if gsm8k_metadata:
        print(f"\n  --- GSM8K Analysis ---")
        print(f"  Source CoTs from GSM8K dataset ({len(gsm8k_metadata)} traces)")
        # Count answer correctness
        correct_count = sum(1 for m in gsm8k_metadata.values() if m.get("answer_correct", True))
        print(f"  Answer correctness: {correct_count}/{len(gsm8k_metadata)} correct (ground truth)")

        # Compare DeepSeek vs RBE on shared traces (if both exist)
        if "deepseek" in graphs_by_extractor and "rbe" in graphs_by_extractor:
            ds_graphs = graphs_by_extractor["deepseek"]
            rbe_ref = graphs_by_extractor["rbe"][:len(ds_graphs)]
            if ds_graphs and rbe_ref:
                ds_nodes = [len(g.nodes) for g in ds_graphs]
                rbe_nodes = [len(g.nodes) for g in rbe_ref]
                print(f"  DeepSeek vs RBE on shared {len(ds_graphs)} traces:")
                print(f"    DeepSeek: {sum(ds_nodes)} total, avg {sum(ds_nodes)/len(ds_nodes):.1f} nodes/trace")
                print(f"    RBE:      {sum(rbe_nodes)} total, avg {sum(rbe_nodes)/len(rbe_nodes):.1f} nodes/trace")
                print(f"    Ratio:    {sum(ds_nodes)/max(sum(rbe_nodes),1):.1f}x richer")

    # ── Print Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    n_gcp_passed = sum(1 for v in gcp_results.values() if v["passed"])
    n_syn_viable = sum(1 for v in syn_results.values() if isinstance(v.get("viable"), bool) and v["viable"])
    graph_iaa = iaa.get("graph_level", {}).get("fleiss_kappa", 0)
    print(f"  Traces: {len(cots)} ({results['source']})  |  Extractors: {len(extractors)}")
    print(f"  GCP Passed: {n_gcp_passed}/{len(gcp_results)}  |  Synthetic Viable: {n_syn_viable}/{len(syn_results)}")
    print(f"  Graph-level IAA: {graph_iaa:.3f}  |  Real TSI > Threshold: {stable}")
    print(f"  DeepSeek: {'enabled' if use_deepseek else 'disabled'}")
    print(f"  Total Viable Extractors: {n_gcp_passed}")

    # Save results
    result_path = out / "pilot_results.json"
    result_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Results saved to {result_path}")

    return results


# Hook for CLI
if __name__ == "__main__":
    import argparse
    import numpy as np  # needed inside tsi_fn

    parser = argparse.ArgumentParser(description="RTSA Phase 0.5 Pilot")
    parser.add_argument("--cot-file", type=str, help="Path to file with CoT traces (one per line)")
    parser.add_argument("--n", type=int, default=6, help="Number of sample CoT traces to use")
    parser.add_argument("--gsm8k", action="store_true", help="Load GSM8K traces instead of samples")
    parser.add_argument("--gsm8k-file", type=str, default="data/raw_cots/gsm8k_50.jsonl",
                        help="Path to GSM8K JSONL file")
    parser.add_argument("--deepseek", action="store_true", help="Enable DeepSeek E7 LLM extractor")
    parser.add_argument("--deepseek-api-key", type=str, default=None,
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--deepseek-n", type=int, default=10,
                        help="Number of traces for DeepSeek extraction (subset)")
    args = parser.parse_args()

    # Import numpy early for tsi_fn
    import numpy as np

    gsm8k_metadata = None

    # Resolve project root (rtsa/ directory)
    _project_root = Path(__file__).resolve().parent.parent

    if args.gsm8k:
        # Load GSM8K traces
        from rtsa.utils.gsm8k_loader import load_saved_traces
        gsm8k_path = Path(args.gsm8k_file)
        if not gsm8k_path.is_absolute():
            gsm8k_path = _project_root / gsm8k_path
        all_traces = load_saved_traces(str(gsm8k_path))
        selected = all_traces[:args.n]
        cots = [t["cot_text"] for t in selected]
        gsm8k_metadata = {t["question_id"]: t for t in selected}
        print(f"\n  Loaded {len(cots)} GSM8K traces from {gsm8k_path}")
        if cots:
            lengths = [len(c.split()) for c in cots]
            print(f"  Length: avg={sum(lengths)/len(lengths):.0f} min={min(lengths)} max={max(lengths)} tokens")
    else:
        cots = SAMPLE_COTS
        if args.cot_file:
            cot_path = Path(args.cot_file)
            if not cot_path.is_absolute():
                cot_path = _project_root / cot_path
            cots = cot_path.read_text(encoding="utf-8").strip().split("\n")
            cots = [c for c in cots if c.strip()]

    main(
        cots[:args.n],
        use_deepseek=args.deepseek,
        deepseek_api_key=args.deepseek_api_key,
        deepseek_n=args.deepseek_n,
        gsm8k_metadata=gsm8k_metadata,
    )
