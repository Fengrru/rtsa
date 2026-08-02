"""
RTSA Demo — Full Pipeline Walkthrough

Shows: extraction → analysis → visualization → comparison

Usage:
    pip install -e .
    python demo.py
"""
import json, os

from rtsa.extractors import create_extractor_deepseek
from rtsa.core.metrics import compute_graph_features, compute_tsi
from rtsa.core.motif_matcher import MotifMatcher
from rtsa.core.ngs_validator import NGSValidator

ext = create_extractor_deepseek()

# ── 1. Extract graphs from two different CoTs ────────────────────────────
cot_verify = (
    "According to the Pythagorean theorem, a squared plus b squared equals c squared. "
    "Substituting a=3 and b=4 gives 9+16=25. "
    "Therefore c=5. "
    "Check: 3^2+4^2=25. This is consistent with the theorem."
)

cot_backtrack = (
    "First, compute the derivative of x squared sin x. "
    "I think the answer is 2x cos x. "
    "No, that is wrong. Let me use the product rule. "
    "Recall: d/dx(u v)=u'v+uv'. "
    "Here u=x squared, v=sin x. So u'=2x, v'=cos x. "
    "Therefore derivative = 2x sin x + x squared cos x. "
    "Check: at x=0, result should be 0, which matches."
)

print("=" * 60)
print("RTSA DEMO: Reasoning Trace Structure Analysis")
print("=" * 60)

for label, cot in [("Verify-only trace", cot_verify), ("Backtrack trace", cot_backtrack)]:
    print(f"\n--- {label} ---")
    g = ext.extract(cot)
    types = [f"{n.id}:{n.type.value}" for n in g.nodes]
    print(f"  Nodes ({len(g.nodes)}): {', '.join(types)}")
    print(f"  Edges ({len(g.edges)}): {g.edges}")

# ── 2. Analyze graph features ────────────────────────────────────────────
g1 = ext.extract(cot_verify)
g2 = ext.extract(cot_backtrack)

print("\n" + "=" * 60)
print("GRAPH FEATURE ANALYSIS")
print("=" * 60)

for label, g in [("Verify-only", g1), ("Backtrack", g2)]:
    metrics = compute_graph_features(g)
    print(f"\n  {label}:")
    print(f"    Nodes: {metrics.n_nodes}, Edges: {metrics.n_edges}")
    print(f"    Depth: {metrics.depth:.2f}, Branching: {metrics.branching:.2f}")
    print(f"    Verify density: {metrics.verify_density:.2f}")

# ── 3. Compare graphs ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GRAPH COMPARISON")
print("=" * 60)

tsi = compute_tsi(g1.to_networkx(), g2.to_networkx())
print(f"\n  TSI:              {tsi.tsi_value:.4f}")
print(f"  Motif similarity: {tsi.motif_similarity:.4f}")
print(f"  WL kernel:        {tsi.wl_similarity:.4f}")
print(f"  Feature sim:      {tsi.feature_similarity:.4f}")

# ── 4. Motif analysis ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MOTIF ANALYSIS")
print("=" * 60)

matcher = MotifMatcher()
for label, g in [("Verify-only", g1), ("Backtrack", g2)]:
    counts = matcher.count_all_motifs(g.to_networkx())
    present = {mid: r.count for mid, r in counts.items() if r.count > 0}
    print(f"\n  {label} motifs: {present}")

# ── 5. Validate ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NGS VALIDATION")
print("=" * 60)

validator = NGSValidator()
for label, g in [("Verify-only", g1), ("Backtrack", g2)]:
    valid, violations = validator.validate(g)
    print(f"\n  {label}: {'PASS' if valid else 'FAIL'} ({len(violations)} warnings)")

# ── 6. JSON export ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("JSON EXPORT (first graph)")
print("=" * 60)
print(json.dumps(g1.to_canonical_dict(), indent=2))
