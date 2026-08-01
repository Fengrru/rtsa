"""Performance-Correlation Benchmark.

Systematic, reproducible evaluation of how graph-level structural metrics
correlate with reasoning performance (correctness or a continuous score).
This mirrors the core contribution of LLM-MindMap (EMNLP 2025) — structure
predicts performance — and turns it into a formal benchmark with proper
multiple-comparison control and uncertainty estimates.

Metric families
---------------
1. ``global``   — 10 GraphMetrics fields (size, depth, branching, verify
   density, backtrack rate, entropy, degree, density, extraction rate);
2. ``type_mix`` — fraction of nodes of each NodeType (6 features);
3. ``shape``    — max out-degree, number of leaf nodes, mean step length.

Statistics
----------
- Spearman rho and p-value per metric;
- Benjamini-Hochberg FDR correction across all metrics (default alpha 0.05);
- optional bootstrap confidence interval on rho (``n_bootstrap > 0``);
- binary labels (correct/incorrect) or continuous performance scores.

Output
------
``PerformanceCorrelationReport`` with ``to_dict()`` (JSON-serializable),
``to_markdown()`` (paper-ready table) and ``summary()`` (console text).

Label sources are handled by the caller; ``synthetic_performance_graphs``
provides a validated pipeline check (redundancy drives a known score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import spearmanr

from core.metrics import compute_graph_features_batch
from core.types import GraphNode, NodeType, ReasoningTraceGraph

Label = Union[bool, float]
Labels = Dict[str, Label]


# ---------------------------------------------------------------------------
# Metric specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """One benchmark metric: how to compute it and which family it belongs to."""

    name: str
    family: str
    field: Optional[str] = None          # GraphMetrics attribute (global family)
    node_type: Optional[NodeType] = None  # NodeType fraction (type_mix family)

    @property
    def label(self) -> str:
        return f"{self.name} [{self.family}]"


def build_metric_specs() -> List[MetricSpec]:
    """All metrics measured by the benchmark, grouped by family."""
    specs: List[MetricSpec] = []
    for f in (
        "n_nodes", "n_edges", "depth", "branching", "verify_density",
        "backtrack_rate", "entropy", "avg_degree", "graph_density",
        "extraction_rate",
    ):
        specs.append(MetricSpec(name=f, family="global", field=f))
    for t in NodeType:
        specs.append(MetricSpec(
            name=f"{t.value.lower()}_ratio", family="type_mix", node_type=t,
        ))
    specs += [
        MetricSpec(name="max_out_degree", family="shape"),
        MetricSpec(name="n_leaf_nodes", family="shape"),
        MetricSpec(name="mean_text_len", family="shape"),
    ]
    return specs


def _type_mix_row(graph: ReasoningTraceGraph,
                  types: Sequence[NodeType]) -> List[float]:
    counts = {t: 0 for t in types}
    for n in graph.nodes:
        counts[n.type] = counts.get(n.type, 0) + 1
    total = max(sum(counts.values()), 1)
    return [counts[t] / total for t in types]


def _shape_row(graph: ReasoningTraceGraph) -> Tuple[float, float, float]:
    out_deg = {n.id: 0 for n in graph.nodes}
    for u, v in graph.edges:
        out_deg[u] = out_deg.get(u, 0) + 1
    max_out = max(out_deg.values()) if out_deg else 0.0
    n_leaves = sum(1 for d in out_deg.values() if d == 0)
    mean_len = float(np.mean([len(n.text.split()) for n in graph.nodes])
                     ) if graph.nodes else 0.0
    return float(max_out), float(n_leaves), mean_len


def performance_metric_matrix(
    graphs: List[ReasoningTraceGraph],
    specs: Optional[List[MetricSpec]] = None,
) -> Tuple[np.ndarray, List[MetricSpec]]:
    """N x M feature matrix where M = len(specs) (default: full benchmark)."""
    specs = specs or build_metric_specs()
    metrics = compute_graph_features_batch(graphs)
    rows = []
    for g, m in zip(graphs, metrics):
        row: List[float] = []
        for spec in specs:
            if spec.family == "type_mix":
                row.append(_type_mix_row(g, [spec.node_type])[0])
            elif spec.family == "shape":
                s_max_out, s_leaves, s_len = _shape_row(g)
                row.append({"max_out_degree": s_max_out,
                            "n_leaf_nodes": s_leaves,
                            "mean_text_len": s_len}[spec.name])
            else:
                row.append(float(getattr(m, spec.field)))
        rows.append(row)
    return np.asarray(rows, dtype=float), specs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (no external dependency)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    return np.clip(q, 0.0, 1.0)


def bootstrap_rho_ci(x: np.ndarray, y: np.ndarray,
                     n_bootstrap: int = 500, seed: int = 42,
                     min_valid: int = 50) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for Spearman rho.

    Returns ``(nan, nan)`` when fewer than ``min_valid`` resamples produced a
    finite rho (e.g. constant columns).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = np.arange(n)
    rhos = []
    for _ in range(n_bootstrap):
        s = rng.choice(idx, size=n, replace=True)
        if np.std(x[s]) == 0.0 or np.std(y[s]) == 0.0:
            continue
        r, _ = spearmanr(x[s], y[s])
        if np.isfinite(r):
            rhos.append(float(r))
    if len(rhos) < min_valid:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class MetricCorrelation:
    """Correlation of a single metric with the performance signal."""

    metric: str
    family: str
    spearman_rho: float
    p_value: float
    q_value: float                    # BH-FDR adjusted
    significant: bool
    direction: str                    # positive | negative | none
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "family": self.family,
            "spearman_rho": self.spearman_rho,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "significant": self.significant,
            "direction": self.direction,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
        }


@dataclass
class PerformanceCorrelationReport:
    """Full benchmark report: per-metric correlations plus context."""

    n_graphs: int
    n_incorrect: int
    label_type: str                    # binary | continuous
    alpha: float
    correlations: List[MetricCorrelation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_graphs": self.n_graphs,
            "n_incorrect": self.n_incorrect,
            "label_type": self.label_type,
            "alpha": self.alpha,
            "correlations": [c.to_dict() for c in self.correlations],
        }

    def to_markdown(self) -> str:
        """Paper-ready table of the benchmark results."""
        lines = [
            "| Metric | Family | Spearman rho | p-value | q (FDR) | Sig | Direction |",
            "|---|---|---:|---:|---:|:---:|---|",
        ]
        for c in self.correlations:
            sig = "*" if c.significant else ""
            lines.append(
                f"| {c.metric} | {c.family} | {c.spearman_rho:.3f} | "
                f"{c.p_value:.4f} | {c.q_value:.4f} | {sig} | {c.direction} |"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        sig = [c for c in self.correlations if c.significant]
        lines = [
            f"Performance-correlation benchmark: {self.n_graphs} graphs, "
            f"{self.n_incorrect} incorrect, {self.label_type} labels, "
            f"alpha={self.alpha}",
            f"  metrics: {len(self.correlations)} | significant: "
            f"{len(sig)} (BH-FDR)",
        ]
        for c in sig:
            ci = (f" 95% CI [{c.ci_lo:.3f}, {c.ci_hi:.3f}]"
                  if c.ci_lo is not None else "")
            lines.append(
                f"  * {c.metric:<22} rho={c.spearman_rho:+.3f} "
                f"q={c.q_value:.4f} ({c.direction}){ci}"
            )
        if not sig:
            lines.append("  (no metric survives FDR correction)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_performance_correlation(
    graphs: List[ReasoningTraceGraph],
    labels: Labels,
    alpha: float = 0.05,
    n_bootstrap: int = 0,
    seed: int = 42,
    default_label: Label = True,
) -> PerformanceCorrelationReport:
    """Run the full benchmark: matrix -> Spearman -> FDR -> optional CI.

    Raises ``ValueError`` when the label column has zero variance (e.g. all
    traces correct) — correlations would be meaningless.
    """
    if not graphs:
        raise ValueError("No graphs provided")
    mat, specs = performance_metric_matrix(graphs)
    y = np.asarray(
        [float(labels.get(g.trace_id, default_label)) for g in graphs],
        dtype=float,
    )
    if float(np.std(y)) == 0.0:
        raise ValueError(
            "Labels have zero variance (e.g. all traces correct). Provide "
            "labels with a mix of outcomes or use synthetic data."
        )
    label_type = "binary" if set(np.unique(y)) <= {0.0, 1.0} else "continuous"

    p_values = []
    rhos = []
    for i in range(mat.shape[1]):
        col = mat[:, i]
        if np.std(col) == 0.0:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(col, y)
            rho, p = float(rho), float(p)
        rhos.append(rho)
        p_values.append(p if np.isfinite(p) else 1.0)

    q_values = _bh_fdr(p_values)
    correlations = []
    for spec, rho, p, q in zip(specs, rhos, p_values, q_values):
        finite = np.isfinite(rho)
        direction = "none"
        if finite and abs(rho) >= 1e-9:
            direction = "positive" if rho > 0 else "negative"
        ci = None
        if n_bootstrap > 0 and finite:
            ci = bootstrap_rho_ci(
                mat[:, specs.index(spec)], y,
                n_bootstrap=n_bootstrap, seed=seed,
            )
        correlations.append(MetricCorrelation(
            metric=spec.name,
            family=spec.family,
            spearman_rho=rho,
            p_value=p,
            q_value=float(q),
            significant=bool(finite and q < alpha),
            direction=direction,
            ci_lo=ci[0] if ci else None,
            ci_hi=ci[1] if ci else None,
        ))
    correlations.sort(key=lambda c: abs(c.spearman_rho), reverse=True)

    n_incorrect = int(np.sum(y == 0.0)) if label_type == "binary" else -1
    return PerformanceCorrelationReport(
        n_graphs=len(graphs),
        n_incorrect=n_incorrect,
        label_type=label_type,
        alpha=alpha,
        correlations=correlations,
    )


# ---------------------------------------------------------------------------
# Synthetic validation set
# ---------------------------------------------------------------------------

def synthetic_performance_graphs(n: int = 60, seed: int = 42,
                                 ) -> List[ReasoningTraceGraph]:
    """Graphs whose redundancy level drives a known performance score.

    Higher redundancy -> more Verify/Backtrack nodes and longer chains ->
    lower ``metadata["performance"]`` (continuous, noisy) and
    ``metadata["correct"]`` (binary threshold at 0.5). Used to validate the
    pipeline: verify_density / backtrack_rate / n_nodes should correlate
    negatively with performance.
    """
    rng = np.random.RandomState(seed)
    graphs = []
    for i in range(n):
        redundancy = rng.uniform(0.0, 1.0)
        base = max(3, int(4 + 8 * redundancy))
        n_verify = int(base * redundancy)
        n_backtrack = int(base * redundancy * 0.3)
        nodes: List[GraphNode] = []
        edges: List[Tuple[int, int]] = []
        prev = 1
        for k in range(base):
            if k < n_backtrack and k > 0:
                ntype = "Backtrack"
            elif k >= base - n_verify:
                ntype = "Verify"
            else:
                ntype = "Transform"
            nodes.append(GraphNode(
                id=k + 1, type=NodeType.from_string(ntype),
                text=f"step {k}",
            ))
            if k > 0:
                edges.append((k, k + 1))
        score = float(np.clip(1.0 - redundancy + rng.normal(0.0, 0.05),
                              0.0, 1.0))
        graphs.append(ReasoningTraceGraph(
            trace_id=f"syn_{i:03d}",
            nodes=nodes,
            edges=edges,
            domain="synthetic",
            metadata={"performance": score, "correct": score >= 0.5},
        ))
    return graphs
