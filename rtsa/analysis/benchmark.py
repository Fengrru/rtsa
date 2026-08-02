"""
Extractor Reliability Benchmark — Direction A (unique differentiator).

Provides a unified benchmarking suite that calibrates and compares
reasoning-trace extractors across three dimensions:

1. Granularity Calibration Protocol (GCP) — precision at sentence level
2. NGS Structural Compliance — validity of produced graphs
3. Cross-Extractor Consistency — Topological Similarity Index (TSI)

This is the feature ReasoningFlow will never build (they don't evaluate
themselves), making it the ideal flagship command for RTSA-as-library.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from rtsa.core.types import NodeType, ReasoningTraceGraph
from rtsa.core.metrics import compute_tsi
from rtsa.core.ngs_validator import NGSValidator, NGSViolation
from rtsa.extractors.gcp_validator import GCPValidator, GCPResult, make_gcp_adapter
from rtsa.extractors.inter_annotator import InterAnnotatorAgreement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractorScore:
    """Scores for a single extractor on a single benchmark dimension."""
    dimension: str
    score: float
    passed: bool
    details: Dict = field(default_factory=dict)


@dataclass
class ExtractorBenchmarkResult:
    """Complete benchmark result for one extractor."""
    extractor_name: str
    n_samples: int
    gcp: Optional[GCPResult] = None
    ngs_pass_rate: float = 0.0
    ngs_violations: List[NGSViolation] = field(default_factory=list)
    mean_tsi_vs_others: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    plan_redundancy_score: float = 0.0  # reasonplan insight: wasteful planning structure

    def summary(self) -> str:
        lines = [
            f"BenchmarkResult: {self.extractor_name}",
            f"  Samples: {self.n_samples}",
        ]
        if self.gcp:
            lines.append(
                f"  GCP: mean={self.gcp.mean_gcs:.3f} min={self.gcp.min_gcs:.3f} "
                f"passed={self.gcp.passed}"
            )
        lines.append(f"  NGS pass rate: {self.ngs_pass_rate:.1%}")
        lines.append(f"  Mean TSI vs peers: {self.mean_tsi_vs_others:.3f}")
        lines.append(f"  Overall score: {self.overall_score:.3f}  passed={self.passed}")
        if self.plan_redundancy_score > 0.05:
            lines.append(
                f"  Plan redundancy: {self.plan_redundancy_score:.3f}  "
                f"(reasonplan insight: excessive planning structure hurts efficiency)"
            )
        return "\n".join(lines)


@dataclass
class BenchmarkReport:
    """Aggregate report across all extractors."""
    results: Dict[str, ExtractorBenchmarkResult]
    winner: Optional[str] = None
    ranking: List[Tuple[str, float]] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        serializable = {
            "winner": self.winner,
            "ranking": self.ranking,
            "extractors": {
                name: {
                    "n_samples": r.n_samples,
                    "gcp_mean": r.gcp.mean_gcs if r.gcp else None,
                    "gcp_passed": r.gcp.passed if r.gcp else None,
                    "ngs_pass_rate": r.ngs_pass_rate,
                    "mean_tsi": r.mean_tsi_vs_others,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                }
                for name, r in self.results.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Benchmark report saved to {path}")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RTSA Extractor Reliability Benchmark Report",
            "=" * 60,
        ]
        for rank, (name, score) in enumerate(self.ranking, 1):
            marker = " ★" if name == self.winner else ""
            lines.append(f"#{rank}  {name}: {score:.3f}{marker}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------

class ExtractorBenchmark:
    """Run the full RTSA extractor reliability benchmark.

    Usage:
        bench = ExtractorBenchmark()
        report = bench.run({
            "rbe": rule_based_extractor,
            "sbe": syntax_based_extractor,
            "llm": llm_extractor,
        }, graphs={...})
        print(report.summary())
    """

    def __init__(
        self,
        gcp_validator: Optional[GCPValidator] = None,
        ngs_validator: Optional[NGSValidator] = None,
    ):
        self.gcp = gcp_validator or GCPValidator()
        self.ngs = ngs_validator or NGSValidator(strict=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        extractors: Dict[str, Callable[[str], ReasoningTraceGraph]],
        sample_texts: Optional[List[str]] = None,
        graphs: Optional[Dict[str, List[ReasoningTraceGraph]]] = None,
        run_gcp: bool = True,
        run_ngs: bool = True,
        run_tsi: bool = True,
    ) -> BenchmarkReport:
        """Run benchmark suite.

        Args:
            extractors: mapping name → extractor function
            sample_texts: texts for GCP calibration (sentence-level)
            graphs: pre-extracted graphs per extractor for NGS/TSI
            run_gcp: whether to run Granularity Calibration Protocol
            run_ngs: whether to run NGS structural validation
            run_tsi: whether to compute cross-extractor TSI

        Returns:
            BenchmarkReport with aggregated scores and ranking
        """
        results: Dict[str, ExtractorBenchmarkResult] = {}

        for name, extractor in extractors.items():
            logger.info(f"Benchmarking extractor: {name}")
            result = ExtractorBenchmarkResult(extractor_name=name, n_samples=0)

            # --- GCP (sentence-level calibration) ---
            if run_gcp and sample_texts:
                # Wrap extractor for GCP (sentence → List[NodeType])
                def sentence_adapter(sentence: str) -> List[NodeType]:
                    # Try to extract a mini-graph from a single sentence
                    # and return its node-type sequence
                    try:
                        mini_graph = extractor(sentence)
                        return [n.type for n in sorted(mini_graph.nodes, key=lambda n: n.id)]
                    except Exception:
                        return []

                gcp_result = self.gcp.calibrate_extractor(
                    make_gcp_adapter(sentence_adapter),
                    extractor_name=name,
                )
                result.gcp = gcp_result

            # --- NGS (graph-level structural compliance) ---
            if run_ngs and graphs and name in graphs:
                extractor_graphs = graphs[name]
                result.n_samples = len(extractor_graphs)
                valid_count = 0
                all_violations: List[NGSViolation] = []
                for g in extractor_graphs:
                    is_valid, violations = self.ngs.validate(g)
                    if is_valid:
                        valid_count += 1
                    all_violations.extend(violations)
                result.ngs_pass_rate = valid_count / max(len(extractor_graphs), 1)
                result.ngs_violations = all_violations
                result.plan_redundancy_score = self._compute_plan_redundancy(extractor_graphs)

            results[name] = result

        # --- TSI (cross-extractor consistency) ---
        if run_tsi and graphs:
            self._compute_tsi_matrix(results, graphs)

        # --- Aggregate scoring ---
        self._compute_overall_scores(results)

        # --- Ranking ---
        ranking = sorted(
            ((name, r.overall_score) for name, r in results.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        winner = ranking[0][0] if ranking else None

        report = BenchmarkReport(results=results, winner=winner, ranking=ranking)
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # TSI computation
    # ------------------------------------------------------------------

    def _compute_tsi_matrix(
        self,
        results: Dict[str, ExtractorBenchmarkResult],
        graphs: Dict[str, List[ReasoningTraceGraph]],
    ) -> None:
        """Compute pairwise mean TSI between extractors."""
        names = list(results.keys())
        if len(names) < 2:
            return

        # Build per-extractor graph lists (must be same-length for fair comparison)
        lens = [len(graphs[n]) for n in names if n in graphs]
        if not lens:
            return
        min_len = min(lens)
        if min_len == 0:
            return

        for name in names:
            if name not in graphs:
                continue
            my_graphs = graphs[name][:min_len]
            tsi_scores: List[float] = []
            for other in names:
                if other == name or other not in graphs:
                    continue
                other_graphs = graphs[other][:min_len]
                for g1, g2 in zip(my_graphs, other_graphs):
                    try:
                        tsi = compute_tsi(g1.to_networkx(), g2.to_networkx())
                        tsi_scores.append(tsi.tsi_value)
                    except Exception:
                        pass
            if tsi_scores:
                results[name].mean_tsi_vs_others = float(np.mean(tsi_scores))

    # ------------------------------------------------------------------
    # Plan-redundancy insight (reasonplan negative result)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_plan_redundancy(graphs: List[ReasoningTraceGraph]) -> float:
        """Measure planning-structure bloat (Branch/Backtrack without productive children).

        Inspired by reasonplan negative results: excessive planning overhead
        in reasoning traces degrades effective output quality.
        """
        if not graphs:
            return 0.0
        scores = []
        for g in graphs:
            n = len(g.nodes)
            if n == 0:
                continue
            plan_nodes = [node for node in g.nodes if node.type.value in ("Branch", "Backtrack")]
            if not plan_nodes:
                scores.append(0.0)
                continue
            wasteful = 0
            for node in plan_nodes:
                children = [e[1] for e in g.edges if e[0] == node.id]
                child_types = {c.type.value for c in g.nodes if c.id in children}
                if not child_types.intersection({"Transform", "Retrieve", "Compare"}):
                    wasteful += 1
            scores.append(wasteful / n)
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Overall scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_scores(
        results: Dict[str, ExtractorBenchmarkResult],
    ) -> None:
        """Aggregate per-extractor scores into a single 0-1 score.

        Weights:
            GCP passed & mean   → 40%
            NGS pass rate       → 35%
            TSI consistency     → 25%
        """
        for r in results.values():
            scores = []
            weights = []

            if r.gcp is not None:
                gcp_score = r.gcp.mean_gcs if r.gcp.passed else r.gcp.mean_gcs * 0.5
                scores.append(gcp_score)
                weights.append(0.40)

            if r.n_samples > 0:
                scores.append(r.ngs_pass_rate)
                weights.append(0.35)

                if r.mean_tsi_vs_others > 0:
                    scores.append(r.mean_tsi_vs_others)
                    weights.append(0.25)

            if scores:
                r.overall_score = float(np.average(scores, weights=weights))
            else:
                r.overall_score = 0.0

            # Pass threshold: overall >= 0.70 AND NGS pass rate >= 0.60
            r.passed = r.overall_score >= 0.70 and r.ngs_pass_rate >= 0.60


# ---------------------------------------------------------------------------
# Convenience one-liner
# ---------------------------------------------------------------------------

def benchmark_extractors(
    extractors: Dict[str, Callable[[str], ReasoningTraceGraph]],
    sample_texts: Optional[List[str]] = None,
    graphs: Optional[Dict[str, List[ReasoningTraceGraph]]] = None,
    output_path: Optional[str] = None,
) -> BenchmarkReport:
    """Run full benchmark and optionally save report to JSON."""
    bench = ExtractorBenchmark()
    report = bench.run(extractors, sample_texts=sample_texts, graphs=graphs)
    if output_path:
        report.to_json(output_path)
    return report
