"""RTSA analysis layer — value-added modules built on core graph primitives.

Provides:
  - prune      : redundancy detection & CoT optimization
  - fingerprint: LLM authorship attribution via graph structure
  - benchmark  : extractor reliability benchmarking (GCP + NGS + TSI)
"""
from extractors.analysis import (
    CostEstimator, CostBreakdown, estimate_project_cost,
    kruskal_wallis_test, bootstrap_ci, cohens_d, partial_correlation,
)

from .prune import (
    PruneConfig,
    RedundancyRegion,
    PruningReport,
    RedundancyAnalyzer,
    prune_graph,
)

from .fingerprint import (
    ModelSignature,
    FingerprintMatchResult,
    ModelFingerprint,
    enroll_model,
    identify_author,
)

from .benchmark import (
    ExtractorScore,
    ExtractorBenchmarkResult,
    BenchmarkReport,
    ExtractorBenchmark,
    benchmark_extractors,
)

__all__ = [
    # Re-exports from extractors.analysis (legacy)
    "CostEstimator", "CostBreakdown", "estimate_project_cost",
    "kruskal_wallis_test", "bootstrap_ci", "cohens_d", "partial_correlation",
    # Prune module
    "PruneConfig",
    "RedundancyRegion",
    "PruningReport",
    "RedundancyAnalyzer",
    "prune_graph",
    # Fingerprint module
    "ModelSignature",
    "FingerprintMatchResult",
    "ModelFingerprint",
    "enroll_model",
    "identify_author",
    # Benchmark module
    "ExtractorScore",
    "ExtractorBenchmarkResult",
    "BenchmarkReport",
    "ExtractorBenchmark",
    "benchmark_extractors",
]
