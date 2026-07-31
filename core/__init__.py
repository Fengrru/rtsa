"""Core module: graph schema, NGS validation, motif matching, TSI metrics."""

from .types import (
    GraphNode,
    MotifEntry,
    MOTIF_CATALOG,
    MOTIF_LOOKUP,
    NodeType,
    ReasoningTraceGraph,
)
from .ngs_validator import NGSValidator, NGSViolation
from .motif_matcher import MotifMatcher, MotifMatchResult
from .robust_tsi import RobustTSI, UnsupervisedTSI
from .metrics import GraphMetrics, compute_graph_features

__all__ = [
    "GraphNode",
    "NodeType",
    "ReasoningTraceGraph",
    "MotifEntry",
    "MOTIF_CATALOG",
    "MOTIF_LOOKUP",
    "NGSValidator",
    "NGSViolation",
    "MotifMatcher",
    "MotifMatchResult",
    "RobustTSI",
    "UnsupervisedTSI",
    "GraphMetrics",
    "compute_graph_features",
]
