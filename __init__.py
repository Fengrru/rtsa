"""
RTSA: Reasoning Trace Structure Analysis Toolkit.

Quick start:
    from extractors import create_extractor_deepseek
    ext = create_extractor_deepseek()
    graph = ext.extract("Calculate 5+3=8. Check the result.")
    print(graph.nodes)
"""

__version__ = "3.4.0"

# Convenience re-exports when running from the project directory
from core.types import ReasoningTraceGraph, GraphNode, NodeType, MotifEntry
from core.metrics import compute_graph_features, compute_tsi, compute_pairwise_tsi
from core.motif_matcher import MotifMatcher
from core.ngs_validator import NGSValidator
from core.robust_tsi import UnsupervisedTSI
from extractors import create_extractor_deepseek, RuleBasedExtractor, SyntaxBasedExtractor

__all__ = [
    "ReasoningTraceGraph", "GraphNode", "NodeType", "MotifEntry",
    "compute_graph_features", "compute_tsi", "compute_pairwise_tsi",
    "MotifMatcher", "NGSValidator", "UnsupervisedTSI",
    "create_extractor_deepseek", "RuleBasedExtractor", "SyntaxBasedExtractor",
]
