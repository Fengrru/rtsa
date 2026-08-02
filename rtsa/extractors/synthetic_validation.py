"""Synthetic Ground Truth Validation — Fix 8.

Generates CoT traces with KNOWN graph structure, then tests whether
each extractor can recover the injected structure. Directly answers:
"If structure exists, can we measure it?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)


def _build_syn_trace(desc, sentences, edges=None):
    cot = " ".join(s[0] for s in sentences)
    nodes = []
    for i, (sent, ntype) in enumerate(sentences):
        start = cot.find(sent) if sent in cot else 0
        end = start + len(sent)
        nodes.append(GraphNode(id=i + 1, type=ntype, span=(start, end)))
    if edges is None:
        edges = [(i + 1, i + 2) for i in range(len(nodes) - 1)]
    g = ReasoningTraceGraph(
        trace_id=f"syn_{desc}", extractor="ground_truth", domain="synthetic",
        nodes=nodes, edges=edges, metadata={"is_synthetic": True},
    )
    return SyntheticTrace(cot_text=cot, ground_truth=g, description=desc)


@dataclass
class SyntheticTrace:
    cot_text: str
    ground_truth: ReasoningTraceGraph
    description: str


@dataclass
class SyntheticValidationResult:
    extractor_name: str
    n_traces: int
    n_successful_extractions: int
    extraction_rate: float
    node_type_accuracy: float
    node_count_error: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    graph_edit_distance_mean: float
    per_trace_accuracy: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 10 synthetic traces covering diverse structures
# ---------------------------------------------------------------------------

SYNTHETIC_TRACES = [
    _build_syn_trace("chain3", [
        ("According to Pythagoras, a squared plus b squared equals c squared.", NodeType.RETRIEVE),
        ("Substituting a=3 and b=4 gives 9 plus 16 equals c squared which is 25.", NodeType.TRANSFORM),
        ("Check: c equals 5 is the correct result.", NodeType.VERIFY),
    ]),
    _build_syn_trace("branch_two_paths", [
        ("If x is positive, we add 1 to it.", NodeType.BRANCH),
        ("For the positive case, y equals x plus 1.", NodeType.TRANSFORM),
        ("For the negative case, y equals x minus 1.", NodeType.TRANSFORM),
    ], edges=[(1, 2), (1, 3)]),
    _build_syn_trace("backtrack_recover", [
        ("Wait, I made a mistake in the previous step.", NodeType.BACKTRACK),
        ("Recall the correct formula: area equals pi r squared.", NodeType.RETRIEVE),
        ("Plugging in r equals 5 gives area equals 25 pi.", NodeType.TRANSFORM),
    ]),
    _build_syn_trace("compare_approaches", [
        ("Recall method A uses the quadratic formula.", NodeType.RETRIEVE),
        ("Recall method B uses completing the square.", NodeType.RETRIEVE),
        ("Compare: A is faster but B gives more insight.", NodeType.COMPARE),
        ("Using method A, we compute x equals the result of the formula.", NodeType.TRANSFORM),
    ]),
    _build_syn_trace("diamond_verify", [
        ("First, compute the derivative: f prime equals 2x.", NodeType.TRANSFORM),
        ("Second, compute the integral: the antiderivative is x squared plus C.", NodeType.TRANSFORM),
        ("Verify that both approaches are consistent with the fundamental theorem.", NodeType.VERIFY),
    ], edges=[(1, 3), (2, 3)]),
    _build_syn_trace("long_chain", [
        ("According to Newton's second law, F equals ma.", NodeType.RETRIEVE),
        ("Given F equals 10 and m equals 2, we compute a equals 5.", NodeType.TRANSFORM),
        ("Check: 10 equals 2 times 5, consistent with the law.", NodeType.VERIFY),
        ("Now compare this with the alternative approach using energy.", NodeType.COMPARE),
        ("The energy approach gives the same result, confirming our calculation.", NodeType.VERIFY),
    ]),
    _build_syn_trace("nested_branch", [
        ("Consider two cases: n is even or n is odd.", NodeType.BRANCH),
        ("If n is even, then n equals 2k.", NodeType.TRANSFORM),
        ("If n is odd, we further consider whether n equals 4m plus 1 or 4m plus 3.", NodeType.BRANCH),
        ("For n equals 4m plus 1, the result follows from Fermat's theorem.", NodeType.TRANSFORM),
        ("For n equals 4m plus 3, we need a different approach.", NodeType.TRANSFORM),
    ], edges=[(1, 2), (1, 3), (3, 4), (3, 5)]),
    _build_syn_trace("retrieve_transform_repeat", [
        ("From the definition of limits, for any epsilon greater than 0 there exists delta.", NodeType.RETRIEVE),
        ("Setting epsilon to 0.01, we find delta equals 0.001.", NodeType.TRANSFORM),
        ("By the squeeze theorem, the limit must be 0.", NodeType.RETRIEVE),
        ("Therefore, the sequence converges to 0.", NodeType.TRANSFORM),
        ("Verify: the error bound is less than 0.01 for all n greater than 1000.", NodeType.VERIFY),
    ]),
    _build_syn_trace("single_node", [
        ("The answer is 42.", NodeType.TRANSFORM),
    ], edges=[]),
    _build_syn_trace("empty_trace", [], edges=[]),
]


class SyntheticValidator:
    def __init__(self):
        self.traces = SYNTHETIC_TRACES
        logger.info(f"Loaded {len(self.traces)} synthetic traces for validation")

    def validate_extractor(
        self, extractor: Callable[[str], ReasoningTraceGraph], extractor_name: str
    ) -> SyntheticValidationResult:
        n_total = len(self.traces)
        n_success = 0
        node_accuracies, node_count_errors = [], []
        edge_precisions, edge_recalls, geds = [], [], []
        per_trace = {}

        for trace in self.traces:
            if not trace.ground_truth.nodes:
                pred = extractor(trace.cot_text)
                n_success += 1
                per_trace[trace.description] = 1.0 if not pred.nodes else 0.0
                continue
            try:
                pred = extractor(trace.cot_text)
            except Exception as e:
                logger.warning(f"Extractor {extractor_name} failed on {trace.description}: {e}")
                per_trace[trace.description] = 0.0
                continue
            if not pred.nodes:
                per_trace[trace.description] = 0.0
                continue
            n_success += 1

            true_types = [n.type for n in trace.ground_truth.nodes]
            pred_types = [n.type for n in pred.nodes]
            matches = sum(1 for i in range(min(len(true_types), len(pred_types)))
                         if true_types[i] == pred_types[i])
            acc = matches / max(len(true_types), 1)
            node_accuracies.append(acc)
            node_count_errors.append(abs(len(pred.nodes) - len(true_types)) / max(len(true_types), 1))

            true_edges = set(trace.ground_truth.edges)
            pred_edges = set(pred.edges)
            tp = len(true_edges & pred_edges)
            fp = len(pred_edges - true_edges)
            fn = len(true_edges - pred_edges)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            edge_precisions.append(prec)
            edge_recalls.append(rec)
            geds.append(abs(len(pred.nodes) - len(true_types)) + abs(len(pred.edges) - len(true_edges)))
            per_trace[trace.description] = acc

        edge_f1s = [
            2 * p * r / max(p + r, 1e-8) for p, r in zip(edge_precisions, edge_recalls)
        ]

        return SyntheticValidationResult(
            extractor_name=extractor_name,
            n_traces=n_total,
            n_successful_extractions=n_success,
            extraction_rate=n_success / max(n_total, 1),
            node_type_accuracy=float(np.mean(node_accuracies)) if node_accuracies else 0.0,
            node_count_error=float(np.mean(node_count_errors)) if node_count_errors else 0.0,
            edge_precision=float(np.mean(edge_precisions)) if edge_precisions else 0.0,
            edge_recall=float(np.mean(edge_recalls)) if edge_recalls else 0.0,
            edge_f1=float(np.mean(edge_f1s)) if edge_f1s else 0.0,
            graph_edit_distance_mean=float(np.mean(geds)) if geds else 0.0,
            per_trace_accuracy=per_trace,
        )

    def validate_all_extractors(
        self, extractors: Dict[str, Callable[[str], ReasoningTraceGraph]]
    ) -> Dict[str, SyntheticValidationResult]:
        return {name: self.validate_extractor(ext, name) for name, ext in extractors.items()}

    @staticmethod
    def is_extractor_viable(result: SyntheticValidationResult) -> bool:
        return (
            result.extraction_rate >= 0.9
            and result.node_type_accuracy >= 0.70
            and result.edge_f1 >= 0.50
        )
