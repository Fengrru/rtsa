"""
Random Baseline Extractor (RBE-Rand) — Extractor E3.

Zero-information anchor: classifies sentences by length only.
If ANY LLM extractor correlates with RBE-Rand (IAA > 0.3),
that indicates length bias in the LLM-based extraction.
"""

from __future__ import annotations

import hashlib
import random
from typing import List

from core.types import GraphNode, NodeType, ReasoningTraceGraph


class RandomBaselineExtractor:
    """
    Assigns node types based SOLELY on sentence length.

    Rules (Section 3.4):
        length <= 5   → Retrieve
        length <= 12  → Transform
        length <= 20  → Verify
        length > 20   → Backtrack

    This is a ZERO-INFORMATION extractor. Any correlation with it
    indicates length confound in the task.
    """

    def __init__(self, name: str = "rbe_rand", seed: int = 42):
        self.name = name
        self.seed = seed
        self._rng = random.Random(seed)

    @staticmethod
    def classify_by_length(sentence: str) -> NodeType:
        """Classify a sentence purely by its word count."""
        length = len(sentence.split())
        if length <= 5:
            return NodeType.RETRIEVE
        elif length <= 12:
            return NodeType.TRANSFORM
        elif length <= 20:
            return NodeType.VERIFY
        else:
            return NodeType.BACKTRACK

    def extract(self, cot_text: str, trace_id: str = "", **metadata) -> ReasoningTraceGraph:
        """Extract using only sentence-length heuristics."""
        from .rule_based import RuleBasedExtractor

        rbe = RuleBasedExtractor()
        sentences = rbe._split_sentences(cot_text)

        if not sentences:
            return ReasoningTraceGraph(
                trace_id=trace_id or "empty",
                extractor=self.name,
                nodes=[],
                edges=[],
                metadata=metadata,
            )

        nodes = []
        char_offset = 0
        for i, s in enumerate(sentences):
            ntype = self.classify_by_length(s)
            span_start = char_offset
            span_end = char_offset + len(s)
            nodes.append(GraphNode(id=i + 1, type=ntype, span=(span_start, span_end), text=s))
            char_offset = span_end + 1

        edges = [(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]

        return ReasoningTraceGraph(
            trace_id=trace_id or f"rberand_{hashlib.md5(cot_text.encode('utf-8')).hexdigest()[:12]}",
            extractor=self.name,
            nodes=nodes,
            edges=edges,
            metadata={
                "cot_length_tokens": len(cot_text.split()),
                "extraction_rate": 1.0,
                "is_zero_information": True,
                **metadata,
            },
        )


class ShuffledTypeExtractor:
    """
    Additional baseline: randomly shuffles node types while preserving graph structure.

    Tests: "Does node type assignment carry information beyond the graph topology?"
    """

    def __init__(self, name: str = "shuffled_type", seed: int = 42):
        self.name = name
        self._rng = random.Random(seed)

    def extract(
        self, reference_graph: ReasoningTraceGraph, trace_id: str = ""
    ) -> ReasoningTraceGraph:
        """
        Shuffle node types of a reference graph randomly.
        Preserves graph topology (nodes, edges), only randomizes type labels.
        """
        import copy

        shuffled = copy.deepcopy(reference_graph)
        shuffled.trace_id = trace_id or f"shuffled_{reference_graph.trace_id}"
        shuffled.extractor = self.name

        # Collect types and shuffle
        types = [n.type for n in shuffled.nodes]
        self._rng.shuffle(types)
        for node, new_type in zip(shuffled.nodes, types):
            node.type = new_type

        return shuffled
