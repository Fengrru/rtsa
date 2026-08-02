"""Semantic step clustering — multi-granularity reasoning analysis.

Inspired by LLM-MindMap (EMNLP 2025, "Mapping the Minds of LLMs"), which
shows that clustering verbose CoT into semantically coherent *steps* before
graph construction captures cross-sentence dependencies that sentence-level
graphs miss.

``StepClusterer`` merges consecutive chain segments of a
``ReasoningTraceGraph`` into step-level clusters while preserving the DAG:

- only chain segments are merged (in-degree 1, out-degree 1, direct edge),
  so the topology stays valid and no cycles can be introduced;
- ``heuristic`` mode merges same-type neighbours and/or textually similar
  neighbours (Jaccard word overlap);
- ``semantic`` mode additionally accepts a pluggable embedder exposing a
  ``similarity(text_a, text_b) -> float`` method and falls back to the
  heuristic rule when no embedder is available.

The output is a new ``ReasoningTraceGraph`` (re-numbered ids, merged text
and spans) that every downstream consumer — NGS validator, motif matcher,
pruner, fingerprint — can process unchanged.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token Jaccard overlap; used as the default lexical similarity."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class StepClusterer:
    """Merge chain segments of a trace graph into step-level clusters.

    Args:
        method: "heuristic" (lexical + type based) or "semantic" (embedder
            similarity with heuristic fallback).
        min_similarity: minimum similarity for merging neighbours of
            different node types.
        embedder: optional object with a ``similarity(a, b) -> float``
            method (e.g. a sentence-transformer wrapper).
    """

    def __init__(
        self,
        method: str = "heuristic",
        min_similarity: float = 0.30,
        embedder: Optional[object] = None,
    ):
        if method not in ("heuristic", "semantic"):
            raise ValueError("method must be 'heuristic' or 'semantic'")
        self.method = method
        self.min_similarity = min_similarity
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(self, graph: ReasoningTraceGraph) -> ReasoningTraceGraph:
        """Return a step-level clustered copy of *graph*."""
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
        if not nodes_sorted:
            return graph

        out_deg = Counter(e[0] for e in graph.edges)
        in_deg = Counter(e[1] for e in graph.edges)
        edge_set = set(graph.edges)

        # Greedy scan: grow a cluster while consecutive nodes form a
        # mergeable chain segment.
        clusters: List[List[int]] = [[nodes_sorted[0].id]]
        for a, b in zip(nodes_sorted, nodes_sorted[1:]):
            mergeable = (
                (a.id, b.id) in edge_set
                and out_deg.get(a.id, 0) == 1
                and in_deg.get(b.id, 0) == 1
                and self._should_merge(a, b)
            )
            if mergeable:
                clusters[-1].append(b.id)
            else:
                clusters.append([b.id])

        return self._rebuild(graph, clusters)

    def merge_chain_segments(
        self, graph: ReasoningTraceGraph, max_segment: int = 5
    ) -> ReasoningTraceGraph:
        """Type-agnostic variant: merge chain segments up to *max_segment*
        nodes regardless of similarity. Useful for a quick coarse pass."""
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
        if not nodes_sorted:
            return graph

        out_deg = Counter(e[0] for e in graph.edges)
        in_deg = Counter(e[1] for e in graph.edges)
        edge_set = set(graph.edges)

        clusters: List[List[int]] = [[nodes_sorted[0].id]]
        for a, b in zip(nodes_sorted, nodes_sorted[1:]):
            chain = (
                (a.id, b.id) in edge_set
                and out_deg.get(a.id, 0) == 1
                and in_deg.get(b.id, 0) == 1
            )
            if chain and len(clusters[-1]) < max_segment:
                clusters[-1].append(b.id)
            else:
                clusters.append([b.id])

        return self._rebuild(graph, clusters)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _should_merge(self, a: GraphNode, b: GraphNode) -> bool:
        if a.type == b.type:
            return True
        if self.method == "semantic" and self.embedder is not None:
            try:
                sim = float(self.embedder.similarity(a.text, b.text))
            except Exception:  # embedder failure -> fall through to lexical
                sim = _jaccard_similarity(a.text, b.text)
        else:
            sim = _jaccard_similarity(a.text, b.text)
        return sim >= self.min_similarity

    def _rebuild(
        self,
        graph: ReasoningTraceGraph,
        clusters: List[List[int]],
    ) -> ReasoningTraceGraph:
        node_map = {n.id: n for n in graph.nodes}
        id_map: Dict[int, int] = {}
        new_nodes: List[GraphNode] = []
        for i, cluster in enumerate(clusters, start=1):
            members = [node_map[nid] for nid in cluster]
            texts = [m.text for m in members if m.text]
            types = [m.type for m in members]
            spans = [m.span for m in members]
            # Majority type; ties resolved by first occurrence order.
            majority = Counter(types).most_common(1)[0][0]
            new_nodes.append(GraphNode(
                id=i,
                type=majority,
                text=" ".join(texts).strip(),
                span=(
                    min(s[0] for s in spans),
                    max(s[1] for s in spans),
                ),
            ))
            for nid in cluster:
                id_map[nid] = i

        new_edges: List[Tuple[int, int]] = []
        seen = set()
        for u, v in graph.edges:
            nu, nv = id_map[u], id_map[v]
            if nu != nv and (nu, nv) not in seen:
                seen.add((nu, nv))
                new_edges.append((nu, nv))

        return ReasoningTraceGraph(
            trace_id=f"{graph.trace_id}_clustered",
            model=graph.model,
            question_id=graph.question_id,
            domain=graph.domain,
            extractor=f"{graph.extractor}_stepclust",
            nodes=new_nodes,
            edges=new_edges,
            metadata={
                **graph.metadata,
                "step_clustering": {
                    "method": self.method,
                    "n_clusters": len(new_nodes),
                    "n_source_nodes": len(graph.nodes),
                },
            },
        )
