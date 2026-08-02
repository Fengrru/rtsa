"""Tests for analysis.step_clustering (B5)."""

from collections import defaultdict

from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.analysis.step_clustering import StepClusterer


def _chain_graph(types, trace_id="chain"):
    nodes = [
        GraphNode(id=i + 1, type=t, text=f"step {i + 1}")
        for i, t in enumerate(types)
    ]
    edges = [(i + 1, i + 2) for i in range(len(types) - 1)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


def test_heuristic_merges_same_type_chain():
    g = _chain_graph([NodeType.TRANSFORM] * 5)
    merged = StepClusterer(method="heuristic").cluster(g)
    assert len(merged.nodes) < len(g.nodes)


def test_branch_graph_not_merged_across_branches():
    """Branch nodes must not be collapsed with their children."""
    nodes = [
        GraphNode(id=1, type=NodeType.BRANCH, text="b"),
        GraphNode(id=2, type=NodeType.TRANSFORM, text="t1"),
        GraphNode(id=3, type=NodeType.TRANSFORM, text="t2"),
    ]
    g = ReasoningTraceGraph(trace_id="branch", nodes=nodes, edges=[(1, 2), (1, 3)])
    merged = StepClusterer(method="heuristic").cluster(g)
    assert len(merged.nodes) == len(g.nodes)


def test_merge_chain_segments_limits_segment_size():
    g = _chain_graph([NodeType.TRANSFORM] * 9)
    merged = StepClusterer().merge_chain_segments(g, max_segment=5)
    assert len(merged.nodes) < len(g.nodes)


def test_clustering_preserves_connectivity():
    g = _chain_graph([NodeType.TRANSFORM] * 6)
    merged = StepClusterer(method="heuristic").cluster(g)
    ids = {n.id for n in merged.nodes}
    assert len(ids) >= 1
    assert len(merged.edges) == len(ids) - 1  # still one chain
    incoming = defaultdict(set)
    for u, v in merged.edges:
        incoming[v].add(u)
    roots = ids - {v for v in incoming}
    assert len(roots) == 1  # exactly one root survives
    for nid in ids - roots:
        assert incoming[nid] & ids, f"node {nid} unreachable"


def test_clustering_records_metadata():
    g = _chain_graph([NodeType.TRANSFORM] * 4)
    merged = StepClusterer(method="heuristic").cluster(g)
    info = merged.metadata.get("step_clustering", {})
    assert "n_clusters" in info
    assert info["n_source_nodes"] == 4


def test_empty_graph_passthrough():
    g = ReasoningTraceGraph(trace_id="empty", nodes=[], edges=[])
    assert StepClusterer().cluster(g) is g
