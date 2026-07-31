"""Tests for MotifMatcher — preset motif matching and data-driven discovery."""

import pytest
import numpy as np
from core.types import (
    GraphNode, NodeType, ReasoningTraceGraph, MotifEntry, MOTIF_LOOKUP,
)
from core.motif_matcher import MotifMatcher, MotifMatchResult


def _make_chain_graph():
    """M1: Retrieve -> Transform -> Verify"""
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
    ]
    edges = [(1, 2), (2, 3)]
    return ReasoningTraceGraph(trace_id="chain", nodes=nodes, edges=edges)


def _make_fork_graph():
    """M2: Retrieve -> Transform, Retrieve -> Transform (fork)"""
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
    ]
    edges = [(1, 2), (1, 3)]
    return ReasoningTraceGraph(trace_id="fork", nodes=nodes, edges=edges)


def _make_diamond_graph():
    """M3: Retrieve -> Transform -> Verify, Retrieve -> Compare -> Verify"""
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.COMPARE),
        GraphNode(id=4, type=NodeType.VERIFY),
    ]
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    return ReasoningTraceGraph(trace_id="diamond", nodes=nodes, edges=edges)


class TestMotifMatcher:
    def test_match_m1_chain(self):
        matcher = MotifMatcher()
        result = matcher.match_preset(_make_chain_graph(), "M1")
        assert result.count == 1
        assert result.motif_id == "M1"

    def test_match_m2_fork(self):
        matcher = MotifMatcher()
        result = matcher.match_preset(_make_fork_graph(), "M2")
        assert result.count >= 1

    def test_match_m3_diamond(self):
        matcher = MotifMatcher()
        result = matcher.match_preset(_make_diamond_graph(), "M3")
        assert result.count >= 1

    def test_match_m5_verify_transform(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.VERIFY),
        ]
        g = ReasoningTraceGraph(trace_id="vt", nodes=nodes, edges=[(1, 2)])
        matcher = MotifMatcher()
        result = matcher.match_preset(g, "M5")
        assert result.count >= 1

    def test_chain_does_not_match_diamond(self):
        matcher = MotifMatcher()
        result = matcher.match_preset(_make_chain_graph(), "M3")
        assert result.count == 0

    def test_unknown_motif_raises(self):
        matcher = MotifMatcher()
        with pytest.raises(ValueError, match="Unknown motif"):
            matcher.match_preset(_make_chain_graph(), "M99")

    def test_match_all_presets(self):
        matcher = MotifMatcher()
        g = _make_chain_graph()
        results = matcher.match_all_presets(g)
        assert len(results) == 11  # 11 preset motifs (M1-M12)
        assert results["M1"].count >= 1

    def test_motif_frequencies(self):
        matcher = MotifMatcher()
        graphs = [_make_chain_graph(), _make_fork_graph()]
        freqs = matcher.compute_motif_frequencies(graphs)
        assert freqs["M1"] == pytest.approx(0.5)  # only chain has M1
        assert 0.0 <= freqs["M2"] <= 1.0

    def test_motif_frequency_vector(self):
        matcher = MotifMatcher()
        vec = matcher.compute_motif_frequency_vector(_make_chain_graph())
        assert vec.shape == (11,)  # M1-M12 (no M4)
        assert vec[0] >= 0  # M1 count (first in sorted order)

    def test_motif_frequency_matrix(self):
        matcher = MotifMatcher()
        graphs = [_make_chain_graph(), _make_fork_graph(), _make_diamond_graph()]
        mat = matcher.compute_motif_frequency_matrix(graphs)
        assert mat.shape == (3, 11)  # M1-M12 (no M4)

    def test_discover_motifs_exhaustive(self):
        matcher = MotifMatcher()
        graphs = [_make_chain_graph(), _make_chain_graph(), _make_fork_graph()]
        discovered = matcher.discover_motifs_exhaustive(graphs, max_size=3, min_frequency=0.3)
        assert len(discovered) > 0
        assert all(isinstance(m, MotifEntry) for m in discovered)
        assert all(m.discovery_method == "exhaustive" for m in discovered)

    def test_deduplicate_overlapping(self):
        sets = [{1, 2}, {2, 3}, {3, 4}, {5, 6}]
        result = MotifMatcher._deduplicate_overlapping(sets)
        assert len(result) >= 2
        # should not have overlapping node sets
        all_nodes = set()
        for ns in result:
            assert not (ns & all_nodes)
            all_nodes |= ns

    def test_motif_frequency_vector_single_graph_no_match(self):
        nodes = [GraphNode(id=1, type=NodeType.BACKTRACK)]
        g = ReasoningTraceGraph(trace_id="none", nodes=nodes, edges=[])
        matcher = MotifMatcher()
        vec = matcher.compute_motif_frequency_vector(g)
        assert vec.shape == (11,)  # M1-M12 (no M4)
        assert np.sum(vec) == 0  # no matches for 1-node graph
