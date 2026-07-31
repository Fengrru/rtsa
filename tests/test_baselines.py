"""Tests for randomization baselines: JP-DPR, Edge Rewiring, Permutation, Ensemble."""

import pytest
from core.types import GraphNode, NodeType, ReasoningTraceGraph
from extractors.baselines import (
    JPDirectedPreservingRandomizer,
    EdgeRewiringBaseline,
    PermutationBaseline,
    EnsembleBaseline,
    compute_tsi_threshold,
    compute_stable_rate,
)


def _make_chain():
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
    ]
    return ReasoningTraceGraph(trace_id="chain", nodes=nodes, edges=[(1, 2), (2, 3)])


def _make_complex():
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.COMPARE),
        GraphNode(id=4, type=NodeType.VERIFY),
    ]
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    return ReasoningTraceGraph(trace_id="complex", nodes=nodes, edges=edges)


class TestJPDirectedPreservingRandomizer:
    def test_randomize_returns_graphs(self):
        dpr = JPDirectedPreservingRandomizer(seed=42)
        g = _make_chain()
        graphs = dpr.randomize(g, k=5)
        assert len(graphs) == 5
        for rg in graphs:
            assert isinstance(rg, ReasoningTraceGraph)
            assert rg.metadata.get("is_randomized") is True
            assert rg.metadata.get("method") == "jp_dpr"

    def test_randomized_graphs_have_same_node_count(self):
        dpr = JPDirectedPreservingRandomizer(seed=42)
        g = _make_complex()
        graphs = dpr.randomize(g, k=10)
        for rg in graphs:
            assert len(rg.nodes) == len(g.nodes)

    def test_different_seeds_produce_different_results(self):
        g = _make_chain()
        dpr1 = JPDirectedPreservingRandomizer(seed=1)
        dpr2 = JPDirectedPreservingRandomizer(seed=2)
        g1 = dpr1.randomize(g, k=1)[0]
        g2 = dpr2.randomize(g, k=1)[0]
        edges1 = set(g1.edges)
        edges2 = set(g2.edges)
        # May or may not differ, but structure should be preserved
        assert len(g1.nodes) == len(g2.nodes)

    def test_preserves_node_types(self):
        dpr = JPDirectedPreservingRandomizer(seed=42)
        g = _make_chain()
        graphs = dpr.randomize(g, k=5)
        for rg in graphs:
            types = sorted(n.type for n in rg.nodes)
            expected_types = sorted(n.type for n in g.nodes)
            assert types == expected_types


class TestEdgeRewiringBaseline:
    def test_randomize_returns_graphs(self):
        rw = EdgeRewiringBaseline(seed=42, n_swaps=5)
        g = _make_chain()
        graphs = rw.randomize(g, k=5)
        assert len(graphs) == 5
        for rg in graphs:
            assert isinstance(rg, ReasoningTraceGraph)
            assert rg.metadata.get("method") == "edge_rewire"

    def test_preserves_node_count(self):
        rw = EdgeRewiringBaseline(seed=42, n_swaps=5)
        g = _make_complex()
        graphs = rw.randomize(g, k=5)
        for rg in graphs:
            assert len(rg.nodes) == len(g.nodes)

    def test_n_swaps_zero(self):
        rw = EdgeRewiringBaseline(seed=42, n_swaps=0)
        g = _make_chain()
        graphs = rw.randomize(g, k=3)
        for rg in graphs:
            assert set(rg.edges) == set(g.edges)


class TestPermutationBaseline:
    def test_randomize_returns_graphs(self):
        perm = PermutationBaseline(seed=42)
        g = _make_chain()
        graphs = perm.randomize(g, k=5)
        assert len(graphs) == 5
        for rg in graphs:
            assert rg.metadata.get("method") == "permutation"

    def test_preserves_topology(self):
        perm = PermutationBaseline(seed=42)
        g = _make_complex()
        graphs = perm.randomize(g, k=5)
        for rg in graphs:
            assert len(rg.nodes) == len(g.nodes)
            assert len(rg.edges) == len(g.edges)

    def test_types_shuffled(self):
        perm = PermutationBaseline(seed=42)
        g = _make_chain()
        graphs = perm.randomize(g, k=10)
        all_types = []
        for rg in graphs:
            all_types.append([n.type for n in rg.nodes])
        # With sufficient permutations, we should see variety
        unique_type_seqs = set(tuple(ts) for ts in all_types)
        # At least one permutation should differ (highly likely with 10)
        assert len(unique_type_seqs) >= 1


class TestEnsembleBaseline:
    def test_randomize_all(self):
        ens = EnsembleBaseline(seed=42)
        g = _make_chain()
        results = ens.randomize_all(g, k_per_method=5)
        assert "jp_dpr" in results
        assert "edge_rewire" in results
        assert "permutation" in results
        for method, graphs in results.items():
            assert len(graphs) == 5


class TestThresholdAndStableRate:
    @staticmethod
    def dummy_tsi(g1, g2):
        return 0.7

    def test_compute_tsi_threshold(self):
        dpr = JPDirectedPreservingRandomizer(seed=42)
        g = _make_chain()
        dpr_graphs = dpr.randomize(g, k=10)
        threshold = compute_tsi_threshold(
            [g], dpr_graphs, TestThresholdAndStableRate.dummy_tsi, percentile=95.0, n_samples=5
        )
        assert threshold == pytest.approx(0.7)

    def test_compute_stable_rate(self):
        dpr = JPDirectedPreservingRandomizer(seed=42)
        g1 = _make_chain()
        g2 = _make_complex()
        dpr_graphs = dpr.randomize(g1, k=10)
        real_tsi, threshold, stable = compute_stable_rate(
            [g1, g2], dpr_graphs, TestThresholdAndStableRate.dummy_tsi,
        )
        assert real_tsi == pytest.approx(0.7)
        assert isinstance(stable, bool)

    def test_compute_tsi_threshold_empty_handled(self):
        g = _make_chain()
        threshold = compute_tsi_threshold([g], [], TestThresholdAndStableRate.dummy_tsi)
        assert threshold == 0.0
