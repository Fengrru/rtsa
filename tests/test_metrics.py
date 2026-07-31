"""Tests for graph-level metrics computation."""

import pytest
import numpy as np
from core.types import GraphNode, NodeType, ReasoningTraceGraph
from core.metrics import (
    compute_graph_features, compute_graph_features_batch,
    compute_feature_matrix, GraphMetrics,
)


def _make_simple_graph(trace_id="t"):
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
    ]
    edges = [(1, 2), (2, 3)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


def _make_branching_graph(trace_id="b"):
    nodes = [
        GraphNode(id=1, type=NodeType.BRANCH),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.TRANSFORM),
    ]
    edges = [(1, 2), (1, 3)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


class TestComputeGraphFeatures:
    def test_basic_metrics(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        assert m.n_nodes == 3
        assert m.n_edges == 2
        assert m.trace_id == "t"

    def test_depth_linear_chain(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        assert m.depth == 2.0  # path length: 1->2->3 = 2 edges

    def test_branching(self):
        g = _make_branching_graph()
        m = compute_graph_features(g)
        assert m.branching > 0

    def test_verify_density(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        assert m.verify_density == pytest.approx(1.0 / 3.0)

    def test_backtrack_rate_zero(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        assert m.backtrack_rate == 0.0

    def test_entropy_nonzero(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        assert m.entropy > 0

    def test_graph_density(self):
        g = _make_simple_graph()
        m = compute_graph_features(g)
        max_edges = 3 * 2 / 2  # n*(n-1)/2 = 3
        assert m.graph_density == pytest.approx(2.0 / 3.0)

    def test_empty_graph_metrics(self):
        g = ReasoningTraceGraph(trace_id="empty", nodes=[], edges=[])
        m = compute_graph_features(g)
        # n_nodes uses max(1, ...) so returns 1 for empty graph
        assert m.n_nodes == 1
        assert m.n_edges == 0
        assert m.depth == 0.0

    def test_single_node_graph(self):
        nodes = [GraphNode(id=1, type=NodeType.TRANSFORM)]
        g = ReasoningTraceGraph(trace_id="s", nodes=nodes, edges=[])
        m = compute_graph_features(g)
        assert m.n_nodes == 1
        assert m.depth == 0.0
        assert m.branching == 0.0


class TestBatchAndMatrix:
    def test_compute_batch(self):
        g1 = _make_simple_graph("a")
        g2 = _make_branching_graph("b")
        metrics = compute_graph_features_batch([g1, g2])
        assert len(metrics) == 2
        assert all(isinstance(m, GraphMetrics) for m in metrics)
        assert metrics[0].trace_id == "a"

    def test_feature_matrix_shape(self):
        g1 = _make_simple_graph("a")
        g2 = _make_branching_graph("b")
        mat = compute_feature_matrix([g1, g2])
        assert mat.shape == (2, 6)
        assert mat.dtype == np.float64

    def test_feature_matrix_values(self):
        g = _make_simple_graph()
        mat = compute_feature_matrix([g])
        assert mat[0, 0] == 3  # n_nodes
        assert mat[0, 1] == 2  # n_edges
