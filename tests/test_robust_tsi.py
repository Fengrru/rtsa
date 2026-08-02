"""Tests for RobustTSI and UnsupervisedTSI."""

import pytest
import numpy as np
from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.core.robust_tsi import (
    extract_level1_features, extract_level1_features_batch,
    RobustTSI, UnsupervisedTSI, bootstrap_tsi_ci, cohens_d,
)


def _make_graph_a():
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
    ]
    return ReasoningTraceGraph(trace_id="a", nodes=nodes, edges=[(1, 2), (2, 3)])


def _make_graph_b():
    nodes = [
        GraphNode(id=1, type=NodeType.RETRIEVE),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.VERIFY),
        GraphNode(id=4, type=NodeType.VERIFY),
    ]
    return ReasoningTraceGraph(trace_id="b", nodes=nodes, edges=[(1, 2), (2, 3), (2, 4)])


def _make_graph_c():
    nodes = [
        GraphNode(id=1, type=NodeType.BRANCH),
        GraphNode(id=2, type=NodeType.TRANSFORM),
        GraphNode(id=3, type=NodeType.TRANSFORM),
    ]
    return ReasoningTraceGraph(trace_id="c", nodes=nodes, edges=[(1, 2), (1, 3)])


class TestLevel1Features:
    def test_extract_single_graph(self):
        g = _make_graph_a()
        feats = extract_level1_features(g.to_networkx())
        assert feats.shape == (4,)
        assert feats[0] > 0  # depth > 0

    def test_extract_batch(self):
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_c()]
        mat = extract_level1_features_batch(graphs)
        assert mat.shape == (3, 4)

    def test_different_graphs_different_features(self):
        g_a = _make_graph_a()
        g_b = _make_graph_b()
        f_a = extract_level1_features(g_a.to_networkx())
        f_b = extract_level1_features(g_b.to_networkx())
        assert not np.allclose(f_a, f_b)


class TestRobustTSI:
    def test_init(self):
        tsi = RobustTSI()
        assert tsi.pca_components == 2
        assert tsi.ridge_alpha == 1.0
        assert not tsi._fitted

    def test_predict_without_fit_raises(self):
        tsi = RobustTSI()
        with pytest.raises(RuntimeError, match="not fitted"):
            tsi.predict_pair(_make_graph_a(), _make_graph_b())

    def test_fit_and_predict(self):
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_c()]
        n = len(graphs)
        n_pairs = n * (n - 1) // 2
        # simple human judgments: similar graphs have higher similarity
        judgments = np.array([0.8, 0.3, 0.4], dtype=np.float64)  # must match n_pairs=3

        tsi = RobustTSI(pca_components=2, ridge_alpha=1.0)
        tsi.fit(graphs, judgments)
        assert tsi._fitted

        sim = tsi.predict_pair(_make_graph_a(), _make_graph_b())
        assert 0.0 <= sim <= 1.0

    def test_fit_wrong_judgment_shape_raises(self):
        tsi = RobustTSI()
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_c()]
        with pytest.raises(ValueError):
            tsi.fit(graphs, np.array([0.5, 0.6]))  # 2 judgments for 3 graphs -> 3 pairs

    def test_pairwise_similarity_matrix(self):
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_c()]
        n = len(graphs)
        judgments = np.ones(n * (n - 1) // 2, dtype=np.float64) * 0.7

        tsi = RobustTSI()
        tsi.fit(graphs, judgments)
        mat = tsi.pairwise_similarity_matrix(graphs)
        assert mat.shape == (3, 3)
        assert np.allclose(np.diag(mat), 1.0)  # self-similarity
        for i in range(n):
            for j in range(n):
                assert 0.0 <= mat[i, j] <= 1.0

    def test_get_alpha(self):
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_c()]
        judgments = np.ones(3, dtype=np.float64) * 0.5
        tsi = RobustTSI()
        assert tsi.get_alpha() == 0.5  # default before fit
        tsi.fit(graphs, judgments)
        alpha = tsi.get_alpha()
        assert 0.0 <= alpha <= 1.0


class TestUnsupervisedTSI:
    def test_init(self):
        usi = UnsupervisedTSI(wl_iterations=3)
        assert usi.wl_iterations == 3

    def test_similarity_range(self):
        usi = UnsupervisedTSI()
        g1 = _make_graph_a()
        g2 = _make_graph_b()
        sim = usi.similarity(g1, g2)
        assert 0.0 <= sim <= 1.0

    def test_identical_graphs_similarity(self):
        usi = UnsupervisedTSI()
        g = _make_graph_a()
        sim = usi.similarity(g, g)
        assert sim > 0.9  # should be very high for identical graphs

    def test_pairwise_similarity_matrix(self):
        usi = UnsupervisedTSI()
        a = _make_graph_a()
        b = _make_graph_b()
        mat = usi.pairwise_similarity_matrix([a, b])
        assert mat.shape == (2, 2)
        assert np.allclose(np.diag(mat), 1.0)

    def test_cross_vocabulary_no_dimension_mismatch(self):
        """Graphs with disjoint node-type vocabularies must not crash
        (WL histograms are aligned on the union vocabulary)."""
        usi = UnsupervisedTSI()
        g1 = _make_graph_a()  # vocab: Retrieve, Transform, Verify
        g2 = _make_graph_c()  # vocab: Branch, Transform
        sim = usi.similarity(g1, g2)
        assert 0.0 <= sim <= 1.0
        mat = usi.pairwise_similarity_matrix([g1, g2])
        assert mat.shape == (2, 2)
        assert not np.any(np.isnan(mat))

    def test_compare_with_supervised(self):
        usi = UnsupervisedTSI()
        graphs = [_make_graph_a(), _make_graph_b(), _make_graph_a()]  # 3 graphs, same type vocab
        u_mat = usi.pairwise_similarity_matrix(graphs)

        s_mat = np.eye(3)
        s_mat[0, 1] = s_mat[1, 0] = 0.8
        s_mat[0, 2] = s_mat[2, 0] = 1.0  # identical graphs
        s_mat[1, 2] = s_mat[2, 1] = 0.8

        result = UnsupervisedTSI.compare_with_supervised(u_mat, s_mat)
        assert "pearson_r" in result
        assert "mean_abs_deviation" in result
        assert "agreement_rate" in result
        # Pearson r can be nan with insufficient variance; check it's a float
        assert isinstance(result["pearson_r"], float)

    def test_node_type_jsd(self):
        usi = UnsupervisedTSI()
        g1 = _make_graph_a()
        g2 = _make_graph_a()  # same
        jsd = usi._node_type_jsd(g1.to_networkx(), g2.to_networkx())
        assert jsd < 0.01  # same distributions

    def test_wl_hash_graph(self):
        usi = UnsupervisedTSI()
        g = _make_graph_a()
        vec = usi._wl_hash_graph(g.to_networkx(), iterations=2)
        assert vec.ndim == 1
        assert np.isclose(np.sum(vec), 1.0)  # normalized

    def test_ged_approx_same_graph(self):
        usi = UnsupervisedTSI()
        g = _make_graph_a()
        ged_sim = usi._graph_edit_distance_approx(g.to_networkx(), g.to_networkx())
        assert ged_sim > 0.9


class TestStatsHelpers:
    """A3: bootstrap confidence intervals and Cohen's d effect size."""

    def test_bootstrap_ci_bounds_and_order(self):
        g1 = _make_graph_a()
        g2 = _make_graph_b()
        mean, lo, hi = bootstrap_tsi_ci(
            UnsupervisedTSI().similarity, g1, g2, n_bootstrap=200,
        )
        assert 0.0 <= lo <= mean <= hi <= 1.0

    def test_bootstrap_ci_deterministic_seed(self):
        g1 = _make_graph_a()
        g2 = _make_graph_b()
        r1 = bootstrap_tsi_ci(
            UnsupervisedTSI().similarity, g1, g2, n_bootstrap=100, seed=7,
        )
        r2 = bootstrap_tsi_ci(
            UnsupervisedTSI().similarity, g1, g2, n_bootstrap=100, seed=7,
        )
        assert r1 == r2

    def test_bootstrap_ci_identical_graphs_narrow(self):
        """Identical graphs give sim=1.0; the CI must stay at the top."""
        g = _make_graph_a()
        mean, lo, hi = bootstrap_tsi_ci(
            UnsupervisedTSI().similarity, g, g, n_bootstrap=200,
        )
        assert 0.95 <= mean <= 1.0
        assert hi == 1.0  # clipped mass at the ceiling

    def test_cohens_d_sign_and_magnitude(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 11.0, 12.0, 13.0])
        d = cohens_d(a, b)
        assert d < 0          # group_a below group_b
        assert abs(d) > 3.0   # clearly separated groups

    def test_cohens_d_identical_groups_zero(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0  # pooled SD = 0

    def test_cohens_d_small_sample_zero(self):
        assert cohens_d(np.array([1.0]), np.array([2.0, 3.0])) == 0.0
