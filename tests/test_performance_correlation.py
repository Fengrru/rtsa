"""Tests for the performance-correlation benchmark (analysis.performance_correlation)."""

import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from analysis.performance_correlation import (
    _bh_fdr,
    bootstrap_rho_ci,
    build_metric_specs,
    performance_metric_matrix,
    run_performance_correlation,
    synthetic_performance_graphs,
)


def test_metric_specs_cover_three_families():
    specs = build_metric_specs()
    families = {s.family for s in specs}
    assert families == {"global", "type_mix", "shape"}
    # 10 global + 6 type-mix + 3 shape
    assert len(specs) == 19
    assert all(s.node_type is not None for s in specs
               if s.family == "type_mix")


def test_metric_matrix_shape_matches_specs():
    graphs = synthetic_performance_graphs(n=10, seed=1)
    mat, specs = performance_metric_matrix(graphs)
    assert mat.shape == (10, len(specs))
    assert np.all(np.isfinite(mat))


def test_type_mix_ratios_sum_to_one():
    graphs = synthetic_performance_graphs(n=5, seed=2)
    mat, specs = performance_metric_matrix(graphs)
    type_idx = [i for i, s in enumerate(specs) if s.family == "type_mix"]
    row_sums = mat[:, type_idx].sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_synthetic_signal_detected_with_correct_direction():
    graphs = synthetic_performance_graphs(n=80, seed=7)
    labels = {g.trace_id: g.metadata["correct"] for g in graphs}
    report = run_performance_correlation(graphs, labels)
    by_name = {c.metric: c for c in report.correlations}
    # Redundancy drives performance down: more nodes/verify -> lower score.
    assert by_name["n_nodes"].spearman_rho < -0.5
    assert by_name["verify_density"].spearman_rho < -0.5
    assert by_name["n_nodes"].significant
    assert by_name["verify_density"].significant
    assert by_name["n_nodes"].direction == "negative"


def test_fdr_controls_false_positives_on_noise():
    rng = np.random.default_rng(0)
    # 19 pure-noise metrics vs pure-noise labels: nothing should survive FDR.
    p = rng.uniform(0.0, 1.0, size=19)
    q = _bh_fdr(p)
    assert q.shape == (19,)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
    assert np.sum(q < 0.05) <= 2  # BH allows at most a few expected FPs


def test_constant_labels_raise_value_error():
    graphs = synthetic_performance_graphs(n=10, seed=3)
    labels = {g.trace_id: True for g in graphs}  # all correct
    with pytest.raises(ValueError, match="zero variance"):
        run_performance_correlation(graphs, labels)


def test_continuous_labels_supported():
    graphs = synthetic_performance_graphs(n=50, seed=5)
    labels = {g.trace_id: g.metadata["performance"] for g in graphs}
    report = run_performance_correlation(graphs, labels)
    assert report.label_type == "continuous"
    assert report.n_incorrect == -1
    assert report.correlations[0].spearman_rho < -0.5


def test_bootstrap_ci_contains_point_estimate():
    x = np.linspace(0.0, 1.0, 60)
    rng = np.random.default_rng(1)
    y = x + rng.normal(0.0, 0.1, size=60)
    lo, hi = bootstrap_rho_ci(x, y, n_bootstrap=200, seed=42)
    rho, _ = spearmanr(x, y)
    assert lo <= rho <= hi
    assert -1.0 <= lo <= hi <= 1.0


def test_report_serializable_to_json_and_markdown(tmp_path):
    graphs = synthetic_performance_graphs(n=40, seed=9)
    labels = {g.trace_id: g.metadata["correct"] for g in graphs}
    report = run_performance_correlation(graphs, labels, n_bootstrap=100)
    payload = json.dumps(report.to_dict())
    assert '"spearman_rho"' in payload
    md = report.to_markdown()
    assert md.startswith("| Metric |")
    assert "verify_density" in md
    summary = report.summary()
    assert "significant" in summary
    assert any(c.ci_lo is not None for c in report.correlations)


def test_metrics_with_zero_variance_report_nan_gracefully():
    # A single-node graph gives zero variance on several shape metrics;
    # the benchmark must not crash and must mark them non-significant.
    graphs = []
    for i in range(6):
        g = synthetic_performance_graphs(n=1, seed=i)[0]
        g.trace_id = f"syn_{i:03d}"  # unique ids (n=1 always yields syn_000)
        graphs.append(g)
    labels = {g.trace_id: bool(i % 2) for i, g in enumerate(graphs)}
    report = run_performance_correlation(graphs, labels)
    assert len(report.correlations) == 19
    for c in report.correlations:
        if not np.isfinite(c.spearman_rho):
            assert not c.significant
