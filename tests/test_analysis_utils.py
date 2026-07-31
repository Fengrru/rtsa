"""Tests for statistical analysis utilities (analysis/__init__)."""

import numpy as np
import pytest
from analysis import (
    CostEstimator, CostBreakdown, estimate_project_cost,
    kruskal_wallis_test, bootstrap_ci, cohens_d, partial_correlation,
)
from extractors.analysis import GPU_PRICING, GEN_SPEED


class TestCostBreakdown:
    def test_add_item(self):
        bd = CostBreakdown("test")
        bd.add("task a", gpu_hours=10.0, api_cost=5.0, human_cost=100.0)
        assert len(bd.items) == 1
        assert bd.items[0]["task"] == "task a"
        assert bd.items[0]["gpu_hours"] == 10.0
        assert bd.items[0]["api_cost"] == 5.0
        assert bd.items[0]["human_cost"] == 100.0
        expected_total = 10 * 0.50 + 5.0 + 100.0
        assert bd.items[0]["total"] == pytest.approx(expected_total)
        assert bd.subtotal == pytest.approx(expected_total)

    def test_multiple_items_sum(self):
        bd = CostBreakdown("multi")
        bd.add("a", gpu_hours=1.0)
        bd.add("b", api_cost=2.0)
        bd.add("c", human_cost=3.0)
        assert len(bd.items) == 3
        assert bd.subtotal == pytest.approx(1.0 * 0.50 + 2.0 + 3.0)


class TestCostEstimator:
    def test_default_gpu_rate(self):
        ce = CostEstimator()
        assert ce.gpu_type == "A100-40GB"
        assert ce.gpu_rate == 0.50

    def test_custom_gpu(self):
        ce = CostEstimator("H100")
        assert ce.gpu_rate == 0.90

    def test_unknown_gpu_fallback(self):
        ce = CostEstimator("unknown")
        assert ce.gpu_rate == 0.50

    def test_est_05_returns_breakdown(self):
        ce = CostEstimator()
        bd = ce.est_05()
        assert bd.phase == "0.5 Pilot"
        assert len(bd.items) > 0
        assert bd.subtotal > 0

    def test_est_0_returns_breakdown(self):
        ce = CostEstimator()
        bd = ce.est_0(n=50)
        assert bd.phase == "0 Extractor Agreement"
        assert len(bd.items) == 5
        assert bd.subtotal > 0

    def test_full_returns_all_phases(self):
        ce = CostEstimator()
        bds = ce.full()
        assert len(bds) == 5
        phases = [b.phase for b in bds]
        assert "0.5 Pilot" in phases
        assert "0 Extractor Agreement" in phases
        assert "1 Topological Existence" in phases
        assert "2 Memorization" in phases
        assert "3 Generalization" in phases

    def test_full_total_positive(self):
        bds, total = estimate_project_cost("RTX4090")
        assert total > 0
        assert len(bds) == 5


class TestKruskalWallis:
    def test_significant(self):
        g1 = np.array([1, 2, 3])
        g2 = np.array([10, 11, 12])
        result = kruskal_wallis_test(g1, g2)
        assert bool(result["significant"]) is True
        assert result["p_value"] < 0.05
        assert result["h_statistic"] > 0

    def test_not_significant(self):
        g1 = np.array([1, 2, 3])
        g2 = np.array([1, 2, 3])
        result = kruskal_wallis_test(g1, g2)
        assert bool(result["significant"]) is False
        assert result["p_value"] > 0.05

    def test_multiple_groups(self):
        g1 = np.array([1, 2, 1, 2])
        g2 = np.array([5, 6, 5, 6])
        g3 = np.array([9, 10, 9, 10])
        result = kruskal_wallis_test(g1, g2, g3)
        assert bool(result["significant"]) is True

    def test_single_group_raises(self):
        with pytest.raises(ValueError, match="Need at least two groups"):
            kruskal_wallis_test(np.array([1, 2, 3]))


class TestBootstrapCI:
    def test_mean_ci_mean_included(self):
        data = np.random.RandomState(42).normal(0, 1, 100)
        stat, lo, hi = bootstrap_ci(data, n_bootstrap=500, seed=42)
        assert lo <= stat <= hi

    def test_ci_narrow_with_large_sample(self):
        data = np.ones(100) * 5.0
        stat, lo, hi = bootstrap_ci(data, n_bootstrap=500, seed=42)
        assert lo == pytest.approx(5.0, abs=0.01)
        assert hi == pytest.approx(5.0, abs=0.01)

    def test_median_ci(self):
        data = np.random.RandomState(0).exponential(scale=2.0, size=200)
        stat, lo, hi = bootstrap_ci(data, statistic=np.median, n_bootstrap=500, seed=0)
        assert lo <= stat <= hi

    def test_reproducible_seed(self):
        data = np.random.RandomState(99).normal(0, 1, 50)
        _, lo1, hi1 = bootstrap_ci(data, n_bootstrap=500, seed=42)
        _, lo2, hi2 = bootstrap_ci(data, n_bootstrap=500, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2


class TestCohensD:
    def test_identical_groups(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([1.0, 2.0, 3.0])
        d = cohens_d(g1, g2)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_large_effect(self):
        g1 = np.array([1.0, 1.5, 1.0])
        g2 = np.array([10.0, 11.0, 10.5])
        d = cohens_d(g1, g2)
        assert abs(d) > 5.0

    def test_negative_effect(self):
        g1 = np.array([10.0, 11.0])
        g2 = np.array([1.0, 1.5])
        d = cohens_d(g1, g2)
        assert d > 0  # g1 > g2

    def test_small_effect(self):
        g1 = np.array([1.0, 3.0, 2.0, 4.0, 1.5])
        g2 = np.array([1.5, 3.5, 2.5, 4.5, 2.0])
        d = cohens_d(g1, g2)
        assert abs(d) < 1.0


class TestPartialCorrelation:
    def test_no_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        z = np.array([5, 1, 4, 2, 3])  # unrelated to x,y relationship
        r = partial_correlation(x, y, z)
        assert r == pytest.approx(-1.0, abs=0.01)

    def test_perfect_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        z = np.array([5, 1, 4, 2, 3])  # unrelated to the perfect x-y relationship
        r = partial_correlation(x, y, z)
        assert r >= 0.99

    def test_z_controls_confound(self):
        x = np.array([1, 2, 3, 4, 5])
        z = np.array([1, 2, 3, 4, 5])  # perfectly correlates with x
        y = x + np.random.RandomState(0).normal(0, 0.1, 5)
        r_raw = np.corrcoef(x, y)[0, 1]
        r_partial = partial_correlation(x, y, z)
        # When z=identical to x, partial correlation is effectively 0
        assert r_partial == pytest.approx(0.0, abs=0.01)
