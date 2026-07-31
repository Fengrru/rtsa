"""Analysis utilities: Cost Estimator (Fix 9) + Statistics."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost Estimator (Fix 9: Budget consistency)
# ---------------------------------------------------------------------------

GPU_PRICING = {"A100-40GB": 0.50, "A100-80GB": 0.65, "H100": 0.90, "RTX4090": 0.35, "L40S": 0.45}
GEN_SPEED = {"7B": 50.0, "32B": 25.0, "72B": 12.0, "8B": 45.0, "70B": 13.0, "16B-moe": 30.0, "47B-moe": 18.0}


@dataclass
class CostBreakdown:
    phase: str
    items: list = field(default_factory=list)
    subtotal: float = 0.0

    def add(self, task: str, gpu_hours: float = 0.0, api_cost: float = 0.0, human_cost: float = 0.0):
        total = gpu_hours * 0.50 + api_cost + human_cost
        self.items.append({"task": task, "gpu_hours": gpu_hours, "api_cost": api_cost,
                           "human_cost": human_cost, "total": total})
        self.subtotal += total


class CostEstimator:
    def __init__(self, gpu_type: str = "A100-40GB"):
        self.gpu_type = gpu_type
        self.gpu_rate = GPU_PRICING.get(gpu_type, 0.50)

    def _cot_gen(self, n, avg_tok=200, ms="7B"):
        tps = GEN_SPEED.get(ms, 25.0)
        gh = n * avg_tok / tps / 3600.0
        api = n * avg_tok / 1000 * 0.015
        return gh, api

    def _lora(self, n=100, at=300, ep=10):
        return n * at * ep / 1e6 * 0.15

    def est_05(self):
        bd = CostBreakdown("0.5 Pilot")
        for ms in ["7B", "32B", "72B"]:
            gh, _ = self._cot_gen(5, 250, ms)
            bd.add(f"CoT gen {ms}", gpu_hours=gh)
        bd.add("GCP + extraction", gpu_hours=5.0)
        bd.add("Robust-TSI + motif test", gpu_hours=3.0)
        bd.add("LLM extraction API", api_cost=5.0)
        return bd

    def est_0(self, n=100):
        bd = CostBreakdown("0 Extractor Agreement")
        gh, _ = self._cot_gen(n, 200, "72B")
        bd.add(f"CoT gen {n}", gpu_hours=gh)
        bd.add("RBE/SBE/RBE-Rand x{n}", gpu_hours=3.0)
        bd.add("LLM extraction x{n}", api_cost=25.0)
        bd.add("Human annotation 100x3", human_cost=500.0)
        bd.add("IAA computation", gpu_hours=2.0)
        return bd

    def est_1(self):
        bd = CostBreakdown("1 Topological Existence")
        for ms in ["7B", "32B", "72B"]:
            gh, _ = self._cot_gen(1200, 200, ms)
            bd.add(f"CoT gen 1200x{ms}", gpu_hours=gh)
        bd.add("Extraction 3,600x", gpu_hours=30.0)
        bd.add("JP-DPR randomization", gpu_hours=50.0)
        bd.add("Motif analysis + TSI", gpu_hours=40.0)
        return bd

    def est_2(self):
        bd = CostBreakdown("2 Memorization")
        bd.add("LoRA training", gpu_hours=self._lora())
        gh, _ = self._cot_gen(500, 200, "7B")
        bd.add("Evaluation gen 500", gpu_hours=gh)
        bd.add("Morphology annotation 30x3", human_cost=150.0)
        bd.add("Extraction + analysis", gpu_hours=20.0)
        return bd

    def est_3(self, n=300):
        bd = CostBreakdown("3 Generalization")
        gh, _ = self._cot_gen(n, 250, "7B")
        bd.add(f"OOD CoT gen {n}", gpu_hours=gh)
        bd.add(f"Extraction {n}", gpu_hours=n * 0.01)
        bd.add("ER-Corrected modeling", gpu_hours=5.0)
        return bd

    def full(self):
        bds = [self.est_05(), self.est_0(), self.est_1(), self.est_2(), self.est_3()]
        t = sum(b.subtotal for b in bds)
        logger.info(f"Total estimated project cost: ${t:,.2f} (GPU: {self.gpu_type} @ ${self.gpu_rate}/hr)")
        return bds


def estimate_project_cost(gpu_type: str = "A100-40GB") -> Tuple[List[CostBreakdown], float]:
    est = CostEstimator(gpu_type)
    bds = est.full()
    return bds, sum(b.subtotal for b in bds)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def kruskal_wallis_test(*groups: np.ndarray) -> dict:
    if len(groups) < 2:
        raise ValueError("Need at least two groups to perform Kruskal-Wallis test")
    h, p = stats.kruskal(*groups)
    return {"h_statistic": float(h), "p_value": float(p), "significant": bool(p < 0.05)}


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=2000, confidence=0.95, seed=42):
    rng = np.random.RandomState(seed)
    sv = float(statistic(data))
    bv = [float(statistic(data[rng.choice(len(data), len(data), replace=True)])) for _ in range(n_bootstrap)]
    a = (1.0 - confidence) / 2.0
    return sv, float(np.percentile(bv, 100 * a)), float(np.percentile(bv, 100 * (1 - a)))


def cohens_d(g1, g2):
    m1, m2 = np.mean(g1), np.mean(g2)
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    ps = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return float((m1 - m2) / ps) if ps > 1e-10 else 0.0


def partial_correlation(x, y, z):
    r_xy = float(np.corrcoef(x, y)[0, 1])
    r_xz = float(np.corrcoef(x, z)[0, 1])
    r_yz = float(np.corrcoef(y, z)[0, 1])
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    return float((r_xy - r_xz * r_yz) / denom) if denom > 1e-10 else 0.0
