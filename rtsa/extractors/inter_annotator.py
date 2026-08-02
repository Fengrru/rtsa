"""Inter-Annotator Agreement for Human Annotation (E7) — Fix 3.

Validates gold-standard reliability via multi-annotator Fleiss' Kappa.
CRITICAL: All GCP and IAA conclusions depend on this validation.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

ANNOTATION_GUIDELINES = """
# RTSA Human Annotation Guidelines
## Task: Identify atomic reasoning operations in CoT traces.
## Types: Retrieve, Transform, Compare, Verify, Branch, Backtrack.
## NGS Rules: 1 node = 1 operation. No consecutive Transforms.
## Each theorem ref = new Retrieve. Multi-verify = 1 Verify.
## Branch = condition itself. Violations invalidate annotation.
"""


class InterAnnotatorAgreement:
    _RELIABILITY = {
        (0.81, 1.01): "excellent",
        (0.61, 0.81): "good",
        (0.41, 0.61): "moderate",
        (0.21, 0.41): "fair",
        (0.00, 0.21): "poor",
    }

    def __init__(self, categories=None):
        self.categories = categories or [
            "Retrieve", "Transform", "Compare", "Verify", "Branch", "Backtrack"
        ]

    def fleiss_kappa(self, annotations: np.ndarray) -> float:
        n, m = annotations.shape
        k = len(self.categories)
        n_ij = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                n_ij[i, j] = np.sum(annotations[i, :] == j)
        p_j = np.sum(n_ij, axis=0) / (n * m)
        P_i = (np.sum(n_ij ** 2, axis=1) - m) / (m * (m - 1))
        P_bar = np.mean(P_i)
        P_e = np.sum(p_j ** 2)
        if P_e >= 1.0:
            return 1.0
        return float((P_bar - P_e) / (1.0 - P_e))

    def fleiss_kappa_bootstrap(
        self, annotations: np.ndarray, n_bootstrap: int = 1000, seed: int = 42
    ) -> Tuple[float, Tuple[float, float]]:
        rng = np.random.RandomState(seed)
        n = annotations.shape[0]
        kappas = [
            self.fleiss_kappa(annotations[rng.choice(n, n, replace=True)])
            for _ in range(n_bootstrap)
        ]
        k = self.fleiss_kappa(annotations)
        ci_low = float(np.percentile(kappas, 2.5))
        ci_high = float(np.percentile(kappas, 97.5))
        return k, (ci_low, ci_high)

    def cohens_kappa(self, rater1: np.ndarray, rater2: np.ndarray) -> float:
        k = len(self.categories)
        n = len(rater1)
        cm = np.zeros((k, k), dtype=np.float64)
        for i in range(n):
            cm[int(rater1[i]), int(rater2[i])] += 1
        p_o = np.trace(cm) / n
        row_sums = cm.sum(axis=1) / n
        col_sums = cm.sum(axis=0) / n
        p_e = np.sum(row_sums * col_sums)
        if p_e >= 1.0:
            return 1.0
        return float((p_o - p_e) / (1.0 - p_e))

    def pairwise_cohens_kappa_all(
        self, annotations: np.ndarray
    ) -> Dict[str, float]:
        m = annotations.shape[1]
        results = {}
        for i in range(m):
            for j in range(i + 1, m):
                kappa = self.cohens_kappa(annotations[:, i], annotations[:, j])
                results[f"annotator_{i}_vs_{j}"] = kappa
        return results

    def analyze(
        self,
        annotations: np.ndarray,
    ) -> dict:
        n_samples, n_annotators = annotations.shape
        if n_annotators < 2:
            return {
                "fleiss_kappa": 1.0,
                "ci": (1.0, 1.0),
                "pairwise": {},
                "is_reliable": False,
                "level": "unknown",
            }
        fk, fk_ci = self.fleiss_kappa_bootstrap(annotations)
        pairwise = self.pairwise_cohens_kappa_all(annotations)
        is_reliable = fk >= 0.60
        level = next(
            (l for (lo, hi), l in self._RELIABILITY.items() if lo <= fk < hi),
            "poor",
        )
        if not is_reliable:
            logger.warning(
                f"E7 GOLD STANDARD UNRELIABLE: Fleiss' Kappa = {fk:.3f} < 0.60. "
                f"All IAA and GCP conclusions are conditional on fixing annotator disagreement."
            )
        else:
            logger.info(
                f"E7 gold standard validated: Fleiss' Kappa = {fk:.3f} "
                f"(95% CI: [{fk_ci[0]:.3f}, {fk_ci[1]:.3f}]), Reliability: {level}"
            )
        return {
            "fleiss_kappa": fk,
            "ci": fk_ci,
            "pairwise": pairwise,
            "is_reliable": is_reliable,
            "level": level,
        }


AnnotatorAgreementResult = type(
    "AnnotatorAgreementResult",
    (),
    {"__annotations__": {"n_annotators": int, "n_samples": int}},
)
