"""Experiments: Phase 0.5 Pilot + Phase 3 ER-Corrected Generalization.

Fix 6: Power analysis for minimum sample size.
Fix 7: Multi-family model registry (Qwen, Llama, DeepSeek, Mixtral).
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from rtsa.core.metrics import compute_graph_features
from rtsa.core.motif_matcher import MotifMatcher
from rtsa.core.types import ReasoningTraceGraph
from .gcp_validator import GCPValidator, GCS_CORPUS_FULL, make_gcp_adapter
from .synthetic_validation import SyntheticValidator
from .agreement import compute_full_iaa
from .baselines import JPDirectedPreservingRandomizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0.5 Pilot
# ---------------------------------------------------------------------------

@dataclass
class PilotResult:
    n_cots: int
    gcp_results: Dict[str, bool]
    synthetic_results: Dict[str, bool]
    preliminary_iaa: Dict[str, Dict[str, float]]
    dpr_runtime_seconds: float
    decision: str
    recommendation: str


class Phase05Pilot:
    """2-week circuit breaker: verify core objects measurable before full commitment."""

    def __init__(self):
        self.gcp_validator = GCPValidator(corpus=GCS_CORPUS_FULL)
        self.synthetic_validator = SyntheticValidator()

    def run(self, cots: List[str], extractors: Dict[str, object]) -> PilotResult:
        logger.info(f"Phase 0.5 Pilot: {len(cots)} CoT traces, {len(extractors)} extractors")

        # Step 1: GCP calibration (use make_gcp_adapter for sentence-level extractors)
        gcp_results = {}
        for name, ext in extractors.items():
            if hasattr(ext, "classify_sentence"):
                wrapped = make_gcp_adapter(ext.classify_sentence)
            else:
                wrapped = lambda s, e=ext: [n.type for n in e.extract(s).nodes]
            r = self.gcp_validator.calibrate_extractor(wrapped, name)
            gcp_results[name] = r.passed

        # Step 2: Synthetic validation
        synthetic_results = {}
        for name, ext in extractors.items():
            r = self.synthetic_validator.validate_extractor(lambda t, e=ext: e.extract(t), name)
            synthetic_results[name] = self.synthetic_validator.is_extractor_viable(r)

        # Step 3: Full extraction
        graphs_by_extractor = {}
        for name, ext in extractors.items():
            gs = []
            for i, cot in enumerate(cots):
                try:
                    gs.append(ext.extract(cot, trace_id=f"pilot_{i}"))
                except Exception as e:
                    logger.warning(f"Extraction failed for {name} on trace {i}: {e}")
            graphs_by_extractor[name] = gs

        # Step 4: Preliminary IAA
        passed = {n: graphs_by_extractor[n] for n in graphs_by_extractor if gcp_results.get(n, False)}
        preliminary_iaa = compute_full_iaa(passed) if passed else {}

        # Step 5: JP-DPR timing
        dpr = JPDirectedPreservingRandomizer()
        t0 = time.time()
        if graphs_by_extractor:
            sample_graphs = list(graphs_by_extractor.values())[0][:5]
            for g in sample_graphs:
                if g.nodes:
                    dpr.randomize(g, k=10)
        dpr_runtime = time.time() - t0

        # Step 6: Go/No-Go
        graph_iaa = preliminary_iaa.get("graph_level", {}).get("fleiss_kappa", 0.0)
        n_gcp = sum(gcp_results.values())
        n_syn = sum(synthetic_results.values())

        if graph_iaa > 0.5 and n_gcp >= 2 and n_syn >= 1 and dpr_runtime < 300:
            decision, recommendation = "GO", f"Pilot passed: IAA={graph_iaa:.3f}>0.5, {n_gcp} GCP-passed, {n_syn} synthetic-viable. Proceed to full Phase 0."
        elif graph_iaa < 0.3 or n_gcp == 0:
            decision, recommendation = "NO-GO", f"Pilot failed: IAA={graph_iaa:.3f}<0.3 or no GCP-passed extractors. TERMINATE. Loss: ~$15 + 2 weeks."
        else:
            decision, recommendation = "CONDITIONAL_GO", f"Marginal: IAA={graph_iaa:.3f} in [0.3, 0.5]. Reduce Phase 0 to 50 CoT."

        return PilotResult(len(cots), gcp_results, synthetic_results, preliminary_iaa, dpr_runtime, decision, recommendation)


# ---------------------------------------------------------------------------
# Multi-Family Model Registry (Fix 7)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "qwen2.5-7b": {"family": "qwen", "size_b": 7, "type": "dense"},
    "qwen2.5-32b": {"family": "qwen", "size_b": 32, "type": "dense"},
    "qwen2.5-72b": {"family": "qwen", "size_b": 72, "type": "dense"},
    "llama-3-8b": {"family": "llama", "size_b": 8, "type": "dense"},
    "llama-3-70b": {"family": "llama", "size_b": 70, "type": "dense"},
    "deepseek-v2-lite": {"family": "deepseek", "size_b": 16, "type": "moe"},
    "mixtral-8x7b": {"family": "mixtral", "size_b": 47, "type": "moe"},
}

MODEL_FAMILIES = sorted(set(m["family"] for m in MODEL_REGISTRY.values()))


def get_models_by_family(family: str) -> List[str]:
    return [n for n, i in MODEL_REGISTRY.items() if i["family"] == family]


def get_cross_family_models(n_per_family: int = 1) -> List[str]:
    return [get_models_by_family(f)[0] for f in MODEL_FAMILIES if get_models_by_family(f)]


# ---------------------------------------------------------------------------
# Power Analysis (Fix 6)
# ---------------------------------------------------------------------------

@dataclass
class PowerAnalysisResult:
    target_effect_size: float
    alpha: float
    power: float
    min_samples_per_group: int
    min_total_samples: int
    with_er_correction: int
    recommendation: str


def power_analysis_logistic(
    n_features: int = 5,
    target_auc_improvement: float = 0.05,
    baseline_auc: float = 0.60,
    alpha: float = 0.05,
    power: float = 0.80,
    extraction_rate: float = 0.60,
) -> PowerAnalysisResult:
    from scipy.stats import norm

    def auc_to_cohens_h(a1, a2):
        return abs(norm.ppf(a2) - norm.ppf(a1))

    effect_size = auc_to_cohens_h(baseline_auc, baseline_auc + target_auc_improvement)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n_per_group = int(np.ceil((z_alpha + z_beta) ** 2 / (effect_size ** 2 + 1e-10) + n_features * 10))
    n_total = n_per_group * 2
    n_total_er = int(np.ceil(n_total / max(extraction_rate, 0.1)))
    sufficient = n_total_er <= 500

    return PowerAnalysisResult(
        target_effect_size=effect_size,
        alpha=alpha,
        power=power,
        min_samples_per_group=n_per_group,
        min_total_samples=n_total,
        with_er_correction=n_total_er,
        recommendation=(
            f"Requires >={n_total_er} OOD samples with ER={extraction_rate:.0%}. "
            f"{'Sufficient for Phase 3.' if sufficient else 'INSUFFICIENT: increase sample size or accept lower power.'}"
        ),
    )


# ---------------------------------------------------------------------------
# ER-Corrected Prediction (Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class ERCorrectedResult:
    extraction_rate: float
    n_total: int
    n_extractable: int
    auc_topology: float
    auc_baseline_length: float
    auc_baseline_difficulty: float
    auc_baseline_combined: float
    auc_topology_combined: float
    auc_improvement: float
    is_significant: bool
    sensitivity_pessimistic: float
    sensitivity_optimistic: float


class ERCorrectedPredictor:
    def __init__(self, motif_matcher=None, n_folds: int = 5, seed: int = 42):
        self.mm = motif_matcher or MotifMatcher()
        self.n_folds = n_folds
        self.seed = seed

    def extract_features(self, graph, cot_length: int = 0, difficulty: float = 0.5) -> np.ndarray:
        n_motifs = len(self.mm.preset_motifs)
        if graph is None or not graph.nodes:
            return np.zeros(n_motifs + 4 + 2, dtype=np.float64)
        motif_vec = self.mm.compute_motif_frequency_vector(graph)
        metrics = compute_graph_features(graph)
        l1 = np.array([metrics.depth, metrics.branching, metrics.verify_density, metrics.entropy])
        return np.concatenate([motif_vec, l1, np.array([cot_length, difficulty])])

    def fit_predict(self, graphs, targets, cot_lengths, difficulties) -> ERCorrectedResult:
        n_total = len(graphs)
        n_extractable = sum(1 for g in graphs if g is not None and g.nodes)
        er = n_extractable / max(n_total, 1)

        X_topo = np.array([self.extract_features(g, l, d) for g, l, d in zip(graphs, cot_lengths, difficulties)])
        X_baseline = np.column_stack([cot_lengths, difficulties])
        X_combined = np.column_stack([X_topo, X_baseline])

        X_topo_s = StandardScaler().fit_transform(X_topo)
        X_baseline_s = StandardScaler().fit_transform(X_baseline)
        X_combined_s = StandardScaler().fit_transform(X_combined)

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        clf = LogisticRegression(max_iter=1000, random_state=self.seed)

        def cv_auc(X, y):
            preds = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
            return float(roc_auc_score(y, preds[:, 1]))

        auc_topo = cv_auc(X_topo_s, targets)
        auc_len = cv_auc(X_baseline_s[:, [0]], targets)
        auc_diff = cv_auc(X_baseline_s[:, [1]], targets)
        auc_b3 = cv_auc(X_baseline_s, targets)
        auc_t2 = cv_auc(X_combined_s, targets)

        improvement = auc_t2 - auc_b3
        is_sig = improvement > 0.05

        targets_pess = targets.copy()
        targets_opt = targets.copy()
        for i, g in enumerate(graphs):
            if g is None or not g.nodes:
                targets_pess[i] = 0
                targets_opt[i] = 1
        auc_pess = cv_auc(X_combined_s, targets_pess)
        auc_opt = cv_auc(X_combined_s, targets_opt)

        return ERCorrectedResult(
            extraction_rate=er, n_total=n_total, n_extractable=n_extractable,
            auc_topology=auc_topo, auc_baseline_length=auc_len,
            auc_baseline_difficulty=auc_diff, auc_baseline_combined=auc_b3,
            auc_topology_combined=auc_t2, auc_improvement=improvement,
            is_significant=is_sig, sensitivity_pessimistic=auc_pess,
            sensitivity_optimistic=auc_opt,
        )
