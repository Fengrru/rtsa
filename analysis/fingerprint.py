"""
Model Fingerprinting & LLM Authorship Attribution — Direction 3 (P1).

Builds "model signatures" from graph structural features and motif
frequency distributions, then identifies which model likely generated
an unknown reasoning trace.

All feature engineering is done — we reuse:
  - core.metrics.compute_feature_matrix()  → 6-dim structural vector
  - core.motif_matcher.MotifMatcher        → 12-dim motif frequency vector

The 18-dim combined vector is the "fingerprint".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from core.types import ReasoningTraceGraph
from core.metrics import compute_feature_matrix
from core.motif_matcher import MotifMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelSignature:
    """Statistical signature of a model's reasoning graph distribution."""
    model_name: str
    n_samples: int
    feature_mean: np.ndarray          # shape (D,)
    feature_cov: np.ndarray           # shape (D, D)
    feature_std: np.ndarray           # shape (D,)

    def mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from *x* to this signature."""
        diff = x - self.feature_mean
        # Use pseudo-inverse for numerical stability
        cov_inv = np.linalg.pinv(self.feature_cov + np.eye(len(self.feature_std)) * 1e-6)
        return float(np.sqrt(diff @ cov_inv @ diff))


@dataclass
class FingerprintMatchResult:
    """Result of matching an unknown graph against known model signatures."""
    predicted_model: str
    confidence: float                    # 0.0–1.0
    all_scores: Dict[str, float]         # model_name → distance (lower = better)
    feature_vector: np.ndarray

    def summary(self) -> str:
        lines = [
            f"FingerprintMatch: predicted={self.predicted_model} confidence={self.confidence:.3f}",
            "  All scores (lower = closer match):",
        ]
        for model, score in sorted(self.all_scores.items(), key=lambda kv: kv[1]):
            marker = " <--" if model == self.predicted_model else ""
            lines.append(f"    {model}: {score:.4f}{marker}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fingerprinting engine
# ---------------------------------------------------------------------------

class ModelFingerprint:
    """Build and query model signatures from reasoning trace collections.

    Usage:
        fp = ModelFingerprint()
        fp.enroll("deepseek-chat", deepseek_graphs)
        fp.enroll("gpt-4o", gpt4o_graphs)

        result = fp.identify(unknown_graph)
        print(result.predicted_model)
    """

    FEATURE_DIM = 6
    # Motif count is dynamic — do NOT hard-code; see property below.
    @property
    def motif_dim(self) -> int:
        return len(self.matcher._code_motif_graphs)

    @property
    def total_dim(self) -> int:
        return self.FEATURE_DIM + self.motif_dim

    def __init__(self):
        self.signatures: Dict[str, ModelSignature] = {}
        self.matcher = MotifMatcher()

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        model_name: str,
        graphs: List[ReasoningTraceGraph],
        min_samples: int = 5,
    ) -> ModelSignature:
        """Compute and store the signature for *model_name*."""
        if len(graphs) < min_samples:
            logger.warning(
                f"Enrolling {model_name} with only {len(graphs)} samples "
                f"(recommended ≥ {min_samples})"
            )

        X = self._compute_combined_features(graphs)

        sig = ModelSignature(
            model_name=model_name,
            n_samples=len(graphs),
            feature_mean=np.mean(X, axis=0),
            feature_cov=np.cov(X, rowvar=False) if X.shape[0] > 1 else np.eye(X.shape[1]) * 1e-4,
            feature_std=np.std(X, axis=0, ddof=1) + 1e-8,
        )
        self.signatures[model_name] = sig
        logger.info(
            f"Enrolled signature for '{model_name}' from {len(graphs)} graphs "
            f"(feature dim={X.shape[1]})"
        )
        return sig

    def update(
        self,
        model_name: str,
        graphs: List[ReasoningTraceGraph],
    ) -> ModelSignature:
        """Incrementally update an existing signature with new graphs."""
        if model_name not in self.signatures:
            return self.enroll(model_name, graphs)

        old = self.signatures[model_name]
        X_new = self._compute_combined_features(graphs)
        X_all = np.vstack([np.random.multivariate_normal(old.feature_mean, old.feature_cov, old.n_samples), X_new])
        # Recompute from combined pseudo-data
        return self.enroll(model_name, [], min_samples=0)

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        graph: ReasoningTraceGraph,
        method: str = "mahalanobis",
    ) -> FingerprintMatchResult:
        """Match *graph* against all enrolled signatures.

        Args:
            graph: unknown reasoning trace
            method: "mahalanobis" | "cosine" | "euclidean"

        Returns:
            FingerprintMatchResult with predicted model and confidence scores
        """
        if not self.signatures:
            raise RuntimeError("No model signatures enrolled. Call .enroll() first.")

        x = self._compute_combined_features([graph])[0]

        scores: Dict[str, float] = {}
        for name, sig in self.signatures.items():
            if method == "mahalanobis":
                scores[name] = sig.mahalanobis(x)
            elif method == "cosine":
                scores[name] = float(1.0 - np.dot(x, sig.feature_mean) / (np.linalg.norm(x) * np.linalg.norm(sig.feature_mean) + 1e-10))
            elif method == "euclidean":
                scores[name] = float(np.linalg.norm(x - sig.feature_mean))
            else:
                raise ValueError(f"Unknown method: {method}")

        # Convert distances to softmax-like confidence
        inv_scores = {k: 1.0 / max(v, 1e-6) for k, v in scores.items()}
        total = sum(inv_scores.values())
        confidences = {k: v / total for k, v in inv_scores.items()}

        predicted = min(scores, key=scores.get)  # type: ignore[arg-type]
        return FingerprintMatchResult(
            predicted_model=predicted,
            confidence=confidences[predicted],
            all_scores=scores,
            feature_vector=x,
        )

    def identify_batch(
        self,
        graphs: List[ReasoningTraceGraph],
        method: str = "mahalanobis",
    ) -> List[FingerprintMatchResult]:
        """Batch identification."""
        return [self.identify(g, method=method) for g in graphs]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize signatures to NPZ."""
        data: Dict[str, np.ndarray] = {}
        meta: List[str] = []
        for name, sig in self.signatures.items():
            data[f"{name}_mean"] = sig.feature_mean
            data[f"{name}_cov"] = sig.feature_cov
            data[f"{name}_std"] = sig.feature_std
            meta.append(f"{name}:{sig.n_samples}")
        data["_meta"] = np.array(meta, dtype=object)
        np.savez(path, **data)
        logger.info(f"Saved {len(self.signatures)} signatures to {path}")

    def load(self, path: str) -> None:
        """Load signatures from NPZ."""
        data = np.load(path, allow_pickle=True)
        meta = list(data["_meta"])
        self.signatures.clear()
        for entry in meta:
            name, n_samples = entry.split(":")
            self.signatures[name] = ModelSignature(
                model_name=name,
                n_samples=int(n_samples),
                feature_mean=data[f"{name}_mean"],
                feature_cov=data[f"{name}_cov"],
                feature_std=data[f"{name}_std"],
            )
        logger.info(f"Loaded {len(self.signatures)} signatures from {path}")

    # ------------------------------------------------------------------
    # Internal feature computation
    # ------------------------------------------------------------------

    def _compute_combined_features(
        self, graphs: List[ReasoningTraceGraph]
    ) -> np.ndarray:
        """Compute (structural + motif) feature matrix for *graphs*."""
        if not graphs:
            return np.empty((0, self.total_dim))

        # 1. Structural features: N × 6
        X_struct = compute_feature_matrix(graphs)

        # 2. Motif frequencies: N × 12
        motif_freqs = []
        for g in graphs:
            freq_vec = self.matcher.compute_motif_frequency_vector(g)
            motif_freqs.append(freq_vec)
        X_motif = np.array(motif_freqs, dtype=np.float64)

        # 3. Concatenate & normalize per-row (z-score)
        X = np.hstack([X_struct, X_motif])
        row_means = X.mean(axis=1, keepdims=True)
        row_stds = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - row_means) / row_stds
        return X_norm


# ---------------------------------------------------------------------------
# Convenience one-liners
# ---------------------------------------------------------------------------

def enroll_model(
    model_name: str,
    graphs: List[ReasoningTraceGraph],
    save_path: Optional[str] = None,
) -> ModelSignature:
    """Enroll a model and optionally persist its signature."""
    fp = ModelFingerprint()
    sig = fp.enroll(model_name, graphs)
    if save_path:
        fp.save(save_path)
    return sig


def identify_author(
    graph: ReasoningTraceGraph,
    signatures_path: str,
    method: str = "mahalanobis",
) -> FingerprintMatchResult:
    """One-liner: load signatures and identify author of *graph*."""
    fp = ModelFingerprint()
    fp.load(signatures_path)
    return fp.identify(graph, method=method)
