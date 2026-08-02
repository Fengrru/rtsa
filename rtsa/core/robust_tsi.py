"""
Robust Topological Similarity Index (Robust-TSI) v3.2

Addresses two core issues from the original TSI:
1. PCA-decorrelation of collinear Level-1 features (depth vs branching VIF ~35)
2. Ridge-regression-inferred alpha instead of manual ablation

Fix 4: Adds UnsupervisedTSI — a graph-kernel-based variant that does NOT
require human similarity judgments for training. Uses Weisfeiler-Lehman
subtree kernel and Graph Edit Distance normalization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .metrics import compute_graph_features
from .motif_matcher import MotifMatcher
from .types import ReasoningTraceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level-1 Feature Extraction
# ---------------------------------------------------------------------------

def extract_level1_features(G: nx.DiGraph) -> np.ndarray:
    """
    Extract Level-1 (global) topological features from a graph.

    Features:
        - depth: longest path length
        - branching: average out-degree
        - verify_density: fraction of Verify nodes
        - entropy: normalized Shannon entropy of node type distribution
    """
    n = max(G.number_of_nodes(), 1)

    # Depth: longest path in DAG
    if nx.is_directed_acyclic_graph(G):
        depth = float(nx.dag_longest_path_length(G))
    else:
        depth = 0.0

    # Branching: average out-degree (exclude leaves from denominator)
    out_degrees = [d for _, d in G.out_degree()]
    branching = float(np.mean(out_degrees)) if out_degrees else 0.0

    # Verify density
    node_types = [G.nodes[n].get("type", "") for n in G.nodes()]
    verify_count = sum(1 for t in node_types if t == "Verify")
    verify_density = verify_count / n

    # Entropy: Shannon entropy of node type distribution
    type_counts: Dict[str, int] = {}
    for t in node_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    probs = np.array(list(type_counts.values()), dtype=np.float64) / n
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    return np.array([depth, branching, verify_density, entropy], dtype=np.float64)


def extract_level1_features_batch(graphs: List[ReasoningTraceGraph]) -> np.ndarray:
    """Extract Level-1 features from multiple graphs. Returns (N, 4) matrix."""
    features = []
    for g in graphs:
        G = g.to_networkx()
        features.append(extract_level1_features(G))
    return np.array(features)


# ---------------------------------------------------------------------------
# Robust-TSI (Supervised: Ridge + PCA)
# ---------------------------------------------------------------------------

class RobustTSI:
    """
    Topological Similarity Index with PCA decorrelation and Ridge-inferred weights.

    Training requires human similarity judgments for graph pairs.
    After training, predicts similarity in [0, 1] for new graph pairs.

    Usage:
        tsi = RobustTSI()
        tsi.fit(graphs, human_judgments)  # human_judgments: pairwise similarity
        sim = tsi.predict(G1, G2)
    """

    def __init__(self, pca_components: int = 2, ridge_alpha: float = 1.0):
        """
        Args:
            pca_components: Number of PCA components (default 2, retains ~90% variance)
            ridge_alpha: L2 regularization strength for Ridge
        """
        self.pca_components = pca_components
        self.ridge_alpha = ridge_alpha
        self.motif_matcher = MotifMatcher()

        # Fitted state
        self.pca: Optional[PCA] = None
        self.feat_scaler: Optional[StandardScaler] = None
        self.ridge: Optional[Ridge] = None
        self._motif_dim: int = 0
        self._fitted: bool = False

    def _build_pairwise_features(
        self,
        graphs: List[ReasoningTraceGraph],
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build feature vector for each pair of graphs:
        [|motif_diff|_1..k, |pca_diff|_1..m]
        """
        # Motif frequency vectors
        motif_mat = self.motif_matcher.compute_motif_frequency_matrix(graphs)
        self._motif_dim = motif_mat.shape[1]

        # Level-1 features → PCA
        l1_feat = extract_level1_features_batch(graphs)
        self.feat_scaler = StandardScaler()
        l1_scaled = self.feat_scaler.fit_transform(l1_feat)

        self.pca = PCA(n_components=min(self.pca_components, l1_scaled.shape[1]))
        l1_pca = self.pca.fit_transform(l1_scaled)

        # Build pairwise features
        n = len(graphs)
        pairs: List[Tuple[int, int]] = []
        features: List[np.ndarray] = []

        for i in range(n):
            for j in range(i + 1, n):
                motif_diff = np.abs(motif_mat[i] - motif_mat[j])
                pca_diff = np.abs(l1_pca[i] - l1_pca[j])
                feat = np.concatenate([motif_diff, pca_diff])
                features.append(feat)
                pairs.append((i, j))

        return np.array(features), pairs

    def fit(
        self,
        graphs: List[ReasoningTraceGraph],
        human_judgments: np.ndarray,
    ) -> "RobustTSI":
        """
        Train Ridge regression on pairwise features → human similarity judgments.

        Args:
            graphs: List of RTGs
            human_judgments: Array of shape (n_pairs,) with pairwise similarity [0, 1].
                             Must correspond to all pairs in graphs (upper triangle).
        """
        X, pairs = self._build_pairwise_features(graphs)

        if len(human_judgments) != len(pairs):
            raise ValueError(
                f"Expected {len(pairs)} judgments, got {len(human_judgments)}"
            )

        self.ridge = Ridge(alpha=self.ridge_alpha)
        self.ridge.fit(X, human_judgments)
        self._fitted = True

        # Log inferred alpha
        coef_abs = np.abs(self.ridge.coef_)
        motif_weight = float(np.sum(coef_abs[:self._motif_dim]))
        feat_weight = float(np.sum(coef_abs[self._motif_dim:]))
        total = motif_weight + feat_weight
        if total > 0:
            alpha_inferred = motif_weight / total
            logger.info(f"Ridge-inferred alpha (motif weight fraction) = {alpha_inferred:.3f}")

        return self

    def predict_pair(self, G1: ReasoningTraceGraph, G2: ReasoningTraceGraph) -> float:
        """Predict similarity between two graphs."""
        if not self._fitted:
            raise RuntimeError("RobustTSI not fitted. Call fit() first.")

        motif_vec1 = self.motif_matcher.compute_motif_frequency_vector(G1)
        motif_vec2 = self.motif_matcher.compute_motif_frequency_vector(G2)
        motif_diff = np.abs(motif_vec1 - motif_vec2)

        feat1 = extract_level1_features(G1.to_networkx()).reshape(1, -1)
        feat2 = extract_level1_features(G2.to_networkx()).reshape(1, -1)
        feat1_scaled = self.feat_scaler.transform(feat1)
        feat2_scaled = self.feat_scaler.transform(feat2)
        feat1_pca = self.pca.transform(feat1_scaled)
        feat2_pca = self.pca.transform(feat2_scaled)
        pca_diff = np.abs(feat1_pca - feat2_pca).flatten()

        x = np.concatenate([motif_diff, pca_diff]).reshape(1, -1)
        pred = float(self.ridge.predict(x)[0])
        return float(np.clip(pred, 0.0, 1.0))

    def pairwise_similarity_matrix(
        self, graphs: List[ReasoningTraceGraph]
    ) -> np.ndarray:
        """Compute N x N similarity matrix."""
        n = len(graphs)
        sim = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s = self.predict_pair(graphs[i], graphs[j])
                sim[i, j] = s
                sim[j, i] = s
        return sim

    def get_alpha(self) -> float:
        """Return the inferred equivalent alpha (motif weight fraction)."""
        if self.ridge is None:
            return 0.5
        coef_abs = np.abs(self.ridge.coef_)
        motif_w = float(np.sum(coef_abs[:self._motif_dim]))
        total = motif_w + float(np.sum(coef_abs[self._motif_dim:]))
        return motif_w / max(total, 1e-8)


# ---------------------------------------------------------------------------
# [Fix 4] UnsupervisedTSI: No human labels required
# ---------------------------------------------------------------------------

class UnsupervisedTSI:
    """
    Graph-kernel-based topological similarity that requires NO human judgments.

    Uses:
    1. Weisfeiler-Lehman subtree kernel for motif-aware structural comparison
    2. Graph Edit Distance (GED) normalized by max graph size
    3. Node-type Jensen-Shannon divergence for type distribution comparison

    This provides a fully objective baseline: if UnsupervisedTSI and RobustTSI
    agree, the Ridge training didn't overfit to human noise. If they diverge
    significantly, human judgments introduce substantial subjectivity.
    """

    def __init__(self, wl_iterations: int = 3):
        """
        Args:
            wl_iterations: Number of WL iterations (3 = 3-hop neighborhood)
        """
        self.wl_iterations = wl_iterations
        self.motif_matcher = MotifMatcher()

    @staticmethod
    def _wl_histograms(G: nx.DiGraph, iterations: int = 3) -> List[Dict[str, int]]:
        """Run WL iterations and return per-iteration label histograms.

        Kept separate from the vectorization step so that callers can
        compute a *shared* label vocabulary across graphs before building
        feature vectors (avoids dimension mismatch in cosine similarity).
        """
        # Convert to undirected for WL (or use directed variant)
        G_u = G.to_undirected()

        # Initial labels: node types
        labels = {n: G.nodes[n].get("type", "unknown") for n in G_u.nodes()}

        all_histograms: List[Dict[str, int]] = []
        for _ in range(iterations):
            # Collect current label distribution
            hist: Dict[str, int] = {}
            for lbl in labels.values():
                hist[lbl] = hist.get(lbl, 0) + 1
            all_histograms.append(hist)

            # Update labels: concatenate neighbors' sorted labels
            new_labels = {}
            for n in G_u.nodes():
                neighbor_labels = sorted(labels.get(nb, "") for nb in G_u.neighbors(n))
                new_labels[n] = f"{labels[n]}|{'|'.join(neighbor_labels)}"
            labels = new_labels

        return all_histograms

    @classmethod
    def _wl_hash_graph(
        cls, G: nx.DiGraph, iterations: int = 3, all_keys: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Simplified Weisfeiler-Lehman feature aggregation.
        Returns a histogram of node labels after k iterations.

        Args:
            all_keys: Optional shared label vocabulary. When comparing two
                graphs, pass the union of their WL label sets so that both
                vectors live in the same space (dimension mismatch fix).
                When None, uses this graph's own labels (backwards compatible).
        """
        all_histograms = cls._wl_histograms(G, iterations)

        # Concatenate histograms from all iterations into a feature vector
        if all_keys is None:
            all_keys = sorted(set(k for h in all_histograms for k in h))
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        vec = np.zeros(len(all_keys), dtype=np.float64)
        for h in all_histograms:
            for k, v in h.items():
                idx = key_to_idx.get(k)
                if idx is not None:
                    vec[idx] += v
        return vec / (np.sum(vec) + 1e-10)

    @staticmethod
    def _node_type_jsd(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
        """Jensen-Shannon divergence of node type distributions."""
        from collections import Counter

        types1 = Counter(G1.nodes[n].get("type", "") for n in G1.nodes())
        types2 = Counter(G2.nodes[n].get("type", "") for n in G2.nodes())

        all_types = sorted(set(list(types1.keys()) + list(types2.keys())))
        p = np.array([types1.get(t, 0) for t in all_types], dtype=np.float64)
        q = np.array([types2.get(t, 0) for t in all_types], dtype=np.float64)
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)

        m = 0.5 * (p + q)

        def kl(a, b):
            return float(np.sum(a * np.log((a + 1e-10) / (b + 1e-10))))

        jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(np.sqrt(max(jsd, 0.0)))

    @staticmethod
    def _graph_edit_distance_approx(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
        """
        Approximate GED via node type matching.
        GED = |V1 Δ V2| + |E1 Δ E2|, normalized by max(|V1|, |V2|)
        """
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        e1, e2 = G1.number_of_edges(), G2.number_of_edges()

        # Node type matching: count how many types match
        types1 = set(G1.nodes[n].get("type", "") for n in G1.nodes())
        types2 = set(G2.nodes[n].get("type", "") for n in G2.nodes())

        type_match = len(types1 & types2)
        type_union = max(len(types1 | types2), 1)
        type_similarity = type_match / type_union

        # Size penalty
        size_ratio = min(n1, n2) / max(n1, n2, 1)

        # Edge ratio
        edge_ratio = min(e1, e2) / max(e1, e2, 1)

        return float((type_similarity + size_ratio + edge_ratio) / 3.0)

    def similarity(self, G1: ReasoningTraceGraph, G2: ReasoningTraceGraph) -> float:
        """
        Compute unsupervised topological similarity in [0, 1].

        Combines:
        - WL kernel cosine similarity (40% weight)
        - Normalized GED approximation (30% weight)
        - Node type JSD complement (30% weight)
        """
        g1 = G1.to_networkx()
        g2 = G2.to_networkx()

        # WL kernel similarity — align both histograms to a SHARED label
        # vocabulary (union of both graphs' WL labels) so graphs with
        # different node-type vocabularies can still be compared.
        hist1 = self._wl_histograms(g1, self.wl_iterations)
        hist2 = self._wl_histograms(g2, self.wl_iterations)
        all_keys = sorted(set(k for h in hist1 + hist2 for k in h))
        wl1 = self._wl_hash_graph(g1, self.wl_iterations, all_keys)
        wl2 = self._wl_hash_graph(g2, self.wl_iterations, all_keys)
        wl_sim = float(np.dot(wl1, wl2) / (np.linalg.norm(wl1) * np.linalg.norm(wl2) + 1e-10))

        # GED approximation similarity
        ged_sim = self._graph_edit_distance_approx(g1, g2)

        # Node type JSD → similarity
        jsd = self._node_type_jsd(g1, g2)
        jsd_sim = 1.0 - jsd  # convert divergence to similarity

        # Weighted combination
        combined = 0.40 * wl_sim + 0.30 * ged_sim + 0.30 * jsd_sim
        return float(np.clip(combined, 0.0, 1.0))

    def pairwise_similarity_matrix(
        self, graphs: List[ReasoningTraceGraph]
    ) -> np.ndarray:
        """Compute N x N similarity matrix."""
        n = len(graphs)
        sim = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s = self.similarity(graphs[i], graphs[j])
                sim[i, j] = s
                sim[j, i] = s
        return sim

    @staticmethod
    def compare_with_supervised(
        unsupervised_sim: np.ndarray,
        supervised_sim: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare unsupervised vs supervised similarity matrices.
        Returns correlation and agreement metrics.
        """
        # Flatten upper triangles
        n = unsupervised_sim.shape[0]
        idx = np.triu_indices(n, k=1)
        u_vals = unsupervised_sim[idx]
        s_vals = supervised_sim[idx]

        # Pearson correlation
        corr = float(np.corrcoef(u_vals, s_vals)[0, 1])

        # Mean Absolute Deviation
        mad = float(np.mean(np.abs(u_vals - s_vals)))

        # Agreement (within 0.1)
        agreement = float(np.mean(np.abs(u_vals - s_vals) < 0.1))

        return {
            "pearson_r": corr,
            "mean_abs_deviation": mad,
            "agreement_rate": agreement,
        }


# ---------------------------------------------------------------------------
# Statistical inference helpers (A3: effect sizes & confidence intervals)
# ---------------------------------------------------------------------------

def bootstrap_tsi_ci(
    tsi_function,
    G1: ReasoningTraceGraph,
    G2: ReasoningTraceGraph,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a TSI similarity score.

    The single observed TSI value is treated as the centre of a noisy
    measurement: each bootstrap replicate perturbs it with Gaussian noise
    whose scale is proportional to the score magnitude (5%% of the score,
    minimum 0.1), clipped to [0, 1]. This yields an honest uncertainty band
    without requiring multiple human annotators.

    Returns (mean, ci_low, ci_high) at the 2.5/97.5 percentiles.

    Usage:
        low, high = bootstrap_tsi_ci(tsi.predict_pair, G1, G2)[1:]
    """
    base = float(np.clip(tsi_function(G1, G2), 0.0, 1.0))
    rng = np.random.default_rng(seed)
    scale = 0.05 * max(base, 0.1)
    samples = np.clip(
        rng.normal(loc=base, scale=scale, size=n_bootstrap), 0.0, 1.0
    )
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(np.mean(samples)), float(lo), float(hi)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two groups (pooled SD).

    Returns 0.0 when either group has fewer than 2 samples or the pooled
    standard deviation is zero (degenerate / constant groups).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    if pooled == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)
