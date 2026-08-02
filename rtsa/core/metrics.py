"""
Topology Metrics for Reasoning Trace Graphs (RTSA v3.2 — merged from v3.1).

Provides three layers of graph similarity:
1. **Level-1 Global Features**: depth, branching, verify density, entropy, per-type counts
2. **Weisfeiler-Lehman Subtree Kernel** — O(m·h) polynomial-time graph similarity
   (replaces NP-hard GED; Shervashidze et al., 2011)
3. **Topology Similarity Index (TSI)** — parameterized combination of motif JSD and feature distance

Also includes statistical utilities: bootstrap CI, stable rate, TDI, partial correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon

from .types import MOTIF_LOOKUP, MotifEntry, NodeType, ReasoningTraceGraph


# ---------------------------------------------------------------------------
# Level 1: Global Graph Features
# ---------------------------------------------------------------------------

@dataclass
class Level1Features:
    """Global descriptive graph features with per-type breakdown."""

    depth: float
    branching_factor: float
    verification_density: float
    backtrack_rate: float
    topological_entropy: float
    n_nodes: int
    n_edges: int
    n_retrieve: int = 0
    n_transform: int = 0
    n_compare: int = 0
    n_verify: int = 0
    n_branch: int = 0
    n_backtrack: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert to a numpy feature vector for downstream ML."""
        return np.array(
            [
                self.depth,
                self.branching_factor,
                self.verification_density,
                self.backtrack_rate,
                self.topological_entropy,
                self.n_nodes,
                self.n_edges,
            ],
            dtype=np.float64,
        )

    def normalize_vector(self) -> np.ndarray:
        """Normalize to [0, 1] range using heuristic bounds."""
        v = self.to_vector()
        max_vals = np.array([30.0, 5.0, 1.0, 1.0, 3.0, 100.0, 200.0])
        return np.clip(v / max_vals, 0.0, 1.0)


def compute_level1_features(G: nx.DiGraph) -> Level1Features:
    """Extract Level-1 global features from a reasoning trace graph.

    Args:
        G: A networkx.DiGraph representing an RTG.

    Returns:
        Level1Features dataclass with all computed metrics.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Node type counts
    type_counts: Dict[str, int] = {
        "Retrieve": 0, "Transform": 0, "Compare": 0,
        "Verify": 0, "Branch": 0, "Backtrack": 0,
    }
    for _, attrs in G.nodes(data=True):
        ntype = attrs.get("type", "Transform")
        if ntype in type_counts:
            type_counts[ntype] += 1

    # Depth: longest path length
    if n > 0 and nx.is_directed_acyclic_graph(G):
        try:
            depth = float(nx.dag_longest_path_length(G))
        except nx.NetworkXError:
            depth = 0.0
    else:
        depth = 0.0

    # Branching factor: mean out-degree (excluding leaves)
    out_degrees = [d for _, d in G.out_degree() if d > 0]
    branching_factor = float(np.mean(out_degrees)) if out_degrees else 0.0

    # Verification density
    verification_density = type_counts["Verify"] / n if n > 0 else 0.0

    # Backtrack rate
    backtrack_rate = type_counts["Backtrack"] / n if n > 0 else 0.0

    # Topological entropy: H = -sum(p_i * log(p_i))
    freqs = np.array(list(type_counts.values()), dtype=np.float64)
    freqs = freqs / freqs.sum()
    freqs = freqs[freqs > 0]
    entropy = -np.sum(freqs * np.log2(freqs)) if len(freqs) > 0 else 0.0

    return Level1Features(
        depth=depth,
        branching_factor=branching_factor,
        verification_density=verification_density,
        backtrack_rate=backtrack_rate,
        topological_entropy=entropy,
        n_nodes=n,
        n_edges=m,
        n_retrieve=type_counts["Retrieve"],
        n_transform=type_counts["Transform"],
        n_compare=type_counts["Compare"],
        n_verify=type_counts["Verify"],
        n_branch=type_counts["Branch"],
        n_backtrack=type_counts["Backtrack"],
    )


# ---------------------------------------------------------------------------
# GraphMetrics (v3.2 compact form, used by robust_tsi)
# ---------------------------------------------------------------------------

@dataclass
class GraphMetrics:
    """Aggregate topological metrics for a single RTG."""
    trace_id: str
    n_nodes: int
    n_edges: int
    depth: float
    branching: float
    verify_density: float
    backtrack_rate: float
    entropy: float
    avg_degree: float = 0.0
    graph_density: float = 0.0
    extraction_rate: float = 1.0


def compute_graph_features(graph: ReasoningTraceGraph) -> GraphMetrics:
    """Compute all graph-level metrics for a single RTG (v3.2 format)."""
    G = graph.to_networkx()
    n = max(G.number_of_nodes(), 1)
    m = G.number_of_edges()

    if nx.is_directed_acyclic_graph(G) and n > 0:
        depth = float(nx.dag_longest_path_length(G))
    else:
        depth = 0.0

    out_degs = [d for _, d in G.out_degree()]
    branching = float(np.mean(out_degs)) if out_degs else 0.0

    node_types = [G.nodes[n].get("type", "Transform") for n in G.nodes()]
    type_counts: Dict[str, int] = {}
    for t in node_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    verify_density = type_counts.get(NodeType.VERIFY.value, 0) / n
    backtrack_rate = type_counts.get(NodeType.BACKTRACK.value, 0) / n

    probs = np.array(list(type_counts.values()), dtype=np.float64) / n
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    avg_degree = float(np.mean([d for _, d in G.degree()])) if n > 0 else 0.0
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    graph_density = m / max_edges

    return GraphMetrics(
        trace_id=graph.trace_id,
        n_nodes=n, n_edges=m,
        depth=depth, branching=branching,
        verify_density=verify_density, backtrack_rate=backtrack_rate,
        entropy=entropy, avg_degree=avg_degree, graph_density=graph_density,
    )


def compute_graph_features_batch(graphs: List[ReasoningTraceGraph]) -> List[GraphMetrics]:
    """Compute metrics for a batch of graphs."""
    return [compute_graph_features(g) for g in graphs]


def compute_feature_matrix(graphs: List[ReasoningTraceGraph]) -> np.ndarray:
    """Compute numeric feature matrix (N x 6) for downstream ML."""
    metrics = compute_graph_features_batch(graphs)
    return np.array([
        [m.n_nodes, m.n_edges, m.depth, m.branching, m.verify_density, m.entropy]
        for m in metrics
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Weisfeiler-Lehman Subtree Kernel (FIX #1 — v3.1)
# ---------------------------------------------------------------------------

def _get_node_labels(G: nx.DiGraph, attr: str = "type") -> Dict[Any, str]:
    """Extract initial string labels from node attributes."""
    labels = {}
    for nid, attrs in G.nodes(data=True):
        labels[nid] = str(attrs.get(attr, "unknown"))
    return labels


def _wl_refine(G: nx.DiGraph, labels: Dict[Any, str]) -> Dict[Any, str]:
    """One iteration of WL label refinement.

    Each node's new label = old_label + sorted(neighbor_labels).
    For directed graphs, we use both incoming and outgoing neighbors.
    """
    new_labels = {}
    for nid in G.nodes():
        old_label = labels.get(nid, "0")
        in_labels = sorted(labels.get(pred, "") for pred in G.predecessors(nid))
        out_labels = sorted(labels.get(succ, "") for succ in G.successors(nid))
        neighbor_str = ",".join(in_labels + out_labels)
        new_labels[nid] = f"{old_label}|{neighbor_str}"
    return new_labels


def weisfeiler_lehman_kernel(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    h: int = 3,
    node_attr: str = "type",
) -> float:
    """Compute normalized WL subtree kernel similarity between two graphs.

    Complexity: O(h · m) where h = iterations, m = total edges.

    Args:
        G1, G2: Input graphs (directed).
        h: Number of WL iterations (default 3 — captures up to 3-hop neighborhoods).
        node_attr: Node attribute key for initial labels.

    Returns:
        Cosine similarity in [0, 1]. 1 = isomorphic, 0 = completely different.
    """
    if G1.number_of_nodes() == 0 and G2.number_of_nodes() == 0:
        return 1.0
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return 0.0

    labels1 = _get_node_labels(G1, node_attr)
    labels2 = _get_node_labels(G2, node_attr)

    all_counts1: Dict[str, int] = {}
    all_counts2: Dict[str, int] = {}

    for iteration in range(h + 1):
        for lbl in labels1.values():
            all_counts1[lbl] = all_counts1.get(lbl, 0) + 1
        for lbl in labels2.values():
            all_counts2[lbl] = all_counts2.get(lbl, 0) + 1
        if iteration < h:
            labels1 = _wl_refine(G1, labels1)
            labels2 = _wl_refine(G2, labels2)

    all_keys = set(all_counts1.keys()) | set(all_counts2.keys())
    vec1 = np.array([all_counts1.get(k, 0) for k in all_keys], dtype=np.float64)
    vec2 = np.array([all_counts2.get(k, 0) for k in all_keys], dtype=np.float64)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 and norm2 == 0:
        return 1.0
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def wl_pairwise_similarity_matrix(graphs: List[nx.DiGraph], h: int = 3) -> np.ndarray:
    """Compute pairwise WL kernel similarity matrix."""
    n = len(graphs)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = weisfeiler_lehman_kernel(graphs[i], graphs[j], h=h)
            sim[i, j] = s
            sim[j, i] = s
    return sim


# ---------------------------------------------------------------------------
# Motif-Based Spectral Distance
# ---------------------------------------------------------------------------

def motif_spectral_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute similarity from motif frequency vectors using (1 - JSD)."""
    jsd = jensenshannon(vec1, vec2, base=2.0)
    if np.isnan(jsd):
        return 1.0
    return float(1.0 - jsd)


# ---------------------------------------------------------------------------
# Feature Distance
# ---------------------------------------------------------------------------

def feature_distance(f1: Level1Features, f2: Level1Features) -> float:
    """Compute normalized Euclidean distance between Level-1 feature vectors."""
    v1 = f1.normalize_vector()
    v2 = f2.normalize_vector()
    return float(np.linalg.norm(v1 - v2) / np.sqrt(len(v1)))


# ---------------------------------------------------------------------------
# Topology Similarity Index (TSI) — Section 5.4
# ---------------------------------------------------------------------------

@dataclass
class TopologySimilarityIndex:
    """Complete TSI computation for a pair of graphs."""

    tsi_value: float
    alpha: float
    motif_similarity: float
    wl_similarity: float
    feature_similarity: float
    structure_similarity: float


def compute_tsi(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    alpha: float = 0.75,
    matcher: Optional[Any] = None,
    wl_iterations: int = 3,
) -> TopologySimilarityIndex:
    """Compute the parameterized Topology Similarity Index.

    TSI_alpha(G1, G2) = alpha * (1 - JSD(M1, M2)) + (1-alpha) * (1 - d_feat)

    Structure-level similarity uses WL kernel as a tractable GED replacement.

    Args:
        G1, G2: The two RTGs to compare.
        alpha: Weight in [0, 1]. Motif-dominant when alpha > 0.5.
        matcher: Optional pre-initialized MotifMatcher.
        wl_iterations: WL kernel iterations.

    Returns:
        TopologySimilarityIndex with all component scores.
    """
    from .motif_matcher import MotifMatcher

    if matcher is None:
        matcher = MotifMatcher()

    # Motif similarity: 1 - JSD
    motif_vec1 = matcher.compute_motif_distribution(G1)
    motif_vec2 = matcher.compute_motif_distribution(G2)
    motif_sim = motif_spectral_similarity(motif_vec1, motif_vec2)

    # Feature similarity: 1 - normalized Euclidean distance
    feats1 = compute_level1_features(G1)
    feats2 = compute_level1_features(G2)
    feat_dist = feature_distance(feats1, feats2)
    feat_sim = 1.0 - feat_dist

    # WL kernel similarity (replaces GED)
    wl_sim = weisfeiler_lehman_kernel(G1, G2, h=wl_iterations)
    structure_sim = wl_sim

    # Combined TSI
    tsi_value = alpha * motif_sim + (1.0 - alpha) * feat_sim

    return TopologySimilarityIndex(
        tsi_value=tsi_value,
        alpha=alpha,
        motif_similarity=motif_sim,
        wl_similarity=wl_sim,
        feature_similarity=feat_sim,
        structure_similarity=structure_sim,
    )


def compute_pairwise_tsi(
    graphs: List[nx.DiGraph],
    alpha: float = 0.75,
    matcher: Optional[Any] = None,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise TSI, motif similarity, and WL similarity matrices.

    Returns:
        (tsi_matrix, motif_sim_matrix, wl_sim_matrix) each (n, n).
    """
    from .motif_matcher import MotifMatcher

    if matcher is None:
        matcher = MotifMatcher()

    n = len(graphs)
    motif_vecs = matcher.batch_compute_motif_vectors(graphs, normalize=True, n_jobs=n_jobs)
    feat_vecs = np.array(
        [compute_level1_features(G).normalize_vector() for G in graphs]
    )

    tsi_matrix = np.eye(n)
    motif_sim_matrix = np.eye(n)
    wl_sim_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            jsd = jensenshannon(motif_vecs[i], motif_vecs[j], base=2.0)
            msim = 1.0 - float(jsd) if not np.isnan(jsd) else 1.0
            motif_sim_matrix[i, j] = msim
            motif_sim_matrix[j, i] = msim

            feat_dist = np.linalg.norm(feat_vecs[i] - feat_vecs[j]) / np.sqrt(feat_vecs.shape[1])
            fsim = 1.0 - float(feat_dist)

            wsim = weisfeiler_lehman_kernel(graphs[i], graphs[j])
            wl_sim_matrix[i, j] = wsim
            wl_sim_matrix[j, i] = wsim

            tsi = alpha * msim + (1.0 - alpha) * fsim
            tsi_matrix[i, j] = tsi
            tsi_matrix[j, i] = tsi

    return tsi_matrix, motif_sim_matrix, wl_sim_matrix


# ---------------------------------------------------------------------------
# Motif Distribution
# ---------------------------------------------------------------------------

class MotifDistribution:
    """Wraps a motif frequency vector with comparison utilities."""

    def __init__(self, vector: np.ndarray, motif_ids: Optional[List[str]] = None):
        self.vector = vector
        self.motif_ids = motif_ids or [f"M{i}" for i in range(1, len(vector) + 1)]

    def compare(self, other: "MotifDistribution") -> float:
        """JSD between two motif distributions."""
        return float(jensenshannon(self.vector, other.vector, base=2.0))

    def dominant_motif(self) -> str:
        """Return the motif ID with the highest frequency."""
        idx = int(np.argmax(self.vector))
        return self.motif_ids[idx]

    def entropy(self) -> float:
        """Shannon entropy of the motif distribution."""
        v = self.vector[self.vector > 0]
        if len(v) == 0:
            return 0.0
        return float(-np.sum(v * np.log2(v)))


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    point_estimate = statistic_fn(data)
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, n)]
        boot_stats.append(statistic_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(boot_stats, 100 * alpha)
    upper = np.percentile(boot_stats, 100 * (1 - alpha))
    return lower, point_estimate, upper


def stable_rate(tsi_values: np.ndarray, threshold: float) -> float:
    """Compute Stable Rate: fraction of graphs with TSI > threshold."""
    if len(tsi_values) == 0:
        return 0.0
    return float(np.mean(tsi_values > threshold))


def topology_diversity_index(pairwise_tsi: np.ndarray) -> float:
    """Compute Topology Diversity Index (TDI).

    TDI = mean pairwise TSI across different questions.
    If TDI > 0.8, all questions collapse to one structure (undesirable).
    If TDI < 0.5, different questions produce genuinely different topologies.
    """
    n = pairwise_tsi.shape[0]
    if n <= 1:
        return 1.0
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(pairwise_tsi[triu_indices]))


def partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> Tuple[float, float]:
    """Compute partial correlation between x and y, controlling for z."""
    from scipy.stats import pearsonr

    beta_x = np.cov(x, z)[0, 1] / np.var(z) if np.var(z) > 1e-10 else 0.0
    resid_x = x - beta_x * z
    beta_y = np.cov(y, z)[0, 1] / np.var(z) if np.var(z) > 1e-10 else 0.0
    resid_y = y - beta_y * z

    r, p = pearsonr(resid_x, resid_y)
    return r, p
