"""
Inter-Extractor Agreement (IAA) — 3-layer computation (v3.2 merged).

Merged from v3.1 + v3.2. Three levels:
1. **Graph-Level IAA**: Levenshtein alignment → Fleiss' Kappa on node-type sequences.
2. **Motif-Level IAA**: Mean Pearson correlation between motif frequency vectors.
3. **Structure-Level IAA**: WL kernel similarity (replaces NP-hard GED).

Plus: length-bias detection (RBE-Rand), syntax-artifact detection (SBE-LLM),
and the full Decision Framework from Section 3.8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from core.metrics import weisfeiler_lehman_kernel
from core.motif_matcher import MotifMatcher
from core.types import NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence Alignment for Graph-Level IAA
# ---------------------------------------------------------------------------

def get_node_type_sequence(graph) -> List[str]:
    """Extract node type sequence via topological sort.

    Accepts both ReasoningTraceGraph and networkx.DiGraph.
    """
    if isinstance(graph, ReasoningTraceGraph):
        if not graph.nodes:
            return []
        G = graph.to_networkx()
    else:
        G = graph

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXError:
        topo = sorted(G.nodes())
    return [G.nodes[n].get("type", "Transform") for n in topo]


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def levenshtein_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Normalized Levenshtein similarity: 1 - dist / max(len)."""
    dist = levenshtein_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2), 1)
    return 1.0 - dist / max_len


def levenshtein_alignment(
    seq1: List[str], seq2: List[str],
) -> Tuple[List[str], List[str]]:
    """Align two sequences via Levenshtein edit distance with backtracking.

    Pads with '<gap>' tokens to handle length mismatches.
    This enables Fleiss' Kappa on variable-length sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    aligned1, aligned2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if seq1[i - 1] == seq2[j - 1] else 1):
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned1.append(seq1[i - 1])
            aligned2.append("<gap>")
            i -= 1
        else:
            aligned1.append("<gap>")
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return aligned1, aligned2


# ---------------------------------------------------------------------------
# Fleiss' Kappa for Multiple Raters
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings: np.ndarray) -> float:
    """Compute Fleiss' Kappa for multiple raters.

    Args:
        ratings: Array of shape (n_items, n_raters) where each entry
            is a category index (0, 1, ..., k-1).

    Returns:
        Fleiss' Kappa value in [-1, 1].
    """
    n_items, n_raters = ratings.shape
    if n_items == 0 or n_raters < 2:
        return 1.0

    categories = sorted(set(ratings.flatten()))
    n_categories = len(categories)
    if n_categories <= 1:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((n_items, n_categories), dtype=np.int32)
    for i in range(n_items):
        for j in range(n_raters):
            cat = ratings[i, j]
            counts[i, cat_to_idx[cat]] += 1

    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    p_j = np.sum(counts, axis=0) / (n_items * n_raters)
    P_e = np.sum(p_j**2)

    if abs(P_e - 1.0) < 1e-9:
        return 1.0

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


# ---------------------------------------------------------------------------
# Decision Framework (Section 3.8)
# ---------------------------------------------------------------------------

class PhaseDecision(str, Enum):
    PROCEED_HIGH = "proceed_high"
    PROCEED_CAVEAT = "proceed_caveat"
    TERMINATE = "terminate"
    FLAG_LENGTH = "flag_length"
    FLAG_SYNTAX = "flag_syntax"


@dataclass
class DecisionReport:
    """Complete decision report for Experiment 0."""

    decisions: List[PhaseDecision] = field(default_factory=list)
    graph_level_iaa: float = 0.0
    motif_level_iaa: float = 0.0
    structure_level_iaa: float = 0.0
    llm_rand_correlation: float = 0.0
    sbe_llm_correlation: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def should_proceed(self) -> bool:
        return PhaseDecision.TERMINATE not in self.decisions

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "EXPERIMENT 0 — DECISION REPORT",
            "=" * 60,
            f"  Graph-Level IAA:      {self.graph_level_iaa:.3f}",
            f"  Motif-Level IAA:      {self.motif_level_iaa:.3f}",
            f"  Structure-Level IAA:  {self.structure_level_iaa:.3f}",
            f"  LLM-Rand Correlation: {self.llm_rand_correlation:.3f}",
            f"  SBE-LLM Correlation:  {self.sbe_llm_correlation:.3f}",
            "-" * 60,
            f"  Decisions: {[d.value for d in self.decisions]}",
            f"  Proceed to Phase 1:   {self.should_proceed()}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IAA Computation Engine (from v3.1)
# ---------------------------------------------------------------------------

ACTION_TYPES = [
    NodeType.RETRIEVE.value,
    NodeType.TRANSFORM.value,
    NodeType.COMPARE.value,
    NodeType.VERIFY.value,
    NodeType.BRANCH.value,
    NodeType.BACKTRACK.value,
]
ACTION_TO_IDX = {at: i for i, at in enumerate(ACTION_TYPES)}


class IAAComputer:
    """Computes all three levels of inter-extractor agreement."""

    def __init__(self, matcher: Optional[MotifMatcher] = None):
        self.matcher = matcher or MotifMatcher()

    # ------------------------------------------------------------------
    # Graph-Level IAA
    # ------------------------------------------------------------------

    def compute_graph_level_iaa(
        self, extractor_graphs: Dict[str, List],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute Fleiss' Kappa on aligned node-type sequences."""
        extractor_ids = list(extractor_graphs.keys())
        n_extractors = len(extractor_ids)
        if n_extractors < 2:
            return 1.0, {"error": "Need at least 2 extractors"}

        n_traces = min(len(graphs) for graphs in extractor_graphs.values())
        all_aligned_sequences: List[List[List[str]]] = []

        for trace_idx in range(n_traces):
            ref_graphs = extractor_graphs[extractor_ids[0]]
            ref_seq = get_node_type_sequence(ref_graphs[trace_idx])

            extractor_seqs: List[List[str]] = [ref_seq]
            for eid in extractor_ids[1:]:
                other_graphs = extractor_graphs[eid]
                other_seq = get_node_type_sequence(other_graphs[trace_idx])
                aligned_ref, aligned_other = levenshtein_alignment(ref_seq, other_seq)
                ref_seq = aligned_ref
                extractor_seqs.append(aligned_other)
            all_aligned_sequences.append(extractor_seqs)

        kappas = []
        for aligned_seqs in all_aligned_sequences:
            if len(aligned_seqs[0]) == 0:
                continue
            n_positions = len(aligned_seqs[0])
            ratings = np.zeros((n_positions, n_extractors), dtype=np.int32)
            for e_idx, seq in enumerate(aligned_seqs):
                for p_idx, token in enumerate(seq):
                    if token in ACTION_TO_IDX:
                        ratings[p_idx, e_idx] = ACTION_TO_IDX[token]
                    elif token == "<gap>":
                        ratings[p_idx, e_idx] = -1

            valid_mask = np.all(ratings >= 0, axis=1)
            valid_ratings = ratings[valid_mask]
            if valid_ratings.shape[0] >= 3:
                k = fleiss_kappa(valid_ratings)
                kappas.append(k)

        mean_kappa = float(np.mean(kappas)) if kappas else 0.0

        return mean_kappa, {
            "n_traces": n_traces,
            "n_extractors": n_extractors,
            "per_trace_kappas": kappas,
        }

    # ------------------------------------------------------------------
    # Motif-Level IAA
    # ------------------------------------------------------------------

    def compute_motif_level_iaa(
        self, extractor_graphs: Dict[str, List],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute mean Pearson correlation between motif frequency vectors."""
        from scipy.stats import pearsonr

        extractor_ids = list(extractor_graphs.keys())
        n_traces = min(len(graphs) for graphs in extractor_graphs.values())

        motif_vecs: Dict[str, np.ndarray] = {}
        for eid in extractor_ids:
            graphs = extractor_graphs[eid][:n_traces]
            nx_graphs = [
                g.to_networkx() if hasattr(g, "to_networkx") else g for g in graphs
            ]
            vecs = self.matcher.batch_compute_motif_vectors(nx_graphs, normalize=True)
            motif_vecs[eid] = vecs

        correlations = []
        pair_details = {}
        for i in range(len(extractor_ids)):
            for j in range(i + 1, len(extractor_ids)):
                e1, e2 = extractor_ids[i], extractor_ids[j]
                n_dims = motif_vecs[e1].shape[1]
                dim_corrs = []
                for d in range(n_dims):
                    v1, v2 = motif_vecs[e1][:, d], motif_vecs[e2][:, d]
                    if np.std(v1) > 1e-8 and np.std(v2) > 1e-8:
                        r, _ = pearsonr(v1, v2)
                        dim_corrs.append(r if not np.isnan(r) else 0.0)
                pair_corr = float(np.mean(dim_corrs)) if dim_corrs else 0.0
                correlations.append(pair_corr)
                pair_details[f"{e1}-{e2}"] = pair_corr

        mean_correlation = float(np.mean(correlations)) if correlations else 0.0
        return mean_correlation, {"n_pairs": len(correlations), "pair_correlations": pair_details}

    # ------------------------------------------------------------------
    # Structure-Level IAA (WL Kernel — replaces GED)
    # ------------------------------------------------------------------

    def compute_structure_level_iaa(
        self, extractor_graphs: Dict[str, List], wl_iterations: int = 3,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute mean WL kernel similarity between all extractor pairs."""
        extractor_ids = list(extractor_graphs.keys())
        n_traces = min(len(graphs) for graphs in extractor_graphs.values())

        similarities = []
        pair_details = {}

        for i in range(len(extractor_ids)):
            for j in range(i + 1, len(extractor_ids)):
                e1, e2 = extractor_ids[i], extractor_ids[j]
                graphs1 = extractor_graphs[e1][:n_traces]
                graphs2 = extractor_graphs[e2][:n_traces]

                trace_sims = []
                for g1, g2 in zip(graphs1, graphs2):
                    g1_nx = g1.to_networkx() if hasattr(g1, "to_networkx") else g1
                    g2_nx = g2.to_networkx() if hasattr(g2, "to_networkx") else g2
                    sim = weisfeiler_lehman_kernel(g1_nx, g2_nx, h=wl_iterations)
                    trace_sims.append(sim)

                mean_pair_sim = float(np.mean(trace_sims))
                similarities.append(mean_pair_sim)
                pair_details[f"{e1}-{e2}"] = mean_pair_sim

        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        return mean_similarity, {
            "n_pairs": len(similarities),
            "pair_similarities": pair_details,
            "wl_iterations": wl_iterations,
        }

    # ------------------------------------------------------------------
    # Full IAA Report
    # ------------------------------------------------------------------

    def compute_full_iaa(
        self,
        extractor_graphs: Dict[str, List],
        include_rbe_rand: bool = True,
        include_sbe: bool = True,
    ) -> DecisionReport:
        """Compute all IAA metrics and generate a DecisionReport."""
        decisions = []

        graph_iaa, graph_diag = self.compute_graph_level_iaa(extractor_graphs)
        motif_iaa, motif_diag = self.compute_motif_level_iaa(extractor_graphs)
        struct_iaa, struct_diag = self.compute_structure_level_iaa(extractor_graphs)

        llm_rand_corr = 0.0
        sbe_llm_corr = 0.0

        # Decision logic (Section 3.8)
        if graph_iaa > 0.6 and motif_iaa > 0.6:
            decisions.append(PhaseDecision.PROCEED_HIGH)
        elif graph_iaa >= 0.4 or motif_iaa >= 0.4:
            decisions.append(PhaseDecision.PROCEED_CAVEAT)
        else:
            decisions.append(PhaseDecision.TERMINATE)

        # RBE-Rand check
        if include_rbe_rand and "rbe-rand" in extractor_graphs:
            llm_ids = [e for e in extractor_graphs if e not in ("rbe-rand", "rbe", "sbe")]
            for llm_id in llm_ids:
                subset = {
                    "rbe-rand": extractor_graphs["rbe-rand"],
                    llm_id: extractor_graphs[llm_id],
                }
                _, motif_diag_rand = self.compute_motif_level_iaa(subset)
                corr_val = motif_diag_rand.get("pair_correlations", {}).get(
                    f"rbe-rand-{llm_id}", 0.0
                )
                llm_rand_corr = max(llm_rand_corr, corr_val)
            if llm_rand_corr > 0.3:
                decisions.append(PhaseDecision.FLAG_LENGTH)

        # SBE syntax check
        if include_sbe and "sbe" in extractor_graphs:
            llm_ids = [e for e in extractor_graphs if e not in ("rbe-rand", "rbe", "sbe", "human")]
            for llm_id in llm_ids:
                subset = {"sbe": extractor_graphs["sbe"], llm_id: extractor_graphs[llm_id]}
                _, motif_diag_sbe = self.compute_motif_level_iaa(subset)
                corr_val = motif_diag_sbe.get("pair_correlations", {}).get(
                    f"sbe-{llm_id}", 0.0
                )
                sbe_llm_corr = max(sbe_llm_corr, corr_val)
            if sbe_llm_corr > 0.7:
                decisions.append(PhaseDecision.FLAG_SYNTAX)

        return DecisionReport(
            decisions=decisions,
            graph_level_iaa=graph_iaa,
            motif_level_iaa=motif_iaa,
            structure_level_iaa=struct_iaa,
            llm_rand_correlation=llm_rand_corr,
            sbe_llm_correlation=sbe_llm_corr,
            details={
                "graph_diagnostics": graph_diag,
                "motif_diagnostics": motif_diag,
                "structure_diagnostics": struct_diag,
            },
        )


# ---------------------------------------------------------------------------
# Simplified IAA functions (from v3.2, for quick access)
# ---------------------------------------------------------------------------

def graph_level_iaa(
    graphs_by_extractor: Dict[str, List[ReasoningTraceGraph]],
    n_samples: int = 100,
) -> Dict[str, float]:
    """Graph-level IAA: node-type sequences → Levenshtein alignment → Fleiss' Kappa."""
    extractor_names = list(graphs_by_extractor.keys())
    n_extractors = len(extractor_names)
    if n_extractors < 2:
        return {"fleiss_kappa": 1.0, "mean_pairwise_similarity": 1.0}

    pairwise_sims = []
    n_graphs = min(len(v) for v in graphs_by_extractor.values())

    for idx in range(min(n_graphs, n_samples)):
        seqs = {}
        for ename in extractor_names:
            if idx < len(graphs_by_extractor[ename]):
                seqs[ename] = get_node_type_sequence(graphs_by_extractor[ename][idx])
            else:
                seqs[ename] = []
        for e1, e2 in combinations(extractor_names, 2):
            sim = levenshtein_similarity(seqs[e1], seqs[e2])
            pairwise_sims.append(sim)

    mean_pairwise_sim = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0
    P_e = 1.0 / 6.0
    kappa = (mean_pairwise_sim - P_e) / (1.0 - P_e) if P_e < 1.0 else 1.0
    kappa = max(-1.0, min(1.0, kappa))

    return {
        "fleiss_kappa": float(kappa),
        "mean_pairwise_similarity": float(mean_pairwise_sim),
    }


def motif_level_iaa(
    graphs_by_extractor: Dict[str, List[ReasoningTraceGraph]],
) -> Dict[str, float]:
    """Motif-level IAA: Pearson correlation of average motif frequency vectors."""
    matcher = MotifMatcher()
    extractor_names = list(graphs_by_extractor.keys())
    if len(extractor_names) < 2:
        return {"mean_pearson_r": 1.0}

    freq_vectors = {}
    for ename in extractor_names:
        vectors = [
            matcher.compute_motif_frequency_vector(g)
            for g in graphs_by_extractor[ename]
        ]
        freq_vectors[ename] = np.mean(vectors, axis=0) if vectors else np.zeros(
            len(matcher.preset_motifs)
        )

    pearson_rs = []
    for e1, e2 in combinations(extractor_names, 2):
        v1, v2 = freq_vectors[e1], freq_vectors[e2]
        if np.std(v1) > 0 and np.std(v2) > 0:
            r = float(np.corrcoef(v1, v2)[0, 1])
            pearson_rs.append(max(-1.0, min(1.0, r)))

    return {"mean_pearson_r": float(np.mean(pearson_rs)) if pearson_rs else 0.0}


def structure_level_iaa(
    graphs_by_extractor: Dict[str, List[ReasoningTraceGraph]],
) -> Dict[str, float]:
    """Structure-level IAA: approximated Graph Edit Distance similarity."""
    extractor_names = list(graphs_by_extractor.keys())
    if len(extractor_names) < 2:
        return {"mean_ged_similarity": 1.0}

    ged_sims = []
    n_graphs = min(len(v) for v in graphs_by_extractor.values())

    for idx in range(n_graphs):
        graphs_at_idx = {}
        for ename in extractor_names:
            if idx < len(graphs_by_extractor[ename]):
                graphs_at_idx[ename] = graphs_by_extractor[ename][idx]

        for e1, e2 in combinations(graphs_at_idx.keys(), 2):
            g1, g2 = graphs_at_idx[e1], graphs_at_idx[e2]
            sim = _graph_edit_distance_approx(g1, g2)
            ged_sims.append(sim)

    return {"mean_ged_similarity": float(np.mean(ged_sims)) if ged_sims else 0.0}


def _graph_edit_distance_approx(
    g1: ReasoningTraceGraph, g2: ReasoningTraceGraph,
) -> float:
    """Approximate normalized GED for two RTGs."""
    G1, G2 = g1.to_networkx(), g2.to_networkx()
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    m1, m2 = G1.number_of_edges(), G2.number_of_edges()

    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    # Node-type distribution similarity
    type_set = sorted(
        set(G1.nodes[n].get("type", "") for n in G1.nodes())
        | set(G2.nodes[n].get("type", "") for n in G2.nodes())
    )
    if not type_set:
        type_sim = 1.0
    else:
        p = np.array([sum(1 for n in G1.nodes() if G1.nodes[n].get("type", "") == t)
                       for t in type_set], dtype=np.float64)
        q = np.array([sum(1 for n in G2.nodes() if G2.nodes[n].get("type", "") == t)
                       for t in type_set], dtype=np.float64)
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        m = 0.5 * (p + q)

        def _kl(a, b):
            return float(np.sum(a * np.log((a + 1e-10) / (b + 1e-10))))

        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        type_sim = 1.0 - float(np.sqrt(max(jsd, 0.0)))

    size_sim = min(n1, n2) / max(n1, n2)
    dens1 = m1 / max(n1 * (n1 - 1) / 2, 1)
    dens2 = m2 / max(n2 * (n2 - 1) / 2, 1)
    edge_sim = 1.0 - abs(dens1 - dens2) / max(dens1, dens2, 1e-10)

    sim = (type_sim + size_sim + edge_sim) / 3.0
    return float(max(0.0, min(1.0, sim)))


# ---------------------------------------------------------------------------
# Bias Detection
# ---------------------------------------------------------------------------

def detect_length_bias(
    llm_extractor_graphs: Dict[str, List[ReasoningTraceGraph]],
    random_baseline_graphs: List[ReasoningTraceGraph],
    threshold: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    """Detect length bias: test if any LLM extractor correlates with RBE-Rand."""
    results = {}
    random_seqs = [get_node_type_sequence(g) for g in random_baseline_graphs]

    for ename, graphs in llm_extractor_graphs.items():
        llm_seqs = [get_node_type_sequence(g) for g in graphs[:len(random_seqs)]]
        sims = [levenshtein_similarity(rs, ls) for rs, ls in zip(random_seqs, llm_seqs)]
        corr = float(np.mean(sims)) if sims else 0.0
        results[ename] = {
            "mean_similarity_to_random": corr,
            "length_bias_detected": corr > threshold,
        }
        if corr > threshold:
            logger.warning(
                f"LENGTH BIAS DETECTED: {ename} correlates with RBE-Rand "
                f"(r = {corr:.3f} > {threshold}). Add length control experiment."
            )

    return results


def detect_syntax_artifact(
    sbe_graphs: List[ReasoningTraceGraph],
    llm_graphs: Dict[str, List[ReasoningTraceGraph]],
    threshold: float = 0.7,
) -> Dict[str, Dict[str, float]]:
    """Detect syntax artifacts: test if SBE correlates strongly with LLMs."""
    sbe_seqs = [get_node_type_sequence(g) for g in sbe_graphs]
    results = {}

    for ename, graphs in llm_graphs.items():
        llm_seqs = [get_node_type_sequence(g) for g in graphs[:len(sbe_seqs)]]
        sims = [levenshtein_similarity(ss, ls) for ss, ls in zip(sbe_seqs, llm_seqs)]
        corr = float(np.mean(sims)) if sims else 0.0
        results[ename] = {
            "sbe_correlation": corr,
            "syntax_artifact_detected": corr > threshold,
        }

    return results


def compute_full_iaa(
    graphs_by_extractor: Dict[str, List[ReasoningTraceGraph]],
) -> Dict[str, Dict[str, float]]:
    """Compute all three layers of IAA and return comprehensive report."""
    return {
        "graph_level": graph_level_iaa(graphs_by_extractor),
        "motif_level": motif_level_iaa(graphs_by_extractor),
        "structure_level": structure_level_iaa(graphs_by_extractor),
    }
