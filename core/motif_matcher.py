"""
Motif Catalog & Pattern Matching for Reasoning Trace Graphs (v3.2 merged).

Merged from v3.1 + v3.2:
- Preset motif catalog (M1–M8) with NetworkX pattern graphs
- Type-aware subgraph isomorphism matching via DiGraphMatcher
- Data-driven motif discovery (exhaustive enumeration + canonical labeling)
- MotifCatalog registry for presets + discovered motifs
- Laplace-smoothed motif distribution for JSD computation
- Code-format motif matching (MotifEntry-based, from v3.2 types)
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon

from .types import MOTIF_LOOKUP, MotifEntry, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset Motif Catalog (Section 5.2) — NetworkX-pattern-based
# ---------------------------------------------------------------------------

@dataclass
class Motif:
    """A single graph motif pattern (NetworkX-based, from v3.1).

    Attributes:
        motif_id: Identifier (M1, M2, ..., M8 for presets).
        pattern: A networkx.DiGraph representing the pattern.
        description: Human-readable description.
        size: Number of nodes in the motif.
    """
    motif_id: str
    pattern: nx.DiGraph
    description: str
    size: int


def _make_chain(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def _make_fork(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(0)
    for i in range(1, n):
        G.add_node(i)
        G.add_edge(0, i)
    return G


def _make_diamond() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    return G


def _make_verify_after_transform() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edge(0, 1)
    return G


def _make_backtrack_recover() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edge(0, 1)
    return G


def _make_branch_explore() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2)])
    return G


def _make_multi_verify() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 2), (1, 2)])
    return G


def _make_loop(n: int = 3) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


def _with_pos_attrs(G: nx.DiGraph) -> nx.DiGraph:
    """Annotate pattern nodes with _pos attribute for type-aware matching."""
    for nid in list(G.nodes()):
        G.nodes[nid]["_pos"] = int(nid)
    return G


PRESET_MOTIFS: Dict[str, Motif] = {
    "M1": Motif(motif_id="M1", pattern=_with_pos_attrs(_make_chain(3)),
                description="Chain(3): A → B → C", size=3),
    "M2": Motif(motif_id="M2", pattern=_with_pos_attrs(_make_fork(3)),
                description="Fork(3): A → B, A → C", size=3),
    "M3": Motif(motif_id="M3", pattern=_with_pos_attrs(_make_diamond()),
                description="Diamond(4): A → B → D, A → C → D", size=4),
    "M4": Motif(motif_id="M4", pattern=_with_pos_attrs(_make_loop(3)),
                description="Loop(3): A → B → C → A (detected but rejected for DAGs)", size=3),
    "M5": Motif(motif_id="M5", pattern=_with_pos_attrs(_make_verify_after_transform()),
                description="Verify-after-Transform: Transform → Verify", size=2),
    "M6": Motif(motif_id="M6", pattern=_with_pos_attrs(_make_backtrack_recover()),
                description="Backtrack-Recover: Backtrack → Retrieve/Transform", size=2),
    "M7": Motif(motif_id="M7", pattern=_with_pos_attrs(_make_branch_explore()),
                description="Branch-Explore: Branch → T1, Branch → T2", size=3),
    "M8": Motif(motif_id="M8", pattern=_with_pos_attrs(_make_multi_verify()),
                description="Multi-Verify: Verify ← multiple parents", size=3),
    "M9": Motif(motif_id="M9", pattern=_with_pos_attrs(_make_verify_after_transform()),
                 description="Branch → Transform", size=2),
    "M10": Motif(motif_id="M10", pattern=_with_pos_attrs(_make_verify_after_transform()),
                  description="Transform → Transform", size=2),
    "M11": Motif(motif_id="M11", pattern=_with_pos_attrs(_make_verify_after_transform()),
                  description="Retrieve → Transform", size=2),
    "M12": Motif(motif_id="M12", pattern=_with_pos_attrs(_make_verify_after_transform()),
                  description="Transform → Branch", size=2),
}

# Motifs that require node-type matching (not just topology)
TYPE_DEPENDENT_MOTIFS = {"M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"}

# Motifs that match purely on topology (ignore node types)
TOPOLOGY_ONLY_MOTIFS = {"M1", "M2", "M3"}


def _get_type_constraints(motif_id: str) -> Optional[Dict[int, Set[str]]]:
    """Return node type constraints for type-dependent motifs."""
    constraints: Dict[str, Dict[int, Set[str]]] = {
        "M5": {0: {"Transform"}, 1: {"Verify"}},
        "M6": {0: {"Backtrack"}, 1: {"Retrieve", "Transform"}},
        "M7": {0: {"Branch"}},
        "M8": {2: {"Verify"}},
        "M9": {0: {"Branch"}, 1: {"Transform"}},
        "M10": {0: {"Transform"}, 1: {"Transform"}},
        "M11": {0: {"Retrieve"}, 1: {"Transform"}},
        "M12": {0: {"Transform"}, 1: {"Branch"}},
    }
    return constraints.get(motif_id)


# ---------------------------------------------------------------------------
# Motif Match Result
# ---------------------------------------------------------------------------

@dataclass
class MotifMatchResult:
    """Result of matching a single motif against a graph."""
    motif_id: str
    count: int
    matched_subgraphs: List[FrozenSet[int]] = field(default_factory=list)
    graph_trace_id: str = ""
    node_sets: List[Set[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Motif Matcher (merged v3.1 + v3.2)
# ---------------------------------------------------------------------------

class MotifMatcher:
    """Efficient motif enumeration engine.

    Supports both preset-matching (NetworkX DiGraphMatcher with type
    constraints) and data-driven discovery via exhaustive enumeration.
    """

    def __init__(
        self,
        preset_motifs: Optional[Dict[str, Motif]] = None,
        max_motif_size: int = 4,
    ):
        self.preset_motifs = preset_motifs or PRESET_MOTIFS
        self.max_motif_size = max_motif_size

        # Also build code-format motif graphs from MotifEntry catalog
        self._code_motif_graphs: Dict[str, nx.DiGraph] = {}
        self._build_code_motif_graphs()

    def _build_code_motif_graphs(self):
        """Convert MotifEntry objects (from types.py) to NetworkX DiGraphs."""
        for motif in MOTIF_LOOKUP.values():
            G = nx.DiGraph()
            for i, ntype in enumerate(motif.node_types):
                G.add_node(i, type=ntype.value)
            G.add_edges_from(motif.edge_list)
            self._code_motif_graphs[motif.motif_id] = G

    # ------------------------------------------------------------------
    # Preset Motif Matching (type-aware, from v3.1)
    # ------------------------------------------------------------------

    def count_motif(self, G: nx.DiGraph, motif_id: str) -> MotifMatchResult:
        """Count occurrences of a specific preset motif in graph G.

        Uses DiGraphMatcher with type-aware node_match for TYPE_DEPENDENT_MOTIFS.
        """
        if motif_id not in self.preset_motifs:
            raise ValueError(f"Unknown motif '{motif_id}'. Available: {list(self.preset_motifs)}")

        motif = self.preset_motifs[motif_id]

        if motif_id == "M4":
            return MotifMatchResult(motif_id=motif_id, count=0)

        if motif.size > G.number_of_nodes():
            return MotifMatchResult(motif_id=motif_id, count=0)

        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            G, motif.pattern, node_match=self._type_aware_node_match(motif_id)
        )

        matched: List[FrozenSet[int]] = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            matched_nodes = frozenset(mapping.values())
            matched.append(matched_nodes)

        unique_matches = list(set(matched))

        return MotifMatchResult(
            motif_id=motif_id, count=len(unique_matches), matched_subgraphs=unique_matches
        )

    def _type_aware_node_match(self, motif_id: str):
        """Build a node_match function that respects type constraints."""
        type_constraints = _get_type_constraints(motif_id)

        if type_constraints is None:
            return lambda n1, n2: True

        def node_match(n1: dict, n2: dict) -> bool:
            pos = n2.get("_pos")
            if pos is None or pos not in type_constraints:
                return True
            allowed_types = type_constraints[pos]
            actual_type = n1.get("type")
            return actual_type in allowed_types

        return node_match

    def count_all_motifs(
        self, G: nx.DiGraph, motif_ids: Optional[List[str]] = None,
    ) -> Dict[str, MotifMatchResult]:
        """Count all specified motifs in graph G."""
        if motif_ids is None:
            motif_ids = list(self.preset_motifs.keys())
        return {mid: self.count_motif(G, mid) for mid in motif_ids}

    # ------------------------------------------------------------------
    # Motif Frequency Vectors
    # ------------------------------------------------------------------

    def compute_motif_vector(
        self, G: nx.DiGraph, motif_ids: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute a motif frequency vector for graph G."""
        if motif_ids is None:
            motif_ids = sorted(self._code_motif_graphs.keys())

        counts = [self.count_motif(G, mid).count for mid in motif_ids]
        vec = np.array(counts, dtype=np.float64)

        if normalize:
            total = vec.sum()
            if total > 0:
                vec /= total
            else:
                vec = np.ones(len(counts)) / len(counts)
        return vec

    def compute_motif_distribution(
        self, G: nx.DiGraph, smooth: bool = True, epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Compute a Laplace-smoothed motif probability distribution."""
        motif_ids = sorted(self._code_motif_graphs.keys())
        vec = self.compute_motif_vector(G, motif_ids, normalize=False)
        if smooth:
            vec = vec + epsilon
        return vec / vec.sum()

    def compare_motif_distributions(self, G1: nx.DiGraph, G2: nx.DiGraph) -> float:
        """Compute JSD between motif distributions of two graphs."""
        p = self.compute_motif_distribution(G1)
        q = self.compute_motif_distribution(G2)
        return float(jensenshannon(p, q, base=2.0))

    def batch_compute_motif_vectors(
        self, graphs: List[nx.DiGraph], motif_ids: Optional[List[str]] = None,
        normalize: bool = True, n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute motif vectors for a batch of graphs."""
        if n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.compute_motif_vector)(G, motif_ids, normalize)
                    for G in graphs
                )
                return np.array(results)
            except ImportError:
                pass
        return np.array(
            [self.compute_motif_vector(G, motif_ids, normalize) for G in graphs]
        )

    def compute_pairwise_jsd_matrix(self, graphs: List[nx.DiGraph]) -> np.ndarray:
        """Compute pairwise JSD matrix for a set of graphs."""
        n = len(graphs)
        motif_vecs = self.batch_compute_motif_vectors(graphs, normalize=True)
        jsd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                jsd = jensenshannon(motif_vecs[i], motif_vecs[j], base=2.0)
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd
        return jsd_matrix

    # ------------------------------------------------------------------
    # Code-format Motif Matching (MotifEntry-based, from v3.2)
    # ------------------------------------------------------------------

    def match_preset(
        self, graph: ReasoningTraceGraph, motif_id: str,
    ) -> MotifMatchResult:
        """Match a preset motif against a ReasoningTraceGraph (code format).

        For topology-only motifs (M1-M3), ignores node types and matches
        purely on graph structure. For type-dependent motifs (M5-M8),
        requires exact node type matching.
        """
        if motif_id not in self._code_motif_graphs:
            raise ValueError(f"Unknown motif: {motif_id}")

        G = graph.to_networkx()
        motif_G = self._code_motif_graphs[motif_id]

        if motif_id in TOPOLOGY_ONLY_MOTIFS:
            def node_match(n1, n2):
                return True
        else:
            def node_match(n1, n2):
                return n1.get("type", "") == n2.get("type", "")

        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            G, motif_G, node_match=node_match
        )

        occurrences = list(matcher.subgraph_isomorphisms_iter())
        node_sets = [set(occ.values()) for occ in occurrences]
        node_sets = self._deduplicate_overlapping(node_sets)

        return MotifMatchResult(
            motif_id=motif_id,
            graph_trace_id=graph.trace_id,
            count=len(node_sets),
            node_sets=node_sets,
        )

    def match_all_presets(
        self, graph: ReasoningTraceGraph,
    ) -> Dict[str, MotifMatchResult]:
        """Match all preset motifs against a single graph."""
        return {mid: self.match_preset(graph, mid) for mid in self._code_motif_graphs}

    def compute_motif_frequencies(
        self, graphs: List[ReasoningTraceGraph],
    ) -> Dict[str, float]:
        """Compute per-motif frequency across a set of graphs."""
        counter: Counter = Counter()
        for g in graphs:
            for motif_id in self._code_motif_graphs:
                result = self.match_preset(g, motif_id)
                if result.count > 0:
                    counter[motif_id] += 1
        n = max(len(graphs), 1)
        return {mid: counter[mid] / n for mid in self._code_motif_graphs}

    def compute_motif_frequency_vector(
        self, graph: ReasoningTraceGraph,
    ) -> np.ndarray:
        """Compute per-motif count vector for a single graph."""
        motif_ids = sorted(self._code_motif_graphs.keys())
        vec = np.zeros(len(motif_ids), dtype=np.float32)
        for i, mid in enumerate(motif_ids):
            result = self.match_preset(graph, mid)
            vec[i] = float(result.count)
        return vec

    def compute_motif_frequency_matrix(
        self, graphs: List[ReasoningTraceGraph],
    ) -> np.ndarray:
        """Compute N_graphs x N_motifs matrix of motif counts."""
        return np.array([self.compute_motif_frequency_vector(g) for g in graphs])

    # ------------------------------------------------------------------
    # Data-Driven Motif Discovery (from v3.1)
    # ------------------------------------------------------------------

    def discover_motifs_exhaustive(
        self, graphs: List[ReasoningTraceGraph],
        max_size: int = 3, min_frequency: float = 0.05,
    ) -> List[MotifEntry]:
        """Exhaustively enumerate all connected subgraphs up to max_size nodes."""
        subgraph_counter: Counter = Counter()
        subgraph_examples: Dict[tuple, Tuple[tuple, tuple]] = {}

        for g in graphs:
            G = g.to_networkx()
            seen_in_graph: Set[tuple] = set()
            for size in range(2, max_size + 1):
                for nodes_subset in itertools.combinations(G.nodes(), size):
                    sub = G.subgraph(nodes_subset)
                    if nx.is_weakly_connected(sub):
                        sorted_nodes = sorted(sub.nodes())
                        types = tuple(
                            G.nodes[n]["type"] for n in sorted_nodes
                        )
                        edges = tuple(sorted(
                            (sorted_nodes.index(u), sorted_nodes.index(w))
                            for u, w in sub.edges()
                        ))
                        key = (types, edges)
                        if key not in seen_in_graph:
                            seen_in_graph.add(key)
                            subgraph_counter[key] += 1
                            subgraph_examples[key] = (types, edges)

        n_graphs = max(len(graphs), 1)
        discovered = []
        for motif_idx, ((types, edges), count) in enumerate(subgraph_counter.items()):
            freq = count / n_graphs
            if freq >= min_frequency:
                discovered.append(MotifEntry(
                    motif_id=f"D{motif_idx + 1}",
                    pattern_name=f"Discovered-{len(types)}",
                    description=f"Data-driven motif: {'->'.join(types)}",
                    size=len(types),
                    node_types=[NodeType.from_string(t) for t in types],
                    edge_list=list(edges),
                    frequency=freq,
                    discovery_method="exhaustive",
                ))

        logger.info(
            f"Discovered {len(discovered)} motifs (min_freq={min_frequency}) "
            f"from {len(graphs)} graphs"
        )
        return sorted(discovered, key=lambda m: m.frequency, reverse=True)

    # ------------------------------------------------------------------
    # Motif Discovery (NetworkX-graph-based, from v3.1)
    # ------------------------------------------------------------------

    class MotifDiscoverer:
        """Wrapper for data-driven motif discovery.

        For small graphs (< 50 nodes), uses brute-force enumeration of all
        3-node connected subgraphs with isomorphism-based deduplication.
        """

        def __init__(self, min_frequency: float = 0.05, max_size: int = 3):
            self.min_frequency = min_frequency
            self.max_size = max_size

        def discover(
            self, graphs: List[nx.DiGraph],
            existing_motif_ids: Optional[Set[str]] = None,
        ) -> List[Motif]:
            """Discover frequent motifs from a set of graphs."""
            if existing_motif_ids is None:
                existing_motif_ids = set()

            n_graphs = len(graphs)
            min_count = max(1, int(n_graphs * self.min_frequency))

            all_subgraphs: List[nx.DiGraph] = []
            for G in graphs:
                subgraphs = self._enumerate_subgraphs(G, self.max_size)
                all_subgraphs.extend(subgraphs)

            clusters: Dict[str, List[nx.DiGraph]] = {}
            for sg in all_subgraphs:
                canonical = self._canonical_label(sg)
                if canonical not in clusters:
                    clusters[canonical] = []
                clusters[canonical].append(sg)

            discovered = []
            next_id = 9
            for canonical, instances in clusters.items():
                if len(instances) >= min_count:
                    motif_id = f"M{next_id}"
                    if motif_id not in existing_motif_ids:
                        discovered.append(Motif(
                            motif_id=motif_id,
                            pattern=instances[0],
                            description=f"Discovered motif {motif_id} (freq={len(instances) / n_graphs:.2%})",
                            size=instances[0].number_of_nodes(),
                        ))
                        next_id += 1
            return discovered

        def _enumerate_subgraphs(
            self, G: nx.DiGraph, max_size: int,
        ) -> List[nx.DiGraph]:
            subgraphs = []
            nodes = list(G.nodes())
            for size in range(2, max_size + 1):
                for combo in itertools.combinations(nodes, size):
                    subg = G.subgraph(combo).copy()
                    if nx.is_weakly_connected(subg) and subg.number_of_edges() > 0:
                        subgraphs.append(subg)
            return subgraphs

        def _canonical_label(self, G: nx.DiGraph) -> str:
            edges = sorted((int(u), int(v)) for u, v in G.edges())
            nodes = sorted(int(n) for n in G.nodes())
            return f"V{len(nodes)}_E{'_'.join(f'{u}-{v}' for u, v in edges)}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_overlapping(
        node_sets: List[Set[int]], max_iter: int = 100,
    ) -> List[Set[int]]:
        """Greedily select a non-overlapping subset of node sets."""
        if len(node_sets) <= 1:
            return node_sets
        sorted_sets = sorted(node_sets, key=len, reverse=True)
        selected: List[Set[int]] = []
        used_nodes: Set[int] = set()
        for ns in sorted_sets:
            if not ns & used_nodes:
                selected.append(ns)
                used_nodes |= ns
        return selected


# ---------------------------------------------------------------------------
# Motif Catalog Registry (from v3.1)
# ---------------------------------------------------------------------------

class MotifCatalog:
    """Unified motif registry combining presets and discovered motifs."""

    def __init__(self):
        self.motifs: Dict[str, Motif] = dict(PRESET_MOTIFS)

    def register(self, motif: Motif) -> None:
        self.motifs[motif.motif_id] = motif

    def register_batch(self, motifs: List[Motif]) -> None:
        for m in motifs:
            self.register(m)

    def get_preset_ids(self) -> List[str]:
        return [f"M{i}" for i in range(1, 9)]

    def get_all_ids(self) -> List[str]:
        return sorted(self.motifs.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)

    def get(self, motif_id: str) -> Motif:
        if motif_id not in self.motifs:
            raise KeyError(f"Motif '{motif_id}' not in catalog.")
        return self.motifs[motif_id]
