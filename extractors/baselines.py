"""Baselines: JP-DPR, Edge Rewiring, Permutation, Ensemble (Fix 5).

Four randomization strategies for null-distribution testing:
1. JP-DPR: Jump-Preserving Degree-Preserving Randomization (4-tier fallback)
2. EdgeRewiring: Randomly swaps edge targets preserving degree distribution
3. Permutation: Shuffles node types preserving graph topology
4. Ensemble: Combines all three for robust null hypothesis testing
"""

import random
from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from core.types import GraphNode, NodeType, ReasoningTraceGraph


class JPDirectedPreservingRandomizer:
    """JP-DPR: preserves original edge jump-distance distribution."""

    def __init__(self, seed: int = 42):
        # Use a private Random instance so repeated construction with the
        # same seed is reproducible WITHOUT mutating the global random state.
        self._rng = random.Random(seed)

    def randomize(self, graph: ReasoningTraceGraph, k: int = 100) -> List[ReasoningTraceGraph]:
        G = graph.to_networkx()
        topo = list(nx.topological_sort(G))
        idx_map = {node: i for i, node in enumerate(topo)}
        edge_jumps = {(u, v): idx_map[v] - idx_map[u] for u, v in G.edges()}
        jdc = Counter(edge_jumps.values())
        jdk = list(jdc.keys())
        jdw = list(jdc.values())
        random_graphs = []

        for _ in range(k):
            shuffled = topo.copy()
            self._rng.shuffle(shuffled)
            idx_new = {node: i for i, node in enumerate(shuffled)}
            # MultiDiGraph so that two original edges which pick the same
            # target stay distinct — otherwise the edge count silently
            # shrinks and the null distribution is biased.
            Gr = nx.MultiDiGraph()
            Gr.add_nodes_from(G.nodes(data=True))
            for (u, v), target_d in edge_jumps.items():
                iu = idx_new[u]
                candidates = [n for n in shuffled[iu + 1:] if idx_new[n] - iu == target_d]
                if not candidates:
                    candidates = [n for n in shuffled[iu + 1:] if abs(idx_new[n] - iu - target_d) <= 2]
                if not candidates:
                    sd = self._rng.choices(jdk, weights=jdw)[0]
                    candidates = [n for n in shuffled[iu + 1:] if idx_new[n] - iu == sd]
                if not candidates:
                    candidates = shuffled[iu + 1:]
                if not candidates:
                    # ``u`` sits at (or near) the end of the shuffled order,
                    # so no forward nodes exist. Without this fallback the
                    # edge would be silently dropped.
                    candidates = [n for n in shuffled if n != u]
                # Adding (u, n) creates a cycle iff n can already reach u.
                # The end-of-order fallback may pick nodes that come *before*
                # u in the shuffled order; those backward edges break the
                # forward-only invariant the earlier tiers rely on, so every
                # candidate set (not just the fallback) must be cycle-checked.
                safe = [n for n in candidates if not nx.has_path(Gr, n, u)]
                if not safe and len(candidates) < len(shuffled) - 1:
                    # Every forward candidate already leads back into u
                    # (earlier fallback edges can route into it), yet a
                    # cycle-free target may still exist elsewhere — retry
                    # over every node except u before dropping the edge.
                    candidates = [n for n in shuffled if n != u]
                    safe = [n for n in candidates if not nx.has_path(Gr, n, u)]
                if safe:
                    Gr.add_edge(u, self._rng.choice(safe))
            random_graphs.append(self._to_rtg(Gr, graph, f"{graph.trace_id}_dpr_{_}"))
        return random_graphs

    @staticmethod
    def _to_rtg(G, ref, tid):
        topo = list(nx.topological_sort(G))
        nodes = []
        for i, n in enumerate(topo):
            ntype_str = G.nodes[n].get("type", "Transform")
            ntype = NodeType.from_string(ntype_str)
            nodes.append(GraphNode(id=i + 1, type=ntype))
        old_to_new = {old: new.id for old, new in zip(topo, nodes)}
        # DiGraph edges are (u, v); MultiDiGraph edges are (u, v, key).
        edges = [(old_to_new[e[0]], old_to_new[e[1]]) for e in G.edges()]
        return ReasoningTraceGraph(
            trace_id=tid, model=ref.model, question_id=ref.question_id,
            domain=ref.domain, extractor=f"{ref.extractor}_jp_dpr",
            nodes=nodes, edges=edges,
            metadata={"is_randomized": True, "method": "jp_dpr", **ref.metadata},
        )


class EdgeRewiringBaseline:
    """Randomly rewires edges while maintaining DAG property and degree distribution."""

    def __init__(self, seed: int = 42, n_swaps: int = 10):
        self.seed = seed
        self.n_swaps = n_swaps
        self._rng = random.Random(seed)

    def randomize(self, graph: ReasoningTraceGraph, k: int = 100) -> List[ReasoningTraceGraph]:
        G_orig = graph.to_networkx()
        random_graphs = []
        for _ in range(k):
            G = G_orig.copy()
            edges = list(G.edges())
            for _ in range(min(self.n_swaps, len(edges) // 2)):
                if len(edges) < 2:
                    break
                e1, e2 = self._rng.sample(edges, 2)
                a, b = e1
                c, d = e2
                if a != d and c != b and not G.has_edge(a, d) and not G.has_edge(c, b):
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                    if not nx.is_directed_acyclic_graph(G):
                        G.remove_edge(a, d)
                        G.remove_edge(c, b)
                        G.add_edge(a, b)
                        G.add_edge(c, d)
            random_graphs.append(self._to_rtg(G, graph, f"{graph.trace_id}_rw_{_}"))
        return random_graphs

    @staticmethod
    def _to_rtg(G, ref, tid):
        topo = list(nx.topological_sort(G))
        nodes = []
        for i, n in enumerate(topo):
            ntype_str = G.nodes[n].get("type", "Transform")
            ntype = NodeType.from_string(ntype_str)
            nodes.append(GraphNode(id=i + 1, type=ntype))
        old_to_new = {old: new.id for old, new in zip(topo, nodes)}
        # DiGraph edges are (u, v); MultiDiGraph edges are (u, v, key).
        edges = [(old_to_new[e[0]], old_to_new[e[1]]) for e in G.edges()]
        return ReasoningTraceGraph(
            trace_id=tid, model=ref.model, question_id=ref.question_id,
            domain=ref.domain, extractor=f"{ref.extractor}_rw",
            nodes=nodes, edges=edges,
            metadata={"is_randomized": True, "method": "edge_rewire", **ref.metadata},
        )


class PermutationBaseline:
    """Shuffles node types while preserving graph topology."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def randomize(self, graph: ReasoningTraceGraph, k: int = 100) -> List[ReasoningTraceGraph]:
        import copy
        random_graphs = []
        for _ in range(k):
            permuted = copy.deepcopy(graph)
            permuted.trace_id = f"{graph.trace_id}_perm_{_}"
            types = [n.type for n in permuted.nodes]
            self._rng.shuffle(types)
            for node, new_type in zip(permuted.nodes, types):
                node.type = new_type
            permuted.metadata["is_randomized"] = True
            permuted.metadata["method"] = "permutation"
            random_graphs.append(permuted)
        return random_graphs


class EnsembleBaseline:
    """Combines JP-DPR, EdgeRewiring, and Permutation for robust null testing."""

    def __init__(self, seed: int = 42):
        self.methods = {
            "jp_dpr": JPDirectedPreservingRandomizer(seed=seed),
            "edge_rewire": EdgeRewiringBaseline(seed=seed),
            "permutation": PermutationBaseline(seed=seed),
        }

    def randomize_all(self, graph: ReasoningTraceGraph, k_per_method: int = 50):
        return {name: method.randomize(graph, k=k_per_method) for name, method in self.methods.items()}


def compute_tsi_threshold(
    real_graphs: List[ReasoningTraceGraph],
    dpr_graphs: List[ReasoningTraceGraph],
    tsi_function,
    percentile: float = 95.0,
    n_samples: int = 50,
) -> float:
    vals = []
    for rg in real_graphs:
        sample_size = min(n_samples, len(dpr_graphs))
        sampled = random.sample(dpr_graphs, sample_size) if len(dpr_graphs) > sample_size else dpr_graphs
        for dg in sampled:
            try:
                vals.append(tsi_function(rg, dg))
            except Exception:
                continue
    return float(np.percentile(vals, percentile)) if vals else 0.0


def compute_stable_rate(
    real_graphs: List[ReasoningTraceGraph],
    dpr_graphs: List[ReasoningTraceGraph],
    tsi_function,
    percentile: float = 95.0,
) -> Tuple[float, float, bool]:
    rvals = []
    for i in range(len(real_graphs)):
        for j in range(i + 1, len(real_graphs)):
            try:
                rvals.append(tsi_function(real_graphs[i], real_graphs[j]))
            except Exception:
                continue
    real_tsi = float(np.mean(rvals)) if rvals else 0.0
    threshold = compute_tsi_threshold(real_graphs, dpr_graphs, tsi_function, percentile)
    return real_tsi, threshold, real_tsi > threshold
