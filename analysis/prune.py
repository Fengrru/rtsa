"""
Redundancy Analysis & CoT Pruning — Direction 1 (P0).

Provides heuristic redundancy detection and pruning recommendations
for reasoning trace graphs. Built entirely on existing core.metrics
and core.motif_matcher — no new graph-theoretic primitives needed.

Design goal: input a ReasoningTraceGraph → output a diagnostic report
that tells you *which* nodes are redundant, *why*, and *how much*
you save by pruning them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.types import GraphNode, NodeType, ReasoningTraceGraph
from core.metrics import GraphMetrics, compute_graph_features, compute_feature_matrix
from core.motif_matcher import MotifMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PruneConfig:
    """Tunable thresholds for redundancy detection."""
    # Verification rules
    verify_density_high: float = 0.40          # excessive verification
    verify_late_stage_ratio: float = 0.60      # >60% verifies in last 30% nodes

    # Branch rules
    branch_min_outgoing: int = 2               # already enforced by NGS R5
    branch_utilization_min: float = 0.50       # at least 50% of branches have >1 successor

    # Transform chain rules
    max_consecutive_transforms: int = 3        # >3 consecutive T → flag

    # General structural redundancy
    max_depth_to_nodes_ratio: float = 0.25     # shallow & wide = potential bloat
    entropy_low_threshold: float = 0.50        # low type diversity = repetitive

    # Token estimation (rough heuristic)
    avg_tokens_per_node: int = 35

    # Signal-enhanced redundancy scoring (optional adapters)
    use_calibration_signal: bool = False
    calibration_weight: float = 0.35
    use_prm_signal: bool = False
    prm_weight: float = 0.30

    # Pruning execution threshold
    min_confidence_threshold: float = 0.50  # only prune regions with confidence >= this

    # Domain-specific threshold overrides, keyed by ``graph.domain``, e.g.
    # {"math": {"verify_density_high": 0.45, "max_consecutive_transforms": 4}}.
    # Structural error signatures are domain-dependent (CRV, arXiv 2510.09312),
    # so long-verbose domains may need laxer thresholds than terse ones.
    domain_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def resolve_for_domain(self, domain: str) -> "PruneConfig":
        """Return a copy with this domain's overrides applied (self if none)."""
        overrides = self.domain_overrides.get(domain, {})
        if not overrides:
            return self
        valid = {k: v for k, v in overrides.items() if hasattr(self, k)}
        return replace(self, **valid)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RedundancyRegion:
    """A contiguous or logical region flagged as redundant."""
    region_type: str                           # e.g. "excessive_verify", "dead_branch"
    node_ids: List[int]
    confidence: float                          # 0.0–1.0
    description: str
    estimated_token_savings: int
    suggested_action: str                      # "delete", "merge", "review"


@dataclass
class PruningReport:
    """Complete diagnostic + optimization report for a single graph."""
    trace_id: str
    original_n_nodes: int
    original_n_edges: int
    features: GraphMetrics
    redundancy_regions: List[RedundancyRegion]
    pruned_graph: Optional[ReasoningTraceGraph] = None
    total_estimated_savings: int = 0
    structural_integrity_score: float = 1.0    # post-prune DAG validity

    def savings_range(self, uncertainty: float = 0.40) -> Tuple[int, int]:
        """Heuristic error band around the estimated token savings.

        Token estimates are rough (``avg_tokens_per_node``), so the report
        surfaces an uncertainty interval instead of a false-precision point
        estimate. Default +/-40% covers the heuristic nature of the estimator.
        """
        lo = int(self.total_estimated_savings * (1.0 - uncertainty))
        hi = int(self.total_estimated_savings * (1.0 + uncertainty))
        return lo, hi

    def summary(self) -> str:
        lo, hi = self.savings_range()
        lines = [
            f"PruningReport for {self.trace_id}",
            f"  Nodes: {self.original_n_nodes} → "
            f"{len(self.pruned_graph.nodes) if self.pruned_graph else 'N/A'}",
            f"  Redundancy regions: {len(self.redundancy_regions)}",
            f"  Est. token savings: {self.total_estimated_savings} "
            f"(range {lo}-{hi})",
            f"  Integrity score: {self.structural_integrity_score:.2f}",
        ]
        for r in self.redundancy_regions:
            lines.append(
                f"  [{r.region_type}] nodes={r.node_ids} conf={r.confidence:.2f} "
                f"action={r.suggested_action} save={r.estimated_token_savings}tok"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Redundancy Analyzer
# ---------------------------------------------------------------------------

class RedundancyAnalyzer:
    """Heuristic redundancy detector for ReasoningTraceGraphs.

    All detection rules are composable — each returns a list of
    RedundancyRegion, and the top-level ``analyze()`` merges them.
    """

    def __init__(self, config: Optional[PruneConfig] = None,
                 calibration_adapter=None, prm_adapter=None):
        self.config = config or PruneConfig()
        self.matcher = MotifMatcher()
        self._cal_adapt = calibration_adapter
        self._prm_adapt = prm_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        graph: ReasoningTraceGraph,
        apply_pruning: bool = False,
    ) -> PruningReport:
        """Run full redundancy analysis on *graph*.

        Args:
            graph: input reasoning trace graph
            apply_pruning: if True, produce a pruned copy of the graph

        Per-domain threshold overrides (``PruneConfig.domain_overrides``)
        are resolved from ``graph.domain`` on every call.
        """
        self.config = self.config.resolve_for_domain(graph.domain)
        features = compute_graph_features(graph)
        regions: List[RedundancyRegion] = []

        regions.extend(self._detect_excessive_verification(graph, features))
        regions.extend(self._detect_underutilized_branches(graph, features))
        regions.extend(self._detect_redundant_transform_chains(graph))
        regions.extend(self._detect_structural_bloat(graph, features))

        # ------------------------------------------------------------------
        # Signal fusion: optional metacog + PRM enhancement
        # ------------------------------------------------------------------
        if self.config.use_calibration_signal or self.config.use_prm_signal:
            regions = self._fuse_signal_scores(graph, regions)

        # Only regions that are actually prunable (delete/merge) contribute
        # to the estimated savings; "review" regions are advisory only.
        total_savings = sum(
            r.estimated_token_savings for r in regions
            if r.suggested_action in ("delete", "merge")
        )

        pruned = None
        integrity = 1.0
        if apply_pruning and regions:
            pruned, integrity = self._apply_pruning(graph, regions)

        return PruningReport(
            trace_id=graph.trace_id,
            original_n_nodes=len(graph.nodes),
            original_n_edges=len(graph.edges),
            features=features,
            redundancy_regions=regions,
            pruned_graph=pruned,
            total_estimated_savings=total_savings,
            structural_integrity_score=integrity,
        )

    def analyze_batch(
        self,
        graphs: List[ReasoningTraceGraph],
        apply_pruning: bool = False,
    ) -> List[PruningReport]:
        """Batch analysis with shared MotifMatcher (faster)."""
        return [self.analyze(g, apply_pruning=apply_pruning) for g in graphs]

    # ------------------------------------------------------------------
    # Detection rules (heuristic)
    # ------------------------------------------------------------------

    def _detect_excessive_verification(
        self, graph: ReasoningTraceGraph, features: GraphMetrics
    ) -> List[RedundancyRegion]:
        """Flag clusters of Verify nodes, especially in late stages."""
        cfg = self.config
        regions: List[RedundancyRegion] = []
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
        n = len(nodes_sorted)
        if n == 0:
            return regions

        verify_ids = [node.id for node in nodes_sorted if node.type == NodeType.VERIFY]
        if not verify_ids:
            return regions

        # Rule A: global verify density
        verify_density = len(verify_ids) / n
        if verify_density > cfg.verify_density_high:
            # Group contiguous verify blocks
            blocks = self._to_contiguous_blocks(verify_ids)
            for block in blocks:
                conf = min(1.0, verify_density / 0.60)
                regions.append(RedundancyRegion(
                    region_type="excessive_verification",
                    node_ids=block,
                    confidence=conf,
                    description=f"Verify density {verify_density:.2f} exceeds threshold {cfg.verify_density_high}",
                    estimated_token_savings=len(block) * cfg.avg_tokens_per_node,
                    suggested_action="merge" if len(block) > 1 else "review",
                ))

        # Rule B: late-stage verify concentration (position-based, NOT id-based)
        late_cutoff_idx = int(n * 0.70)
        late_verify_ids = [
            node.id
            for i, node in enumerate(nodes_sorted)
            if node.type == NodeType.VERIFY and i >= late_cutoff_idx
        ]
        if late_verify_ids and len(late_verify_ids) / max(len(verify_ids), 1) > cfg.verify_late_stage_ratio:
            conf = 0.75
            regions.append(RedundancyRegion(
                region_type="late_stage_verification",
                node_ids=late_verify_ids,
                confidence=conf,
                description=f"{len(late_verify_ids)}/{len(verify_ids)} verifies appear in last 30% of graph",
                estimated_token_savings=len(late_verify_ids) * cfg.avg_tokens_per_node // 2,
                suggested_action="review",
            ))

        return regions

    def _detect_underutilized_branches(
        self, graph: ReasoningTraceGraph, features: GraphMetrics
    ) -> List[RedundancyRegion]:
        """Flag Branch nodes that don't actually fork meaningfully."""
        cfg = self.config
        regions: List[RedundancyRegion] = []

        branch_nodes = [n for n in graph.nodes if n.type == NodeType.BRANCH]
        if not branch_nodes:
            return regions

        underutilized: List[int] = []
        for node in branch_nodes:
            outgoing = [e[1] for e in graph.edges if e[0] == node.id]
            if len(outgoing) < 2:
                underutilized.append(node.id)
            else:
                # Check if branches reconverge immediately (shallow fork)
                depths = []
                for succ in outgoing:
                    d = self._max_depth_from(graph, succ, visited=set())
                    depths.append(d)
                if max(depths, default=0) <= 1:
                    underutilized.append(node.id)

        if underutilized:
            util_rate = 1.0 - len(underutilized) / len(branch_nodes)
            if util_rate < cfg.branch_utilization_min:
                regions.append(RedundancyRegion(
                    region_type="underutilized_branch",
                    node_ids=underutilized,
                    confidence=min(1.0, 1.0 - util_rate),
                    description=f"{len(underutilized)}/{len(branch_nodes)} branches are underutilized",
                    estimated_token_savings=len(underutilized) * cfg.avg_tokens_per_node,
                    suggested_action="merge",
                ))

        return regions

    def _detect_redundant_transform_chains(
        self, graph: ReasoningTraceGraph
    ) -> List[RedundancyRegion]:
        """Flag long chains of consecutive Transform nodes (NGS R2 violation signal)."""
        cfg = self.config
        regions: List[RedundancyRegion] = []
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)

        chains: List[List[int]] = []
        current: List[int] = []
        for node in nodes_sorted:
            if node.type == NodeType.TRANSFORM:
                current.append(node.id)
            else:
                if len(current) > cfg.max_consecutive_transforms:
                    chains.append(current)
                current = []
        if len(current) > cfg.max_consecutive_transforms:
            chains.append(current)

        for chain in chains:
            # Only flag the *excess* nodes (keep the first few as anchor)
            excess = chain[cfg.max_consecutive_transforms:]
            regions.append(RedundancyRegion(
                region_type="redundant_transform_chain",
                node_ids=excess,
                confidence=0.70,
                description=f"Transform chain length {len(chain)} > threshold {cfg.max_consecutive_transforms}",
                estimated_token_savings=len(excess) * cfg.avg_tokens_per_node,
                suggested_action="merge",
            ))

        return regions

    def _detect_structural_bloat(
        self, graph: ReasoningTraceGraph, features: GraphMetrics
    ) -> List[RedundancyRegion]:
        """Flag graphs that are shallow & wide (low depth-to-nodes ratio) or low-entropy."""
        cfg = self.config
        regions: List[RedundancyRegion] = []
        n = features.n_nodes
        if n == 0:
            return regions

        depth_ratio = features.depth / max(n, 1)
        if depth_ratio < cfg.max_depth_to_nodes_ratio and n > 5:
            regions.append(RedundancyRegion(
                region_type="shallow_wide_bloat",
                node_ids=[],
                confidence=0.55,
                description=f"Depth/nodes ratio {depth_ratio:.2f} suggests shallow-wide bloat",
                estimated_token_savings=int(n * cfg.avg_tokens_per_node * 0.15),
                suggested_action="review",
            ))

        if features.entropy < cfg.entropy_low_threshold and n > 5:
            regions.append(RedundancyRegion(
                region_type="low_type_diversity",
                node_ids=[],
                confidence=0.60,
                description=f"Type entropy {features.entropy:.2f} < {cfg.entropy_low_threshold}",
                estimated_token_savings=int(n * cfg.avg_tokens_per_node * 0.10),
                suggested_action="review",
            ))

        return regions

    # ------------------------------------------------------------------
    # Pruning engine
    # ------------------------------------------------------------------

    def _apply_pruning(
        self,
        graph: ReasoningTraceGraph,
        regions: List[RedundancyRegion],
    ) -> Tuple[ReasoningTraceGraph, float]:
        """Execute pruning operations and return pruned graph + integrity score.

        Only regions whose confidence >= config.min_confidence_threshold are acted upon.
        After deletion, unreachable orphan nodes are cleaned up to preserve DAG validity.
        """
        # Collect nodes to delete (respect confidence threshold)
        to_delete: Set[int] = set()
        merge_texts: Dict[int, List[str]] = {}  # kept node id -> absorbed texts
        node_text_map = {n.id: n.text for n in graph.nodes}
        for r in regions:
            if r.confidence < self.config.min_confidence_threshold:
                continue
            if r.suggested_action == "delete":
                to_delete.update(r.node_ids)
            elif r.suggested_action == "merge":
                if r.region_type == "redundant_transform_chain":
                    # node_ids are the *excess* nodes only — the chain anchor
                    # lives outside this region, so all of them are redundant.
                    to_delete.update(r.node_ids)
                elif len(r.node_ids) > 1:
                    # Merge region: keep the first node, absorb the rest's text
                    keep = r.node_ids[0]
                    for extra in r.node_ids[1:]:
                        to_delete.add(extra)
                        txt = node_text_map.get(extra)
                        if txt:
                            merge_texts.setdefault(keep, []).append(txt)

        if not to_delete:
            return graph, 1.0

        # Initial keep set after direct deletions (absorbing merged text)
        kept_nodes = []
        for n in graph.nodes:
            if n.id in to_delete:
                continue
            if n.id in merge_texts:
                extra = " ".join(merge_texts[n.id])
                kept_nodes.append(n.model_copy(update={"text": (n.text + " " + extra).strip()}))
            else:
                kept_nodes.append(n)
        kept_ids = {n.id for n in kept_nodes}
        pruned_edges = [e for e in graph.edges if e[0] in kept_ids and e[1] in kept_ids]

        # Reroute around deleted blocks: connect each kept predecessor to
        # every kept successor whose path consists only of deleted nodes.
        # Without this, a merge anchor can become unreachable when another
        # region deletes its in-between chain — and would then be removed
        # as an "orphan", defeating the merge (e.g. a Transform chain
        # collapsed right before the Verify block that kept its anchor).
        existing = {(u, v) for u, v in pruned_edges}
        del_edges = [
            (u, v) for u, v in graph.edges if u in to_delete and v in to_delete
        ]
        for p in kept_ids:
            for d in [v for u, v in graph.edges if u == p and v in to_delete]:
                stack = [d]
                seen: Set[int] = set()
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    for s in [
                        v for u, v in graph.edges
                        if u == cur and v in kept_ids
                    ]:
                        if p != s and (p, s) not in existing:
                            pruned_edges.append((p, s))
                            existing.add((p, s))
                    for nxt in [v for u, v in del_edges if u == cur]:
                        stack.append(nxt)

        # Clean up orphan nodes (no path from root)
        root_id = graph.nodes[0].id if graph.nodes else 0
        reachable: Set[int] = set()
        stack = [root_id]
        while stack:
            curr = stack.pop()
            if curr in reachable or curr not in kept_ids:
                continue
            reachable.add(curr)
            for e in pruned_edges:
                if e[0] == curr and e[1] in kept_ids:
                    stack.append(e[1])

        # Remove unreachable nodes
        kept_nodes = [n for n in kept_nodes if n.id in reachable]
        kept_ids = {n.id for n in kept_nodes}
        pruned_edges = [e for e in pruned_edges if e[0] in kept_ids and e[1] in kept_ids]

        # Integrity: DAG property + reachability
        integrity = self._compute_integrity(kept_nodes, pruned_edges, root_id)

        pruned_graph = ReasoningTraceGraph(
            trace_id=f"{graph.trace_id}_pruned",
            nodes=kept_nodes,
            edges=pruned_edges,
            domain=graph.domain,
            model=graph.model,
            extractor=graph.extractor,
        )

        return pruned_graph, integrity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_contiguous_blocks(ids: List[int]) -> List[List[int]]:
        if not ids:
            return []
        ids_sorted = sorted(ids)
        blocks: List[List[int]] = [[ids_sorted[0]]]
        for x in ids_sorted[1:]:
            if x == blocks[-1][-1] + 1:
                blocks[-1].append(x)
            else:
                blocks.append([x])
        return blocks

    def _fuse_signal_scores(
        self,
        graph: ReasoningTraceGraph,
        regions: List[RedundancyRegion],
    ) -> List[RedundancyRegion]:
        """Re-weight region confidence using optional metacog / PRM signals.

        Higher returned confidence = stronger redundancy suspicion.
        Both adapters return *quality* scores (high = good step).
        We invert them to obtain *redundancy* scores.
        """
        node_map = {n.id: n for n in graph.nodes}
        sorted_nodes = sorted(graph.nodes, key=lambda n: n.id)
        prev_map: Dict[int, Optional[GraphNode]] = {}
        for i, n in enumerate(sorted_nodes):
            prev_map[n.id] = sorted_nodes[i - 1] if i > 0 else None

        # Lazy-init adapters
        if self._cal_adapt is None and self.config.use_calibration_signal:
            from analysis.signal_adapters import make_calibration_adapter
            self._cal_adapt = make_calibration_adapter()
        if self._prm_adapt is None and self.config.use_prm_signal:
            from analysis.signal_adapters import make_prm_adapter
            self._prm_adapt = make_prm_adapter()

        for region in regions:
            if not region.node_ids:
                continue

            fused_scores: List[float] = []
            fused_weights: List[float] = []

            # Structural confidence is already a redundancy score
            w_struct = max(
                0.0,
                1.0
                - (self.config.calibration_weight if self.config.use_calibration_signal else 0.0)
                - (self.config.prm_weight if self.config.use_prm_signal else 0.0),
            )
            fused_scores.append(region.confidence)
            fused_weights.append(w_struct)

            if self.config.use_calibration_signal and self._cal_adapt:
                cal_vals = [
                    self._cal_adapt.score(node_map[nid].text, node_map[nid].type.value)
                    for nid in region.node_ids
                    if nid in node_map
                ]
                if cal_vals:
                    avg_cal = sum(cal_vals) / len(cal_vals)
                    fused_scores.append(1.0 - avg_cal)  # invert: low confidence -> redundant
                    fused_weights.append(self.config.calibration_weight)

            if self.config.use_prm_signal and self._prm_adapt:
                prm_vals = []
                for nid in region.node_ids:
                    if nid not in node_map:
                        continue
                    node = node_map[nid]
                    prev = prev_map.get(nid)
                    prm_vals.append(
                        self._prm_adapt.score(node.text, prev.text if prev else None)
                    )
                if prm_vals:
                    avg_prm = sum(prm_vals) / len(prm_vals)
                    fused_scores.append(1.0 - avg_prm)  # invert: low reward -> redundant
                    fused_weights.append(self.config.prm_weight)

            total_w = sum(fused_weights)
            if total_w > 0:
                region.confidence = min(
                    1.0,
                    sum(s * w for s, w in zip(fused_scores, fused_weights)) / total_w,
                )

        return regions

    def _max_depth_from(
        self, graph: ReasoningTraceGraph, start_id: int, visited: Optional[Set[int]] = None
    ) -> int:
        """Longest path depth from *start_id*.

        ``visited`` tracks the current DFS *path* (nodes are removed on
        backtrack) so sibling branches do not block each other — a shared
        global visited set would systematically underestimate depth in
        multi-branch graphs.
        """
        if visited is None:
            visited = set()
        if start_id in visited:
            return 0
        visited.add(start_id)
        children = [e[1] for e in graph.edges if e[0] == start_id]
        depth = 1 + max(
            (self._max_depth_from(graph, c, visited) for c in children),
            default=0,
        )
        visited.discard(start_id)
        return depth

    @staticmethod
    def _compute_integrity(
        nodes: List[GraphNode], edges: List[Tuple[int, int]], root_id: int
    ) -> float:
        """Simple integrity score: 1.0 if DAG + all nodes reachable from root."""
        if not nodes:
            return 0.0
        node_ids = {n.id for n in nodes}

        # DAG check: no cycles via DFS
        def has_cycle(curr: int, path: Set[int]) -> bool:
            if curr in path:
                return True
            path.add(curr)
            for e in edges:
                if e[0] == curr and e[1] in node_ids:
                    if has_cycle(e[1], path):
                        return True
            path.discard(curr)
            return False

        if has_cycle(root_id, set()):
            return 0.0

        # Reachability
        reachable: Set[int] = set()
        stack = [root_id]
        while stack:
            curr = stack.pop()
            if curr in reachable:
                continue
            reachable.add(curr)
            for e in edges:
                if e[0] == curr and e[1] in node_ids:
                    stack.append(e[1])

        return len(reachable) / max(len(node_ids), 1)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def prune_graph(
    graph: ReasoningTraceGraph,
    apply: bool = True,
    config: Optional[PruneConfig] = None,
) -> PruningReport:
    """One-liner: analyze + optionally prune a single graph."""
    analyzer = RedundancyAnalyzer(config=config)
    return analyzer.analyze(graph, apply_pruning=apply)
