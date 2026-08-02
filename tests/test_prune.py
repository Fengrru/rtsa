"""Tests for analysis.prune — redundancy detection & CoT pruning."""

import pytest
from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.analysis.prune import (
    PruneConfig,
    RedundancyAnalyzer,
    prune_graph,
)


def _make_graph(nodes_spec, edges):
    """Helper: build a ReasoningTraceGraph from compact specs.
    
    *nodes_spec* is a list of (index, NodeType) where index is 0-based.
    GraphNode ids are automatically mapped to 1-based to satisfy schema.
    """
    nodes = [GraphNode(id=i + 1, type=t, text=f"{t.value}_{i}") for i, t in nodes_spec]
    return ReasoningTraceGraph(
        trace_id="test_graph",
        nodes=nodes,
        edges=edges,
    )


def test_empty_graph():
    g = ReasoningTraceGraph(trace_id="empty", nodes=[], edges=[])
    report = prune_graph(g, apply=False)
    assert report.redundancy_regions == []
    assert report.total_estimated_savings == 0


def test_excessive_verification():
    """10 nodes, 5 verifies → density 0.5 > threshold 0.4."""
    nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
    edges = [(i + 1, i + 2) for i in range(9)]
    g = _make_graph(nodes, edges)

    analyzer = RedundancyAnalyzer(config=PruneConfig(verify_density_high=0.40))
    report = analyzer.analyze(g)

    verify_regions = [r for r in report.redundancy_regions if r.region_type == "excessive_verification"]
    assert len(verify_regions) >= 1
    assert verify_regions[0].confidence > 0.5


def test_underutilized_branch():
    """Branch with only 1 outgoing edge → underutilized."""
    nodes = [
        (0, NodeType.RETRIEVE),
        (1, NodeType.BRANCH),
        (2, NodeType.TRANSFORM),
        (3, NodeType.VERIFY),
    ]
    edges = [(1, 2), (2, 3), (3, 4)]
    g = _make_graph(nodes, edges)

    analyzer = RedundancyAnalyzer(config=PruneConfig())
    report = analyzer.analyze(g)

    branch_regions = [r for r in report.redundancy_regions if r.region_type == "underutilized_branch"]
    assert len(branch_regions) >= 1


def test_redundant_transform_chain():
    """5 consecutive TRANSFORM nodes → flag the excess."""
    nodes = [(i, NodeType.TRANSFORM) for i in range(5)]
    edges = [(i + 1, i + 2) for i in range(4)]
    g = _make_graph(nodes, edges)

    analyzer = RedundancyAnalyzer(config=PruneConfig(max_consecutive_transforms=3))
    report = analyzer.analyze(g)

    chain_regions = [r for r in report.redundancy_regions if r.region_type == "redundant_transform_chain"]
    assert len(chain_regions) >= 1
    # excess = nodes 3,4 (first 3 kept)
    assert len(chain_regions[0].node_ids) == 2


def test_prune_apply_deletes_nodes():
    """Applying pruning should reduce node count."""
    nodes = [(i, NodeType.TRANSFORM) for i in range(5)]
    edges = [(i + 1, i + 2) for i in range(4)]
    g = _make_graph(nodes, edges)

    report = prune_graph(g, apply=True, config=PruneConfig(max_consecutive_transforms=3))

    assert report.pruned_graph is not None
    assert len(report.pruned_graph.nodes) < len(g.nodes)
    assert report.structural_integrity_score > 0.0


def test_prune_apply_removes_all_excess_chain_nodes():
    """Redundant transform chain: ALL excess nodes are deleted (not just one)."""
    nodes = [(i, NodeType.TRANSFORM) for i in range(5)]
    edges = [(i + 1, i + 2) for i in range(4)]
    g = _make_graph(nodes, edges)

    report = prune_graph(g, apply=True, config=PruneConfig(max_consecutive_transforms=3))

    assert report.pruned_graph is not None
    # first 3 kept as anchor, excess nodes (4th & 5th) both removed
    assert len(report.pruned_graph.nodes) == 3


def test_prune_merge_keeps_anchor_and_merges_text():
    """Excessive verification merge: one anchor node kept, absorbed text merged in."""
    nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
    edges = [(i + 1, i + 2) for i in range(9)]
    g = _make_graph(nodes, edges)

    report = prune_graph(g, apply=True, config=PruneConfig(verify_density_high=0.40))

    assert report.pruned_graph is not None
    pruned = report.pruned_graph
    assert len(pruned.nodes) < len(g.nodes)
    verify_nodes = [n for n in pruned.nodes if n.type == NodeType.VERIFY]
    assert len(verify_nodes) == 1  # merge collapses the contiguous block
    assert "Verify_5" in verify_nodes[0].text  # anchor text retained
    assert "Verify_9" in verify_nodes[0].text  # absorbed text merged


def test_shallow_wide_bloat():
    """Depth 2 with 10 nodes → shallow-wide bloat."""
    nodes = [(0, NodeType.RETRIEVE)] + [(i, NodeType.TRANSFORM) for i in range(1, 10)]
    edges = [(1, i + 1) for i in range(1, 10)]
    g = _make_graph(nodes, edges)

    analyzer = RedundancyAnalyzer(config=PruneConfig(max_depth_to_nodes_ratio=0.25))
    report = analyzer.analyze(g)

    bloat_regions = [r for r in report.redundancy_regions if r.region_type == "shallow_wide_bloat"]
    assert len(bloat_regions) >= 1


def test_batch_analysis():
    """Batch mode should return same number of reports."""
    graphs = [
        _make_graph([(i, NodeType.TRANSFORM) for i in range(5)], [(i + 1, i + 2) for i in range(4)]),
        _make_graph([(i, NodeType.TRANSFORM) for i in range(3)], [(i + 1, i + 2) for i in range(2)]),
    ]
    analyzer = RedundancyAnalyzer(config=PruneConfig(max_consecutive_transforms=3))
    reports = analyzer.analyze_batch(graphs)
    assert len(reports) == len(graphs)


class TestSavingsRange:
    """A3: heuristic uncertainty band on estimated token savings."""

    def test_range_centered_on_estimate(self):
        nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
        edges = [(i + 1, i + 2) for i in range(9)]
        g = _make_graph(nodes, edges)
        report = prune_graph(g, apply=True, config=PruneConfig(verify_density_high=0.40))
        lo, hi = report.savings_range()
        assert lo <= report.total_estimated_savings <= hi

    def test_custom_uncertainty(self):
        nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
        edges = [(i + 1, i + 2) for i in range(9)]
        g = _make_graph(nodes, edges)
        report = prune_graph(g, apply=True, config=PruneConfig(verify_density_high=0.40))
        lo, hi = report.savings_range(uncertainty=0.25)
        assert lo <= report.total_estimated_savings <= hi

    def test_summary_includes_range(self):
        nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
        edges = [(i + 1, i + 2) for i in range(9)]
        g = _make_graph(nodes, edges)
        report = prune_graph(g, apply=True, config=PruneConfig(verify_density_high=0.40))
        assert "(range" in report.summary()

    def test_zero_savings_zero_range(self):
        g = _make_graph(
            [(i, NodeType.TRANSFORM) for i in range(3)],
            [(i + 1, i + 2) for i in range(2)],
        )
        report = prune_graph(g, apply=False)
        assert report.savings_range() == (0, 0)


class TestDomainOverrides:
    """B8: per-domain adaptive thresholds."""

    def test_resolve_for_domain_applies_override(self):
        cfg = PruneConfig(
            verify_density_high=0.40,
            domain_overrides={"math": {"verify_density_high": 0.60}},
        )
        resolved = cfg.resolve_for_domain("math")
        assert resolved.verify_density_high == 0.60
        assert resolved is not cfg  # immutable copy

    def test_resolve_for_domain_unknown_domain_keeps_base(self):
        cfg = PruneConfig(
            verify_density_high=0.40,
            domain_overrides={"math": {"verify_density_high": 0.60}},
        )
        assert cfg.resolve_for_domain("biology") is cfg

    def test_unknown_override_key_ignored(self):
        cfg = PruneConfig(
            verify_density_high=0.40,
            domain_overrides={"math": {"no_such_param": 1}},
        )
        resolved = cfg.resolve_for_domain("math")
        assert resolved.verify_density_high == 0.40

    def test_analyze_uses_domain_override(self):
        """A lax domain threshold must suppress verification flags."""
        nodes = [(i, NodeType.VERIFY if i >= 5 else NodeType.TRANSFORM) for i in range(10)]
        edges = [(i + 1, i + 2) for i in range(9)]
        g = _make_graph(nodes, edges)
        g.domain = "math"
        cfg = PruneConfig(
            verify_density_high=0.40,
            domain_overrides={"math": {"verify_density_high": 0.90}},
        )
        report = prune_graph(g, apply=False, config=cfg)
        verify_regions = [
            r for r in report.redundancy_regions
            if r.region_type == "excessive_verification"
        ]
        assert verify_regions == []
