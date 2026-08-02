"""Tests for core types: NodeType, GraphNode, ReasoningTraceGraph, MotifEntry."""

import pytest
from rtsa.core.types import (
    NodeType, GraphNode, ReasoningTraceGraph,
    MotifEntry, MOTIF_CATALOG, MOTIF_LOOKUP,
)


class TestNodeType:
    def test_valid_set_contains_six_types(self):
        s = NodeType.valid_set()
        assert s == {"Retrieve", "Transform", "Compare", "Verify", "Branch", "Backtrack"}

    def test_from_string_case_insensitive(self):
        assert NodeType.from_string("retrieve") == NodeType.RETRIEVE
        assert NodeType.from_string("TRANSFORM") == NodeType.TRANSFORM
        assert NodeType.from_string("  verify  ") == NodeType.VERIFY
        assert NodeType.from_string("BaCkTrAcK") == NodeType.BACKTRACK

    def test_from_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown node type"):
            NodeType.from_string("unknown")

    def test_enum_values_match_strings(self):
        assert NodeType.RETRIEVE.value == "Retrieve"
        assert NodeType.TRANSFORM.value == "Transform"


class TestGraphNode:
    def test_create_valid_node(self):
        n = GraphNode(id=1, type=NodeType.RETRIEVE, span=(0, 10))
        assert n.id == 1
        assert n.type == NodeType.RETRIEVE
        assert n.span == (0, 10)

    def test_default_span(self):
        n = GraphNode(id=1, type=NodeType.TRANSFORM)
        assert n.span == (0, 0)

    def test_negative_span_raises(self):
        with pytest.raises(ValueError):
            GraphNode(id=1, type=NodeType.TRANSFORM, span=(-1, 5))

    def test_span_start_gt_end_raises(self):
        with pytest.raises(ValueError):
            GraphNode(id=1, type=NodeType.TRANSFORM, span=(10, 5))

    def test_model_dump_simple(self):
        n = GraphNode(id=3, type=NodeType.VERIFY, span=(20, 30))
        dumped = n.model_dump_simple()
        assert dumped == {"id": 3, "type": "Verify", "span": [20, 30]}

    def test_id_must_be_ge_1(self):
        with pytest.raises(Exception):
            GraphNode(id=0, type=NodeType.TRANSFORM)


class TestReasoningTraceGraph:
    def _make_simple_graph(self, trace_id="test_1"):
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.VERIFY),
        ]
        edges = [(1, 2), (2, 3)]
        return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)

    def test_create_valid_graph(self):
        g = self._make_simple_graph()
        assert g.trace_id == "test_1"
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_edges_reference_invalid_node_raises(self):
        nodes = [GraphNode(id=1, type=NodeType.RETRIEVE)]
        with pytest.raises(ValueError, match="not in node set"):
            ReasoningTraceGraph(trace_id="bad", nodes=nodes, edges=[(1, 99)])

    def test_to_networkx_returns_dag(self):
        g = self._make_simple_graph()
        G = g.to_networkx()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.nodes[1]["type"] == "Retrieve"

    def test_validate_dag_passes(self):
        g = self._make_simple_graph()
        assert g.validate_dag() is True

    def test_validate_dag_detects_cycle(self):
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        edges = [(1, 2), (2, 1)]
        g = ReasoningTraceGraph(trace_id="cycle", nodes=nodes, edges=edges)
        assert g.validate_dag() is False

    def test_validate_no_isolates_single_node(self):
        nodes = [GraphNode(id=1, type=NodeType.RETRIEVE)]
        g = ReasoningTraceGraph(trace_id="single", nodes=nodes, edges=[])
        assert g.validate_no_isolates() is True

    def test_validate_no_isolates_detects_orphan(self):
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.VERIFY),
        ]
        edges = [(1, 2)]  # node 3 is isolated
        g = ReasoningTraceGraph(trace_id="orphan", nodes=nodes, edges=edges)
        assert g.validate_no_isolates() is False

    def test_is_valid_empty_nodes(self):
        g = ReasoningTraceGraph(trace_id="empty", nodes=[], edges=[])
        valid, errors = g.is_valid()
        assert not valid
        assert "no nodes" in " ".join(errors).lower()

    def test_is_valid_with_cycle(self):
        nodes = [GraphNode(id=1, type=NodeType.RETRIEVE), GraphNode(id=2, type=NodeType.TRANSFORM)]
        g = ReasoningTraceGraph(trace_id="cyc", nodes=nodes, edges=[(1, 2), (2, 1)])
        valid, errors = g.is_valid()
        assert not valid

    def test_to_canonical_dict(self):
        g = self._make_simple_graph()
        d = g.to_canonical_dict()
        assert d["trace_id"] == "test_1"
        assert "graph" in d
        assert len(d["graph"]["nodes"]) == 3
        assert d["graph"]["edges"] == [[1, 2], [2, 3]]

    def test_default_fields(self):
        nodes = [GraphNode(id=1, type=NodeType.RETRIEVE)]
        g = ReasoningTraceGraph(trace_id="t", nodes=nodes, edges=[])
        assert g.model == ""
        assert g.domain == ""
        assert g.extractor == ""
        assert g.metadata == {}

    def test_from_json_null_span(self):
        """span: null in JSON must deserialize to (0, 0), not crash."""
        data = {
            "trace_id": "t1",
            "graph": {
                "nodes": [{"id": 1, "type": "Retrieve", "span": None}],
                "edges": [],
            },
        }
        g = ReasoningTraceGraph.from_json(data)
        assert g.nodes[0].span == (0, 0)

    def test_from_json_roundtrip(self):
        g = self._make_simple_graph()
        d = g.to_canonical_dict()
        g2 = ReasoningTraceGraph.from_json(d)
        assert g2.trace_id == g.trace_id
        assert [n.type for n in g2.nodes] == [n.type for n in g.nodes]
        assert g2.edges == g.edges


class TestMotifCatalog:
    def test_catalog_has_twelve_entries(self):
        assert len(MOTIF_CATALOG) == 12  # M1-M12 (M4 = Loop, rejected for DAGs)

    def test_catalog_ids(self):
        ids = {m.motif_id for m in MOTIF_CATALOG}
        assert ids == {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"}

    def test_motif_lookup(self):
        assert MOTIF_LOOKUP["M1"].pattern_name == "Chain(3)"
        assert MOTIF_LOOKUP["M3"].size == 4
        assert MOTIF_LOOKUP["M4"].pattern_name == "Loop(3)"
        assert MOTIF_LOOKUP["M5"].discovery_method == "preset"

    def test_motif_entry_creation(self):
        m = MotifEntry(
            motif_id="X1", pattern_name="Test", description="desc",
            size=3, node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY],
            edge_list=[(0, 1), (1, 2)],
        )
        assert m.motif_id == "X1"
        assert m.frequency == 0.0
        assert m.discovery_method == "preset"
