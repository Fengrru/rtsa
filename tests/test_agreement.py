"""Tests for Inter-Annotator Agreement (3-layer IAA) and bias detection."""

import pytest
import numpy as np
from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.extractors.agreement import (
    levenshtein_distance, levenshtein_similarity,
    get_node_type_sequence, graph_level_iaa, motif_level_iaa,
    structure_level_iaa, compute_full_iaa,
    detect_length_bias, detect_syntax_artifact,
    _graph_edit_distance_approx,
)


def _make_graph(trace_id="t", node_types=None, edges=None):
    nodes = [
        GraphNode(id=i + 1, type=nt)
        for i, nt in enumerate(node_types or [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY])
    ]
    edges = edges or [(i + 1, i + 2) for i in range(len(nodes) - 1)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein_distance(["A", "B", "C"], ["A", "B", "C"]) == 0

    def test_one_substitution(self):
        assert levenshtein_distance(["A", "B", "C"], ["A", "X", "C"]) == 1

    def test_one_deletion(self):
        assert levenshtein_distance(["A", "B", "C"], ["A", "C"]) == 1

    def test_empty(self):
        assert levenshtein_distance([], []) == 0
        assert levenshtein_distance(["A"], []) == 1

    def test_similarity_perfect(self):
        assert levenshtein_similarity(["A", "B"], ["A", "B"]) == 1.0


class TestNodeTypeSequence:
    def test_linear_chain(self):
        g = _make_graph(node_types=[NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY])
        seq = get_node_type_sequence(g)
        assert len(seq) == 3

    def test_empty_graph(self):
        g = ReasoningTraceGraph(trace_id="e", nodes=[], edges=[])
        seq = get_node_type_sequence(g)
        assert seq == []


class TestGraphLevelIAA:
    def test_single_extractor(self):
        graphs = {"rbe": [_make_graph("a")]}
        result = graph_level_iaa(graphs)
        assert "fleiss_kappa" in result

    def test_two_extractors_identical(self):
        g = _make_graph()
        graphs = {"rbe": [g], "sbe": [g]}
        result = graph_level_iaa(graphs)
        assert result["mean_pairwise_similarity"] >= 0.99

    def test_two_extractors_different(self):
        g1 = _make_graph("a", [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY])
        g2 = _make_graph("b", [NodeType.BACKTRACK, NodeType.RETRIEVE, NodeType.TRANSFORM])
        graphs = {"rbe": [g1], "sbe": [g2]}
        result = graph_level_iaa(graphs)
        assert 0.0 <= result["mean_pairwise_similarity"] <= 1.0


class TestMotifLevelIAA:
    def test_single_extractor(self):
        graphs = {"rbe": [_make_graph()]}
        result = motif_level_iaa(graphs)
        assert "mean_pearson_r" in result

    def test_two_extractors(self):
        g = _make_graph()
        graphs = {"rbe": [g, g], "sbe": [g, g]}
        result = motif_level_iaa(graphs)
        assert -1.0 <= result["mean_pearson_r"] <= 1.0


class TestStructureLevelIAA:
    def test_single_extractor(self):
        graphs = {"rbe": [_make_graph()]}
        result = structure_level_iaa(graphs)
        assert "mean_ged_similarity" in result

    def test_identical_graphs(self):
        g = _make_graph()
        graphs = {"rbe": [g], "sbe": [g]}
        result = structure_level_iaa(graphs)
        assert result["mean_ged_similarity"] >= 0.9

    def test_ged_approx_identical(self):
        g = _make_graph()
        sim = _graph_edit_distance_approx(g, g)
        assert sim > 0.95

    def test_ged_approx_one_empty(self):
        g1 = _make_graph()
        g2 = ReasoningTraceGraph(trace_id="empty", nodes=[], edges=[])
        sim = _graph_edit_distance_approx(g1, g2)
        assert sim == 0.0


class TestFullIAA:
    def test_returns_three_layers(self):
        g = _make_graph()
        graphs = {"rbe": [g], "sbe": [g]}
        result = compute_full_iaa(graphs)
        assert "graph_level" in result
        assert "motif_level" in result
        assert "structure_level" in result


class TestBiasDetection:
    def test_length_bias_detection(self):
        llm_graphs = {"gpt4": [_make_graph() for _ in range(3)]}
        rand_graphs = [_make_graph("r") for _ in range(3)]
        result = detect_length_bias(llm_graphs, rand_graphs, threshold=0.3)
        assert "gpt4" in result
        assert "length_bias_detected" in result["gpt4"]

    def test_syntax_artifact_detection(self):
        sbe_graphs = [_make_graph("s") for _ in range(3)]
        llm_graphs = {"gpt4": [_make_graph() for _ in range(3)]}
        result = detect_syntax_artifact(sbe_graphs, llm_graphs, threshold=0.7)
        assert "gpt4" in result
        assert "syntax_artifact_detected" in result["gpt4"]
