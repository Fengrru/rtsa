"""Tests for analysis.fingerprint — LLM authorship attribution."""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from core.types import GraphNode, NodeType, ReasoningTraceGraph
from analysis.fingerprint import ModelFingerprint, enroll_model, identify_author


def _make_graph(n_nodes, trace_id="g"):
    """Helper: create a simple chain graph with *n_nodes* TRANSFORM nodes."""
    nodes = [GraphNode(id=i + 1, type=NodeType.TRANSFORM, text=f"step_{i}") for i in range(n_nodes)]
    edges = [(i + 1, i + 2) for i in range(n_nodes - 1)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


def test_enroll_minimum():
    """Enrolling with 5 graphs should produce a valid signature."""
    graphs = [_make_graph(n_nodes=i + 3, trace_id=f"g{i}") for i in range(5)]
    fp = ModelFingerprint()
    sig = fp.enroll("test-model", graphs)

    assert sig.model_name == "test-model"
    assert sig.n_samples == 5
    assert sig.feature_mean.shape[0] == fp.total_dim


def test_identify_basic():
    """A graph from the enrolled model should be identified correctly."""
    model_a_graphs = [_make_graph(n_nodes=5 + i, trace_id=f"a{i}") for i in range(8)]
    model_b_graphs = [_make_graph(n_nodes=10 + i, trace_id=f"b{i}") for i in range(8)]

    fp = ModelFingerprint()
    fp.enroll("model-a", model_a_graphs)
    fp.enroll("model-b", model_b_graphs)

    unknown = _make_graph(n_nodes=6, trace_id="unknown")
    result = fp.identify(unknown)

    assert result.predicted_model in ("model-a", "model-b")
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.all_scores) == 2


def test_identify_strong_signal():
    """Very different graph sizes should produce high-confidence separation."""
    model_a_graphs = [_make_graph(n_nodes=5, trace_id=f"a{i}") for i in range(10)]
    model_b_graphs = [_make_graph(n_nodes=20, trace_id=f"b{i}") for i in range(10)]

    fp = ModelFingerprint()
    fp.enroll("small-model", model_a_graphs)
    fp.enroll("large-model", model_b_graphs)

    unknown_small = _make_graph(n_nodes=5, trace_id="u")
    result = fp.identify(unknown_small)

    assert result.predicted_model == "small-model"
    assert result.confidence > 0.5


def test_persistence():
    """Save and load signatures should preserve state."""
    graphs = [_make_graph(n_nodes=5, trace_id=f"g{i}") for i in range(5)]
    fp = ModelFingerprint()
    fp.enroll("persist-model", graphs)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        fp.save(path)

        fp2 = ModelFingerprint()
        fp2.load(path)
        assert "persist-model" in fp2.signatures
        assert fp2.signatures["persist-model"].n_samples == 5
    finally:
        Path(path).unlink(missing_ok=True)


def test_one_liner_identify():
    """Test the convenience one-liner identify_author."""
    graphs = [_make_graph(n_nodes=5, trace_id=f"g{i}") for i in range(5)]

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        enroll_model("oneliner-model", graphs, save_path=path)
        unknown = _make_graph(n_nodes=5, trace_id="u")
        result = identify_author(unknown, path)
        assert result.predicted_model == "oneliner-model"
    finally:
        Path(path).unlink(missing_ok=True)


def test_empty_enroll_warning():
    """Enrolling with < min_samples should log a warning but still work."""
    graphs = [_make_graph(n_nodes=5, trace_id="g0")]
    fp = ModelFingerprint()
    sig = fp.enroll("tiny-model", graphs, min_samples=5)
    assert sig.n_samples == 1


def test_batch_identify():
    """Batch identification should return a list of results."""
    graphs = [_make_graph(n_nodes=5, trace_id=f"g{i}") for i in range(5)]
    fp = ModelFingerprint()
    fp.enroll("batch-model", graphs)

    unknowns = [_make_graph(n_nodes=5, trace_id=f"u{i}") for i in range(3)]
    results = fp.identify_batch(unknowns)
    assert len(results) == 3
    for r in results:
        assert r.predicted_model == "batch-model"
