"""Tests for analysis.benchmark — extractor reliability benchmarking."""

import tempfile
from pathlib import Path

import pytest

from core.types import GraphNode, NodeType, ReasoningTraceGraph
from analysis.benchmark import ExtractorBenchmark, BenchmarkReport, benchmark_extractors


def _make_graph(n_nodes=5, trace_id="g"):
    """Helper: create a valid chain graph that satisfies NGS rules.
    
    NGS R2 (no_consecutive_repeat) requires no two adjacent nodes share
    the same type. We alternate Retrieve → Transform → Verify → Transform ...
    to guarantee compliance.
    """
    types_cycle = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY, NodeType.COMPARE]
    nodes = [
        GraphNode(id=i + 1, type=types_cycle[i % len(types_cycle)], text=f"step_{i}")
        for i in range(n_nodes)
    ]
    edges = [(i + 1, i + 2) for i in range(n_nodes - 1)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


def test_benchmark_basic():
    """Two extractors with identical graphs should have high TSI."""
    graphs_a = [_make_graph(n_nodes=5, trace_id=f"a{i}") for i in range(3)]
    graphs_b = [_make_graph(n_nodes=5, trace_id=f"b{i}") for i in range(3)]

    bench = ExtractorBenchmark()
    report = bench.run(
        {"ext-a": lambda t: t, "ext-b": lambda t: t},
        graphs={"ext-a": graphs_a, "ext-b": graphs_b},
        run_gcp=False,
        run_ngs=True,
        run_tsi=True,
    )

    assert "ext-a" in report.results
    assert "ext-b" in report.results
    # Identical graphs → high TSI
    assert report.results["ext-a"].mean_tsi_vs_others > 0.9
    assert report.results["ext-b"].mean_tsi_vs_others > 0.9


def test_benchmark_ngs_only():
    """NGS-only mode should compute pass rates."""
    valid = [_make_graph(n_nodes=5, trace_id=f"v{i}") for i in range(4)]
    bench = ExtractorBenchmark()
    report = bench.run(
        {"ext": lambda t: t},
        graphs={"ext": valid},
        run_gcp=False,
        run_ngs=True,
        run_tsi=False,
    )

    r = report.results["ext"]
    assert r.ngs_pass_rate == 1.0  # all valid chain graphs pass NGS (warnings don't count)
    # R1 atomicity may produce warnings for short text; that's OK.
    assert all(v.severity == "warning" for v in r.ngs_violations)


def test_benchmark_report_json():
    """Report should serialize to JSON correctly."""
    graphs = [_make_graph(n_nodes=5, trace_id=f"g{i}") for i in range(3)]
    bench = ExtractorBenchmark()
    report = bench.run(
        {"ext": lambda t: t},
        graphs={"ext": graphs},
        run_gcp=False,
        run_ngs=True,
        run_tsi=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    try:
        report.to_json(path)
        data = Path(path).read_text(encoding="utf-8")
        assert "ext" in data
        assert "overall_score" in data
    finally:
        Path(path).unlink(missing_ok=True)


def test_benchmark_ranking():
    """Ranking should be sorted by overall_score descending."""
    graphs_a = [_make_graph(n_nodes=5, trace_id=f"a{i}") for i in range(5)]
    # b has invalid graphs: two consecutive TRANSFORM nodes violate NGS R2
    graphs_b = [
        ReasoningTraceGraph(
            trace_id=f"b{i}",
            nodes=[
                GraphNode(id=1, type=NodeType.RETRIEVE, text="retrieve information"),
                GraphNode(id=2, type=NodeType.TRANSFORM, text="transform step one"),
                GraphNode(id=3, type=NodeType.TRANSFORM, text="transform step two"),  # R2 violation
                GraphNode(id=4, type=NodeType.VERIFY, text="verify result here"),
            ],
            edges=[(1, 2), (2, 3), (3, 4)],
        )
        for i in range(5)
    ]

    bench = ExtractorBenchmark()
    report = bench.run(
        {"good": lambda t: t, "bad": lambda t: t},
        graphs={"good": graphs_a, "bad": graphs_b},
        run_gcp=False,
        run_ngs=True,
        run_tsi=False,
    )

    assert report.ranking[0][0] == "good"
    assert report.winner == "good"
    assert report.results["good"].passed is True
    # bad extractor fails NGS (consecutive transforms = error)
    assert report.results["bad"].ngs_pass_rate == 0.0


def test_benchmark_one_liner():
    """Convenience function benchmark_extractors should work."""
    graphs = [_make_graph(n_nodes=5, trace_id=f"g{i}") for i in range(2)]
    report = benchmark_extractors(
        {"ext": lambda t: t},
        graphs={"ext": graphs},
    )
    assert isinstance(report, BenchmarkReport)
    assert "ext" in report.results
