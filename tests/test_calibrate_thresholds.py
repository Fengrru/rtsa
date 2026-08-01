"""Tests for experiments.calibrate_thresholds (A1)."""

import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.prune import PruneConfig, RedundancyAnalyzer
from experiments.calibrate_thresholds import (
    PARAM_GRID, aggregate, predict_redundant, run,
    synthetic_annotated_graphs,
)


def test_synthetic_annotations_have_ground_truth():
    graphs, annotations = synthetic_annotated_graphs(n=20, seed=1)
    assert len(graphs) == len(annotations) == 20
    assert any(len(a) > 0 for a in annotations)  # some redundancy exists
    for g, ann in zip(graphs, annotations):
        ids = {n.id for n in g.nodes}
        assert ann <= ids  # labels only reference real nodes


def test_aggregate_counts():
    preds = [{1, 2}, {3}]
    gts = [{1, 4}, {3}]
    agg = aggregate(preds, gts)
    assert agg["tp"] == 2
    assert agg["fp"] == 1
    assert agg["fn"] == 1
    assert abs(agg["precision"] - 2 / 3) < 1e-9
    assert abs(agg["recall"] - 2 / 3) < 1e-9


def test_aggregate_empty_is_zero():
    agg = aggregate([set()], [set()])
    assert agg["f1"] == 0.0
    assert agg["precision"] == 0.0


def test_predict_redundant_returns_set():
    graphs, _ = synthetic_annotated_graphs(n=10, seed=2)
    analyzer = RedundancyAnalyzer(config=PruneConfig())
    pred = predict_redundant(analyzer, graphs[0])
    assert isinstance(pred, set)


def test_run_coordinate_descent_returns_best_params():
    graphs, annotations = synthetic_annotated_graphs(n=30, seed=3)
    report = run(graphs, annotations, iterations=1, metric="f1")
    assert set(report["best_params"].keys()) == set(PARAM_GRID.keys())
    assert "best_aggregate" in report
    assert "history" in report
    assert report["n_graphs"] == 30


def test_grid_params_exist_on_pruneconfig():
    names = {f.name for f in fields(PruneConfig)}
    assert set(PARAM_GRID.keys()) <= names
