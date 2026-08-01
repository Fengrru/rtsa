"""Tests for analysis.step_classifier (B7)."""

import numpy as np
import pytest

from core.types import GraphNode, NodeType, ReasoningTraceGraph
from analysis.step_classifier import (
    StepDiagnosis, StepFeatureExtractor, StepCorrectnessClassifier,
)


def _graph(types, trace_id="g"):
    nodes = [
        GraphNode(id=i + 1, type=t, text=f"s{i}")
        for i, t in enumerate(types)
    ]
    edges = [(i + 1, i + 2) for i in range(len(types) - 1)]
    return ReasoningTraceGraph(trace_id=trace_id, nodes=nodes, edges=edges)


class TestStepFeatureExtractor:
    def test_shape_and_names(self):
        g = _graph([NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY])
        node_ids, X, names = StepFeatureExtractor.extract(g)
        assert node_ids == [1, 2, 3]
        assert X.shape == (3, len(names))
        assert names == StepFeatureExtractor.FEATURE_NAMES
        # 11 structural features + one-hot per node type
        assert len(names) == 11 + len(list(NodeType))

    def test_empty_graph(self):
        g = ReasoningTraceGraph(trace_id="e", nodes=[], edges=[])
        node_ids, X, names = StepFeatureExtractor.extract(g)
        assert node_ids == []
        assert X.shape == (0, len(names))

    def test_position_ratio_ordered(self):
        g = _graph([NodeType.TRANSFORM] * 4)
        _, X, _ = StepFeatureExtractor.extract(g)
        ratios = X[:, 3]
        assert ratios[0] < ratios[-1]

    def test_depth_from_root(self):
        g = _graph([NodeType.TRANSFORM] * 4)
        _, X, _ = StepFeatureExtractor.extract(g)
        assert X[:, 2].tolist() == [1.0, 2.0, 3.0, 4.0]


class TestStepCorrectnessClassifier:
    def test_train_and_predict(self):
        graphs = [
            _graph([NodeType.TRANSFORM, NodeType.TRANSFORM]),
            _graph([NodeType.TRANSFORM, NodeType.VERIFY]),
        ]
        labels = [{1: False, 2: False}, {1: True, 2: True}]
        clf = StepCorrectnessClassifier().fit(graphs, labels)
        probs = clf.predict_proba_error(graphs[0])
        assert set(probs.keys()) == {1, 2}
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_requires_labels(self):
        clf = StepCorrectnessClassifier()
        with pytest.raises(ValueError):
            clf.fit([_graph([NodeType.TRANSFORM])], [{}])

    def test_analyze_returns_diagnoses(self):
        g = _graph([NodeType.TRANSFORM, NodeType.TRANSFORM, NodeType.VERIFY])
        clf = StepCorrectnessClassifier().fit(
            [g], [{1: False, 2: False, 3: True}],
        )
        diags = clf.analyze(g, threshold=0.5)
        assert len(diags) == 3
        assert all(isinstance(d, StepDiagnosis) for d in diags)
        assert all(isinstance(d.failure_modes, list) for d in diags)
        assert all(d.rule_violations >= 0 for d in diags)

    def test_feature_importance_nonempty(self):
        graphs = [
            _graph([NodeType.TRANSFORM] * 3),
            _graph([NodeType.VERIFY] * 3),
        ]
        labels = [
            {1: False, 2: False, 3: False},
            {1: True, 2: True, 3: True},
        ]
        clf = StepCorrectnessClassifier().fit(graphs, labels)
        imp = clf.feature_importance()
        assert len(imp) == len(StepFeatureExtractor.FEATURE_NAMES)

    def test_save_load_roundtrip(self, tmp_path):
        graphs = [
            _graph([NodeType.TRANSFORM] * 3),
            _graph([NodeType.VERIFY] * 3),
        ]
        labels = [
            {1: False, 2: False, 3: False},
            {1: True, 2: True, 3: True},
        ]
        clf = StepCorrectnessClassifier().fit(graphs, labels)
        path = str(tmp_path / "clf.pkl")
        clf.save(path)
        loaded = StepCorrectnessClassifier.load(path)
        assert loaded.predict_proba_error(graphs[0]) == clf.predict_proba_error(graphs[0])
