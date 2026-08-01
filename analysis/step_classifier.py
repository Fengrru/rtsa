"""Step-level correctness diagnosis — black-box structural verifier.

Inspired by CRV ("Verifying Chain-of-Thought Reasoning via Its
Computational Graph", Meta FAIR, arXiv 2510.09312), which shows that
*structural* features of a reasoning step's graph neighbourhood carry a
strong signal of step correctness — but without CRV's requirement of
white-box model access.

``StepCorrectnessClassifier`` learns to predict per-node error probability
from purely structural features (degrees, depth, type mix, NGS-violation
flags, position in the trace). It is trained on step-level labels
(``{node_id: is_correct}``) such as the ones produced by
``experiments/annotate_steps.py`` (strong-LLM judge + manual spot-check).

The classifier complements the NGS rule-based validator: rules are
deterministic and explainable, while the classifier generalizes patterns
beyond the hand-written rules.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.types import GraphNode, NodeType, ReasoningTraceGraph
from core.ngs_validator import NGSValidator, classify_failure_mode

logger = logging.getLogger(__name__)

_NODE_TYPES = list(NodeType)


# ---------------------------------------------------------------------------
# Feature extraction (per node)
# ---------------------------------------------------------------------------

class StepFeatureExtractor:
    """Extract a fixed-size structural feature vector per graph node.

    Features are deliberately *structural* (no text embeddings): degrees,
    depth, type mix, NGS-violation flags, position. This keeps the verifier
    black-box and cheap, mirroring CRV's finding that graph statistics and
    topological features are the most predictive.
    """

    FEATURE_NAMES: List[str] = [
        "in_degree", "out_degree", "depth_from_root", "position_ratio",
        "text_len", "text_words", "verify_no_incoming", "branch_underfork",
        "consecutive_transform", "has_backtrack_edge", "branching_ratio",
        *[f"type_{t.value.lower()}" for t in _NODE_TYPES],
    ]

    @classmethod
    def extract(
        cls, graph: ReasoningTraceGraph
    ) -> Tuple[List[int], np.ndarray, List[str]]:
        """Return (node_ids, feature_matrix, feature_names)."""
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
        n = len(nodes_sorted)
        if n == 0:
            return [], np.zeros((0, len(cls.FEATURE_NAMES))), cls.FEATURE_NAMES

        node_ids = [nd.id for nd in nodes_sorted]
        id_to_idx = {nd.id: i for i, nd in enumerate(nodes_sorted)}
        out_edges: Dict[int, List[int]] = {}
        in_edges: Dict[int, List[int]] = {}
        for u, v in graph.edges:
            out_edges.setdefault(u, []).append(v)
            in_edges.setdefault(v, []).append(u)

        # Longest-path depth from the root (first node in id order).
        depth: Dict[int, int] = {}
        for nd in nodes_sorted:
            preds = in_edges.get(nd.id, [])
            depth[nd.id] = 1 + max(
                (depth[p] for p in preds if p in depth), default=0
            )

        type_one_hot = {
            t: np.zeros(len(_NODE_TYPES), dtype=float) for t in _NODE_TYPES
        }
        for t in _NODE_TYPES:
            type_one_hot[t][_NODE_TYPES.index(t)] = 1.0

        X = np.zeros((n, len(cls.FEATURE_NAMES)), dtype=float)
        for nd in nodes_sorted:
            i = id_to_idx[nd.id]
            outs = out_edges.get(nd.id, [])
            ins = in_edges.get(nd.id, [])
            row = X[i]
            row[0] = len(ins)
            row[1] = len(outs)
            row[2] = depth[nd.id]
            row[3] = i / max(n - 1, 1)
            row[4] = len(nd.text)
            row[5] = len(nd.text.split())
            row[6] = 1.0 if nd.type == NodeType.VERIFY and not ins else 0.0
            row[7] = 1.0 if nd.type == NodeType.BRANCH and len(outs) < 2 else 0.0
            # Consecutive Transform: a direct Transform neighbour either way.
            row[8] = 1.0 if (
                nd.type == NodeType.TRANSFORM
                and any(
                    nodes_sorted[id_to_idx[p]].type == NodeType.TRANSFORM
                    for p in ins if p in id_to_idx
                )
            ) or (
                nd.type == NodeType.TRANSFORM
                and any(
                    nodes_sorted[id_to_idx[s]].type == NodeType.TRANSFORM
                    for s in outs if s in id_to_idx
                )
            ) else 0.0
            row[9] = 1.0 if any(
                nodes_sorted[id_to_idx[p]].type == NodeType.BACKTRACK
                for p in ins if p in id_to_idx
            ) or any(
                nodes_sorted[id_to_idx[s]].type == NodeType.BACKTRACK
                for s in outs if s in id_to_idx
            ) else 0.0
            row[10] = len(outs) / max(n - 1, 1)
            row[11:] = type_one_hot[nd.type]

        return node_ids, X, cls.FEATURE_NAMES


# ---------------------------------------------------------------------------
# Diagnosis records
# ---------------------------------------------------------------------------

@dataclass
class StepDiagnosis:
    """Per-node diagnosis produced by ``StepCorrectnessClassifier.analyze``."""
    node_id: int
    prob_error: float
    is_error: bool
    failure_modes: List[str] = field(default_factory=list)
    rule_violations: int = 0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class StepCorrectnessClassifier:
    """Gradient-boosted structural verifier for reasoning steps.

    Args:
        model: optional sklearn classifier with ``fit``/``predict_proba``.
            Defaults to a GradientBoostingClassifier.
        threshold: default decision threshold for ``analyze``.
    """

    def __init__(self, model=None, threshold: float = 0.5):
        if model is None:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42,
            )
        self.model = model
        self.threshold = threshold
        self.feature_names_: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        graphs: Sequence[ReasoningTraceGraph],
        labels: Sequence[Dict[int, bool]],
    ) -> "StepCorrectnessClassifier":
        """Train on step-level labels.

        Args:
            graphs: trace graphs to train on.
            labels: per-graph dict ``{node_id: is_correct}`` (nodes without
                a label are skipped).
        """
        X: List[List[float]] = []
        y: List[int] = []
        for g, lab in zip(graphs, labels):
            node_ids, feats, names = StepFeatureExtractor.extract(g)
            for nid, row in zip(node_ids, feats):
                if nid in lab:
                    X.append(row.tolist())
                    y.append(0 if lab[nid] else 1)  # 1 = error
        if not X:
            raise ValueError(
                "No labelled nodes provided — run "
                "`experiments/annotate_steps.py` first or pass labels."
            )
        self.feature_names_ = names
        self.model.fit(np.asarray(X), np.asarray(y))
        logger.info(f"StepCorrectnessClassifier trained on {len(X)} labelled nodes")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba_error(self, graph: ReasoningTraceGraph) -> Dict[int, float]:
        """Return ``{node_id: P(error)}`` for every node."""
        if not graph.nodes:
            return {}
        node_ids, feats, _ = StepFeatureExtractor.extract(graph)
        probs = self.model.predict_proba(feats)
        # predict_proba columns are [P(ok), P(error)] after binary fit.
        error_col = 1 if probs.shape[1] > 1 else 0
        return {
            nid: float(probs[i, error_col])
            for i, nid in enumerate(node_ids)
        }

    def analyze(
        self,
        graph: ReasoningTraceGraph,
        threshold: Optional[float] = None,
    ) -> List[StepDiagnosis]:
        """Diagnose every node: error probability + NGS failure modes."""
        thr = self.threshold if threshold is None else threshold
        probs = self.predict_proba_error(graph)
        _, violations = NGSValidator().validate(graph)
        by_node: Dict[int, List[str]] = {}
        for v in violations:
            for nid in v.node_indices:
                by_node.setdefault(nid, []).append(v.failure_mode or v.rule.value)
        counts: Dict[int, int] = {}
        for v in violations:
            for nid in v.node_indices:
                counts[nid] = counts.get(nid, 0) + 1

        return [
            StepDiagnosis(
                node_id=nid,
                prob_error=p,
                is_error=p >= thr,
                failure_modes=sorted(set(by_node.get(nid, []))),
                rule_violations=counts.get(nid, 0),
            )
            for nid, p in sorted(probs.items())
        ]

    def feature_importance(self) -> Dict[str, float]:
        """Return feature-name -> importance mapping (sklearn models)."""
        if not hasattr(self.model, "feature_importances_"):
            return {}
        names = self.feature_names_ or StepFeatureExtractor.FEATURE_NAMES
        return {
            name: float(imp)
            for name, imp in zip(names, self.model.feature_importances_)
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Persist the classifier to *path* (pickle)."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "names": self.feature_names_}, f)
        return path

    @classmethod
    def load(cls, path: str) -> "StepCorrectnessClassifier":
        """Load a classifier saved with :meth:`save`."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(model=payload["model"])
        obj.feature_names_ = payload.get("names", [])
        return obj
