"""
Rule-Based Extractor (RBE) — Extractor E1 (v3.2 merged).

Merged from v3.1 + v3.2:
- Weighted keyword scoring (strong=1.0, weak=0.5) from v3.1
- Negation detection from v3.1
- Chinese keyword support from v3.2
- NGS R2 inline merging from v3.2
- Full CalibrationReport (P/R/F1 + LaTeX) from v3.1
- Confidence scoring from v3.1
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.types import GraphNode, NodeType, ReasoningTraceGraph


# ---------------------------------------------------------------------------
# Helper: detect CJK-only keywords
# ---------------------------------------------------------------------------

_CJK = set(
    chr(cp) for cp in range(0x4E00, 0x9FFF + 1)
) | set(chr(cp) for cp in range(0x3000, 0x303F + 1))


def _is_cjk_only(s: str) -> bool:
    return all(c in _CJK or c in "。" for c in s)


# ---------------------------------------------------------------------------
# Keyword Rules (merged English + Chinese, with weights)
# ---------------------------------------------------------------------------

# Each entry: (NodeType, [(keyword, weight), ...])
# Weight: 1.0 = strong signal, 0.5 = weak/ambiguous
# NOTE: Single-word English verbs are auto-expanded with inflectional suffixes
# (-s, -ed, -ing, -es) during compilation. Phrases and CJK keywords are exact-matched.
KEYWORD_RULES: List[Tuple[NodeType, List[Tuple[str, float]]]] = [
    (NodeType.RETRIEVE, [
        # Multi-word / phrase patterns (exact match only)
        ("according to", 1.0), ("by theorem", 1.0), ("from the definition", 1.0),
        ("recall that", 1.0), ("we know that", 0.5), ("the rule", 0.5),
        ("property states", 0.5), ("by definition", 0.5),
        ("the theorem states", 0.5), ("as established", 0.5),
        # Single-word patterns (auto-inflected: -s, -ed, -ing, -es)
        ("formula", 1.0), ("theorem", 1.0), ("recall", 1.0),
        ("definition", 0.5), ("rule", 0.5),
        # Chinese
        ("根据", 1.0), ("由定义", 1.0), ("根据定理", 1.0), ("由公式", 1.0),
        ("我们知道", 1.0), ("由性质", 1.0), ("由引理", 1.0), ("根据性质", 1.0),
        ("回忆", 0.5),
    ]),
    (NodeType.TRANSFORM, [
        # Multi-word patterns
        ("plug in", 0.5), ("otherwise", 0.5),
        # Single-word patterns (auto-inflected)
        ("calculate", 1.0), ("compute", 1.0), ("simplify", 1.0),
        ("substitute", 1.0), ("derive", 1.0), ("solve", 0.5),
        ("evaluate", 0.5), ("apply", 0.5),
        # Chinese
        ("计算", 1.0), ("化简", 1.0), ("代入", 1.0), ("推导", 1.0),
        ("求解", 1.0), ("展开", 1.0), ("整理得", 1.0), ("可得", 1.0),
        ("解得", 1.0), ("得到", 1.0),
    ]),
    (NodeType.COMPARE, [
        # Multi-word patterns
        ("larger than", 1.0), ("smaller than", 1.0),
        ("greater than", 1.0), ("less than", 1.0), ("difference between", 1.0),
        ("same as", 0.5), ("similar to", 0.5), ("vs.", 0.5),
        # Single-word patterns (auto-inflected)
        ("compare", 1.0),
        # Chinese
        ("比较", 1.0), ("大于", 1.0), ("小于", 1.0), ("大于等于", 1.0),
        ("小于等于", 1.0), ("对比", 1.0), ("相比之下", 1.0), ("两者差异", 1.0),
        ("优于", 1.0), ("劣于", 1.0),
    ]),
    (NodeType.VERIFY, [
        # Multi-word patterns
        ("make sure", 0.5), ("let me check", 0.5),
        ("check if", 1.0), ("verify if", 1.0), ("test if", 0.5),
        # Single-word patterns (auto-inflected)
        ("check", 1.0), ("verify", 1.0), ("confirm", 1.0),
        ("test", 0.5), ("indeed", 0.5), ("consistent", 1.0),
        ("correct", 0.5), ("valid", 0.5), ("matches", 0.5),
        ("satisfies", 0.5), ("holds", 0.5),
        # Chinese
        ("验证", 1.0), ("检验", 1.0), ("确认", 1.0), ("检查", 1.0),
        ("符合", 1.0), ("匹配", 1.0), ("正确", 1.0), ("成立", 1.0),
        ("满足", 1.0), ("一致", 1.0),
    ]),
    (NodeType.BRANCH, [
        # Multi-word patterns
        ("in that case", 0.5), ("when then", 0.5),
        ("on the other hand", 0.5),
        # Single-word patterns (auto-inflected)
        ("if", 1.0), ("suppose", 1.0), ("assume", 1.0),
        ("case", 0.5), ("alternatively", 0.5),
        # Chinese
        ("如果", 1.0), ("假设", 1.0), ("假定", 1.0), ("则", 1.0),
        ("否则", 1.0), ("当", 1.0), ("若", 1.0),
        ("分类讨论", 1.0), ("考虑", 1.0), ("分情况", 1.0),
    ]),
    (NodeType.BACKTRACK, [
        # Multi-word patterns
        ("let me rethink", 0.5), ("scratch that", 0.5),
        ("let me re-evaluate", 0.5),
        ("i made a mistake", 0.5), ("that's wrong", 0.5),
        ("let me try", 0.5),
        # Single-word patterns (auto-inflected)
        ("wait", 1.0), ("actually", 1.0), ("correction", 1.0),
        ("instead", 1.0), ("reconsider", 1.0),
        ("oops", 1.0), ("mistake", 0.5), ("wrong", 0.5),
        # Chinese
        ("等等", 1.0), ("不对", 1.0), ("错误", 1.0), ("更正", 1.0),
        ("重新考虑", 1.0), ("我错了", 1.0), ("搞错了", 1.0),
        ("重新计算", 1.0), ("实际上", 1.0), ("或者", 0.5),
    ]),
]

# Negation patterns (v3.2 — handle inflected forms)
# Verbs in negation patterns use a broad word-stem pattern so that
# "not checking", "not verified", "no need to compare", etc. all match.
_NEG_VERB = r"(?:check(?:ing|s|ed)?|verify(?:ing|s|ied|fies)?|comput(?:e|ing|es|ed)|calculat(?:e|ing|es|ed)|compar(?:e|ing|es|ed))"
NEGATION_PATTERNS = [
    rf"\bnot\s+(?:\w+\s+){{0,2}}{_NEG_VERB}\b",
    rf"\bnot\s+need\s+to\s+{_NEG_VERB}\b",
    r"\bno\s+need\s+to\b",
    rf"\bwithout\s+{_NEG_VERB}\b",
]


def _is_negated(sentence: str) -> bool:
    """Check if a keyword match occurs in a negation context."""
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


def _make_inflection_pattern(kw: str) -> str:
    """Build regex matching a keyword with common English inflectional forms.

    For single-word English keywords, generates a pattern that also matches
    common suffixes (-s, -ed, -ing, -es). For verbs ending in 'e', also
    generates the -ing form (which drops the 'e'). For verbs ending in
    consonant+'y', generates -ies/-ied forms.
    """
    # Multi-word phrases, CJK: exact match only
    if " " in kw or _is_cjk_only(kw):
        return r"\b" + re.escape(kw) + r"\b"

    base = re.escape(kw)

    # Verbs ending in 'e'
    if kw.endswith("e") and len(kw) > 2 and kw[-2] not in "aeiou":
        stem = re.escape(kw[:-1])
        return rf"\b(?:{base}|{base}s|{base}d|{stem}ing)\b"

    # Verbs ending in consonant + 'y'
    if kw.endswith("y") and len(kw) > 2 and kw[-2] not in "aeiou":
        stem = re.escape(kw[:-1])
        return rf"\b(?:{base}|{base}s|{stem}ied|{stem}ies|{base}ing)\b"

    # Regular single-word verbs
    return rf"\b(?:{base}|{base}s|{base}ed|{base}ing|{base}es)\b"


# ---------------------------------------------------------------------------
# Classification Result
# ---------------------------------------------------------------------------

# Priority for tie-breaking: more specific operations win over generic Transform.
# When multiple types have equal scores, the highest priority type is chosen.
_TYPE_PRIORITY: Dict[NodeType, int] = {
    NodeType.BRANCH: 6,
    NodeType.BACKTRACK: 5,
    NodeType.COMPARE: 4,
    NodeType.VERIFY: 3,
    NodeType.RETRIEVE: 2,
    NodeType.TRANSFORM: 1,
}


@dataclass
class RBEClassification:
    """Result of classifying a single sentence."""
    sentence: str
    node_type: NodeType
    confidence: float  # 0–1
    matched_keywords: List[str] = field(default_factory=list)
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Rule-Based Extractor (merged)
# ---------------------------------------------------------------------------

class RuleBasedExtractor:
    """Deterministic keyword + heuristic CoT graph extractor.

    Strategy:
    1. Split CoT text into sentences (English + Chinese punctuation)
    2. Classify each sentence using weighted keyword matching
    3. Apply NGS R2 (merge consecutive Transforms)
    4. Build sequential + heuristic edges
    """

    def __init__(self, name: str = "rbe", confidence_threshold: float = 0.3):
        self.name = name
        self.confidence_threshold = confidence_threshold

        # Precompile regex patterns with inflection support
        self._compiled: List[Tuple[NodeType, List[Tuple[re.Pattern, float]]]] = []
        for ntype, rules in KEYWORD_RULES:
            patterns = []
            for kw, weight in rules:
                if _is_cjk_only(kw):
                    patterns.append((re.compile(re.escape(kw)), weight))
                else:
                    pattern_str = _make_inflection_pattern(kw)
                    patterns.append((
                        re.compile(pattern_str, re.IGNORECASE), weight
                    ))
            self._compiled.append((ntype, patterns))

    # ------------------------------------------------------------------
    # Sentence classification
    # ------------------------------------------------------------------

    def classify_sentence(self, sentence: str) -> NodeType:
        """Classify a single sentence using weighted keyword rules.

        Algorithm:
        1. Scan for negation — if negated, fallback.
        2. Match all keyword patterns, accumulating weighted scores.
        3. Select the action type with the highest score.
        4. Fallback: Transform (default) or Backtrack (if ?/uncertainty).
        """
        sentence_lower = sentence.lower().strip()

        # Check negation
        if _is_negated(sentence_lower):
            return NodeType.TRANSFORM

        # Score each action type
        scores: Dict[NodeType, float] = {nt: 0.0 for nt in NodeType}

        for ntype, patterns in self._compiled:
            for pat, weight in patterns:
                if pat.search(sentence_lower):
                    scores[ntype] += weight

        # Mathematical equation pattern (e.g., "n=pq", "x=5") → strong Transform signal
        if re.search(
            r'(?<![<!>=])\w+\s*=\s*\w+(?![=<>])',
            sentence_lower,
        ):
            scores[NodeType.TRANSFORM] += 0.5  # moderate boost, won't overpower keywords at 1.0

        # Find best match (prioritizing specific operations over generic Transform)
        best_type = max(scores, key=lambda k: (scores[k], _TYPE_PRIORITY.get(k, 0)))
        best_score = scores[best_type]

        if best_score >= self.confidence_threshold:
            return best_type

        # Fallback
        if re.search(
            r'\?|uncertain|not sure|maybe|perhaps|hmm|不确定|也许|可能|怀疑',
            sentence_lower,
        ):
            return NodeType.BACKTRACK
        return NodeType.TRANSFORM

    def classify_sentence_detailed(self, sentence: str) -> RBEClassification:
        """Classify with full metadata (confidence, matched keywords)."""
        sentence_lower = sentence.lower().strip()

        if _is_negated(sentence_lower):
            return RBEClassification(
                sentence=sentence, node_type=NodeType.TRANSFORM,
                confidence=0.3, matched_keywords=[], is_fallback=True,
            )

        scores: Dict[NodeType, float] = {nt: 0.0 for nt in NodeType}
        matched_all: Dict[NodeType, List[str]] = {nt: [] for nt in NodeType}

        for ntype, patterns in self._compiled:
            for pat, weight in patterns:
                if pat.search(sentence_lower):
                    scores[ntype] += weight
                    matched_all[ntype].append(pat.pattern)

        best_type = max(scores, key=lambda k: (scores[k], _TYPE_PRIORITY.get(k, 0)))
        best_score = scores[best_type]
        total_score = sum(scores.values())

        confidence = best_score / total_score if total_score > 0 else 0.0

        if best_score >= self.confidence_threshold and confidence >= 0.3:
            return RBEClassification(
                sentence=sentence, node_type=best_type,
                confidence=confidence, matched_keywords=matched_all[best_type],
                is_fallback=False,
            )

        # Fallback
        if re.search(
            r'\?|uncertain|not sure|maybe|perhaps|hmm|不确定|也许|可能|怀疑',
            sentence_lower,
        ):
            fallback_type = NodeType.BACKTRACK
        else:
            fallback_type = NodeType.TRANSFORM

        return RBEClassification(
            sentence=sentence, node_type=fallback_type,
            confidence=0.2, matched_keywords=[], is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Full extraction
    # ------------------------------------------------------------------

    def extract(
        self, cot_text: str, trace_id: str = "", **metadata,
    ) -> ReasoningTraceGraph:
        """Extract a ReasoningTraceGraph from CoT text."""
        sentences = self._split_sentences(cot_text)
        if not sentences:
            return ReasoningTraceGraph(
                trace_id=trace_id or "empty", extractor=self.name,
                nodes=[], edges=[], metadata=metadata,
            )

        raw_types = [self.classify_sentence(s) for s in sentences]
        merged_types = self._merge_consecutive_transforms(raw_types)

        # Build nodes with spans
        nodes = []
        char_offset = 0
        for i, mtype in enumerate(merged_types):
            # Find sentence span
            if i < len(sentences):
                sent = sentences[i]
                start = cot_text.find(sent, char_offset) if char_offset < len(cot_text) else char_offset
                end = start + len(sent) if start >= 0 else char_offset + len(sent)
                char_offset = max(0, end)
            else:
                start, end = 0, 0
            nodes.append(GraphNode(
                id=i + 1, type=mtype,
                span=(max(0, start), max(0, end)),
                text=sent,
            ))

        # Build edges: sequential chain + Branch forward edges
        edges = [(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]
        for i, node in enumerate(nodes):
            if node.type == NodeType.BRANCH:
                targets = [n for n in nodes[i + 1:i + 3]]
                for t in targets:
                    edge = (node.id, t.id)
                    if edge not in edges:
                        edges.append(edge)

        return ReasoningTraceGraph(
            trace_id=trace_id or f"rbe_{hash(cot_text) % 100000}",
            extractor=self.name, nodes=nodes, edges=edges,
            metadata={
                "cot_length_tokens": len(cot_text.split()),
                "extraction_rate": 1.0,
                "n_sentences": len(sentences),
                **metadata,
            },
        )

    # ------------------------------------------------------------------
    # Sentence splitting (supports English + Chinese)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences with bilingual support."""
        text = text.replace("\n\n", "\n").replace("\r\n", "\n").replace("\r", "\n")

        raw = re.split(
            r'(?<=[.!?。！？])\s*(?=[A-Z\u4e00-\u9fff])',
            text,
        )

        sentences = []
        for part in raw:
            sub_parts = part.strip().split("\n")
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    sentences.append(sp)
        return sentences

    # ------------------------------------------------------------------
    # NGS R2: Merge consecutive Transform nodes
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_consecutive_transforms(types: List[NodeType]) -> List[NodeType]:
        """NGS Rule 2: No consecutive Transform nodes."""
        if not types:
            return []
        merged = [types[0]]
        for t in types[1:]:
            if t == NodeType.TRANSFORM and merged[-1] == NodeType.TRANSFORM:
                continue
            merged.append(t)
        return merged


# ---------------------------------------------------------------------------
# Calibration System (from v3.1)
# ---------------------------------------------------------------------------

@dataclass
class PerClassMetrics:
    """Precision, recall, F1 for a single action type."""
    action_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class CalibrationReport:
    """Complete calibration report for an extractor against gold standard."""
    extractor_id: str
    n_samples: int
    per_class: Dict[str, PerClassMetrics] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    class_labels: List[str] = field(default_factory=list)
    class_imbalance_warning: bool = False
    unreliable_classes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Calibration Report: {self.extractor_id}",
            f"  Samples: {self.n_samples}",
            f"  Overall Accuracy: {self.overall_accuracy:.3f}",
            f"  Class Imbalance Warning: {self.class_imbalance_warning}",
            f"  Unreliable Classes (F1 < 0.5): {self.unreliable_classes}",
            "-" * 60,
            f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "-" * 60,
        ]
        for label in self.class_labels:
            m = self.per_class.get(label, PerClassMetrics(label))
            lines.append(f"  {label:<13} {m.precision:>10.3f} {m.recall:>10.3f} {m.f1:>10.3f}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[h]", r"\centering",
            r"\caption{Extractor Calibration: " + self.extractor_id + r"}",
            r"\label{tab:cal_" + self.extractor_id + r"}",
            r"\begin{tabular}{lccc}", r"\toprule",
            r"Action Type & Precision & Recall & F1 \\", r"\midrule",
        ]
        for label in self.class_labels:
            m = self.per_class.get(label, PerClassMetrics(label))
            lines.append(f"  {label} & {m.precision:.3f} & {m.recall:.3f} & {m.f1:.3f} \\\\")
        lines.extend([
            r"\midrule",
            rf"  Overall & — & — & {self.overall_accuracy:.3f} \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ])
        return "\n".join(lines)


def calibrate_extractor(
    extractor: RuleBasedExtractor,
    gold_sentences: List[str],
    gold_labels: List[str],
    extractor_id: str = "rbe",
) -> CalibrationReport:
    """Calibrate an RBE extractor against human-annotated gold standard.

    Computes per-class precision, recall, F1, and identifies unreliable
    action types (F1 < 0.5) that require caveats in analysis.
    """
    if len(gold_sentences) != len(gold_labels):
        raise ValueError(
            f"Length mismatch: {len(gold_sentences)} sentences vs {len(gold_labels)} labels"
        )

    all_labels = sorted(set(gold_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n_classes = len(all_labels)

    per_class = {lbl: PerClassMetrics(lbl) for lbl in all_labels}
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)

    correct = 0
    for sentence, gold_label in zip(gold_sentences, gold_labels):
        pred = extractor.classify_sentence_detailed(sentence)
        pred_label = pred.node_type.value
        gold_idx = label_to_idx[gold_label]

        if pred_label in label_to_idx:
            pred_idx = label_to_idx[pred_label]
            confusion[pred_idx, gold_idx] += 1

        if pred_label == gold_label:
            correct += 1
            if gold_label in per_class:
                per_class[gold_label].true_positives += 1
        else:
            if pred_label in per_class:
                per_class[pred_label].false_positives += 1
            if gold_label in per_class:
                per_class[gold_label].false_negatives += 1

    overall_accuracy = correct / len(gold_sentences) if gold_sentences else 0.0

    gold_counts = {lbl: gold_labels.count(lbl) for lbl in all_labels}
    max_class_ratio = max(gold_counts.values()) / len(gold_labels) if gold_labels else 0.0
    class_imbalance_warning = max_class_ratio > 0.5

    unreliable = [
        lbl for lbl in all_labels
        if per_class[lbl].f1 < 0.5 or gold_counts.get(lbl, 0) < 5
    ]

    return CalibrationReport(
        extractor_id=extractor_id,
        n_samples=len(gold_sentences),
        per_class=per_class,
        overall_accuracy=overall_accuracy,
        confusion_matrix=confusion,
        class_labels=all_labels,
        class_imbalance_warning=class_imbalance_warning,
        unreliable_classes=unreliable,
    )
