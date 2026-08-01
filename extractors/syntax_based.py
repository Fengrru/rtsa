"""
Syntax-Based Extractor (SBE) — Extractor E2.

Uses dependency parsing + POS tagging to classify reasoning operations.
Requires spaCy with a transformer or large model for dependency parsing.
Falls back gracefully if spaCy is not available.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import re

from core.types import GraphNode, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    spacy = None  # type: ignore


# ---------------------------------------------------------------------------
# Syntax-to-NodeType mapping rules
# ---------------------------------------------------------------------------

# Root verbs → NodeType
ROOT_VERB_MAP: Dict[str, NodeType] = {
    # TRANSFORM: computation, manipulation, derivation
    "calculate": NodeType.TRANSFORM,
    "compute": NodeType.TRANSFORM,
    "derive": NodeType.TRANSFORM,
    "solve": NodeType.TRANSFORM,
    "simplify": NodeType.TRANSFORM,
    "substitute": NodeType.TRANSFORM,
    "evaluate": NodeType.TRANSFORM,
    "find": NodeType.TRANSFORM,
    "determine": NodeType.TRANSFORM,
    "integrate": NodeType.TRANSFORM,
    "differentiate": NodeType.TRANSFORM,
    "multiply": NodeType.TRANSFORM,
    "divide": NodeType.TRANSFORM,
    "add": NodeType.TRANSFORM,
    "subtract": NodeType.TRANSFORM,
    "factor": NodeType.TRANSFORM,
    "expand": NodeType.TRANSFORM,
    "rewrite": NodeType.TRANSFORM,
    "convert": NodeType.TRANSFORM,
    "express": NodeType.TRANSFORM,
    "use": NodeType.TRANSFORM,
    "perform": NodeType.TRANSFORM,
    "apply": NodeType.TRANSFORM,
    "obtain": NodeType.TRANSFORM,
    "get": NodeType.TRANSFORM,
    "plug": NodeType.TRANSFORM,  # "plug in"
    "set": NodeType.TRANSFORM,
    # VERIFY: checking, validation
    "check": NodeType.VERIFY,
    "verify": NodeType.VERIFY,
    "confirm": NodeType.VERIFY,
    "validate": NodeType.VERIFY,
    "ensure": NodeType.VERIFY,
    "prove": NodeType.VERIFY,
    "test": NodeType.VERIFY,
    "examine": NodeType.VERIFY,
    "show": NodeType.VERIFY,
    "demonstrate": NodeType.VERIFY,
    # COMPARE: comparison operations
    "compare": NodeType.COMPARE,
    "contrast": NodeType.COMPARE,
    "distinguish": NodeType.COMPARE,
    "weigh": NodeType.COMPARE,  # "weigh options"
    # RETRIEVE: recall, reference
    "recall": NodeType.RETRIEVE,
    "remember": NodeType.RETRIEVE,
    "refer": NodeType.RETRIEVE,
    "invoke": NodeType.RETRIEVE,
    "quote": NodeType.RETRIEVE,
    "state": NodeType.RETRIEVE,
    "note": NodeType.RETRIEVE,
    "define": NodeType.RETRIEVE,
    # BRANCH: conditional reasoning
    "suppose": NodeType.BRANCH,
    "assume": NodeType.BRANCH,
    "consider": NodeType.BRANCH,
    "hypothesize": NodeType.BRANCH,
    "posit": NodeType.BRANCH,
    # BACKTRACK: correction, retraction
    "reconsider": NodeType.BACKTRACK,
    "correct": NodeType.BACKTRACK,
    "revise": NodeType.BACKTRACK,
    "retract": NodeType.BACKTRACK,
    "revert": NodeType.BACKTRACK,
}


# Keywords indicating theorem/reference patterns → RETRIEVE
THEOREM_KEYWORDS = frozenset({
    "theorem", "law", "rule", "lemma", "definition", "formula",
    "principle", "axiom", "corollary", "property", "method",
})


class SyntaxBasedExtractor:
    """
    Extracts RTG using dependency parsing of each CoT sentence.

    Architecture (Section 3.3):
    1. Parse sentence to dependency tree
    2. Root = "calculate"/"compute"/"derive" → Transform
    3. Root = "check"/"verify"/"confirm" → Verify
    4. Conditional dependency (mark/if) → Branch
    5. Negation + correction (neg + advmod) → Backtrack
    6. Otherwise → Transform
    """

    def __init__(self, name: str = "sbe", spacy_model: str = "en_core_web_sm"):
        self.name = name
        self._nlp: Optional[object] = None

        if _SPACY_AVAILABLE and spacy is not None:
            try:
                self._nlp = spacy.load(spacy_model)
                logger.info(f"SBE loaded spaCy model: {spacy_model}")
            except Exception:
                logger.warning(
                    f"spaCy model '{spacy_model}' not found. "
                    f"Run: python -m spacy download {spacy_model}. "
                    f"SBE will fall back to keyword classification."
                )
                self._nlp = None
        else:
            logger.warning("spaCy not installed. SBE falls back to rule-based classification.")

    # ------------------------------------------------------------------
    # Sentence classification
    # ------------------------------------------------------------------

    def classify_sentence(self, sentence: str) -> NodeType:
        """
        Classify a single sentence using dependency parse analysis.
        Falls back to keyword matching if spaCy unavailable.
        """
        if self._nlp is None:
            return self._classify_keyword_fallback(sentence)

        doc = self._nlp(sentence.strip())

        if len(doc) == 0:
            return NodeType.TRANSFORM

        sentence_lower = sentence.strip().lower()
        first_token = doc[0]
        first_lemma = first_token.lemma_.lower()

        # 0a. "Wait" at sentence start → BACKTRACK (math CoT correction signal)
        if first_lemma == "wait":
            return NodeType.BACKTRACK

        # 0b. "Actually" at sentence start (adverb) → BACKTRACK (correction signal)
        if first_lemma == "actually" and first_token.pos_ == "ADV":
            return NodeType.BACKTRACK

        # 0c. Theorem reference: "By/From/Via/According to ... Theorem/Law/Rule" → RETRIEVE
        if (first_lemma in ("by", "via", "from", "according")
                or sentence_lower.startswith("according to")) and any(
            token.lemma_.lower() in THEOREM_KEYWORDS for token in doc
        ):
            return NodeType.RETRIEVE

        # 0d. "I need to reconsider/correct/revise" → BACKTRACK
        if sentence_lower.startswith("i need to") and any(
            token.lemma_.lower() in ("reconsider", "correct", "revise", "retract", "rethink")
            for token in doc
        ):
            return NodeType.BACKTRACK

        # 0e. "I see an error/mistake" → BACKTRACK
        if "i see an error" in sentence_lower or "i made a mistake" in sentence_lower:
            return NodeType.BACKTRACK

        # 0f. "Case N:" at sentence start → BRANCH (case-based reasoning)
        if first_lemma == "case" and len(doc) > 1 and doc[1].like_num:
            return NodeType.BRANCH

        # 0g. "The [theorem_keyword] is/are/was..." → RETRIEVE (fact/definition recall)
        if any(token.lemma_.lower() in THEOREM_KEYWORDS for token in doc):
            root = next((t for t in doc if t.dep_ == "ROOT"), None)
            if root is not None and root.lemma_.lower() in ("be",):
                return NodeType.RETRIEVE

        # 1. Check root verb
        root = next((token for token in doc if token.dep_ == "ROOT"), None)
        if root is not None:
            root_lemma = root.lemma_.lower()
            if root_lemma in ROOT_VERB_MAP:
                # Special: "use" → RETRIEVE if referencing theorem/formula
                if root_lemma == "use":
                    for child in root.children:
                        if child.dep_ in ("dobj", "attr", "pobj", "prep"):
                            child_text = child.text.lower()
                            subtree_text = " ".join(t.text.lower() for t in child.subtree)
                            if any(kw in child_text or kw in subtree_text for kw in THEOREM_KEYWORDS):
                                return NodeType.RETRIEVE
                    return NodeType.TRANSFORM
                return ROOT_VERB_MAP[root_lemma]

        # 1b. Check for imperative "let us/let's + verb" → Transform
        for token in doc:
            if token.lemma_.lower() in ("let",) and token.dep_ == "ROOT":
                # "Let us/Let's compute..." pattern
                for child in token.rights:
                    if child.dep_ in ("dobj", "nsubj", "xcomp", "ccomp"):
                        child_lemma = child.lemma_.lower()
                        # "Let me/us try a different/another..." → BACKTRACK
                        if child_lemma == "try" and any(
                            t.lemma_.lower() in ("different", "another", "other", "else", "new")
                            for t in doc
                        ):
                            return NodeType.BACKTRACK
                        for verb in ["calculate", "compute", "simplify", "derive",
                                      "determine", "find", "consider", "recall",
                                      "assume", "suppose", "check", "verify"]:
                            if child_lemma == verb:
                                base_map = ROOT_VERB_MAP.get(verb, NodeType.TRANSFORM)
                                return base_map
                return NodeType.TRANSFORM  # default for "let us" patterns

        # 2. Check for conditional dependencies (Branch)
        for token in doc:
            if token.dep_ in ("mark",) and token.lemma_.lower() in ("if", "when", "whether"):
                # Check if this is "check if" or "verify if" (VERIFY, not BRANCH)
                head = token.head
                if head is not None and head.lemma_.lower() in (
                    "check", "verify", "test", "see", "determine",
                    "examine", "evaluate", "confirm", "validate",
                ):
                    return NodeType.VERIFY
                return NodeType.BRANCH
            if token.lemma_.lower() in ("if", "suppose", "assume") and token.pos_ in ("SCONJ", "VERB"):
                return NodeType.BRANCH

        # 3. Check for Backtrack patterns
        has_negation = any(token.dep_ == "neg" for token in doc)
        has_correction = any(
            token.lemma_.lower() in ("actually", "instead", "rather", "however",
                                      "wait", "wrong", "mistake", "instead", "correct")
            for token in doc
        )
        has_neg_word = any(
            token.lemma_.lower() in ("no", "not", "never", "wrong", "incorrect")
            for token in doc
        )
        if (has_negation and has_correction) or (has_neg_word and has_correction):
            return NodeType.BACKTRACK

        # 3b. Standalone backtrack words (mid-sentence correction signals)
        if any(
            token.lemma_.lower() in ("wait", "actually", "oops", "mistake", "error")
            for token in doc
        ):
            return NodeType.BACKTRACK

        # 4. Check for retrieve patterns
        for token in doc:
            if token.lemma_.lower() in ("recall", "remember", "define", "state", "note"):
                if token.pos_ == "VERB":
                    return NodeType.RETRIEVE

        # 5. Check for compare patterns
        for token in doc:
            if token.lemma_.lower() in ("compare", "contrast"):
                return NodeType.COMPARE
        # Check for comparative adjectives
        for token in doc:
            if token.tag_ in ("JJR", "RBR"):  # comparative adjective/adverb
                return NodeType.COMPARE

        # 6. Check for verify patterns (beyond root verb)
        for token in doc:
            if token.lemma_.lower() in ("check", "verify", "confirm", "validate", "test"):
                if token.pos_ == "VERB":
                    return NodeType.VERIFY
        # 6b. Check for verify adjectives/results
        for token in doc:
            if token.lemma_.lower() in ("consistent",):
                return NodeType.VERIFY

        # 7. Default: Transform
        return NodeType.TRANSFORM

    @staticmethod
    def _classify_keyword_fallback(sentence: str) -> NodeType:
        """Lightweight keyword fallback when spaCy is unavailable."""
        from .rule_based import RuleBasedExtractor
        rbe = RuleBasedExtractor()
        return rbe.classify_sentence(sentence)

    # ------------------------------------------------------------------
    # Full extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _split_complex_sentence(sentence: str) -> List[str]:
        """Split a complex sentence into sub-operation segments.

        Handles implicit transitions within a sentence that indicate
        multiple reasoning operations (e.g., "Then/Now/Thus/Therefore").
        Delegates to GCP validator's transition pattern splitting.
        """
        from .gcp_validator import _split_multi_op_sentence
        return _split_multi_op_sentence(sentence)

    @staticmethod
    def _merge_with_segments(
        types: List[NodeType],
        segments: List[str],
    ) -> Tuple[List[NodeType], List[List[str]]]:
        """Merge consecutive Transform / Verify / Backtrack nodes while tracking
        which source segments each merged node covers.

        Preserves Transforms that belong to different conditional branches
        ("otherwise" / "else" / "case N" / "if") and collapses repeated
        Verify / Backtrack signals (NGS R4 / multi-backtrack), exactly like
        ``_merge_consecutive_transforms_smart`` + ``_merge_same_type`` but
        without losing segment-to-node alignment.
        """
        if not types:
            return [], []
        merged_types: List[NodeType] = []
        merged_segments: List[List[str]] = []
        for i, t in enumerate(types):
            seg = segments[i] if i < len(segments) else ""
            if merged_types:
                last_t = merged_types[-1]
                seg_lower = seg.lstrip().lower()
                same_transform = (
                    t == NodeType.TRANSFORM and last_t == NodeType.TRANSFORM
                )
                keep_transform = same_transform and seg_lower.startswith(
                    ("otherwise", "else", "case ", "if ", "when ", "whether ")
                )
                same_mergeable = (
                    t == last_t and t in (NodeType.VERIFY, NodeType.BACKTRACK)
                )
                if (same_transform and not keep_transform) or same_mergeable:
                    merged_segments[-1].append(seg)
                    continue
            merged_types.append(t)
            merged_segments.append([seg])
        return merged_types, merged_segments

    def extract(
        self, cot_text: str, trace_id: str = "",
        model: str = "", question_id: str = "", domain: str = "",
        **metadata,
    ) -> ReasoningTraceGraph:
        """
        Extract a ReasoningTraceGraph using syntax-based classification.
        Splits complex sentences into multiple operations, applies
        context-aware post-processing (branch refinement, transform merging).
        """
        from .rule_based import RuleBasedExtractor
        from .gcp_validator import _refine_branch_context

        # Use RBE's sentence splitting
        rbe = RuleBasedExtractor()
        sentences = rbe._split_sentences(cot_text)

        if not sentences:
            return ReasoningTraceGraph(
                trace_id=trace_id or "empty",
                extractor=self.name,
                nodes=[],
                edges=[],
                model=model, question_id=question_id, domain=domain,
                metadata=metadata,
            )

        # Split complex sentences into finer operation segments
        all_segments: List[str] = []
        for sent in sentences:
            segments = self._split_complex_sentence(sent)
            all_segments.extend(segments)

        # Classify each segment
        raw_types = [self.classify_sentence(s) for s in all_segments]

        # Context-aware branch refinement (types stay 1:1 with segments here)
        raw_types = _refine_branch_context(raw_types, all_segments)

        # Merge consecutive Transforms / Verify / Backtrack, keeping the
        # segment mapping so text and spans never shift out of alignment.
        merged_types, merged_segments = self._merge_with_segments(
            raw_types, all_segments
        )

        # Build nodes with improved span tracking
        nodes = []
        char_offset = 0
        for i, mtype in enumerate(merged_types):
            sents = merged_segments[i]
            text = " ".join(sents)
            first = sents[0]
            last = sents[-1]
            start = cot_text.find(first, char_offset)
            if start < 0:
                start = char_offset
            last_start = cot_text.find(last, start)
            end = (last_start + len(last)) if last_start >= 0 else start + len(text)
            char_offset = max(0, end)
            nodes.append(GraphNode(
                id=i + 1, type=mtype, span=(max(0, start), max(0, end)), text=text,
            ))

        # Build sequential edges
        edges = [(nodes[i].id, nodes[i + 1].id) for i in range(len(nodes) - 1)]

        # Add Branch forward edges (skip to next non-Transform for Branch nodes)
        for i, node in enumerate(nodes):
            if node.type == NodeType.BRANCH:
                targets = [n for n in nodes[i + 1:i + 3]]
                for t in targets:
                    edge = (node.id, t.id)
                    if edge not in edges:
                        edges.append(edge)

        return ReasoningTraceGraph(
            trace_id=trace_id or f"sbe_{hashlib.md5(cot_text.encode('utf-8')).hexdigest()[:12]}",
            extractor=self.name,
            model=model, question_id=question_id, domain=domain,
            nodes=nodes,
            edges=edges,
            metadata={
                "cot_length_tokens": len(cot_text.split()),
                "extraction_rate": 1.0,
                "n_sentences": len(sentences),
                "n_segments": len(all_segments),
                **metadata,
            },
        )
