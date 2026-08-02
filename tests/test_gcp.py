"""Tests for GCP Validator — Granularity Calibration Protocol."""

import pytest
import numpy as np
from rtsa.core.types import NodeType
from rtsa.extractors.gcp_validator import (
    GCPValidator, compute_gcs, GCSSentence,
    GCS_CORPUS_FULL, GCPResult, make_gcp_adapter,
)
from rtsa.extractors.rule_based import RuleBasedExtractor


class TestComputeGCS:
    def test_perfect_match(self):
        pred = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY]
        gold = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY]
        score = compute_gcs(pred, gold)
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        pred = [NodeType.RETRIEVE, NodeType.TRANSFORM]
        gold = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY]
        score = compute_gcs(pred, gold)
        assert 0.0 < score < 1.0

    def test_no_match(self):
        pred = [NodeType.BACKTRACK]
        gold = [NodeType.RETRIEVE, NodeType.TRANSFORM]
        score = compute_gcs(pred, gold)
        assert score < 1.0

    def test_length_penalty(self):
        pred = [NodeType.RETRIEVE, NodeType.TRANSFORM, NodeType.VERIFY, NodeType.COMPARE]
        gold = [NodeType.RETRIEVE, NodeType.TRANSFORM]
        score = compute_gcs(pred, gold)
        assert score < 0.7  # significant penalty for length mismatch

    def test_empty_pred(self):
        score = compute_gcs([], [NodeType.RETRIEVE])
        assert score == 0.0

    def test_empty_both(self):
        score = compute_gcs([], [])
        assert score == 0.0


class TestGCS_CORPUS_FULL:
    def test_has_30_sentences(self):
        assert len(GCS_CORPUS_FULL) == 30

    def test_unique_ids(self):
        ids = [s.id for s in GCS_CORPUS_FULL]
        assert len(ids) == len(set(ids))

    def test_categories_coverage(self):
        cats = {s.category for s in GCS_CORPUS_FULL}
        assert "single_type" in cats
        assert "transition" in cats
        assert "complex" in cats
        assert "edge_case" in cats

    def test_all_gold_types_valid(self):
        for s in GCS_CORPUS_FULL:
            for t in s.gold_sequence:
                assert isinstance(t, NodeType)


class TestGCPValidator:
    @pytest.fixture
    def validator(self):
        return GCPValidator()

    def test_calibrate_rbe(self, validator):
        rbe = RuleBasedExtractor()
        wrapped = lambda s: [rbe.classify_sentence(s)]
        result = validator.calibrate_extractor(wrapped, "rbe")
        assert isinstance(result, GCPResult)
        assert result.extractor_name == "rbe"
        assert len(result.gcs_scores) == 30
        assert 0.0 <= result.mean_gcs <= 1.0
        assert result.bootstrap_ci[0] <= result.bootstrap_ci[1]

    def test_calibrate_all(self, validator):
        rbe = RuleBasedExtractor()
        wrapped = lambda s: [rbe.classify_sentence(s)]
        extractors = {
            "rbe": wrapped,
        }
        results = validator.calibrate_all(extractors)
        assert "rbe" in results
        assert isinstance(results["rbe"], GCPResult)

    def test_pass_thresholds_defined(self, validator):
        assert validator.PASS_MEAN_THRESHOLD == 0.80
        assert validator.PASS_MIN_THRESHOLD == 0.60
        assert validator.PASS_CI_LOWER_THRESHOLD == 0.70

    def test_get_passed_extractors(self, validator):
        rbe = RuleBasedExtractor()
        wrapped = lambda s: [rbe.classify_sentence(s)]
        results = validator.calibrate_all({"rbe": wrapped})
        passed = validator.get_passed_extractors(results)
        assert isinstance(passed, list)
        if passed:
            assert all(isinstance(n, str) for n in passed)

    def test_category_analysis(self, validator):
        rbe = RuleBasedExtractor()
        wrapped = lambda s: [rbe.classify_sentence(s)]
        result = validator.calibrate_extractor(wrapped, "rbe")
        cat_analysis = validator.category_analysis(result)
        assert len(cat_analysis) > 0
        for key in cat_analysis:
            assert key.startswith("category_")

    def test_calibrate_with_custom_corpus(self, validator):
        mini_corpus = [GCS_CORPUS_FULL[0], GCS_CORPUS_FULL[1]]
        validator = GCPValidator(corpus=mini_corpus)
        rbe = RuleBasedExtractor()
        wrapped = lambda s: [rbe.classify_sentence(s)]
        result = validator.calibrate_extractor(wrapped, "rbe")
        assert len(result.gcs_scores) == 2

    def test_extractor_exception_handled(self, validator):
        def failing_extractor(sentence):
            raise RuntimeError("simulated failure")
        result = validator.calibrate_extractor(failing_extractor, "fail")
        assert result.mean_gcs == 0.0
        assert "GCS-001" in result.failure_details[0] if result.failure_details else True


class TestMakeGcpAdapter:
    """Regression: implicit Transform insertion must keep ``segments`` aligned
    with ``types`` so branch-aware merging still works."""

    def test_verify_that_inserts_implicit_transform(self):
        rbe = RuleBasedExtractor()
        adapter = make_gcp_adapter(rbe.classify_sentence)
        types = adapter("Verify that the derivative of x squared is 2x.")
        assert types[0] == NodeType.TRANSFORM  # implicit Transform precedes Verify
        assert types[-1] == NodeType.VERIFY

    def test_otherwise_branches_preserved_after_insert(self):
        """After an implicit-Transform insert, an 'otherwise' Transform must
        NOT be collapsed with the previous Transform (GCS-003 semantics).
        Before the fix the segment list drifted and the smart merge fell
        back to merging ALL consecutive Transforms, losing the branch."""
        rbe = RuleBasedExtractor()
        adapter = make_gcp_adapter(rbe.classify_sentence)
        types = adapter("Verify that x equals 2; otherwise compute y; otherwise compute z")
        assert types == [NodeType.TRANSFORM, NodeType.VERIFY, NodeType.TRANSFORM, NodeType.TRANSFORM]

    def test_gcs003_branch_sequence(self):
        """GCS-003 gold [Branch, Transform, Transform] is preserved end-to-end."""
        rbe = RuleBasedExtractor()
        adapter = make_gcp_adapter(rbe.classify_sentence)
        types = adapter("If x > 0, then y = x + 1; otherwise y = x - 1.")
        assert types == [NodeType.BRANCH, NodeType.TRANSFORM, NodeType.TRANSFORM]
