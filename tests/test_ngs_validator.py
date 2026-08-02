"""Tests for NGS Validator — all six iron rules."""

import pytest
from rtsa.core.types import GraphNode, NodeType, ReasoningTraceGraph
from rtsa.core.ngs_validator import (
    NGSValidator, NGSRule, NGSViolation,
    NGSRobustnessTester, classify_failure_mode,
    FAILURE_MODE_TAXONOMY,
)


def _make_graph(trace_id="t", nodes=None, edges=None):
    return ReasoningTraceGraph(
        trace_id=trace_id, nodes=nodes or [], edges=edges or [],
    )


class TestNGSValidator:
    def test_atomicity_no_violations(self):
        g = _make_graph(nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        v = NGSValidator.check_atomicity(g)
        assert v == []

    def test_consecutive_transform_violation(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        violations = NGSValidator.check_no_consecutive_repeat(g)
        assert len(violations) == 1
        assert violations[0].rule == NGSRule.NO_CONSECUTIVE_REPEAT
        assert violations[0].severity == "error"
        assert 1 in violations[0].node_indices

    def test_consecutive_transform_non_adjacent_ok(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.VERIFY),
            GraphNode(id=3, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2), (2, 3)])
        violations = NGSValidator.check_no_consecutive_repeat(g)
        assert len(violations) == 0

    def test_reference_atomicity_always_empty(self):
        g = _make_graph(nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        assert NGSValidator.check_reference_atomicity(g) == []

    def test_verify_no_incoming_warning(self):
        nodes = [
            GraphNode(id=1, type=NodeType.VERIFY),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        violations = NGSValidator.check_verify_scope(g)
        assert len(violations) == 1
        assert violations[0].severity == "warning"

    def test_verify_with_incoming_ok(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.VERIFY),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        violations = NGSValidator.check_verify_scope(g)
        assert len(violations) == 0

    def test_branch_without_fork_warning(self):
        nodes = [
            GraphNode(id=1, type=NodeType.BRANCH),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        violations = NGSValidator.check_branch_encoding(g)
        assert len(violations) == 1

    def test_branch_with_fork_ok(self):
        nodes = [
            GraphNode(id=1, type=NodeType.BRANCH),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2), (1, 3)])
        violations = NGSValidator.check_branch_encoding(g)
        assert len(violations) == 0

    def test_ngs_compliance_empty(self):
        g = _make_graph()
        violations = NGSValidator.check_ngs_compliance(g)
        assert len(violations) == 1
        assert violations[0].rule == NGSRule.NGS_COMPLIANCE


class TestNGSValidatorFull:
    def test_clean_graph_passes(self):
        nodes = [
            GraphNode(id=1, type=NodeType.RETRIEVE),
            GraphNode(id=2, type=NodeType.TRANSFORM),
            GraphNode(id=3, type=NodeType.VERIFY),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2), (2, 3)])
        validator = NGSValidator()
        valid, violations = validator.validate(g)
        assert valid
        assert len(violations) == 0

    def test_consecutive_transform_fails_validation(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        validator = NGSValidator()
        valid, violations = validator.validate(g)
        assert not valid
        assert any(v.rule == NGSRule.NO_CONSECUTIVE_REPEAT for v in violations)

    def test_non_strict_passes_with_warnings(self):
        nodes = [
            GraphNode(id=1, type=NodeType.VERIFY),
        ]
        g = _make_graph(nodes=nodes, edges=[])
        validator = NGSValidator(strict=False)
        valid, violations = validator.validate(g)
        assert valid  # only warnings, no errors

    def test_validate_batch(self):
        g1 = _make_graph("t1", nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        g2 = _make_graph("t2", nodes=[GraphNode(id=1, type=NodeType.TRANSFORM), GraphNode(id=2, type=NodeType.TRANSFORM)], edges=[(1, 2)])
        validator = NGSValidator()
        results = validator.validate_batch([g1, g2])
        assert results["t1"][0] is True
        assert results["t2"][0] is False


class TestNGSRobustness:
    def test_all_variants_defined(self):
        variants = NGSRobustnessTester.VARIANTS
        assert "strict" in variants
        assert "standard" in variants
        assert "relaxed" in variants
        assert "ultra_relaxed" in variants

    def test_run_robustness_check(self):
        g = _make_graph(nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        tester = NGSRobustnessTester()
        results = tester.run_robustness_check([g])
        assert len(results) == 4
        assert results["strict"].pass_rate >= 0

    def test_stability_score_range(self):
        g = _make_graph(nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        tester = NGSRobustnessTester()
        tester.run_robustness_check([g])
        score = tester.compute_stability_score()
        assert 0.0 <= score <= 1.0

    def test_build_validator_for_variant(self):
        v = NGSRobustnessTester.build_validator_for_variant("strict")
        assert isinstance(v, NGSValidator)
        assert v.strict is True

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            NGSRobustnessTester.build_validator_for_variant("nonexistent")

    def test_variant_relaxation_actually_applies(self):
        """Variant configs must be wired into validation: strict rejects
        consecutive Transforms while relaxed allows them."""
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])

        strict_v = NGSRobustnessTester.build_validator_for_variant("strict")
        relaxed_v = NGSRobustnessTester.build_validator_for_variant("relaxed")

        valid_strict, _ = strict_v.validate(g)
        valid_relaxed, _ = relaxed_v.validate(g)

        assert not valid_strict
        assert valid_relaxed

    def test_ultra_relaxed_allows_empty_graph(self):
        """min_nodes_per_graph=0 must skip the R6 compliance check so the
        empty graph is no longer rejected."""
        g = _make_graph()
        ultra_v = NGSRobustnessTester.build_validator_for_variant("ultra_relaxed")
        valid, violations = ultra_v.validate(g)
        assert valid
        assert not any(v.rule == NGSRule.NGS_COMPLIANCE for v in violations)


class TestFailureModes:
    """B6: failure-mode taxonomy classification."""

    def test_taxonomy_has_all_modes(self):
        assert len(FAILURE_MODE_TAXONOMY) == 7
        assert "fragmented_step" in FAILURE_MODE_TAXONOMY
        assert "dependency_violation" in FAILURE_MODE_TAXONOMY
        assert "empty_graph" in FAILURE_MODE_TAXONOMY

    def test_classify_groups_every_violation(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        _, violations = NGSValidator().validate(g)
        grouped = classify_failure_mode(violations)
        total = sum(len(v) for v in grouped.values())
        assert total == len(violations)
        for v in violations:
            assert v.failure_mode  # every violation carries a mode

    def test_no_violations_empty_groups(self):
        g = _make_graph(nodes=[GraphNode(id=1, type=NodeType.RETRIEVE)])
        _, violations = NGSValidator().validate(g)
        assert classify_failure_mode(violations) == {}

    def test_violation_records_failure_mode(self):
        nodes = [
            GraphNode(id=1, type=NodeType.TRANSFORM),
            GraphNode(id=2, type=NodeType.TRANSFORM),
        ]
        g = _make_graph(nodes=nodes, edges=[(1, 2)])
        _, violations = NGSValidator().validate(g)
        assert violations[0].failure_mode in FAILURE_MODE_TAXONOMY
