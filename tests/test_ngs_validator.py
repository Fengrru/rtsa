"""Tests for NGS Validator — all six iron rules."""

import pytest
from core.types import GraphNode, NodeType, ReasoningTraceGraph
from core.ngs_validator import (
    NGSValidator, NGSRule, NGSViolation,
    NGSRobustnessTester,
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
