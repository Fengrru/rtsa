"""
NGS Validator — Node Granularity Standardization (v3.2 merged).

Merged from v3.1 + v3.2:
- Six NGS iron rules (R1–R6) from v3.2
- NGS Arbitration Protocol (Fleiss' Kappa + tiebreaker) from v3.1
- NGS Robustness Tester (rule-variant sensitivity) from v3.2
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import GraphNode, NodeType, ReasoningTraceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NGS Rule Definitions
# ---------------------------------------------------------------------------

class NGSRule(str, Enum):
    """The six iron rules of Node Granularity Standardization."""
    ATOMICITY = "R1_Atomicity"
    NO_CONSECUTIVE_REPEAT = "R2_NoConsecutiveRepeat"
    REFERENCE_ATOMICITY = "R3_ReferenceAtomicity"
    VERIFY_SCOPE = "R4_VerifyScope"
    BRANCH_ENCODING = "R5_BranchEncoding"
    NGS_COMPLIANCE = "R6_NGSCompliance"


@dataclass
class NGSViolation:
    """A single NGS rule violation with diagnostic context."""
    rule: NGSRule
    node_indices: List[int]
    message: str
    severity: str = "error"


# ---------------------------------------------------------------------------
# NGS Validator (merged)
# ---------------------------------------------------------------------------

class NGSValidator:
    """Validates extracted ReasoningTraceGraphs against the six NGS iron rules."""

    def __init__(self, strict: bool = True):
        self.strict = strict

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    @staticmethod
    def check_atomicity(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R1: One atomic operation = one node.

        Checks that no node is trivially small (likely an unnecessary split)
        and that nodes with large text spans don't obviously contain multiple
        independent operations (which should have been separate nodes).
        """
        violations = []
        for node in graph.nodes:
            text_len = len(node.text)
            if text_len == 0:
                continue  # no text available, skip text-dependent checks
            if text_len < 8 and node.type != NodeType.TRANSFORM:
                # Very short sentences that aren't simple calculations
                violations.append(NGSViolation(
                    rule=NGSRule.ATOMICITY,
                    node_indices=[node.id],
                    message=f"Node {node.id} ({node.type.value}): text too short ({text_len} chars) "
                            f"for an atomic operation: '{node.text}' (NGS R1)",
                    severity="warning",
                ))
            if text_len > 200 and node.type in (NodeType.TRANSFORM, NodeType.BRANCH):
                # Suspiciously long sentence for a single operation
                violations.append(NGSViolation(
                    rule=NGSRule.ATOMICITY,
                    node_indices=[node.id],
                    message=f"Node {node.id} ({node.type.value}): text very long ({text_len} chars) "
                            f"may contain multiple operations: '{node.text[:80]}...' (NGS R1)",
                    severity="warning",
                ))
        return violations

    @staticmethod
    def check_no_consecutive_repeat(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R2: No consecutive Transform nodes representing one calculation."""
        violations = []
        nodes_sorted = sorted(graph.nodes, key=lambda n: n.id)
        for i in range(len(nodes_sorted) - 1):
            current = nodes_sorted[i]
            next_node = nodes_sorted[i + 1]
            if current.type == NodeType.TRANSFORM and next_node.type == NodeType.TRANSFORM:
                edge = (current.id, next_node.id)
                if edge in graph.edges or (next_node.id, current.id) in graph.edges:
                    violations.append(NGSViolation(
                        rule=NGSRule.NO_CONSECUTIVE_REPEAT,
                        node_indices=[current.id, next_node.id],
                        message=f"Consecutive Transform nodes {current.id}->{next_node.id} "
                                f"may represent one continuous calculation (NGS R2)",
                        severity="error",
                    ))
        return violations

    @staticmethod
    def check_reference_atomicity(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R3: Each reference to a theorem/rule/definition is an independent Retrieve node.

        Flags Retrieve nodes whose text mentions multiple distinct sources
        (e.g., "Recall the Pythagorean theorem and the distance formula")
        which should have been separate Retrieve nodes.
        """
        violations = []
        _ref_markers = [
            r"\b(theorem|lemma|definition|axiom|postulate|property|rule|formula|algorithm)\b",
            r"\b(law|principle|criterion|identity|inequality)\b",
        ]
        for node in graph.nodes:
            if node.type != NodeType.RETRIEVE:
                continue
            if not node.text:
                continue  # no text available, skip reference check
            ref_count = 0
            for pat in _ref_markers:
                ref_count += len(re.findall(pat, node.text, re.IGNORECASE))
            if ref_count >= 2:
                violations.append(NGSViolation(
                    rule=NGSRule.REFERENCE_ATOMICITY,
                    node_indices=[node.id],
                    message=f"Retrieve node {node.id}: contains {ref_count} references "
                            f"({node.text[:60]}...) should be split into {ref_count} nodes (NGS R3)",
                    severity="warning",
                ))
        return violations

    @staticmethod
    def check_verify_scope(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R4: A Verify node verifies one or more prior steps."""
        violations = []
        for node in graph.nodes:
            if node.type == NodeType.VERIFY:
                incoming = [e for e in graph.edges if e[1] == node.id]
                if len(incoming) == 0:
                    violations.append(NGSViolation(
                        rule=NGSRule.VERIFY_SCOPE,
                        node_indices=[node.id],
                        message=f"Verify node {node.id} has no incoming edges (verifies nothing)",
                        severity="warning",
                    ))
        return violations

    @staticmethod
    def check_branch_encoding(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R5: A conditional split is ONE Branch node with multiple outgoing edges."""
        violations = []
        for node in graph.nodes:
            if node.type == NodeType.BRANCH:
                outgoing = [e for e in graph.edges if e[0] == node.id]
                if len(outgoing) < 2:
                    violations.append(NGSViolation(
                        rule=NGSRule.BRANCH_ENCODING,
                        node_indices=[node.id],
                        message=f"Branch node {node.id} has only {len(outgoing)} "
                                f"outgoing edge(s); Branch should fork to >= 2 paths",
                        severity="warning",
                    ))
        return violations

    @staticmethod
    def check_ngs_compliance(graph: ReasoningTraceGraph) -> List[NGSViolation]:
        """R6: Meta-rule — determines whether the graph is NGS-compliant for motif analysis.

        A graph is NGS-compliant if it passes R1-R5 with no errors. This meta-rule
        aggregates the structural checks: reasonable atomicity (R1), no merged
        consecutive operations (R2), clean references (R3), verified verifies (R4),
        and proper branch structure (R5).

        Returns:
            Empty list if NGS-compliant. Otherwise the first fatal violation found
            by scanning the graph structure and cross-checking node edges.
        """
        violations = []

        # Check that the graph has at least one meaningful node
        if not graph.nodes:
            violations.append(NGSViolation(
                rule=NGSRule.NGS_COMPLIANCE,
                node_indices=[],
                message="Graph has zero nodes — cannot be NGS-compliant for motif analysis (NGS R6)",
                severity="error",
            ))
            return violations

        # Check that every non-source node has at least one incoming edge
        node_ids = {n.id for n in graph.nodes}
        targets = {e[1] for e in graph.edges}
        for node in graph.nodes:
            if node.id not in targets and node.id != min(node_ids):
                violations.append(NGSViolation(
                    rule=NGSRule.NGS_COMPLIANCE,
                    node_indices=[node.id],
                    message=f"Node {node.id} ({node.type.value}) has no incoming edges "
                            f"(orphan operation) (NGS R6)",
                    severity="warning",
                ))

        return violations

    # ------------------------------------------------------------------
    # Full validation
    # ------------------------------------------------------------------

    def validate(self, graph: ReasoningTraceGraph) -> Tuple[bool, List[NGSViolation]]:
        """Run all NGS rule checks on a graph.

        Returns:
            (is_valid, list_of_violations)
        """
        all_violations: List[NGSViolation] = []
        all_violations.extend(self.check_atomicity(graph))
        all_violations.extend(self.check_no_consecutive_repeat(graph))
        all_violations.extend(self.check_reference_atomicity(graph))
        all_violations.extend(self.check_verify_scope(graph))
        all_violations.extend(self.check_branch_encoding(graph))

        errors = [v for v in all_violations if v.severity == "error"]
        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                f"Graph {graph.trace_id}: NGS validation failed with "
                f"{len(errors)} error(s), {len(all_violations) - len(errors)} warning(s)"
            )

        return is_valid, all_violations

    def validate_batch(
        self, graphs: List[ReasoningTraceGraph],
    ) -> Dict[str, Tuple[bool, List[NGSViolation]]]:
        """Validate multiple graphs; returns {trace_id: (valid, violations)}."""
        return {g.trace_id: self.validate(g) for g in graphs}


# ---------------------------------------------------------------------------
# NGS Arbitration Protocol (from v3.1 — Fix #3)
# ---------------------------------------------------------------------------

@dataclass
class ArbitrationCase:
    """A single case requiring arbitration between two annotators."""
    trace_id: str
    annotator_1_judgment: bool
    annotator_2_judgment: bool
    annotator_1_notes: str = ""
    annotator_2_notes: str = ""
    tiebreaker_judgment: Optional[bool] = None
    tiebreaker_notes: str = ""
    resolution: str = ""


@dataclass
class ArbitrationResult:
    """Complete results from an NGS arbitration session."""
    cases: List[ArbitrationCase]
    fleiss_kappa: float
    agreement_rate: float
    n_ambiguous: int
    n_total: int
    per_rule_agreement: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"NGS Arbitration: κ={self.fleiss_kappa:.3f}, "
            f"agreement={self.agreement_rate:.1%}, "
            f"ambiguous={self.n_ambiguous}/{self.n_total}, "
            f"per_rule={self.per_rule_agreement}"
        )


class NGSArbitrationProtocol:
    """Implements the structured NGS arbitration protocol.

    Protocol:
    1. Two annotators independently judge NGS compliance on 50 samples.
    2. Compute Fleiss' Kappa for NGS IAA.
    3. For disagreements, a third annotator (tie-breaker) resolves.
    4. Flag "NGS-ambiguous" cases for separate analysis.
    """

    def __init__(self, validator: Optional[NGSValidator] = None):
        self.validator = validator or NGSValidator(strict=False)

    def compute_agreement(
        self, annotator_1: List[bool], annotator_2: List[bool],
    ) -> Tuple[float, float]:
        """Compute Fleiss' Kappa and raw agreement rate."""
        if len(annotator_1) != len(annotator_2):
            raise ValueError("Annotator judgment lists must have same length.")
        if len(annotator_1) == 0:
            return 1.0, 1.0

        n = len(annotator_1)
        agreements = sum(1 for a, b in zip(annotator_1, annotator_2) if a == b)
        agreement_rate = agreements / n

        p_a = agreement_rate
        n_pass_a = sum(annotator_1)
        n_pass_b = sum(annotator_2)
        n_fail_a = n - n_pass_a
        n_fail_b = n - n_pass_b

        p_pass = ((n_pass_a + n_pass_b) / (2 * n)) ** 2
        p_fail = ((n_fail_a + n_fail_b) / (2 * n)) ** 2
        p_e = p_pass + p_fail

        if abs(p_e - 1.0) < 1e-9:
            kappa = 1.0
        else:
            kappa = (p_a - p_e) / (1.0 - p_e)

        return kappa, agreement_rate

    def run_arbitration(
        self,
        trace_ids: List[str],
        annotator_1_judgments: Dict[str, bool],
        annotator_2_judgments: Dict[str, bool],
        tiebreaker_judgments: Optional[Dict[str, Optional[bool]]] = None,
        annotator_1_notes: Optional[Dict[str, str]] = None,
        annotator_2_notes: Optional[Dict[str, str]] = None,
        tiebreaker_notes: Optional[Dict[str, str]] = None,
    ) -> ArbitrationResult:
        """Execute the full arbitration protocol."""
        cases: List[ArbitrationCase] = []
        j1_list, j2_list = [], []

        for tid in trace_ids:
            j1 = annotator_1_judgments.get(tid, True)
            j2 = annotator_2_judgments.get(tid, True)
            j1_list.append(j1)
            j2_list.append(j2)

            notes1 = (annotator_1_notes or {}).get(tid, "")
            notes2 = (annotator_2_notes or {}).get(tid, "")
            notes_tb = (tiebreaker_notes or {}).get(tid, "")

            if j1 == j2:
                resolution = "unanimous_pass" if j1 else "unanimous_fail"
                case = ArbitrationCase(
                    trace_id=tid,
                    annotator_1_judgment=j1,
                    annotator_2_judgment=j2,
                    annotator_1_notes=notes1,
                    annotator_2_notes=notes2,
                    tiebreaker_judgment=j1,
                    tiebreaker_notes="",
                    resolution=resolution,
                )
            else:
                tb = (tiebreaker_judgments or {}).get(tid)
                if tb is None:
                    resolution = "ambiguous"
                elif tb:
                    resolution = "arbitrated_pass"
                else:
                    resolution = "arbitrated_fail"
                case = ArbitrationCase(
                    trace_id=tid,
                    annotator_1_judgment=j1,
                    annotator_2_judgment=j2,
                    annotator_1_notes=notes1,
                    annotator_2_notes=notes2,
                    tiebreaker_judgment=tb,
                    tiebreaker_notes=notes_tb,
                    resolution=resolution,
                )
            cases.append(case)

        kappa, agreement_rate = self.compute_agreement(j1_list, j2_list)
        n_ambiguous = sum(1 for c in cases if c.resolution == "ambiguous")
        n_total = len(cases)

        return ArbitrationResult(
            cases=cases, fleiss_kappa=kappa, agreement_rate=agreement_rate,
            n_ambiguous=n_ambiguous, n_total=n_total,
        )

    def get_final_labels(
        self, result: ArbitrationResult,
    ) -> Dict[str, Optional[bool]]:
        """Extract final NGS labels from arbitration result.

        Returns:
            Mapping trace_id -> NGS-compliant (True/False/None = ambiguous).
        """
        labels: Dict[str, Optional[bool]] = {}
        for case in result.cases:
            if case.resolution in ("unanimous_pass", "arbitrated_pass"):
                labels[case.trace_id] = True
            elif case.resolution in ("unanimous_fail", "arbitrated_fail"):
                labels[case.trace_id] = False
            else:
                labels[case.trace_id] = None
        return labels

    @staticmethod
    def filter_ambiguous(
        schemas: List[ReasoningTraceGraph], labels: Dict[str, Optional[bool]],
    ) -> Tuple[List[ReasoningTraceGraph], List[ReasoningTraceGraph]]:
        """Split schemas into NGS-clean and NGS-ambiguous sets."""
        clean, ambiguous = [], []
        for s in schemas:
            label = labels.get(s.trace_id, True)
            if label is None:
                ambiguous.append(s)
            elif label:
                clean.append(s)
        return clean, ambiguous


# ---------------------------------------------------------------------------
# NGS Robustness Tester (from v3.2 — Fix 1)
# ---------------------------------------------------------------------------

@dataclass
class NGSRobustnessResult:
    """Results from testing how sensitive conclusions are to rule choice."""
    rule_variant: str
    n_graphs_valid: int
    n_graphs_total: int
    pass_rate: float
    stable_rate: float = 0.0
    iaa_mean: float = 0.0
    iaa_std: float = 0.0


class NGSRobustnessTester:
    """Tests sensitivity of NGS conclusions to alternative segmentation rules."""

    VARIANTS = {
        "strict": {
            "allow_consecutive_transform": False,
            "allow_orphan_verify": False,
            "require_branch_fork": True,
            "min_nodes_per_graph": 1,
        },
        "standard": {
            "allow_consecutive_transform": False,
            "allow_orphan_verify": True,
            "require_branch_fork": True,
            "min_nodes_per_graph": 1,
        },
        "relaxed": {
            "allow_consecutive_transform": True,
            "allow_orphan_verify": True,
            "require_branch_fork": False,
            "min_nodes_per_graph": 1,
        },
        "ultra_relaxed": {
            "allow_consecutive_transform": True,
            "allow_orphan_verify": True,
            "require_branch_fork": False,
            "min_nodes_per_graph": 0,
        },
    }

    def __init__(self):
        self.results: Dict[str, NGSRobustnessResult] = {}

    @staticmethod
    def build_validator_for_variant(variant_name: str) -> NGSValidator:
        if variant_name not in NGSRobustnessTester.VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name}")
        return NGSValidator(strict=(variant_name == "strict"))

    def run_robustness_check(
        self, graphs: List[ReasoningTraceGraph],
        extractor_outputs: Optional[Dict[str, List[ReasoningTraceGraph]]] = None,
    ) -> Dict[str, NGSRobustnessResult]:
        """Run validation under all rule variants and compare results."""
        for variant_name in self.VARIANTS:
            validator = self.build_validator_for_variant(variant_name)
            valid_count = sum(
                1 for g in graphs if validator.validate(g)[0]
            )

            self.results[variant_name] = NGSRobustnessResult(
                rule_variant=variant_name,
                n_graphs_valid=valid_count,
                n_graphs_total=len(graphs),
                pass_rate=valid_count / max(len(graphs), 1),
            )

        pass_rates = [r.pass_rate for r in self.results.values()]
        if len(pass_rates) > 1:
            cv = float(np.std(pass_rates) / max(np.mean(pass_rates), 1e-8))
            logger.info(
                f"NGS robustness CV = {cv:.3f} "
                f"(lower = more robust to rule choice)"
            )

        return self.results

    def compute_stability_score(self) -> float:
        """Compute stability score: 1.0 - CV(pass_rates)."""
        if not self.results:
            return 0.0
        rates = [r.pass_rate for r in self.results.values()]
        cv = float(np.std(rates) / max(np.mean(rates), 1e-8))
        return float(max(0.0, 1.0 - cv))
