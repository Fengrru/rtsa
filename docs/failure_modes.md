# Failure-Mode Taxonomy

RTSA maps the six NGS iron rules (R1-R6) onto an actionable failure-mode
taxonomy inspired by CoT2Graph ("Understanding Failures in LLM Reasoning by
Learning Reasoning Graphs from Chain-of-Thought", OpenReview 0XfuJjhaI5).

Every `NGSViolation` carries a `failure_mode` field; group violations with
`core.ngs_validator.classify_failure_mode()` to consume them by mode.

## Two top-level categories

- **Type I — Structural inefficiency**: the trace wastes steps (overthinking,
  over-splitting, merged operations). These are pruning targets: `analysis/prune.py`
  converts them into `RedundancyRegion`s with `suggested_action="merge"`.
- **Type II — Graph dependency violation**: the causal structure of the trace is
  broken (orphan steps, missing links). These are *correctness* signals: they
  indicate the extraction is unreliable or the reasoning itself is disconnected.

## Mode table

| failure_mode           | NGS rule | Type  | Meaning | Downstream action |
|------------------------|----------|-------|---------|-------------------|
| `fragmented_step`      | R1       | I     | One atomic operation split into trivially small nodes | Merge candidates for `prune` |
| `overloaded_step`      | R1       | I     | One node packs multiple operations | Split recommendation, extraction QC |
| `merged_computation`   | R2       | I     | Consecutive Transforms = one continuous calculation | Merge candidates for `prune` |
| `overloaded_retrieve`  | R3       | I     | One Retrieve references multiple distinct sources | Split recommendation, extraction QC |
| `pseudo_branch`        | R5       | I     | Branch node that never forks into >= 2 paths | Merge candidates for `prune` |
| `dependency_violation` | R4 / R6  | II    | Verify/orphan node with no incoming causal link | Extraction QC, step-level verifier signal |
| `empty_graph`          | R6       | II    | Trace produced zero nodes | Drop from analysis, extraction failure |

## Mapping to CoT2Graph findings

CoT2Graph reports that verification loops start appearing once traces reach
35-40 steps, and structural efficiency decreases roughly linearly with trace
length. The Type I modes above are the graph-level manifestations of that
finding: `merged_computation` and `pseudo_branch` dominate long, oververbose
traces, which is why `PruneConfig` exposes per-domain thresholds for them
(see `analysis/prune.py` `domain_overrides`).

## Usage

```python
from core.ngs_validator import NGSValidator, classify_failure_mode

validator = NGSValidator()
is_valid, violations = validator.validate(graph)
by_mode = classify_failure_mode(violations)
for mode, items in by_mode.items():
    print(mode, len(items))
```
