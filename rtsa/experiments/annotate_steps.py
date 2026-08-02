"""Step-level annotation generator (A1, companion to the B7 classifier).

Turns deterministic NGS rule violations into per-node correctness labels
that ``analysis.step_classifier.StepCorrectnessClassifier.fit`` consumes.
This makes the "labelling loop" explicit and reproducible: annotations are
derived from the rule set by default (zero manual cost), with a pluggable
``--judge`` hook reserved for strong-LLM or human judgments when available.

Output format (JSONL, one record per node):

    {"question_id", "node_id", "is_correct", "failure_modes",
     "rule_violations", "text"}

Example:
    python -m experiments.annotate_steps --traces-file data/raw_cots/gsm8k_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))  # for direct imports

from rtsa.core.ngs_validator import NGSValidator
from rtsa.extractors.rule_based import RuleBasedExtractor
from rtsa.utils.data_loader import load_cot_traces


def annotate_trace(graph, validator: NGSValidator) -> List[Dict]:
    """Produce per-node annotation records for one trace graph.

    A node is ``is_correct=False`` iff it participates in at least one NGS
    violation; the failure modes and violation count are attached so the
    classifier can be analysed per failure mode later.
    """
    violations = validator.validate(graph)[1]
    by_node: Dict[int, List[str]] = {}
    counts: Dict[int, int] = {}
    for v in violations:
        mode = v.failure_mode or v.rule.value
        for nid in v.node_indices:
            by_node.setdefault(nid, []).append(mode)
            counts[nid] = counts.get(nid, 0) + 1

    records = []
    for node in sorted(graph.nodes, key=lambda n: n.id):
        modes = by_node.get(node.id, [])
        records.append({
            "question_id": graph.trace_id,
            "node_id": node.id,
            "is_correct": not modes,
            "failure_modes": sorted(set(modes)),
            "rule_violations": counts.get(node.id, 0),
            "text": node.text,
        })
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-file", type=str,
                        default="data/raw_cots/gsm8k_50.jsonl")
    parser.add_argument("--max-traces", type=int, default=50)
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSONL path (default: "
                             "experiments/results/step_annotations.jsonl)")
    args = parser.parse_args(argv)

    traces = load_cot_traces(args.traces_file)[: args.max_traces]
    extractor = RuleBasedExtractor()
    validator = NGSValidator()

    records: List[Dict] = []
    for i, t in enumerate(traces):
        graph = extractor.extract(
            t.get("cot_text", ""),
            trace_id=t.get("question_id", f"t{i}"),
            answer=t.get("answer", ""),
        )
        records.extend(annotate_trace(graph, validator))

    counts = Counter(r["is_correct"] for r in records)
    print(f"Annotated {len(records)} nodes from {len(traces)} traces")
    print(f"  correct={counts[True]}  error={counts[False]}")

    out_path = Path(args.out) if args.out else (
        Path("experiments/results") / "step_annotations.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
