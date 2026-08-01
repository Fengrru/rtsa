# RTSA — Reasoning Trace Structure Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/Fengrru/rtsa/actions)

> **Extract, analyze, prune, fingerprint, and benchmark Chain-of-Thought reasoning as structured graphs.**

RTSA treats a model's chain-of-thought (CoT) not as a flat text string, but as a **directed acyclic graph (DAG)** of atomic reasoning operations (Retrieve, Transform, Verify, Branch, Backtrack, Compare). This structural perspective enables quantitative answers to questions that are hard to answer on raw text:

- **Is this trace redundant, and exactly where?** — region-level redundancy detection with executable pruning
- **Is this reasoning step correct?** — black-box structural verifier trained on graph features (inspired by CRV)
- **Does structure predict correctness?** — Spearman correlation experiments between graph metrics and answer accuracy
- **Which model wrote this?** — authorship fingerprinting from structural style
- **How similar are two traces?** — supervised (Robust-TSI) and unsupervised (WL-kernel) graph similarity

## Pipeline

```
raw CoT text (JSONL / HuggingFace datasets)
    │  extractors: RBE (rule) · SBE (syntax) · LLM · random baselines
    ▼
ReasoningTraceGraph (typed DAG: Retrieve/Transform/Verify/Branch/Backtrack/Compare)
    │
    ├──► validate    NGS structural rules + failure-mode taxonomy (Type I/II)
    ├──► analyze     graph metrics, motifs, TSI/JSD matrices, structure↔correctness
    ├──► prune       redundancy regions → pruned graph (DAG-preserving)
    ├──► classify    per-step error probability (GradientBoosting on 17 structural features)
    └──► benchmark   GCP · NGS pass rate · TSI · authorship fingerprint
```

Every stage is a Python module; the [unified entrypoint](#unified-experiment-entrypoint) wraps them behind one CLI with versioned result directories.

## Installation

```bash
pip install rtsa                          # or: pip install -e ".[dev]" from source
pip install "rtsa[dev]"                   # pytest, pytest-cov, pytest-xdist
pip install "rtsa[llm]"                   # openai / anthropic / transformers backends
```

**Requires Python >= 3.10.** Core stack: NumPy, SciPy, NetworkX, scikit-learn, Pydantic v2, Matplotlib, spaCy.

## Quick Start (demo)

```bash
# 1. Extract a reasoning graph from CoT text
echo "First, retrieve the value x=5. Then transform: x+1=6. Finally, verify: 6 is correct." > cot.txt
rtsa extract cot.txt --extractor rbe --output graph.json

# 2. Validate against NGS structural rules (with failure-mode taxonomy)
rtsa validate graph.json

# 3. Detect redundancy and prune (structural, or signal-enhanced)
rtsa prune graph.json --apply --output pruned.json
rtsa prune graph.json --apply --use-calibration --use-prm

# 4. Compare two reasoning traces (TSI / motif / WL-kernel)
rtsa compare graph_a.json graph_b.json

# 5. Fingerprint authorship
rtsa fingerprint enroll sample.json
rtsa fingerprint identify mystery.json
```

Or in Python:

```python
from extractors import RuleBasedExtractor
from analysis.prune import RedundancyAnalyzer, PruneConfig
from core.ngs_validator import NGSValidator, classify_failure_mode

text = "Retrieve x=3. Transform: x*2=6. Verify: 6 is even."
graph = RuleBasedExtractor().extract(text, trace_id="demo_001")

valid, violations = NGSValidator().validate(graph)
modes = classify_failure_mode(violations)          # {"overloaded_step": [...], ...}

report = RedundancyAnalyzer(config=PruneConfig()).analyze(graph, apply_pruning=True)
print(f"Regions: {len(report.redundancy_regions)} | "
      f"Savings: {report.total_estimated_savings} tok (range {report.savings_range()}) | "
      f"Integrity: {report.structural_integrity_score}")
pruned = report.pruned_graph
```

## Research Features

### A. Scientific rigor

| Feature | Where | What it does |
|---|---|---|
| Structure-correctness correlation | `experiments/correlation_analysis.py` | Spearman rho between 9 graph metrics and per-trace correctness (synthetic labels or `--labels-file`) |
| Effect size + confidence intervals | `core/robust_tsi.py` | `cohens_d` for group comparisons; `bootstrap_tsi_ci` for TSI uncertainty bands |
| Pruning savings error range | `analysis/prune.py` | `PruningReport.savings_range()` surfaces +/-40% heuristic band instead of false precision |
| Threshold calibration | `experiments/calibrate_thresholds.py` | Coordinate-descent grid scan of every `PruneConfig` threshold against redundancy annotations |
| Step-level annotation pipeline | `experiments/annotate_steps.py` | Deterministic NGS-violation labels consumable by the classifier |

### B. Method alignment (with the current research frontier)

| Feature | Where | Inspired by |
|---|---|---|
| Failure-mode taxonomy (Type I structural inefficiency / Type II dependency violation) | `core/ngs_validator.py` + `docs/failure_modes.md` | CoT2Graph |
| Semantic step clustering (merge chain segments into macro-steps) | `analysis/step_clustering.py` | LLM-MindMap (EMNLP 2025) |
| Step-level correctness classifier (17 structural features, no white-box access) | `analysis/step_classifier.py` | CRV (Meta FAIR, arXiv 2510.09312) |
| Domain-adaptive thresholds | `analysis/prune.py` `PruneConfig.domain_overrides` | CRV's finding that error signatures are domain-dependent |

### C. Engineering

| Feature | Where |
|---|---|
| HuggingFace `datasets` adapter (any CoT dataset, custom mapper, streaming) | `utils/hf_adapter.py` |
| Unified experiment entrypoint with versioned results + `manifest.json` (git hash, python, args) | `experiments/run.py` |
| GitHub Actions CI (Python 3.10-3.12 matrix) | `.github/workflows/ci.yml` |
| OTLP / Langfuse trace exporters (no-op fallback) | `utils/trace_exporters.py` |

## Related Work

RTSA sits at the intersection of three active lines of research:

| Work | Focus | RTSA counterpart |
|---|---|---|
| **LLM-MindMap** (EMNLP 2025) | Cluster CoT into semantic steps, then build graphs; exploration density / branching / convergence ratios predict performance | `analysis/step_clustering.py` + `experiments/correlation_analysis.py` |
| **CRV** (Meta FAIR, arXiv 2510.09312) | Verify reasoning steps from structural graph features (AUROC 70-92%); error signatures are domain-dependent | `analysis/step_classifier.py` + `PruneConfig.domain_overrides` |
| **CoT2Graph** | CoT-to-graph transformation with reasoning-path validation and failure modes | `core/ngs_validator.py` failure-mode taxonomy + `extractors/` |

A capability-by-capability matrix is maintained in [docs/comparison.md](docs/comparison.md).

## Documentation

- [docs/api.md](docs/api.md) — public API reference and module layout
- [docs/comparison.md](docs/comparison.md) — capability matrix vs. LLM-MindMap / CRV / CoT2Graph
- [docs/failure_modes.md](docs/failure_modes.md) — NGS failure-mode taxonomy
- [docs/README.md](docs/README.md) — documentation index
- [CHANGELOG.md](CHANGELOG.md) — version history (Keep a Changelog)

## Unified Experiment Entrypoint

```bash
python -m experiments.run extract   --dataset gsm8k --max-traces 50
python -m experiments.run analyze   --dataset gsm8k
python -m experiments.run prune     --dataset synthetic --n 50
python -m experiments.run calibrate --synthetic
python -m experiments.run annotate
python -m experiments.run all       --dataset gsm8k
```

Each run writes to `experiments/results/runs/<command>_<timestamp>/` with a `manifest.json` recording git commit, Python version, arguments and timestamps. Any dataset can be pulled from HuggingFace instead of the bundled corpus:

```bash
python -m experiments.run analyze --hf-dataset openai/gsm8k --hf-config main --split test
```

## CLI Reference

| Command | Description |
|---|---|
| `rtsa extract <file>` | Extract `ReasoningTraceGraph` from CoT text |
| `rtsa quick <text>` | Quick inline extraction |
| `rtsa visualize <file>` | Plot graph as DAG (PNG / interactive) |
| `rtsa compare <f1> <f2>` | TSI, motif, and WL-kernel comparison |
| `rtsa validate <file>` | Validate against NGS structural rules |
| `rtsa prune <file>` | Redundancy detection & CoT optimization |
| `rtsa benchmark` | Extractor reliability benchmarking (GCP + NGS + TSI) |
| `rtsa gcp` | Granularity Calibration Protocol |
| `rtsa fingerprint enroll/identify` | LLM authorship attribution |

## Signal-Enhanced Pruning

The pruning engine can be enhanced with two optional signals that refine redundancy confidence without adding new CLI commands:

- **Metacognitive Calibration** (`--use-calibration`): estimates per-step confidence from hedge words, numerical density, and length. Low-confidence steps are treated as higher redundancy risk.
- **PRM Signal** (`--use-prm`): estimates process-reward per step by measuring textual novelty vs. the previous step and pronoun coherence. Low-reward steps are boosted as redundant.

Both adapters automatically fall back to robust internal heuristics when external libraries are absent.

## End-to-End Pruning Results

```bash
python -m experiments.end_to_end_prune --dataset synthetic --n 50
python -m experiments.end_to_end_prune --dataset gsm8k --n 50
python -m experiments.end_to_end_prune --dataset mixed --n 50
```

| Condition | Node Compression | Token Savings | NGS Pass Rate |
|---|---|---|---|
| Structural (heuristic, synthetic) | ~12.5% | ~31 tokens/trace | 100% |
| Signal-enhanced (calib+PRM) | ~0.0% | ~0 tokens/trace | 100% |

Signal-enhanced pruning is deliberately **more conservative**: steps carrying substantive signals (numbers, formulas, forward progress) keep their confidence high. On naturally compact GSM8K traces, structural pruning self-limits to ~2% compression.

## Tests

```bash
python -m pytest tests/ -v     # or: pytest -q
```

## Citation

```bibtex
@software{rtsa2026,
  title={RTSA: Reasoning Trace Structure Analysis Toolkit},
  author={Fengrru},
  year={2026},
  url={https://github.com/Fengrru/rtsa}
}
```

## License

[MIT](LICENSE) — Fengrru

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request guidelines.
