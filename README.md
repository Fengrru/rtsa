# RTSA — Reasoning Trace Structure Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

> **Extract, analyze, prune, fingerprint, and benchmark Chain-of-Thought reasoning as structured graphs.**

RTSA treats a model's chain-of-thought (CoT) not as a flat text string, but as a **directed acyclic graph (DAG)** of atomic reasoning operations (Retrieve, Transform, Verify, Branch, Backtrack, Compare). This structural perspective enables:

- **Redundancy detection** — identify and prune wasteful reasoning steps
- **Quality benchmarking** — compare extractors and models on structural fidelity
- **Authorship fingerprinting** — attribute reasoning style to specific LLMs
- **Cross-trace comparison** — measure semantic similarity via graph kernels

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Architecture](#architecture)
- [Signal-Enhanced Pruning](#signal-enhanced-pruning)
- [End-to-End Experiments](#end-to-end-experiments)
- [External Project Integrations](#external-project-integrations)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
# Basic install
pip install rtsa

# With development dependencies
pip install "rtsa[dev]"

# With LLM-backed extractors
pip install "rtsa[llm]"

# Editable install from source
git clone https://github.com/Fengrru/rtsa.git
cd rtsa
pip install -e ".[dev]"
```

**Dependencies**: Python >= 3.10, NumPy, SciPy, NetworkX, scikit-learn, Pydantic v2, Matplotlib, spaCy.

## Quick Start

### 1. Extract a reasoning graph from CoT text

```bash
# From a text file
echo "First, retrieve the value x=5. Then transform: x+1=6. Finally, verify: 6 is correct." > cot.txt
rtsa extract cot.txt --extractor rbe --output graph.json

# Inline quick extraction
rtsa quick "Retrieve A. Transform A->B. Verify B."
```

### 2. Visualize the reasoning graph

```bash
rtsa visualize graph.json --output graph.png
```

### 3. Detect redundancy and prune

```bash
# Structural heuristics only
rtsa prune graph.json --apply --output pruned.json

# With signal enhancement (recommended)
rtsa prune graph.json --apply --use-calibration --use-prm
```

### 4. Compare two reasoning traces

```bash
rtsa compare graph_a.json graph_b.json
```

### 5. Validate against NGS structural rules

```bash
rtsa validate graph.json
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `rtsa extract <file>` | Extract `ReasoningTraceGraph` from CoT text |
| `rtsa quick <text>` | Quick inline extraction |
| `rtsa visualize <file>` | Plot graph as DAG (PNG / interactive) |
| `rtsa compare <f1> <f2>` | TSI, motif, and WL-kernel comparison |
| `rtsa validate <file>` | Validate against NGS structural rules |
| `rtsa prune <file>` | Redundancy detection & CoT optimization |
| `rtsa benchmark` | Extractor reliability benchmarking (GCP + NGS + TSI) |
| `rtsa gcp` | Granularity Calibration Protocol |
| `rtsa fingerprint enroll/identify` | LLM authorship attribution |

## Python API

```python
from extractors import RuleBasedExtractor
from analysis.prune import RedundancyAnalyzer, PruneConfig
from core.types import ReasoningTraceGraph

# Extract
text = "Retrieve x=3. Transform: x*2=6. Verify: 6 is even."
extractor = RuleBasedExtractor()
graph = extractor.extract(text, trace_id="demo_001")

# Analyze
analyzer = RedundancyAnalyzer(config=PruneConfig())
report = analyzer.analyze(graph, apply_pruning=True)

print(f"Regions found: {len(report.redundancy_regions)}")
print(f"Token savings: {report.total_estimated_savings}")
print(f"Integrity: {report.structural_integrity_score}")

# Access pruned graph
pruned = report.pruned_graph
```

## Architecture

```
rtsa/
├── core/              # Graph types, NGS validator, TSI, WL kernel, fingerprint
├── extractors/        # Rule-based, Syntax-based, LLM-based, Random baselines
├── analysis/          # Pruning engine, benchmark suite, signal adapters
├── experiments/       # End-to-end utility experiments, synthetic data generators
├── utils/             # GSM8K loader, visualization helpers
├── data/              # Sample CoT traces and extracted graphs
├── tests/             # pytest suite (14 test modules)
└── cli.py             # Click-based command-line interface
```

## Signal-Enhanced Pruning

The pruning engine can be enhanced with two optional signals that refine redundancy confidence without adding new CLI commands:

```bash
rtsa prune graph.json --apply --use-calibration --use-prm
```

- **Metacognitive Calibration** (`--use-calibration`): Estimates per-step confidence from hedge words, numerical density, and length. Low-confidence steps are treated as higher redundancy risk.
- **PRM Signal** (`--use-prm`): Estimates process-reward for each step by measuring textual novelty vs. the previous step and pronoun coherence. Low-reward steps are boosted as redundant.

Both adapters automatically fall back to robust internal heuristics if the external libraries are not installed, so the enhancement works out-of-the-box.

## End-to-End Experiments

Validate that pruning reduces trace size without breaking structural validity:

```bash
# High-redundancy synthetic traces (demonstrates compression power)
python -m experiments.end_to_end_prune --dataset synthetic --n 50

# Real GSM8K human CoT traces (demonstrates conservative safety)
python -m experiments.end_to_end_prune --dataset gsm8k --n 50

# Mixed (best of both)
python -m experiments.end_to_end_prune --dataset mixed --n 50
```

**Key findings (synthetic, n=50):**

| Condition | Node Compression | Token Savings | NGS Pass Rate |
|-----------|-----------------|---------------|---------------|
| Structural (heuristic) | **12.5%** | ~31 tokens/trace | **100%** |
| Signal-enhanced (calib+PRM) | **0.0%** | ~0 tokens/trace | **100%** |

Signal-enhanced pruning downgrades confidence on steps that contain substantive signals (numbers, formulas, forward progress), making it **more conservative** and avoiding false positives. On naturally compact GSM8K traces, structural pruning self-limits to ~2% compression, proving the system does not over-prune tight reasoning.

## External Project Integrations

These projects are **absorbed into existing modules** rather than becoming new CLI commands:

| External Project | Absorbed Into | Enhancement |
|-----------------|---------------|-------------|
| **metacognitive-calibration** | `analysis/prune.py` | Node-level confidence signal refines redundancy confidence |
| **Reasoning Navigation Engine (PRM)** | `analysis/prune.py` | Per-step process reward refines redundancy confidence |
| **reasonplan** | `analysis/benchmark.py` | Plan-redundancy score validates "excessive planning hurts efficiency" |

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

If you use RTSA in your research, please cite:

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
