# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2026-08-01

### Added

- **Threshold calibration**: `experiments/calibrate_thresholds.py` runs a coordinate-descent scan over every `PruneConfig` threshold against redundancy annotations, with a synthetic annotated-graph generator for offline validation.
- **Step-level annotation pipeline**: `experiments/annotate_steps.py` produces deterministic per-node correctness labels from NGS violations, consumable by the step classifier.
- **Step classifier and clustering**: `analysis/step_classifier.py` predicts per-step error probability from 17 structural features (11 structural + 6 type one-hot) without white-box access; `analysis/step_clustering.py` merges chain segments into macro-steps (LLM-MindMap-style).
- **Statistical rigor helpers**: `bootstrap_tsi_ci` (bootstrap confidence interval for TSI scores) and `cohens_d` (effect size) in `core/robust_tsi.py`; `PruningReport.savings_range()` surfaces a +/-40% heuristic band instead of false precision.
- **Performance-correlation benchmark**: `analysis/performance_correlation.py` measures 19 structural metrics across 3 families (global / type-mix / shape) against correctness or continuous performance scores, with Spearman rho, Benjamini-Hochberg FDR correction, optional bootstrap confidence intervals, and paper-ready Markdown output; wired into the unified entrypoint as `run correlation`.
- **HuggingFace datasets adapter**: `utils/hf_adapter.py` supports any CoT dataset (gsm8k / math / plain parsers with auto-sniffing, custom mappers, streaming iteration, and export back to HF format).
- **Unified experiment entrypoint**: `experiments/run.py` exposes `extract / analyze / prune / calibrate / annotate / correlation / all` behind one CLI, writing versioned run directories with `manifest.json` provenance (git commit, Python version, args, UTC timestamp).
- **Observability**: `utils/trace_exporters.py` provides OTLP and Langfuse trace exporters behind a unified interface, degrading to a no-op exporter when dependencies are missing.
- **GitHub Actions CI**: `.github/workflows/ci.yml` runs the full test suite on Python 3.10 / 3.11 / 3.12.
- **Documentation**: rewritten `README.md` (pipeline diagram, research feature tables, related-work mapping, unified entrypoint); `docs/comparison.md` capability matrix; `docs/api.md` API reference; `experiments/notebooks/end_to_end.ipynb` runnable walkthrough; new `hf` / `obs` / `notebook` / `all` extras in `pyproject.toml`.
- **Tests**: 67 new tests across 9 modules (failure modes, savings range, domain overrides, calibration, HF adapter, exporters, step clustering, step classifier, run CLI, performance correlation).

### Changed

- `README.md` corrected the step-classifier feature count (17, not 21).

### Fixed

- `analysis/step_classifier.py`: generator-expression `NameError` when a step references out-of-graph node ids; now safely skips missing ids.
- `experiments/synthetic_redundant_cots.py`: `_inject_long_transform_chain` accepted an `rng` argument to match the injector protocol.
- `experiments/run.py`: robust iteration over `GraphMetrics` fields, non-numeric `trace_id` handling, and `--synthetic` flag for `calibrate`.

## [3.3.0] - 2026-07-31

### Added

- **Signal-Enhanced Pruning**: `analysis/prune.py` now supports optional metacognitive calibration and PRM (Process-Reward Model) signals.
  - `--use-calibration` CLI flag for confidence-based redundancy refinement.
  - `--use-prm` CLI flag for process-reward-based redundancy refinement.
  - `analysis/signal_adapters.py` provides unified adapter interfaces with robust fallback heuristics when external libraries are not installed.
- **Plan Redundancy Score**: `analysis/benchmark.py` now computes a `plan_redundancy_score` inspired by *reasonplan* negative results. Measures the proportion of `Branch`/`Backtrack` nodes without productive children.
- **End-to-End Pruning Utility Experiment**: New `experiments/end_to_end_prune.py` validates that pruning reduces trace size while preserving structural integrity (NGS validity).
  - Supports three datasets: `synthetic` (controlled redundancy), `gsm8k` (real human CoT), and `mixed`.
- **Synthetic CoT Generator**: `experiments/synthetic_redundant_cots.py` produces 50 diverse math-reasoning traces with injected redundancy patterns for controlled experiments.
- **Pruning Safety Enhancements**:
  - `PruneConfig.min_confidence_threshold` for fine-grained control over pruning aggressiveness.
  - `_apply_pruning` now cleans up unreachable orphan nodes after deletion to guarantee DAG integrity.

### Changed

- `PruneConfig` extended with signal-related fields (`use_calibration_signal`, `calibration_weight`, `use_prm_signal`, `prm_weight`).
- `RedundancyAnalyzer` constructor now accepts optional `calibration_adapter` and `prm_adapter`.
- `ExtractorBenchmarkResult` extended with `plan_redundancy_score`.
- `README.md` reorganized with clearer sections for installation, quick start, and project integrations.

### Fixed

- `end_to_end_prune.py` correctly handles cases where `pruned_graph` is `None` (no redundancy detected), inheriting original metrics instead of defaulting to zero.

## [3.2.0] - 2026-07-15

### Added

- **Granularity Calibration Protocol (GCP)**: Systematic protocol for calibrating reasoning-graph extraction granularity against human judgments.
- **Fingerprint Module**: LLM authorship attribution via stylometric graph features (`rtsa fingerprint enroll/identify`).
- **WL Kernel Comparison**: Weisfeiler-Lehman graph kernel for semantic similarity between reasoning traces.
- **Motif Matcher**: Frequent substructure detection for identifying recurring reasoning patterns.

### Changed

- Refactored `core/types.py` to use Pydantic BaseModel for all graph structures.
- Improved `RuleBasedExtractor` coverage for DeepSeek-style `<think>` tags.

## [3.1.0] - 2026-06-20

### Added

- **NGS Validator**: 13-rule structural validator for reasoning trace graphs.
- **Pruning Engine**: Heuristic-based redundancy detection with four motif detectors (excessive verification, dead branches, long transform chains, structural bloat).
- **TSI (Trace Similarity Index)**: Graph-edit-distance-based similarity metric with robust normalization.

## [3.0.0] - 2026-06-01

### Added

- Initial release of RTSA v3 with modular architecture.
- `extract`, `visualize`, `compare`, `validate`, `prune`, `benchmark` CLI commands.
- Support for Rule-Based, Syntax-Based, and LLM-Based extractors.
- Pilot experiments on GSM8K and synthetic reasoning traces.
