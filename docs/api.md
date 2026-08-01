# API Reference

RTSA is organized into five layers. Everything in `core/`, `analysis/`,
`extractors/`, and `utils/` is importable directly; `experiments/` contains
reproducible experiment scripts rather than a library surface.

## Core Data Model (`core/types.py`)

The single shared representation for all reasoning traces.

| Symbol | Description |
|---|---|
| `NodeType` | Enum: `RETRIEVE`, `TRANSFORM`, `VERIFY`, `BRANCH`, `BACKTRACK`, `COMPARE` |
| `GraphNode` | Pydantic node: `id`, `type`, `text`, `span`, `meta` |
| `ReasoningTraceGraph` | Pydantic graph: `trace_id`, `nodes`, `edges`; helpers `to_networkx()`, `is_valid()`, `to_json()` |

## Layer 1: Core (`core/`)

| Symbol | Description |
|---|---|
| `compute_graph_features(graph)` | Structural feature vector (per-graph metrics) |
| `compute_tsi(G1, G2)` / `compute_pairwise_tsi(graphs)` | Supervised TSI similarity |
| `RobustTSI` | PCA + ridge regression similarity model (`fit`, `predict_pair`, `pairwise_similarity_matrix`) |
| `UnsupervisedTSI` | WL-kernel + JSD + GED similarity, no training data |
| `bootstrap_tsi_ci(tsi_function, G1, G2)` | Bootstrap confidence interval for a TSI score |
| `cohens_d(group_a, group_b)` | Effect size between two score groups |
| `MotifMatcher` | Frequent substructure detection (chain / fork / diamond / verify-after-transform) |
| `NGSValidator` | 13-rule structural validation (`validate(graph)` returns violations) |
| `classify_failure_mode(violations)` | Maps violations to the Type I / Type II failure-mode taxonomy |
| `NGSRule`, `NGSViolation` | Rule enum and violation record |

## Layer 2: Extractors (`extractors/`)

CoT text -> `ReasoningTraceGraph`. All implement `extract(text, trace_id, **metadata)`.

| Symbol | Description |
|---|---|
| `RuleBasedExtractor` | Regex/heuristic extraction (fast, deterministic) |
| `SyntaxBasedExtractor` | spaCy dependency-parse based extraction |
| `LLMExtractor` / `create_extractor_deepseek` / `create_extractor_e4` / `create_extractor_e5` | LLM-supervised extraction |
| `RandomBaselineExtractor` / `ShuffledTypeExtractor` | Random baselines for GCP benchmarking |
| `JPDirectedPreservingRandomizer` / `EdgeRewiringBaseline` / `PermutationBaseline` / `EnsembleBaseline` | Graph randomizations for baseline comparison |

## Layer 3: Analysis (`analysis/`)

| Symbol | Description |
|---|---|
| `RedundancyAnalyzer` / `PruneConfig` / `PruningReport` | Redundancy detection and DAG-preserving pruning; `PruningReport.savings_range()` gives an uncertainty band; `PruneConfig.resolve_for_domain()` applies domain overrides |
| `ExtractorBenchmark` / `benchmark_extractors` | GCP + NGS + TSI + plan-redundancy benchmarking |
| `StepFeatureExtractor` | 17-dimensional per-step features (11 structural + 6 type one-hot) |
| `StepCorrectnessClassifier` | GradientBoosting per-step error probability (`fit`, `predict_proba_error`, `save`, `load`) |
| `StepClusterer` | LLM-MindMap-style merging of chain segments into macro-steps |
| Fingerprint module | `ModelSignature`, `FingerprintMatchResult`; CLI `rtsa fingerprint enroll/identify` |

## Layer 4: Utilities (`utils/`)

| Symbol | Description |
|---|---|
| `load_hf_traces(...)` / `iter_hf_traces(...)` | Load any HuggingFace CoT dataset (gsm8k / math / plain parsers, auto-sniffing, custom mapper, streaming) |
| `to_hf_dataset(records)` / `save_hf_traces(records, path)` | Export traces back to HF format / JSONL |
| `make_trace_exporter(kind=None, **kwargs)` | OTLP / Langfuse exporter with automatic no-op fallback |
| `data_loader` / `gsm8k_loader` / `math_loader` | Bundled corpus loaders |
| `visualize` | DAG plotting helpers |

## Layer 5: Experiments (`experiments/`)

| Entry point | Purpose |
|---|---|
| `python -m experiments.run <cmd>` | Unified entrypoint: `extract`, `analyze`, `prune`, `calibrate`, `annotate`, `all`; versioned run dirs + `manifest.json` |
| `python -m experiments.calibrate_thresholds` | Coordinate-descent threshold calibration on annotated graphs |
| `python -m experiments.annotate_steps` | Generate per-node NGS-violation labels for classifier training |
| `python -m experiments.correlation_analysis` | Spearman structure-correctness correlation (synthetic or labeled data) |
| `python -m experiments.end_to_end_prune` | Dataset-level pruning evaluation (synthetic / gsm8k / mixed) |
| `python -m experiments.synthetic_redundant_cots` | Controlled synthetic CoT corpus with injected redundancy |
| `experiments/notebooks/end_to_end.ipynb` | Runnable full-pipeline walkthrough |

## CLI (`rtsa`)

`extract`, `quick`, `visualize`, `validate`, `gcp`, `compare`, `prune`,
`fingerprint enroll|identify`, `benchmark`. See the CLI Reference table in the
top-level [README](../README.md).
