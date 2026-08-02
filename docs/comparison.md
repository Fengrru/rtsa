# RTSA vs. Related Work — Capability Matrix

RTSA draws on three contemporary lines of research — LLM-MindMap (EMNLP
2025), CRV (Meta FAIR, arXiv 2510.09312) and CoT2Graph — and implements
their core ideas *black-box* (no model-internal access), with an
extensible toolchain on top.

Legend: yes / no / partial / n/a (not applicable to that work's goal)

| Capability | LLM-MindMap | CRV | CoT2Graph | RTSA |
|---|---|---|---|---|
| CoT → graph extraction | yes (semantic clusters) | n/a (white-box compute graph) | yes (rule + LLM mix) | **yes** — 4 extractor families (RBE / SBE / LLM / random baseline) |
| No white-box model access | yes | **no** (needs internal states) | yes | **yes** — purely textual input |
| Graph-level structural metrics | exploration density, branching, convergence ratio | global statistics, node influence, topological paths | graph statistics | **yes** — 9 global metrics + motif frequencies + WL kernel + TSI |
| Step-level structural features | no | yes (3-layer feature stack) | partial | **yes** — 17-dim vector per node (11 structural + 6 type one-hot) |
| Step correctness prediction | no | yes (AUROC 70-92%) | path validity only | **yes** — `StepCorrectnessClassifier` (GradientBoosting, error probability per step) |
| Redundancy detection & pruning | no | no | no | **yes** — region-level detection + DAG-preserving executable pruning |
| Failure-mode taxonomy | no | binary error/ok | reasoning-path failure classes | **yes** — Type I (structural inefficiency) / Type II (dependency violation), 7 modes |
| Domain-adaptive thresholds | no | yes (signatures domain-dependent) | no | **yes** — `PruneConfig.domain_overrides` |
| Structure ↔ correctness correlation | yes (reported) | implicit | no | **yes** — `rtsa/experiments/correlation_analysis.py` (Spearman, p-values) |
| Threshold calibration workflow | no | no | no | **yes** — `rtsa/experiments/calibrate_thresholds.py` (coordinate-descent grid scan) |
| Similarity / authorship fingerprinting | no | no | no | **yes** — supervised Robust-TSI + unsupervised WL-kernel + fingerprint |
| Annotation burden | none | step labels (costly) | none | **zero-annotation path** — NGS-rule labels, optional LLM/human judge |
| Statistical rigor (CI / effect size) | partial | partial | no | **yes** — bootstrap CI, Cohen's d, savings error bands |
| Tooling (CLI, versioned runs, CI) | partial | no | partial | **yes** — `rtsa/experiments/run.py` + manifest.json + GitHub Actions |
| Observability | no | no | no | **yes** — OTLP / Langfuse exporters (optional) |
| Dataset flexibility | fixed benchmarks | fixed | fixed | **any** HuggingFace dataset via `rtsa/utils/hf_adapter.py` |

## Design principles behind the differences

1. **Black-box by default.** CRV's strongest results come from white-box
   features; RTSA keeps everything computable from the text alone so it
   works with any API-only model.
2. **Rules first, models second.** The NGS rule set gives deterministic,
   explainable signals; the learned classifier generalizes beyond them —
   and the annotation pipeline makes both trainable without manual labour.
3. **Uncertainty is surfaced, not hidden.** Correlations report p-values,
   TSI reports bootstrap CIs, pruning reports savings ranges.
4. **Everything is a script.** Every analysis in this table runs from
   `python -m experiments.run ...` and lands in a versioned result
   directory.
