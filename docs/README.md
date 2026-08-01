# RTSA Documentation

This directory collects the reference documentation for the RTSA toolkit. The
top-level [README](../README.md) covers installation, quick start, research
features, and the unified experiment entrypoint.

## Documents

| Document | Audience | What it covers |
|---|---|---|
| [api.md](api.md) | Developers, researchers integrating RTSA | Public API reference, module layout, and core data model |
| [comparison.md](comparison.md) | Evaluators, paper reviewers | Capability-by-capability matrix against LLM-MindMap, CRV, and CoT2Graph |
| [failure_modes.md](failure_modes.md) | Everyone running `rtsa validate` | NGS failure-mode taxonomy (Type I structural inefficiency / Type II dependency violation) |

## Module Layout

```
core/        graph types, metrics, similarity (TSI), motif matching, NGS validation
extractors/  CoT -> ReasoningTraceGraph (rule-based, syntax-based, LLM, random baselines)
analysis/    pruning, benchmarking, fingerprinting, step classifier, step clustering
utils/       data loading (GSM8K / MATH / HuggingFace), trace exporters, visualization
experiments/ reproducible experiments and the unified entrypoint (run.py)
```

## Quick Orientation

- Start with the [API reference](api.md) for the public interface of each module.
- See [failure_modes.md](failure_modes.md) before interpreting validation output.
- See [comparison.md](comparison.md) to understand how RTSA maps to current
  research (LLM-MindMap, CRV, CoT2Graph) and where its capabilities end.
- Run the end-to-end walkthrough notebook at
  `experiments/notebooks/end_to_end.ipynb` to see the whole pipeline in action.
