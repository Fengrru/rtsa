# RTSA — Reasoning Trace Structure Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](https://github.com/Fengrru/rtsa/actions)
[![tests: 322](https://img.shields.io/badge/tests-322%20passing-green.svg)](https://github.com/Fengrru/rtsa/actions)

**把大模型的思维链（Chain-of-Thought）当作结构化图来研究。**

RTSA 将 CoT 文本解析为带类型的 DAG（Retrieve / Transform / Verify / Branch /
Backtrack / Compare），并在此之上提供验证、分析、剪枝、步骤级正确性预测、
指纹识别与基准评测——全部基于纯文本，**无需白盒访问模型内部状态**。

- [快速开始](#快速开始) · [核心能力](#核心能力) · [Pipeline](#pipeline) ·
  [实验入口](#统一实验入口) · [文档](#文档) · [引用](#引用)

## 快速开始

```bash
pip install rtsa
```

三行代码跑通全流程：

```python
from extractors import RuleBasedExtractor
from core.ngs_validator import NGSValidator
from analysis.prune import RedundancyAnalyzer, PruneConfig

text = "Retrieve x=3. Transform: x*2=6. Verify: 6 is even."
graph = RuleBasedExtractor().extract(text, trace_id="demo_001")

valid, violations = NGSValidator().validate(graph)
report = RedundancyAnalyzer(config=PruneConfig()).analyze(graph, apply_pruning=True)
print(report.summary())
```

命令行：

```bash
rtsa extract cot.txt --extractor rbe --output graph.json
rtsa validate graph.json
rtsa prune graph.json --apply --output pruned.json
```

## 能回答什么问题

| 问题 | 答案 |
|---|---|
| 这条推理冗余吗？冗余在哪？ | 区域级冗余检测 + 可执行的 DAG 剪枝（`analysis/prune.py`） |
| 这一步推理正确吗？ | 基于 17 维结构特征的黑盒步骤分类器（`analysis/step_classifier.py`，受 CRV 启发） |
| 结构能预测正确性吗？ | 19 项结构指标与正确性/性能分数的基准评测，FDR 校正 + bootstrap 置信区间（`analysis/performance_correlation.py`） |
| 哪家模型写的？ | 基于结构风格的作者指纹（`rtsa fingerprint`） |
| 两条推理有多像？ | 有监督 Robust-TSI + 无监督 WL-kernel 相似度 |

## 核心能力

| 能力 | 实现 | 备注 |
|---|---|---|
| CoT → 图提取 | `extractors/` | 规则 / 依存句法 / LLM / 随机基线 4 族提取器 |
| 结构验证 | `core/ngs_validator.py` | 13 条 NGS 规则 + Type I/II 失败模式分类（7 类） |
| 冗余剪枝 | `analysis/prune.py` | 4 种冗余检测器，DAG 完整性保持，支持 domain 自适应阈值 |
| 步骤级分析 | `analysis/step_classifier.py` `step_clustering.py` | 17 维特征错误概率 + 语义宏步聚类（LLM-MindMap 式） |
| 相似度与指纹 | `core/robust_tsi.py` `analysis/fingerprint.py` | 有监督 TSI / 无监督 WL-kernel / 作者识别 |
| 统计严谨性 | `core/robust_tsi.py` `analysis/performance_correlation.py` | bootstrap 置信区间、Cohen's d 效应量、剪枝节省误差带、BH-FDR 多重比较校正 |
| 性能相关性基准 | `analysis/performance_correlation.py` | 19 指标 × 3 家族（全局/类型构成/形状），Spearman + FDR + bootstrap rho CI，输出论文表格 |
| 基准评测 | `analysis/benchmark.py` | GCP + NGS 通过率 + TSI + 计划冗余度 |
| 数据集接入 | `utils/hf_adapter.py` | 任意 HuggingFace CoT 数据集（gsm8k / math / 自定义） |
| 可观测性 | `utils/trace_exporters.py` | OTLP / Langfuse 导出，缺依赖自动降级 |
| 可复现实验 | `experiments/run.py` | 统一入口 + 版本化结果目录 + manifest 溯源 |

## Pipeline

```
raw CoT text (JSONL / HuggingFace datasets)
    │  extractors: RBE (rule) · SBE (syntax) · LLM · random baselines
    ▼
ReasoningTraceGraph (typed DAG)
    │
    ├──► validate    NGS structural rules + failure-mode taxonomy
    ├──► analyze     graph metrics, motifs, TSI/JSD, structure↔correctness
    ├──► prune       redundancy regions → pruned graph (DAG-preserving)
    ├──► classify    per-step error probability (GradientBoosting)
    └──► benchmark   GCP · NGS pass rate · TSI · authorship fingerprint
```

## 与研究前沿的关系

| 工作 | 我们的对应实现 |
|---|---|
| **LLM-MindMap**（EMNLP 2025）— 语义步聚类 + 结构指标预测性能 | `analysis/step_clustering.py` + `experiments/correlation_analysis.py` |
| **CRV**（Meta FAIR, arXiv 2510.09312）— 图特征验证推理步 | `analysis/step_classifier.py` + `PruneConfig.domain_overrides` |
| **CoT2Graph** — 推理路径验证与失败模式 | `core/ngs_validator.py` 失败模式分类 |

逐项能力对比矩阵见 [docs/comparison.md](docs/comparison.md)。

## 统一实验入口

```bash
python -m experiments.run extract   --dataset gsm8k --max-traces 50
python -m experiments.run analyze   --dataset gsm8k
python -m experiments.run prune     --dataset synthetic --n 50
python -m experiments.run calibrate --synthetic
python -m experiments.run annotate
python -m experiments.run correlation --synthetic           # 19 指标性能相关性基准
python -m experiments.run all       --dataset gsm8k
```

每次运行写入 `experiments/results/runs/<command>_<timestamp>/`，附 `manifest.json`
（git commit、Python 版本、参数、UTC 时间戳）保证可复现。

## 文档

- [docs/api.md](docs/api.md) — 公共 API 参考与模块布局
- [docs/comparison.md](docs/comparison.md) — 与 LLM-MindMap / CRV / CoT2Graph 的能力矩阵
- [docs/failure_modes.md](docs/failure_modes.md) — NGS 失败模式分类法
- [experiments/notebooks/end_to_end.ipynb](experiments/notebooks/end_to_end.ipynb) — 端到端可运行教程
- [CHANGELOG.md](CHANGELOG.md) — 版本历史（Keep a Changelog）

## 测试

```bash
python -m pytest tests/ -q        # 312 tests
```

## 引用

```bibtex
@software{rtsa2026,
  title={RTSA: Reasoning Trace Structure Analysis Toolkit},
  author={Fengrru},
  year={2026},
  url={https://github.com/Fengrru/rtsa}
}
```

## 贡献与许可

- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 许可证：[MIT](LICENSE)
