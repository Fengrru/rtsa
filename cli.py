"""
RTSA CLI — Reasoning Trace Structure Analysis.

Usage:
    rtsa extract <file>          Extract RTG from CoT text (default: DeepSeek)
    rtsa quick <text...>         Quick inline extraction
    rtsa visualize <file>        Visualize an RTG as a graph
    rtsa validate <file>         Validate graph against NGS rules
    rtsa gcp                     Run GCP calibration
    rtsa compare <f1> <f2>       Compare two RTGs
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="3.3.0")
def main():
    """RTSA: Reasoning Trace Structure Analysis Toolkit."""


@main.command()
@click.argument("cot_file", type=click.Path(exists=True))
@click.option("--extractor", "-e", default="deepseek",
              type=click.Choice(["rbe", "sbe", "deepseek", "rbe_rand"]),
              help="Extractor to use (default: deepseek)")
@click.option("--output", "-o", default=None, help="Output JSON file path")
@click.option("--model", default="deepseek-chat", help="LLM model name (for deepseek extractor)")
def extract(cot_file: str, extractor: str, output: Optional[str], model: str):
    """Extract a Reasoning Trace Graph from CoT text file."""
    cot_text = Path(cot_file).read_text(encoding="utf-8")

    if extractor == "deepseek":
        from extractors.llm_extractor import create_extractor_deepseek
        ext = create_extractor_deepseek(model=model)
    elif extractor == "rbe":
        from extractors.rule_based import RuleBasedExtractor
        ext = RuleBasedExtractor()
    elif extractor == "sbe":
        from extractors.syntax_based import SyntaxBasedExtractor
        ext = SyntaxBasedExtractor()
    else:
        from extractors.random_baseline import RandomBaselineExtractor
        ext = RandomBaselineExtractor()

    click.echo(f"Extracting with {extractor}...")
    graph = ext.extract(cot_text)
    result = json.dumps(graph.to_canonical_dict(), indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(result, encoding="utf-8")
        click.echo(f"Graph saved to {output}")
    else:
        click.echo(result)

    types = [n.type.value for n in graph.nodes]
    click.echo(f"\nSummary: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    click.echo(f"Types: {' -> '.join(types)}")


@main.command()
@click.argument("cot_text", nargs=-1)
@click.option("--model", default="deepseek-chat", help="LLM model name")
def quick(cot_text: tuple, model: str):
    """Quick extraction from inline CoT text using DeepSeek."""
    from extractors.llm_extractor import create_extractor_deepseek

    text = " ".join(cot_text)
    if not text.strip():
        click.echo("Usage: rtsa quick <your CoT text here>")
        return

    ext = create_extractor_deepseek(model=model)
    with click.progressbar(length=1, label="Extracting") as bar:
        graph = ext.extract(text)
        bar.update(1)

    types = [n.type.value for n in graph.nodes]
    click.echo(f"\nNodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
    click.echo(f"Types: {' -> '.join(types)}")


@main.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Save plot to file (PNG)")
@click.option("--title", default=None, help="Plot title")
def visualize(graph_file: str, output: Optional[str], title: Optional[str]):
    """Visualize an RTG JSON file as a DAG."""
    from core.types import ReasoningTraceGraph
    from utils.visualize import plot_graph

    data = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    graph = ReasoningTraceGraph.from_json(data)
    plot_graph(graph, title=title, save_path=output)
    click.echo(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")


@main.command()
@click.argument("graph_file", type=click.Path(exists=True))
def validate(graph_file: str):
    """Validate an RTG JSON file against schema and NGS rules."""
    from core.types import GraphNode, NodeType, ReasoningTraceGraph
    from core.ngs_validator import NGSValidator

    data = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    graph = ReasoningTraceGraph.from_json(data)

    valid_schema, schema_errs = graph.is_valid()
    validator = NGSValidator()
    valid_ngs, violations = validator.validate(graph)

    if valid_schema and valid_ngs:
        click.echo("PASS: Graph is valid and NGS-compliant.")
    else:
        click.echo(f"FAIL: {len(schema_errs)} schema error(s), {len(violations)} NGS violation(s)")
        for e in schema_errs:
            click.echo(f"  [schema] {e}")
        for v in violations:
            click.echo(f"  [ngs:{v.rule.value}] {v.message}")


@main.command()
def gcp():
    """Run Granularity Calibration Protocol on all deterministic extractors."""
    from extractors.gcp_validator import GCPValidator, GCS_CORPUS_FULL
    from extractors.rule_based import RuleBasedExtractor
    from extractors.syntax_based import SyntaxBasedExtractor
    from extractors.random_baseline import RandomBaselineExtractor

    validator = GCPValidator(corpus=GCS_CORPUS_FULL)
    extractors = {
        "rbe": RuleBasedExtractor().classify_sentence,
        "sbe": SyntaxBasedExtractor().classify_sentence,
        "rbe_rand": lambda s: RandomBaselineExtractor().classify_by_length(s),
    }

    click.echo(f"\nGCP Calibration ({len(GCS_CORPUS_FULL)} sentences)\n")
    click.echo(f"{'Extractor':<12} {'Status':<10} {'Mean':>8} {'Min':>8} {'CI Lower':>10} {'CI Upper':>10}")
    click.echo("-" * 60)

    for name, ext_fn in extractors.items():
        result = validator.calibrate_extractor(ext_fn, name)
        status = "PASSED" if result.passed else "FAILED"
        click.echo(
            f"{name:<12} {status:<10} {result.mean_gcs:>8.3f} {result.min_gcs:>8.3f} "
            f"{result.bootstrap_ci[0]:>10.3f} {result.bootstrap_ci[1]:>10.3f}"
        )
        if result.failure_details:
            for fd in result.failure_details[:3]:
                click.echo(f"  -> {fd}")


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
def compare(file1: str, file2: str):
    """Compare two RTGs and compute similarity metrics."""
    from core.types import ReasoningTraceGraph
    from core.metrics import compute_tsi, compute_level1_features, feature_distance

    g1 = ReasoningTraceGraph.from_json(json.loads(Path(file1).read_text()))
    g2 = ReasoningTraceGraph.from_json(json.loads(Path(file2).read_text()))

    nx1, nx2 = g1.to_networkx(), g2.to_networkx()

    tsi_result = compute_tsi(nx1, nx2)
    f1, f2 = compute_level1_features(nx1), compute_level1_features(nx2)
    fd = feature_distance(f1, f2)

    click.echo(f"\nComparing {g1.trace_id} vs {g2.trace_id}")
    click.echo(f"{'Metric':<25} {'Value':>10}")
    click.echo("-" * 37)
    click.echo(f"{'TSI':<25} {tsi_result.tsi_value:>10.4f}")
    click.echo(f"{'Motif Similarity':<25} {tsi_result.motif_similarity:>10.4f}")
    click.echo(f"{'WL Kernel':<25} {tsi_result.wl_similarity:>10.4f}")
    click.echo(f"{'Feature Similarity':<25} {tsi_result.feature_similarity:>10.4f}")
    click.echo(f"{'Feature Distance':<25} {fd:>10.4f}")
    click.echo(f"{'Node count diff':<25} {abs(len(g1.nodes) - len(g2.nodes)):>10}")
    click.echo(f"{'Edge count diff':<25} {abs(len(g1.edges) - len(g2.edges)):>10}")


# ===================================================================
# NEW: Prune — redundancy detection & CoT optimization
# ===================================================================

@main.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--apply", is_flag=True, default=False,
              help="Apply pruning and output pruned graph")
@click.option("--output", "-o", default=None, help="Output JSON file for pruned graph")
@click.option("--report", "-r", default=None, help="Output markdown report file")
@click.option("--verify-threshold", default=0.40, help="Verify density threshold")
@click.option("--transform-chain", default=3, help="Max consecutive transforms before flag")
@click.option("--use-calibration", is_flag=True, default=False,
              help="Use metacognitive-calibration signal to refine redundancy confidence")
@click.option("--use-prm", is_flag=True, default=False,
              help="Use PRM process-reward signal to refine redundancy confidence")
def prune(graph_file: str, apply: bool, output: Optional[str], report: Optional[str],
          verify_threshold: float, transform_chain: int,
          use_calibration: bool, use_prm: bool):
    """Analyze redundancy in a Reasoning Trace Graph and suggest pruning."""
    from core.types import ReasoningTraceGraph
    from analysis.prune import RedundancyAnalyzer, PruneConfig

    data = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    graph = ReasoningTraceGraph.from_json(data)

    config = PruneConfig(
        verify_density_high=verify_threshold,
        max_consecutive_transforms=transform_chain,
        use_calibration_signal=use_calibration,
        use_prm_signal=use_prm,
    )
    analyzer = RedundancyAnalyzer(config=config)
    result = analyzer.analyze(graph, apply_pruning=apply)

    click.echo(f"\nPruning Analysis: {result.trace_id}")
    click.echo("-" * 50)
    click.echo(f"Original: {result.original_n_nodes} nodes, {result.original_n_edges} edges")
    click.echo(f"Redundancy regions found: {len(result.redundancy_regions)}")
    click.echo(f"Estimated token savings: {result.total_estimated_savings}")

    if result.redundancy_regions:
        click.echo(f"\n{'Region':<25} {'Nodes':<15} {'Conf':>6} {'Action':<10} {'Savings':>8}")
        click.echo("-" * 70)
        for r in result.redundancy_regions:
            nodes_str = ",".join(str(n) for n in r.node_ids[:5])
            if len(r.node_ids) > 5:
                nodes_str += "..."
            click.echo(
                f"{r.region_type:<25} {nodes_str:<15} {r.confidence:>6.2f} "
                f"{r.suggested_action:<10} {r.estimated_token_savings:>8}"
            )

    if apply and result.pruned_graph:
        click.echo(f"\nPruned: {result.pruned_graph.n_nodes} nodes, {len(result.pruned_graph.edges)} edges")
        click.echo(f"Structural integrity: {result.structural_integrity_score:.2f}")

        if output:
            pruned_data = result.pruned_graph.to_canonical_dict()
            Path(output).write_text(
                json.dumps(pruned_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            click.echo(f"Pruned graph saved to {output}")

    if report:
        lines = [
            f"# Pruning Report: {result.trace_id}\n",
            f"- **Original nodes**: {result.original_n_nodes}",
            f"- **Original edges**: {result.original_n_edges}",
            f"- **Redundancy regions**: {len(result.redundancy_regions)}",
            f"- **Estimated token savings**: {result.total_estimated_savings}",
        ]
        if apply and result.pruned_graph:
            lines.extend([
                f"- **Pruned nodes**: {result.pruned_graph.n_nodes}",
                f"- **Pruned edges**: {len(result.pruned_graph.edges)}",
                f"- **Integrity score**: {result.structural_integrity_score:.2f}",
            ])
        lines.append("\n## Detected Regions\n")
        for r in result.redundancy_regions:
            lines.append(
                f"- **{r.region_type}** (conf={r.confidence:.2f}): "
                f"nodes={r.node_ids}, action={r.suggested_action}, "
                f"savings={r.estimated_token_savings} tokens"
            )
            lines.append(f"  - {r.description}")
        Path(report).write_text("\n".join(lines), encoding="utf-8")
        click.echo(f"Report saved to {report}")


# ===================================================================
# NEW: Fingerprint — LLM authorship attribution
# ===================================================================

@main.group()
def fingerprint():
    """Model fingerprinting: enroll signatures and identify authors."""


@fingerprint.command("enroll")
@click.option("--model", required=True, help="Model name (e.g. deepseek-chat, gpt-4o)")
@click.option("--graphs", required=True, type=click.Path(exists=True),
              help="Directory or JSONL file containing RTG graphs")
@click.option("--output", "-o", required=True, help="Output .npz signature file")
def fingerprint_enroll(model: str, graphs: str, output: str):
    """Enroll a model signature from a collection of reasoning graphs."""
    from core.types import ReasoningTraceGraph
    from analysis.fingerprint import ModelFingerprint

    graph_list: List[ReasoningTraceGraph] = []
    path = Path(graphs)

    if path.is_file():
        # JSONL
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                graph_list.append(ReasoningTraceGraph.from_json(data))
    elif path.is_dir():
        for json_file in path.glob("*.json"):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            graph_list.append(ReasoningTraceGraph.from_json(data))

    if len(graph_list) < 5:
        click.echo(f"WARNING: only {len(graph_list)} graphs found (recommended >= 5)")

    fp = ModelFingerprint()
    sig = fp.enroll(model, graph_list)
    fp.save(output)

    click.echo(f"\nEnrolled signature for '{model}'")
    click.echo(f"  Samples: {sig.n_samples}")
    click.echo(f"  Feature dim: {len(sig.feature_mean)}")
    click.echo(f"  Saved to: {output}")


@fingerprint.command("identify")
@click.argument("graph_file", type=click.Path(exists=True))
@click.option("--signatures", "-s", required=True, help="Path to .npz signature file")
@click.option("--method", default="mahalanobis",
              type=click.Choice(["mahalanobis", "cosine", "euclidean"]))
def fingerprint_identify(graph_file: str, signatures: str, method: str):
    """Identify which model most likely generated a reasoning graph."""
    from core.types import ReasoningTraceGraph
    from analysis.fingerprint import ModelFingerprint

    data = json.loads(Path(graph_file).read_text(encoding="utf-8"))
    graph = ReasoningTraceGraph.from_json(data)

    fp = ModelFingerprint()
    fp.load(signatures)
    result = fp.identify(graph, method=method)

    click.echo(f"\nAuthorship Identification: {graph.trace_id}")
    click.echo("-" * 40)
    click.echo(f"Predicted model: {result.predicted_model}")
    click.echo(f"Confidence: {result.confidence:.3f}")
    click.echo(f"\nAll scores ({method} distance, lower = closer):")
    for model, score in sorted(result.all_scores.items(), key=lambda kv: kv[1]):
        marker = " <--" if model == result.predicted_model else ""
        click.echo(f"  {model:<20} {score:.4f}{marker}")


# ===================================================================
# NEW: Benchmark — extractor reliability suite
# ===================================================================

@main.command()
@click.option("--extractors", "-e", default="rbe,sbe",
              help="Comma-separated extractor names to benchmark")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Directory containing pre-extracted graphs per extractor")
@click.option("--output", "-o", default=None, help="Output JSON report file")
@click.option("--gcp/--no-gcp", default=True, help="Run GCP calibration")
@click.option("--ngs/--no-ngs", default=True, help="Run NGS validation")
@click.option("--tsi/--no-tsi", default=True, help="Run TSI consistency")
def benchmark(extractors: str, data: str, output: Optional[str],
              gcp: bool, ngs: bool, tsi: bool):
    """Run the RTSA extractor reliability benchmark suite."""
    from core.types import ReasoningTraceGraph
    from analysis.benchmark import ExtractorBenchmark

    extractor_names = [n.strip() for n in extractors.split(",") if n.strip()]
    data_dir = Path(data)

    # Load pre-extracted graphs per extractor
    graphs: Dict[str, List[ReasoningTraceGraph]] = {}
    for name in extractor_names:
        extractor_dir = data_dir / name
        graph_list: List[ReasoningTraceGraph] = []
        if extractor_dir.is_dir():
            for json_file in extractor_dir.glob("*.json"):
                graph_list.append(ReasoningTraceGraph.from_json(
                    json.loads(json_file.read_text(encoding="utf-8"))
                ))
        elif (data_dir / f"{name}.jsonl").exists():
            with open(data_dir / f"{name}.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        graph_list.append(ReasoningTraceGraph.from_json(json.loads(line)))
        graphs[name] = graph_list
        click.echo(f"Loaded {len(graph_list)} graphs for '{name}'")

    # Build real extractor callables so the benchmark can also run GCP
    # when enabled (each callable is an extractor ``extract`` bound method).
    extractor_map = {}
    for name in extractor_names:
        if name == "rbe":
            from extractors.rule_based import RuleBasedExtractor
            extractor_map[name] = RuleBasedExtractor().extract
        elif name == "sbe":
            from extractors.syntax_based import SyntaxBasedExtractor
            extractor_map[name] = SyntaxBasedExtractor().extract
        elif name == "rbe_rand":
            from extractors.random_baseline import RandomBaselineExtractor
            extractor_map[name] = RandomBaselineExtractor().extract
        else:
            # Unknown name (e.g. "deepseek"): keep an identity stub;
            # NGS/TSI consume the pre-extracted graphs, not this callable.
            extractor_map[name] = lambda t, n=name: t

    bench = ExtractorBenchmark()
    report = bench.run(
        extractor_map,
        graphs=graphs,
        run_gcp=False,  # GCP requires sentence-level adapters; skip in CLI for now
        run_ngs=ngs,
        run_tsi=tsi,
    )

    click.echo(report.summary())

    if output:
        report.to_json(output)


if __name__ == "__main__":
    main()
