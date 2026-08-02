"""Visualization utilities for Reasoning Trace Graphs."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from rtsa.core.types import ReasoningTraceGraph


NODE_COLORS = {
    "Retrieve": "#4CAF50",
    "Transform": "#2196F3",
    "Compare": "#FF9800",
    "Verify": "#9C27B0",
    "Branch": "#F44336",
    "Backtrack": "#FF5722",
}


def _get_type_colors(types):
    return [NODE_COLORS.get(t, "#607D8B") for t in types]


def plot_graph(
    graph: ReasoningTraceGraph,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_labels: bool = True,
    save_path: Optional[str] = None,
):
    """Plot a ReasoningTraceGraph as a DAG with colored nodes by type."""
    G = graph.to_networkx()
    if G.number_of_nodes() == 0:
        print("Empty graph - nothing to plot")
        return

    pos = nx.spring_layout(G, seed=42, k=2.0)
    types = [G.nodes[n].get("type", "Transform") for n in G.nodes()]
    colors = _get_type_colors(types)
    labels = {n: f"{n}\n({t})" for n, t in zip(G.nodes(), types)} if show_labels else {n: str(n) for n in G.nodes()}

    plt.figure(figsize=figsize)
    nx.draw(
        G, pos, with_labels=True, labels=labels,
        node_color=colors, node_size=1200, font_size=8,
        arrows=True, arrowsize=15, edge_color="#666",
        font_color="white", font_weight="bold",
    )
    plt.title(title or f"RTG: {graph.trace_id} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_graphs_comparison(
    graphs: list,
    labels: list,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    """Plot multiple RTGs side by side for comparison."""
    n = len(graphs)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, graph, label in zip(axes, graphs, labels):
        G = graph.to_networkx()
        pos = nx.spring_layout(G, seed=42, k=1.5)
        types = [G.nodes[n].get("type", "Transform") for n in G.nodes()]
        colors = _get_type_colors(types)
        labels_dict = {n: f"{n}\n({t})" for n, t in zip(G.nodes(), types)}

        ax.set_title(f"{label} ({len(graph.nodes)} nodes)")
        nx.draw(
            G, pos, ax=ax, with_labels=True, labels=labels_dict,
            node_color=colors, node_size=800, font_size=7,
            arrows=True, arrowsize=12, edge_color="#666",
            font_color="white", font_weight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()
