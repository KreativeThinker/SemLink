"""
Visualization Module.

This module provides functions for creating interactive and static
visualizations of knowledge graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import orjson


# Default color palette for communities
DEFAULT_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
]


# =============================================================================
# Interactive HTML Visualization (Pyvis)
# =============================================================================


def to_pyvis(
    graph: nx.Graph,
    output_path: Path,
    height: str = "800px",
    width: str = "100%",
    bgcolor: str = "#222222",
    font_color: str = "white",
    notebook: bool = False,
) -> Path:
    """
    Create interactive HTML visualization using Pyvis.

    Args:
        graph: NetworkX graph
        output_path: Output HTML file path
        height: Canvas height
        width: Canvas width
        bgcolor: Background color
        font_color: Label font color
        notebook: Whether running in Jupyter notebook

    Returns:
        Path to generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "Pyvis is required for interactive visualization. "
            "Install with: pip install pyvis"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        notebook=notebook,
    )

    # Add nodes
    for node, data in graph.nodes(data=True):
        title = data.get("title", str(node))
        size = data.get("size", 20)
        color = data.get("color", "#97c2fc")
        net.add_node(str(node), label=title, title=title, size=size, color=color)

    # Add edges
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        net.add_edge(str(u), str(v), value=weight, title=f"weight: {weight:.3f}")

    # Configure physics
    net.set_options(
        """
        var options = {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -80000,
                    "centralGravity": 0.3,
                    "springLength": 250,
                    "springConstant": 0.001
                },
                "maxVelocity": 50,
                "stabilization": {"iterations": 150}
            }
        }
        """
    )

    net.save_graph(str(output_path))
    return output_path


def configure_pyvis_physics(
    gravity: float = -80000,
    central_gravity: float = 0.3,
    spring_length: int = 250,
    spring_strength: float = 0.001,
) -> dict[str, Any]:
    """
    Create physics configuration for Pyvis.

    Args:
        gravity: Node repulsion strength
        central_gravity: Pull toward center
        spring_length: Preferred edge length
        spring_strength: Edge spring constant

    Returns:
        Physics configuration dict
    """
    return {
        "gravity": gravity,
        "central_gravity": central_gravity,
        "spring_length": spring_length,
        "spring_strength": spring_strength,
    }


def style_nodes_by_community(
    graph: nx.Graph,
    colors: list[str] | None = None,
) -> dict[str, str]:
    """
    Generate node colors based on community membership.

    Args:
        graph: NetworkX graph with 'community' attribute
        colors: Custom color palette

    Returns:
        Dict mapping node_id to color
    """
    if colors is None:
        colors = DEFAULT_COLORS

    node_colors: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        community = data.get("community", 0)
        color = colors[community % len(colors)]
        node_colors[str(node)] = color

    return node_colors


def style_nodes_by_centrality(
    graph: nx.Graph,
    measure: str = "pagerank",
    min_size: int = 10,
    max_size: int = 50,
) -> dict[str, int]:
    """
    Generate node sizes based on centrality.

    Args:
        graph: NetworkX graph with centrality attributes
        measure: Centrality measure to use
        min_size: Minimum node size
        max_size: Maximum node size

    Returns:
        Dict mapping node_id to size
    """
    values = []
    for node, data in graph.nodes(data=True):
        values.append((str(node), data.get(measure, 0)))

    if not values:
        return {}

    scores = [v[1] for v in values]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score > min_score else 1

    node_sizes: dict[str, int] = {}
    for node_id, score in values:
        normalized = (score - min_score) / score_range
        size = int(min_size + normalized * (max_size - min_size))
        node_sizes[node_id] = size

    return node_sizes


# =============================================================================
# Static Visualization (Matplotlib)
# =============================================================================


def to_matplotlib(
    graph: nx.Graph,
    output_path: Path,
    figsize: tuple[int, int] = (20, 20),
    layout: str = "spring",
    show_labels: bool = True,
    label_top_n: int = 20,
) -> Path:
    """
    Create static PNG visualization using Matplotlib.

    Args:
        graph: NetworkX graph
        output_path: Output PNG file path
        figsize: Figure size in inches
        layout: Layout algorithm ('spring', 'kamada_kawai', 'spectral')
        show_labels: Whether to show node labels
        label_top_n: Only label top N nodes by centrality

    Returns:
        Path to generated PNG file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for static visualization. "
            "Install with: pip install matplotlib"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute layout
    pos = compute_layout(graph, layout)

    # Get node colors and sizes
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        data = graph.nodes[node]
        color = data.get("color", "#97c2fc")
        size = data.get("size", 300)
        node_colors.append(color)
        node_sizes.append(size)

    # Get edge weights for line widths
    edge_weights = [d.get("weight", 0.5) * 3 for _, _, d in graph.edges(data=True)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#222222")
    fig.set_facecolor("#222222")

    # Draw edges
    nx.draw_networkx_edges(
        graph, pos, alpha=0.5, width=edge_weights, edge_color="gray", ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
    )

    # Draw labels
    if show_labels:
        # Get top N nodes by degree if we need to limit labels
        if label_top_n and label_top_n < graph.number_of_nodes():
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:label_top_n]
            labels = {n: graph.nodes[n].get("title", str(n)) for n in top_nodes}
        else:
            labels = {n: graph.nodes[n].get("title", str(n)) for n in graph.nodes()}

        nx.draw_networkx_labels(
            graph, pos, labels, font_size=8, font_color="white", ax=ax
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    return output_path


def compute_layout(
    graph: nx.Graph,
    algorithm: str = "spring",
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """
    Compute node positions using layout algorithm.

    Args:
        graph: NetworkX graph
        algorithm: Layout algorithm name
        **kwargs: Algorithm-specific parameters

    Returns:
        Dict mapping node_id to (x, y) position
    """
    if algorithm == "spring":
        pos = nx.spring_layout(
            graph, k=kwargs.get("k", 2), iterations=kwargs.get("iterations", 50)
        )
    elif algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif algorithm == "spectral":
        pos = nx.spectral_layout(graph)
    elif algorithm == "circular":
        pos = nx.circular_layout(graph)
    elif algorithm == "shell":
        pos = nx.shell_layout(graph)
    else:
        raise ValueError(f"Unknown layout algorithm: {algorithm}")

    return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


# =============================================================================
# D3.js Export
# =============================================================================


def to_d3_json(graph: nx.Graph, output_path: Path) -> Path:
    """
    Export graph in D3.js force-directed format.

    Format:
    {
        "nodes": [{"id": "...", "name": "...", "group": 0, "size": 10}],
        "links": [{"source": "...", "target": "...", "value": 0.5}]
    }

    Args:
        graph: NetworkX graph
        output_path: Output JSON file path

    Returns:
        Path to generated JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = []
    for node, data in graph.nodes(data=True):
        nodes.append(
            {
                "id": str(node),
                "name": data.get("title", str(node)),
                "group": data.get("community", 0),
                "size": data.get("size", 10),
            }
        )

    links = []
    for u, v, data in graph.edges(data=True):
        links.append(
            {"source": str(u), "target": str(v), "value": data.get("weight", 1.0)}
        )

    d3_data = {"nodes": nodes, "links": links}
    output_path.write_bytes(orjson.dumps(d3_data, option=orjson.OPT_INDENT_2))

    return output_path


# =============================================================================
# Obsidian Export
# =============================================================================


def to_obsidian(
    graph: nx.Graph,
    vault_path: Path,
    output_name: str = "graph_data.json",
) -> tuple[Path, Path]:
    """
    Export graph data compatible with Obsidian plugins.

    Creates:
    1. JSON file with graph data
    2. Markdown summary file

    Args:
        graph: NetworkX graph
        vault_path: Path to Obsidian vault
        output_name: Name of JSON output file

    Returns:
        Tuple of (json_path, markdown_path)
    """
    vault_path = Path(vault_path)
    vault_path.mkdir(parents=True, exist_ok=True)

    # Export JSON
    json_path = vault_path / output_name
    to_d3_json(graph, json_path)

    # Generate and save markdown summary
    md_content = generate_obsidian_summary(graph)
    md_path = vault_path / "graph_summary.md"
    md_path.write_text(md_content)

    return json_path, md_path


def generate_obsidian_summary(
    graph: nx.Graph,
    communities: list[set[str]] | None = None,
) -> str:
    """
    Generate Markdown summary for Obsidian.

    Args:
        graph: NetworkX graph
        communities: Optional community sets

    Returns:
        Markdown string
    """
    lines = [
        "# Knowledge Graph Summary",
        "",
        "## Overview",
        f"- **Nodes:** {graph.number_of_nodes()}",
        f"- **Edges:** {graph.number_of_edges()}",
        f"- **Density:** {nx.density(graph):.4f}",
        "",
    ]

    # Top connected nodes
    lines.append("## Most Connected Notes")
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, degree in top_nodes:
        title = graph.nodes[node].get("title", node)
        lines.append(f"- [[{title}]] ({degree} connections)")

    lines.append("")

    # Communities if available
    if communities:
        lines.append("## Communities")
        lines.append(f"Found {len(communities)} communities:")
        for i, community in enumerate(communities[:5]):
            lines.append(f"\n### Community {i + 1} ({len(community)} notes)")
            for node in list(community)[:5]:
                title = graph.nodes[node].get("title", node) if node in graph else node
                lines.append(f"- [[{title}]]")
            if len(community) > 5:
                lines.append(f"- *...and {len(community) - 5} more*")

    return "\n".join(lines)


# =============================================================================
# Subgraph Visualization
# =============================================================================


def visualize_neighborhood(
    graph: nx.Graph,
    node_id: str,
    radius: int = 2,
    output_path: Path | None = None,
) -> Path | None:
    """
    Visualize neighborhood around a specific node.

    Args:
        graph: NetworkX graph
        node_id: Center node
        radius: Hops to include
        output_path: Output path (None for display)

    Returns:
        Output path if saved
    """
    from semlink.core.graph import subgraph_around_node

    subgraph = subgraph_around_node(graph, node_id, radius)

    # Highlight center node
    if node_id in subgraph:
        subgraph.nodes[node_id]["color"] = "#ff0000"
        subgraph.nodes[node_id]["size"] = 40

    if output_path:
        return to_pyvis(subgraph, output_path)
    return None


def visualize_community(
    graph: nx.Graph,
    community_id: int,
    output_path: Path | None = None,
) -> Path | None:
    """
    Visualize a single community.

    Args:
        graph: NetworkX graph with 'community' attribute
        community_id: Community to visualize
        output_path: Output path (None for display)

    Returns:
        Output path if saved
    """
    # Get nodes in community
    community_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("community") == community_id
    ]

    if not community_nodes:
        return None

    subgraph = graph.subgraph(community_nodes).copy()

    if output_path:
        return to_pyvis(subgraph, output_path)
    return None
