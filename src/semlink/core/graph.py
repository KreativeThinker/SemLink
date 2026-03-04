"""
Graph Construction Module.

This module handles building, manipulating, and exporting
knowledge graphs from inferred links.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import orjson

from semlink.core.linker import Edge


def build_graph(
    nodes: list[dict[str, Any]],
    edges: list[Edge],
) -> nx.Graph:
    """
    Build NetworkX graph from nodes and edges.

    Args:
        nodes: List of node dictionaries with 'id' and attributes
        edges: List of Edge objects

    Returns:
        NetworkX Graph object
    """
    graph = nx.Graph()

    # Add nodes with attributes
    for node in nodes:
        node_id = node.get("id")
        if node_id is None:
            continue
        attrs = {k: v for k, v in node.items() if k != "id"}
        graph.add_node(node_id, **attrs)

    # Add edges with all available attributes
    for edge in edges:
        edge_attrs = {
            "weight": edge.weight,
            "method": edge.method,
        }
        if edge.reason:
            edge_attrs["reason"] = edge.reason
        if edge.shared_terms:
            edge_attrs["shared_terms"] = edge.shared_terms

        graph.add_edge(edge.source, edge.target, **edge_attrs)

    return graph


def add_node_attributes(
    graph: nx.Graph,
    attributes: dict[str, dict[str, Any]],
) -> nx.Graph:
    """
    Add attributes to existing nodes.

    Args:
        graph: NetworkX graph
        attributes: Dict mapping node_id to attribute dict

    Returns:
        Modified graph
    """
    for node_id, attrs in attributes.items():
        if node_id in graph:
            for key, value in attrs.items():
                graph.nodes[node_id][key] = value

    return graph


def get_node_attributes(graph: nx.Graph, node_id: str) -> dict[str, Any]:
    """
    Get all attributes for a node.

    Args:
        graph: NetworkX graph
        node_id: Node identifier

    Returns:
        Dictionary of node attributes
    """
    if node_id not in graph:
        return {}
    return dict(graph.nodes[node_id])


def get_neighbors(
    graph: nx.Graph,
    node_id: str,
    include_weights: bool = True,
) -> list[tuple[str, float] | str]:
    """
    Get neighbors of a node.

    Args:
        graph: NetworkX graph
        node_id: Node identifier
        include_weights: Include edge weights

    Returns:
        List of neighbor IDs or (neighbor_id, weight) tuples
    """
    if node_id not in graph:
        return []

    if include_weights:
        return [
            (neighbor, graph[node_id][neighbor].get("weight", 1.0))
            for neighbor in graph.neighbors(node_id)
        ]
    return list(graph.neighbors(node_id))


def subgraph_around_node(
    graph: nx.Graph,
    node_id: str,
    radius: int = 2,
) -> nx.Graph:
    """
    Extract subgraph within radius of a node.

    Args:
        graph: Source graph
        node_id: Center node
        radius: Number of hops

    Returns:
        Subgraph containing nodes within radius
    """
    if node_id not in graph:
        return nx.Graph()

    # Get all nodes within radius using BFS
    nodes_in_radius = {node_id}
    current_frontier = {node_id}

    for _ in range(radius):
        next_frontier = set()
        for node in current_frontier:
            for neighbor in graph.neighbors(node):
                if neighbor not in nodes_in_radius:
                    next_frontier.add(neighbor)
                    nodes_in_radius.add(neighbor)
        current_frontier = next_frontier

    return graph.subgraph(nodes_in_radius).copy()


def validate_graph(graph: nx.Graph) -> dict[str, Any]:
    """
    Validate graph structure and return diagnostics.

    Returns:
        Dictionary with validation results
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # Check connectivity
    if n_nodes == 0:
        n_components = 0
        largest_component_size = 0
    else:
        components = list(nx.connected_components(graph))
        n_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0

    # Check for self-loops
    self_loops = list(nx.selfloop_edges(graph))

    # Degree distribution
    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    # Edge weights
    weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]

    return {
        "valid": n_nodes > 0 and n_edges >= 0,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "largest_component_size": largest_component_size,
        "is_connected": n_components == 1,
        "has_self_loops": len(self_loops) > 0,
        "self_loop_count": len(self_loops),
        "avg_degree": round(avg_degree, 2),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_weight": round(sum(weights) / len(weights), 4) if weights else 0,
        "density": nx.density(graph) if n_nodes > 0 else 0,
    }


# =============================================================================
# Export Functions
# =============================================================================


def export_json(graph: nx.Graph, path: Path) -> None:
    """
    Export graph to JSON format (node-link data).

    Compatible with web visualization and Obsidian plugins.

    Args:
        graph: NetworkX graph
        path: Output file path
    """
    from networkx.readwrite import json_graph

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json_graph.node_link_data(graph)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def export_graphml(graph: nx.Graph, path: Path) -> None:
    """
    Export graph to GraphML format.

    Compatible with Gephi and other graph tools.

    Args:
        graph: NetworkX graph
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(path))


def export_gexf(graph: nx.Graph, path: Path) -> None:
    """
    Export graph to GEXF format.

    Best for Gephi import with dynamic attributes.

    Args:
        graph: NetworkX graph
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, str(path))


def export_edge_list(graph: nx.Graph, path: Path) -> None:
    """
    Export graph as simple edge list CSV.

    Args:
        graph: NetworkX graph
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["source,target,weight"]
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        lines.append(f"{u},{v},{weight}")

    path.write_text("\n".join(lines))


# =============================================================================
# Import Functions
# =============================================================================


def load_json(path: Path) -> nx.Graph:
    """
    Load graph from JSON file.

    Args:
        path: Input file path

    Returns:
        NetworkX graph
    """
    from networkx.readwrite import json_graph

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    data = orjson.loads(path.read_bytes())
    return json_graph.node_link_graph(data)


def load_graphml(path: Path) -> nx.Graph:
    """
    Load graph from GraphML file.

    Args:
        path: Input file path

    Returns:
        NetworkX graph
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    return nx.read_graphml(str(path))


# =============================================================================
# Graph Manipulation
# =============================================================================


def merge_graphs(graphs: list[nx.Graph]) -> nx.Graph:
    """
    Merge multiple graphs into one.

    Args:
        graphs: List of graphs to merge

    Returns:
        Combined graph
    """
    if not graphs:
        return nx.Graph()

    merged = nx.Graph()
    for graph in graphs:
        merged = nx.compose(merged, graph)

    return merged


def prune_graph(
    graph: nx.Graph,
    min_degree: int = 1,
    min_weight: float | None = None,
) -> nx.Graph:
    """
    Remove low-degree nodes and weak edges.

    Args:
        graph: Input graph
        min_degree: Minimum node degree to keep
        min_weight: Minimum edge weight to keep

    Returns:
        Pruned graph
    """
    pruned = graph.copy()

    # Remove weak edges first
    if min_weight is not None:
        edges_to_remove = [
            (u, v)
            for u, v, d in pruned.edges(data=True)
            if d.get("weight", 1.0) < min_weight
        ]
        pruned.remove_edges_from(edges_to_remove)

    # Remove low-degree nodes iteratively
    changed = True
    while changed:
        changed = False
        nodes_to_remove = [n for n, d in pruned.degree() if d < min_degree]
        if nodes_to_remove:
            pruned.remove_nodes_from(nodes_to_remove)
            changed = True

    return pruned


def largest_component(graph: nx.Graph) -> nx.Graph:
    """
    Extract largest connected component.

    Args:
        graph: Input graph

    Returns:
        Subgraph of largest component
    """
    if graph.number_of_nodes() == 0:
        return nx.Graph()

    components = list(nx.connected_components(graph))
    if not components:
        return nx.Graph()

    largest = max(components, key=len)
    return graph.subgraph(largest).copy()
