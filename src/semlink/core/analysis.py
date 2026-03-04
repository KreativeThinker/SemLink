"""
Graph Analysis Module.

This module provides functions for analyzing graph structure,
detecting communities, and computing centrality measures.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def compute_metrics(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute comprehensive graph metrics.

    Metrics include:
    - Node and edge counts
    - Density
    - Number of connected components
    - Average clustering coefficient
    - Transitivity
    - Diameter (for connected graphs)
    - Average path length (for connected graphs)

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary of metrics
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0,
            "n_components": 0,
            "avg_clustering": 0,
            "transitivity": 0,
        }

    components = list(nx.connected_components(graph))
    n_components = len(components)
    is_connected = n_components == 1

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph),
        "n_components": n_components,
        "is_connected": is_connected,
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
    }

    # Compute diameter and avg path length only for connected graphs
    if is_connected and n_nodes > 1:
        metrics["diameter"] = nx.diameter(graph)
        metrics["avg_path_length"] = nx.average_shortest_path_length(graph)
    else:
        # Compute for largest component
        if components:
            largest = max(components, key=len)
            if len(largest) > 1:
                subg = graph.subgraph(largest)
                metrics["largest_component_diameter"] = nx.diameter(subg)
                metrics["largest_component_avg_path"] = nx.average_shortest_path_length(
                    subg
                )

    return metrics


def degree_distribution(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute degree distribution statistics.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with min, max, mean, median, histogram
    """
    degrees = [d for _, d in graph.degree()]

    if not degrees:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "histogram": {}}

    return {
        "min": min(degrees),
        "max": max(degrees),
        "mean": float(np.mean(degrees)),
        "median": float(np.median(degrees)),
        "std": float(np.std(degrees)),
        "histogram": dict(zip(*np.unique(degrees, return_counts=True))),
    }


def weight_distribution(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute edge weight distribution statistics.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with weight statistics
    """
    weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]

    if not weights:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}

    return {
        "min": float(min(weights)),
        "max": float(max(weights)),
        "mean": float(np.mean(weights)),
        "median": float(np.median(weights)),
        "std": float(np.std(weights)),
    }


# =============================================================================
# Community Detection
# =============================================================================


def detect_communities_louvain(
    graph: nx.Graph,
    resolution: float = 1.0,
    weight: str = "weight",
) -> list[set[str]]:
    """
    Detect communities using Louvain algorithm.

    Args:
        graph: NetworkX graph
        resolution: Resolution parameter (higher = more communities)
        weight: Edge attribute to use as weight

    Returns:
        List of sets, each containing node IDs in a community
    """
    from networkx.algorithms.community import louvain_communities

    communities = louvain_communities(graph, resolution=resolution, weight=weight)
    return [set(str(n) for n in c) for c in communities]


def detect_communities_leiden(
    graph: nx.Graph,
    resolution: float = 1.0,
) -> list[set[str]]:
    """
    Detect communities using Leiden algorithm.

    Requires leidenalg package (optional dependency).

    Args:
        graph: NetworkX graph
        resolution: Resolution parameter

    Returns:
        List of community sets
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "Leiden algorithm requires 'leidenalg' and 'igraph' packages. "
            "Install with: pip install leidenalg python-igraph"
        )

    # Convert to igraph
    ig_graph = ig.Graph.from_networkx(graph)

    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
    )

    # Convert back to node IDs
    node_list = list(graph.nodes())
    communities = []
    for community_indices in partition:
        community = {str(node_list[i]) for i in community_indices}
        communities.append(community)

    return communities


def detect_communities_label_propagation(graph: nx.Graph) -> list[set[str]]:
    """
    Detect communities using label propagation.

    Fastest method but lower quality.

    Args:
        graph: NetworkX graph

    Returns:
        List of community sets
    """
    from networkx.algorithms.community import label_propagation_communities

    communities = list(label_propagation_communities(graph))
    return [set(str(n) for n in c) for c in communities]


def add_community_labels(
    graph: nx.Graph,
    communities: list[set[str]],
    attribute: str = "community",
) -> nx.Graph:
    """
    Add community membership as node attribute.

    Args:
        graph: NetworkX graph
        communities: List of community sets
        attribute: Name of attribute to add

    Returns:
        Modified graph
    """
    for i, community in enumerate(communities):
        for node in community:
            if node in graph:
                graph.nodes[node][attribute] = i

    return graph


def community_summary(
    graph: nx.Graph,
    communities: list[set[str]],
) -> list[dict[str, Any]]:
    """
    Generate summary statistics for each community.

    Args:
        graph: NetworkX graph
        communities: List of community sets

    Returns:
        List of dicts with community stats
    """
    summaries = []

    for i, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()

        # Internal density
        internal_density = nx.density(subgraph) if n_nodes > 1 else 0

        # External edges
        external_edges = 0
        for node in community:
            for neighbor in graph.neighbors(node):
                if neighbor not in community:
                    external_edges += 1

        summaries.append(
            {
                "community_id": i,
                "n_nodes": n_nodes,
                "n_internal_edges": n_edges,
                "n_external_edges": external_edges,
                "internal_density": internal_density,
                "nodes": list(community)[:10],  # Sample of nodes
            }
        )

    return summaries


# =============================================================================
# Centrality Measures
# =============================================================================


def compute_centrality(
    graph: nx.Graph,
    measures: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute centrality measures for all nodes.

    Available measures:
    - 'degree': Number of connections
    - 'betweenness': Bridge importance
    - 'pagerank': Link-based importance
    - 'eigenvector': Connected to important nodes

    Args:
        graph: NetworkX graph
        measures: List of measures to compute (None = all)

    Returns:
        Dict mapping measure name to {node_id: score}
    """
    available = ["degree", "betweenness", "pagerank", "eigenvector"]
    if measures is None:
        measures = available

    result: dict[str, dict[str, float]] = {}

    for measure in measures:
        if measure == "degree":
            centrality = nx.degree_centrality(graph)
        elif measure == "betweenness":
            centrality = nx.betweenness_centrality(graph)
        elif measure == "pagerank":
            centrality = nx.pagerank(graph)
        elif measure == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                centrality = nx.eigenvector_centrality_numpy(graph)
        else:
            raise ValueError(f"Unknown centrality measure: {measure}")

        result[measure] = {str(k): float(v) for k, v in centrality.items()}

    return result


def add_centrality_attributes(
    graph: nx.Graph,
    centrality: dict[str, dict[str, float]],
) -> nx.Graph:
    """
    Add centrality scores as node attributes.

    Args:
        graph: NetworkX graph
        centrality: Output from compute_centrality

    Returns:
        Modified graph
    """
    for measure_name, scores in centrality.items():
        for node_id, score in scores.items():
            if node_id in graph:
                graph.nodes[node_id][measure_name] = score

    return graph


def top_nodes_by_centrality(
    graph: nx.Graph,
    measure: str = "pagerank",
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Get top nodes by centrality measure.

    Args:
        graph: NetworkX graph
        measure: Centrality measure name
        top_n: Number of nodes to return

    Returns:
        List of (node_id, score) tuples
    """
    centrality = compute_centrality(graph, measures=[measure])
    scores = centrality[measure]

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:top_n]


# =============================================================================
# Bridge and Cluster Analysis
# =============================================================================


def find_bridges(graph: nx.Graph) -> list[tuple[str, str]]:
    """
    Find bridge edges (whose removal disconnects graph).

    Args:
        graph: NetworkX graph

    Returns:
        List of (source, target) edge tuples
    """
    return [(str(u), str(v)) for u, v in nx.bridges(graph)]


def find_articulation_points(graph: nx.Graph) -> list[str]:
    """
    Find articulation points (whose removal disconnects graph).

    Args:
        graph: NetworkX graph

    Returns:
        List of node IDs
    """
    return [str(n) for n in nx.articulation_points(graph)]


def cluster_coefficient_by_node(graph: nx.Graph) -> dict[str, float]:
    """
    Compute clustering coefficient for each node.

    Args:
        graph: NetworkX graph

    Returns:
        Dict mapping node_id to clustering coefficient
    """
    clustering = nx.clustering(graph)
    return {str(k): float(v) for k, v in clustering.items()}


# =============================================================================
# Analysis Report
# =============================================================================


def generate_analysis_report(
    graph: nx.Graph,
    include_communities: bool = True,
    include_centrality: bool = True,
) -> dict[str, Any]:
    """
    Generate comprehensive analysis report.

    Args:
        graph: NetworkX graph
        include_communities: Run community detection
        include_centrality: Compute centrality measures

    Returns:
        Complete analysis dictionary
    """
    report: dict[str, Any] = {
        "metrics": compute_metrics(graph),
        "degree_distribution": degree_distribution(graph),
        "weight_distribution": weight_distribution(graph),
    }

    if include_communities:
        try:
            communities = detect_communities_louvain(graph)
            report["communities"] = {
                "method": "louvain",
                "n_communities": len(communities),
                "summary": community_summary(graph, communities),
            }
        except Exception as e:
            report["communities"] = {"error": str(e)}

    if include_centrality:
        try:
            centrality = compute_centrality(graph)
            report["centrality"] = {
                "measures": list(centrality.keys()),
                "top_by_pagerank": top_nodes_by_centrality(graph, "pagerank", 5),
                "top_by_betweenness": top_nodes_by_centrality(graph, "betweenness", 5),
            }
        except Exception as e:
            report["centrality"] = {"error": str(e)}

    # Structural features
    report["bridges"] = find_bridges(graph)
    report["articulation_points"] = find_articulation_points(graph)

    return report
