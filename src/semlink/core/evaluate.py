"""
Evaluation and Comparison Module.

This module provides functions for comparing different similarity
methods and evaluating graph quality.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Similarity Distribution Analysis
# =============================================================================


def similarity_distribution(
    similarity_matrix: NDArray[np.float32],
) -> dict[str, Any]:
    """
    Compute distribution statistics for similarity matrix.

    Args:
        similarity_matrix: Pairwise similarity matrix

    Returns:
        Dictionary with distribution statistics
    """
    # Get upper triangle (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    if len(upper_tri) == 0:
        return {"n_pairs": 0}

    return {
        "n_pairs": len(upper_tri),
        "min": float(np.min(upper_tri)),
        "max": float(np.max(upper_tri)),
        "mean": float(np.mean(upper_tri)),
        "median": float(np.median(upper_tri)),
        "std": float(np.std(upper_tri)),
        "percentile_25": float(np.percentile(upper_tri, 25)),
        "percentile_75": float(np.percentile(upper_tri, 75)),
        "percentile_90": float(np.percentile(upper_tri, 90)),
        "percentile_95": float(np.percentile(upper_tri, 95)),
    }


def compare_similarity_distributions(
    matrices: dict[str, NDArray[np.float32]],
) -> dict[str, Any]:
    """
    Compare similarity distributions across methods.

    Args:
        matrices: Dict mapping method name to similarity matrix

    Returns:
        Comparison statistics
    """
    results: dict[str, Any] = {"methods": list(matrices.keys())}

    # Compute stats for each method
    for name, matrix in matrices.items():
        results[name] = similarity_distribution(matrix)

    # Compare rankings at different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    threshold_counts: dict[str, dict[float, int]] = {}

    for name, matrix in matrices.items():
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        threshold_counts[name] = {t: int(np.sum(upper_tri >= t)) for t in thresholds}

    results["threshold_counts"] = threshold_counts

    return results


def plot_similarity_histogram(
    matrices: dict[str, NDArray[np.float32]],
    output_path: Path,
    bins: int = 50,
) -> Path:
    """
    Plot overlaid similarity histograms for multiple methods.

    Args:
        matrices: Dict mapping method name to similarity matrix
        output_path: Output PNG path
        bins: Number of histogram bins

    Returns:
        Output path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, matrix in matrices.items():
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
        ax.hist(upper_tri, bins=bins, alpha=0.5, label=name, density=True)

    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title("Similarity Distribution by Method")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


# =============================================================================
# Graph Quality Metrics
# =============================================================================


def evaluate_graph_quality(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute graph quality metrics.

    Metrics:
    - Modularity (if communities assigned)
    - Average clustering coefficient
    - Edge weight statistics
    - Connectivity metrics

    Args:
        graph: NetworkX graph

    Returns:
        Quality metrics dictionary
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return {"n_nodes": 0, "n_edges": 0}

    # Basic metrics
    metrics: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": nx.density(graph),
        "avg_clustering": nx.average_clustering(graph),
        "transitivity": nx.transitivity(graph),
    }

    # Connectivity
    if n_nodes > 0:
        components = list(nx.connected_components(graph))
        metrics["n_components"] = len(components)
        metrics["largest_component_size"] = max(len(c) for c in components)
        metrics["is_connected"] = len(components) == 1

    # Degree statistics
    degrees = [d for _, d in graph.degree()]
    metrics["avg_degree"] = float(np.mean(degrees))
    metrics["max_degree"] = max(degrees)

    # Edge weight statistics
    weights = [d.get("weight", 1.0) for _, _, d in graph.edges(data=True)]
    if weights:
        metrics["avg_weight"] = float(np.mean(weights))
        metrics["min_weight"] = float(min(weights))
        metrics["max_weight"] = float(max(weights))

    # Modularity if communities exist
    community_attr = "community"
    if any(community_attr in graph.nodes[n] for n in graph.nodes()):
        communities_dict: dict[int, set] = {}
        for n, d in graph.nodes(data=True):
            comm_id = d.get(community_attr, 0)
            if comm_id not in communities_dict:
                communities_dict[comm_id] = set()
            communities_dict[comm_id].add(n)
        communities = list(communities_dict.values())
        metrics["modularity"] = nx.community.modularity(graph, communities)

    return metrics


def compare_graphs(
    graphs: dict[str, nx.Graph],
) -> dict[str, Any]:
    """
    Compare multiple graphs across metrics.

    Args:
        graphs: Dict mapping method name to graph

    Returns:
        Comparison table as dictionary
    """
    results: dict[str, Any] = {"methods": list(graphs.keys())}

    for name, graph in graphs.items():
        results[name] = evaluate_graph_quality(graph)

    return results


# =============================================================================
# Link Quality Assessment
# =============================================================================


def sample_links(
    graph: nx.Graph,
    n_samples: int = 20,
    strategy: str = "random",
) -> list[tuple[str, str, float]]:
    """
    Sample links for qualitative inspection.

    Strategies:
    - 'random': Random sample
    - 'high_weight': Highest weight links
    - 'low_weight': Lowest weight links
    - 'stratified': Sample from weight ranges

    Args:
        graph: NetworkX graph
        n_samples: Number of links to sample
        strategy: Sampling strategy

    Returns:
        List of (source, target, weight) tuples
    """
    if graph.number_of_edges() == 0:
        return []

    # Get all edges with weights
    edges = [(u, v, d.get("weight", 1.0)) for u, v, d in graph.edges(data=True)]

    n_samples = min(n_samples, len(edges))

    if strategy == "random":
        sampled = random.sample(edges, n_samples)

    elif strategy == "high_weight":
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        sampled = sorted_edges[:n_samples]

    elif strategy == "low_weight":
        sorted_edges = sorted(edges, key=lambda x: x[2])
        sampled = sorted_edges[:n_samples]

    elif strategy == "stratified":
        # Divide into weight ranges and sample from each
        sorted_edges = sorted(edges, key=lambda x: x[2])
        n_strata = min(4, n_samples)
        samples_per_stratum = n_samples // n_strata
        remainder = n_samples % n_strata

        strata_size = len(sorted_edges) // n_strata
        sampled = []

        for i in range(n_strata):
            start = i * strata_size
            end = start + strata_size if i < n_strata - 1 else len(sorted_edges)
            stratum = sorted_edges[start:end]

            # Add extra sample to first strata if remainder
            n_from_stratum = samples_per_stratum + (1 if i < remainder else 0)
            n_from_stratum = min(n_from_stratum, len(stratum))

            if stratum:
                sampled.extend(random.sample(stratum, n_from_stratum))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sampled


def link_coherence_score(
    graph: nx.Graph,
    node_texts: dict[str, str],
    embedder: Any,
) -> float:
    """
    Compute semantic coherence of linked nodes.

    Measures whether linked nodes are actually semantically similar.

    Args:
        graph: NetworkX graph
        node_texts: Dict mapping node_id to text content
        embedder: Embedding model for computing similarity

    Returns:
        Average coherence score (0-1)
    """
    if graph.number_of_edges() == 0:
        return 0.0

    # Get edges with both nodes having text
    valid_edges = [
        (u, v) for u, v in graph.edges() if u in node_texts and v in node_texts
    ]

    if not valid_edges:
        return 0.0

    # Collect unique node IDs from valid edges
    unique_nodes = list(set(n for edge in valid_edges for n in edge))
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}

    # Get texts and compute embeddings
    texts = [node_texts[n] for n in unique_nodes]

    # Check if embedder has encode method (SBERT/OpenAI style) or fit_transform (TF-IDF)
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(texts)
    elif hasattr(embedder, "fit_transform"):
        embeddings = embedder.fit_transform(texts)
    else:
        raise ValueError("Embedder must have 'encode' or 'fit_transform' method")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Compute coherence as average similarity of linked pairs
    coherence_scores = []
    for u, v in valid_edges:
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        similarity = float(np.dot(normalized[idx_u], normalized[idx_v]))
        coherence_scores.append(similarity)

    return float(np.mean(coherence_scores))


# =============================================================================
# Method Comparison
# =============================================================================


def compare_methods(
    embeddings: dict[str, NDArray[np.float32]],
    ids: list[str],
    link_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare different embedding methods end-to-end.

    Builds graphs for each method and compares:
    - Similarity distributions
    - Graph structure metrics
    - Overlap in inferred links

    Args:
        embeddings: Dict mapping method name to embedding matrix
        ids: Document IDs
        link_params: Parameters for link inference

    Returns:
        Comprehensive comparison results
    """
    from semlink.core.graph import build_graph
    from semlink.core.linker import HybridStrategy

    if link_params is None:
        link_params = {}

    # Default link parameters
    threshold = link_params.get("threshold", 0.5)
    k = link_params.get("k", 10)

    results: dict[str, Any] = {
        "methods": list(embeddings.keys()),
        "n_docs": len(ids),
        "link_params": {"threshold": threshold, "k": k},
    }

    # Compute similarity matrices
    similarity_matrices: dict[str, NDArray[np.float32]] = {}
    for name, emb in embeddings.items():
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = emb / norms
        similarity_matrices[name] = np.dot(normalized, normalized.T).astype(np.float32)

    # Compare similarity distributions
    results["similarity_distributions"] = compare_similarity_distributions(
        similarity_matrices
    )

    # Build graphs for each method
    graphs: dict[str, nx.Graph] = {}
    linker = HybridStrategy(threshold=threshold, k=k)

    for name, emb in embeddings.items():
        edges = linker.infer_links(emb, ids)
        nodes = [{"id": doc_id} for doc_id in ids]
        graphs[name] = build_graph(nodes, edges)

    results["graphs"] = {name: evaluate_graph_quality(g) for name, g in graphs.items()}

    # Compute pairwise overlaps if multiple methods
    method_names = list(embeddings.keys())
    if len(method_names) >= 2:
        overlaps: dict[str, Any] = {}
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i + 1 :]:
                key = f"{m1}_vs_{m2}"
                overlaps[key] = link_overlap(graphs[m1], graphs[m2])
        results["link_overlaps"] = overlaps

    return results


def link_overlap(
    graph1: nx.Graph,
    graph2: nx.Graph,
) -> dict[str, Any]:
    """
    Compute overlap between two graphs' edges.

    Args:
        graph1: First graph
        graph2: Second graph

    Returns:
        Overlap statistics (Jaccard, common edges, etc.)
    """
    # Convert edges to frozensets for undirected comparison
    edges1 = {frozenset((u, v)) for u, v in graph1.edges()}
    edges2 = {frozenset((u, v)) for u, v in graph2.edges()}

    intersection = edges1 & edges2
    union = edges1 | edges2

    n_common = len(intersection)
    n_only_1 = len(edges1 - edges2)
    n_only_2 = len(edges2 - edges1)
    n_union = len(union)

    # Jaccard similarity
    jaccard = n_common / n_union if n_union > 0 else 0.0

    # Overlap coefficient (normalized by smaller set)
    min_size = min(len(edges1), len(edges2))
    overlap_coef = n_common / min_size if min_size > 0 else 0.0

    return {
        "n_edges_graph1": len(edges1),
        "n_edges_graph2": len(edges2),
        "n_common": n_common,
        "n_only_graph1": n_only_1,
        "n_only_graph2": n_only_2,
        "jaccard_similarity": jaccard,
        "overlap_coefficient": overlap_coef,
    }


# =============================================================================
# Reporting
# =============================================================================


def generate_comparison_report(
    results: dict[str, Any],
    output_path: Path,
    format: str = "markdown",
) -> Path:
    """
    Generate formatted comparison report.

    Args:
        results: Output from compare_methods
        output_path: Output file path
        format: 'markdown' or 'html'

    Returns:
        Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = results.get("methods", [])
    n_docs = results.get("n_docs", 0)
    link_params = results.get("link_params", {})

    if format == "markdown":
        lines = [
            "# SemLink Method Comparison Report",
            "",
            "## Overview",
            "",
            f"- **Number of documents**: {n_docs}",
            f"- **Methods compared**: {', '.join(methods)}",
            f"- **Link parameters**: threshold={link_params.get('threshold', 'N/A')}, k={link_params.get('k', 'N/A')}",
            "",
            "## Similarity Distributions",
            "",
        ]

        # Add similarity stats table
        sim_dist = results.get("similarity_distributions", {})
        if sim_dist and methods:
            lines.append("| Method | Mean | Median | Std | 90th Percentile |")
            lines.append("|--------|------|--------|-----|-----------------|")

            for method in methods:
                stats = sim_dist.get(method, {})
                mean = stats.get("mean", 0)
                median = stats.get("median", 0)
                std = stats.get("std", 0)
                p90 = stats.get("percentile_90", 0)
                lines.append(
                    f"| {method} | {mean:.4f} | {median:.4f} | {std:.4f} | {p90:.4f} |"
                )

            lines.append("")

        # Add graph metrics table
        lines.append("## Graph Metrics")
        lines.append("")

        graphs = results.get("graphs", {})
        if graphs and methods:
            lines.append(
                "| Method | Nodes | Edges | Density | Avg Degree | Avg Clustering |"
            )
            lines.append(
                "|--------|-------|-------|---------|------------|----------------|"
            )

            for method in methods:
                g = graphs.get(method, {})
                nodes = g.get("n_nodes", 0)
                edges = g.get("n_edges", 0)
                density = g.get("density", 0)
                avg_deg = g.get("avg_degree", 0)
                avg_clust = g.get("avg_clustering", 0)
                lines.append(
                    f"| {method} | {nodes} | {edges} | {density:.4f} | {avg_deg:.2f} | {avg_clust:.4f} |"
                )

            lines.append("")

        # Add link overlaps
        overlaps = results.get("link_overlaps", {})
        if overlaps:
            lines.append("## Link Overlaps")
            lines.append("")
            lines.append("| Comparison | Common | Jaccard | Overlap Coef |")
            lines.append("|------------|--------|---------|--------------|")

            for key, overlap in overlaps.items():
                common = overlap.get("n_common", 0)
                jaccard = overlap.get("jaccard_similarity", 0)
                overlap_coef = overlap.get("overlap_coefficient", 0)
                lines.append(
                    f"| {key} | {common} | {jaccard:.4f} | {overlap_coef:.4f} |"
                )

            lines.append("")

        content = "\n".join(lines)

    elif format == "html":
        # Simple HTML wrapper around markdown content
        md_report = generate_comparison_report(results, output_path, "markdown")
        with open(md_report) as f:
            md_content = f.read()

        content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SemLink Comparison Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
    </style>
</head>
<body>
<pre>{md_content}</pre>
</body>
</html>"""

    else:
        raise ValueError(f"Unknown format: {format}. Use 'markdown' or 'html'")

    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def generate_plots(
    results: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """
    Generate all comparison plots.

    Creates:
    - Similarity distribution histograms
    - Graph metric comparison bar charts
    - Edge weight boxplots

    Args:
        results: Comparison results
        output_dir: Directory for outputs

    Returns:
        List of generated file paths
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    methods = results.get("methods", [])

    if not methods:
        return generated

    # 1. Graph metrics bar chart
    graphs = results.get("graphs", {})
    if graphs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Number of edges
        ax = axes[0, 0]
        values = [graphs.get(m, {}).get("n_edges", 0) for m in methods]
        ax.bar(methods, values, color="steelblue")
        ax.set_title("Number of Edges")
        ax.set_ylabel("Edges")

        # Density
        ax = axes[0, 1]
        values = [graphs.get(m, {}).get("density", 0) for m in methods]
        ax.bar(methods, values, color="coral")
        ax.set_title("Graph Density")
        ax.set_ylabel("Density")

        # Average degree
        ax = axes[1, 0]
        values = [graphs.get(m, {}).get("avg_degree", 0) for m in methods]
        ax.bar(methods, values, color="seagreen")
        ax.set_title("Average Degree")
        ax.set_ylabel("Degree")

        # Average clustering
        ax = axes[1, 1]
        values = [graphs.get(m, {}).get("avg_clustering", 0) for m in methods]
        ax.bar(methods, values, color="mediumpurple")
        ax.set_title("Average Clustering Coefficient")
        ax.set_ylabel("Clustering")

        plt.tight_layout()
        metrics_path = output_dir / "graph_metrics.png"
        plt.savefig(metrics_path, dpi=150)
        plt.close()
        generated.append(metrics_path)

    # 2. Similarity distribution stats
    sim_dist = results.get("similarity_distributions", {})
    if sim_dist and methods:
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(methods))
        width = 0.2

        means = [sim_dist.get(m, {}).get("mean", 0) for m in methods]
        medians = [sim_dist.get(m, {}).get("median", 0) for m in methods]
        p90s = [sim_dist.get(m, {}).get("percentile_90", 0) for m in methods]

        ax.bar(x - width, means, width, label="Mean", color="steelblue")
        ax.bar(x, medians, width, label="Median", color="coral")
        ax.bar(x + width, p90s, width, label="90th Percentile", color="seagreen")

        ax.set_xlabel("Method")
        ax.set_ylabel("Similarity Score")
        ax.set_title("Similarity Distribution Statistics")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        sim_stats_path = output_dir / "similarity_stats.png"
        plt.savefig(sim_stats_path, dpi=150)
        plt.close()
        generated.append(sim_stats_path)

    # 3. Link overlap heatmap (if multiple methods)
    overlaps = results.get("link_overlaps", {})
    if overlaps and len(methods) >= 2:
        # Create similarity matrix
        n = len(methods)
        matrix = np.eye(n)  # Diagonal is 1 (self-comparison)

        method_idx = {m: i for i, m in enumerate(methods)}
        for key, overlap in overlaps.items():
            parts = key.split("_vs_")
            if len(parts) == 2:
                i = method_idx.get(parts[0], -1)
                j = method_idx.get(parts[1], -1)
                if i >= 0 and j >= 0:
                    jaccard = overlap.get("jaccard_similarity", 0)
                    matrix[i, j] = jaccard
                    matrix[j, i] = jaccard

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] < 0.5 else "white",
                )

        ax.set_title("Link Overlap (Jaccard Similarity)")
        fig.colorbar(im, ax=ax, label="Jaccard Similarity")

        plt.tight_layout()
        overlap_path = output_dir / "link_overlap_heatmap.png"
        plt.savefig(overlap_path, dpi=150)
        plt.close()
        generated.append(overlap_path)

    return generated
