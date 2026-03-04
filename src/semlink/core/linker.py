"""
Link Inference Module.

This module converts similarity scores into graph edges using
various strategies like thresholding and k-nearest neighbors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Edge:
    """Represents an inferred link between two nodes."""

    source: str
    target: str
    weight: float
    method: str  # 'threshold', 'knn', 'mutual_knn', 'hybrid'

    def to_tuple(self) -> tuple[str, str, float]:
        """Return as (source, target, weight) tuple."""
        return (self.source, self.target, self.weight)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "method": self.method,
        }


class LinkStrategy(ABC):
    """Abstract base class for link inference strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        ...

    @abstractmethod
    def infer_links(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> list[Edge]:
        """
        Infer links from embeddings.

        Args:
            embeddings: Embedding matrix (n_docs, dimensions)
            ids: Document IDs

        Returns:
            List of Edge objects
        """
        ...


class ThresholdStrategy(LinkStrategy):
    """
    Simple threshold-based linking.

    Creates edge if similarity >= threshold.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize threshold strategy.

        Args:
            threshold: Minimum similarity for edge creation
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "threshold"

    def infer_links(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> list[Edge]:
        """Infer links using similarity threshold."""
        sim_matrix = compute_similarity_matrix(embeddings)
        n = len(ids)
        edges: list[Edge] = []

        # Only consider upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    edges.append(
                        Edge(
                            source=ids[i],
                            target=ids[j],
                            weight=float(sim_matrix[i, j]),
                            method=self.name,
                        )
                    )

        return edges


class KNNStrategy(LinkStrategy):
    """
    K-nearest neighbors linking.

    Each node connects to its k most similar neighbors.
    Guarantees minimum connectivity.
    """

    def __init__(self, k: int = 7, metric: str = "cosine") -> None:
        """
        Initialize KNN strategy.

        Args:
            k: Number of neighbors per node
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.k = k
        self.metric = metric

    @property
    def name(self) -> str:
        return "knn"

    def infer_links(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> list[Edge]:
        """Infer links using k-nearest neighbors."""
        sim_matrix = compute_similarity_matrix(embeddings)
        n = len(ids)
        edges: list[Edge] = []
        edge_set: set[tuple[str, str]] = set()

        # For each node, find k nearest neighbors
        for i in range(n):
            # Get indices sorted by similarity (descending)
            neighbors = np.argsort(sim_matrix[i])[::-1][: self.k]

            for j in neighbors:
                if i == j:
                    continue

                # Create canonical edge (smaller id first)
                source, target = (
                    (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                )
                edge_key = (source, target)

                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(
                        Edge(
                            source=source,
                            target=target,
                            weight=float(sim_matrix[i, j]),
                            method=self.name,
                        )
                    )

        return edges


class MutualKNNStrategy(LinkStrategy):
    """
    Mutual k-nearest neighbors linking.

    Only creates edge if both nodes consider each other neighbors.
    Higher precision, sparser graph.
    """

    def __init__(self, k: int = 10, metric: str = "cosine") -> None:
        """
        Initialize mutual KNN strategy.

        Args:
            k: Number of neighbors to consider
            metric: Distance metric
        """
        self.k = k
        self.metric = metric

    @property
    def name(self) -> str:
        return "mutual_knn"

    def infer_links(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> list[Edge]:
        """Infer links using mutual k-nearest neighbors."""
        sim_matrix = compute_similarity_matrix(embeddings)
        n = len(ids)

        # Find k-nearest neighbors for each node
        knn_indices: list[set[int]] = []
        for i in range(n):
            neighbors = np.argsort(sim_matrix[i])[::-1][: self.k]
            knn_indices.append(set(neighbors) - {i})

        # Only create edge if mutual neighbors
        edges: list[Edge] = []
        edge_set: set[tuple[str, str]] = set()

        for i in range(n):
            for j in knn_indices[i]:
                # Check if mutual
                if i in knn_indices[j]:
                    source, target = (
                        (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                    )
                    edge_key = (source, target)

                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append(
                            Edge(
                                source=source,
                                target=target,
                                weight=float(sim_matrix[i, j]),
                                method=self.name,
                            )
                        )

        return edges


class HybridStrategy(LinkStrategy):
    """
    Hybrid strategy combining KNN and threshold.

    1. Start with KNN to ensure connectivity
    2. Add high-similarity edges above threshold
    3. Optionally cap edges per node

    Recommended for most use cases.
    """

    def __init__(
        self,
        k: int = 7,
        threshold: float = 0.5,
        max_edges_per_node: int | None = None,
    ) -> None:
        """
        Initialize hybrid strategy.

        Args:
            k: Minimum neighbors per node (KNN)
            threshold: Additional edges if similarity above this
            max_edges_per_node: Maximum edges per node (None = unlimited)
        """
        self.k = k
        self.threshold = threshold
        self.max_edges_per_node = max_edges_per_node

    @property
    def name(self) -> str:
        return "hybrid"

    def infer_links(
        self,
        embeddings: NDArray[np.float32],
        ids: list[str],
    ) -> list[Edge]:
        """Infer links using hybrid KNN + threshold approach."""
        sim_matrix = compute_similarity_matrix(embeddings)
        n = len(ids)
        edge_dict: dict[tuple[str, str], Edge] = {}

        # Step 1: Add KNN edges (ensure connectivity)
        for i in range(n):
            neighbors = np.argsort(sim_matrix[i])[::-1][: self.k]
            for j in neighbors:
                if i == j:
                    continue
                source, target = (
                    (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                )
                edge_key = (source, target)

                if edge_key not in edge_dict:
                    edge_dict[edge_key] = Edge(
                        source=source,
                        target=target,
                        weight=float(sim_matrix[i, j]),
                        method="knn",
                    )

        # Step 2: Add threshold edges (high similarity)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    source, target = ids[i], ids[j]
                    edge_key = (source, target)

                    if edge_key not in edge_dict:
                        edge_dict[edge_key] = Edge(
                            source=source,
                            target=target,
                            weight=float(sim_matrix[i, j]),
                            method="threshold",
                        )
                    else:
                        # Update method to hybrid if already exists from KNN
                        existing = edge_dict[edge_key]
                        edge_dict[edge_key] = Edge(
                            source=source,
                            target=target,
                            weight=existing.weight,
                            method="hybrid",
                        )

        edges = list(edge_dict.values())

        # Step 3: Optionally cap edges per node
        if self.max_edges_per_node is not None:
            edges = self._cap_edges_per_node(edges, ids)

        return edges

    def _cap_edges_per_node(self, edges: list[Edge], ids: list[str]) -> list[Edge]:
        """Cap edges per node by keeping highest weight."""
        from collections import defaultdict

        # Group edges by node
        node_edges: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            node_edges[edge.source].append(edge)
            node_edges[edge.target].append(edge)

        # Keep track of edges to remove
        edges_to_keep: set[tuple[str, str]] = set()

        for node_id in ids:
            node_edge_list = node_edges[node_id]
            # Sort by weight descending
            node_edge_list.sort(key=lambda e: e.weight, reverse=True)
            # Keep top max_edges_per_node
            for edge in node_edge_list[: self.max_edges_per_node]:
                key = (edge.source, edge.target)
                edges_to_keep.add(key)

        return [e for e in edges if (e.source, e.target) in edges_to_keep]


# Strategy registry
_STRATEGIES: dict[str, type[LinkStrategy]] = {
    "threshold": ThresholdStrategy,
    "knn": KNNStrategy,
    "mutual_knn": MutualKNNStrategy,
    "hybrid": HybridStrategy,
}

LinkStrategyType = Literal["threshold", "knn", "mutual_knn", "hybrid"]


def get_strategy(name: LinkStrategyType, **kwargs) -> LinkStrategy:
    """
    Get link strategy by name.

    Args:
        name: Strategy name
        **kwargs: Strategy-specific parameters

    Returns:
        LinkStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    if name not in _STRATEGIES:
        available = ", ".join(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

    return _STRATEGIES[name](**kwargs)


def list_strategies() -> list[str]:
    """Return list of available strategy names."""
    return list(_STRATEGIES.keys())


def compute_similarity_matrix(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Embedding matrix

    Returns:
        Similarity matrix with zeros on diagonal
    """
    sim_matrix = cosine_similarity(embeddings).astype(np.float32)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def filter_edges(
    edges: list[Edge],
    min_weight: float | None = None,
    max_edges: int | None = None,
) -> list[Edge]:
    """
    Filter edges by weight or count.

    Args:
        edges: List of edges
        min_weight: Minimum edge weight
        max_edges: Maximum number of edges to keep

    Returns:
        Filtered edge list
    """
    filtered = edges

    if min_weight is not None:
        filtered = [e for e in filtered if e.weight >= min_weight]

    if max_edges is not None:
        # Sort by weight descending and take top max_edges
        filtered = sorted(filtered, key=lambda e: e.weight, reverse=True)[:max_edges]

    return filtered


def edge_list_stats(edges: list[Edge]) -> dict:
    """
    Compute statistics about edge list.

    Returns:
        Dictionary with count, weight stats, etc.
    """
    if not edges:
        return {"count": 0}

    weights = [e.weight for e in edges]
    return {
        "count": len(edges),
        "min_weight": min(weights),
        "max_weight": max(weights),
        "mean_weight": sum(weights) / len(weights),
        "methods": list(set(e.method for e in edges)),
    }
