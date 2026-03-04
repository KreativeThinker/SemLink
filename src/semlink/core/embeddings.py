"""
Neural Embedding Module.

This module provides embedding backends for semantic text similarity,
including Sentence-BERT and optional OpenAI embeddings.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class EmbedderBase(ABC):
    """Abstract base class for embedding backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return embedder name."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        ...

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embedding matrix (n_texts, dimensions)
        """
        ...

    def compute_similarity_matrix(
        self, embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Embedding matrix (n_docs, dimensions)

        Returns:
            Similarity matrix (n_docs, n_docs)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(embeddings).astype(np.float32)


class SBERTEmbedder(EmbedderBase):
    """
    Sentence-BERT embedding backend.

    Uses sentence-transformers library for local embedding generation.
    """

    # Available models with their characteristics
    MODELS = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_tokens": 256,
            "speed": "fast",
            "quality": "good",
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "max_tokens": 384,
            "speed": "medium",
            "quality": "better",
        },
        "paraphrase-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_tokens": 256,
            "speed": "fast",
            "quality": "good",
        },
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Initialize SBERT embedder.

        Args:
            model_name: Model to use (see MODELS)
            device: Device to run on ('cpu', 'cuda', or None for auto)
            normalize: L2 normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimensions = self.MODELS.get(model_name, {}).get("dimensions", 384)

    @property
    def name(self) -> str:
        return f"sbert:{self.model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SBERT embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts using Sentence-BERT."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def benchmark(self, texts: list[str], batch_size: int = 32) -> dict:
        """
        Benchmark encoding speed and memory.

        Args:
            texts: Sample texts to benchmark
            batch_size: Batch size

        Returns:
            Dictionary with timing and memory stats
        """
        import tracemalloc

        n_texts = len(texts)

        # Warm up
        _ = self.encode(texts[: min(10, n_texts)], show_progress=False)

        # Benchmark
        tracemalloc.start()
        start_time = time.time()
        _ = self.encode(texts, batch_size=batch_size, show_progress=False)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end_time - start_time
        return {
            "n_texts": n_texts,
            "batch_size": batch_size,
            "total_seconds": round(elapsed, 3),
            "texts_per_second": round(n_texts / elapsed, 2),
            "peak_memory_mb": round(peak / 1024 / 1024, 2),
        }

    @classmethod
    def list_models(cls) -> list[str]:
        """Return list of available model names."""
        return list(cls.MODELS.keys())


class OpenAIEmbedder(EmbedderBase):
    """
    OpenAI embedding backend.

    Uses OpenAI API for cloud-based embedding generation.
    Requires openai package and API key.
    """

    MODELS = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "price_per_1m": 0.02,
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "price_per_1m": 0.13,
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "price_per_1m": 0.10,
        },
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        """
        Initialize OpenAI embedder.

        Args:
            model_name: Model to use
            dimensions: Output dimensions (only for v3 models)
        """
        self.model_name = model_name
        self._dimensions = dimensions or self.MODELS[model_name]["dimensions"]
        self._client = None
        self._total_tokens = 0

    @property
    def name(self) -> str:
        return f"openai:{self.model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI()
        return self._client

    def encode(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Encode texts using OpenAI API."""
        client = self._load_client()

        all_embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch = texts[i * batch_size : (i + 1) * batch_size]

            kwargs = {"input": batch, "model": self.model_name}
            # Only v3 models support custom dimensions
            if "text-embedding-3" in self.model_name and self._dimensions:
                kwargs["dimensions"] = self._dimensions

            response = client.embeddings.create(**kwargs)

            # Track usage
            if hasattr(response, "usage"):
                self._total_tokens += response.usage.total_tokens

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def estimate_cost(self, texts: list[str]) -> dict:
        """
        Estimate embedding cost before processing.

        Args:
            texts: Texts to embed

        Returns:
            Dictionary with token count and estimated cost
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for cost estimation. "
                "Install with: pip install tiktoken"
            )

        # Use cl100k_base encoding (used by embedding models)
        enc = tiktoken.get_encoding("cl100k_base")

        total_tokens = sum(len(enc.encode(text)) for text in texts)
        price = self.MODELS[self.model_name]["price_per_1m"]
        estimated_cost = total_tokens * price / 1_000_000

        return {
            "total_tokens": total_tokens,
            "price_per_1m_tokens": price,
            "estimated_cost_usd": round(estimated_cost, 6),
        }

    def get_usage_stats(self) -> dict:
        """Get cumulative usage statistics."""
        price = self.MODELS[self.model_name]["price_per_1m"]
        return {
            "total_tokens_used": self._total_tokens,
            "total_cost_usd": round(self._total_tokens * price / 1_000_000, 6),
        }


# Embedder type literal for factory
EmbedderType = Literal["sbert", "openai"]


def create_embedder(
    embedder_type: EmbedderType,
    model_name: str | None = None,
    **kwargs,
) -> EmbedderBase:
    """
    Factory function to create embedder by type.

    Args:
        embedder_type: 'sbert' or 'openai'
        model_name: Specific model name (uses default if None)
        **kwargs: Additional embedder-specific arguments

    Returns:
        EmbedderBase instance

    Raises:
        ValueError: If embedder type is unknown
    """
    if embedder_type == "sbert":
        model = model_name or "all-MiniLM-L6-v2"
        return SBERTEmbedder(model_name=model, **kwargs)
    elif embedder_type == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedder(model_name=model, **kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def save_embeddings(
    embeddings: NDArray[np.float32],
    ids: list[str],
    path: Path,
    metadata: dict | None = None,
) -> None:
    """
    Save embeddings to .npz file.

    Args:
        embeddings: Embedding matrix
        ids: Document IDs corresponding to rows
        path: Output path
        metadata: Optional metadata dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "embeddings": embeddings,
        "ids": np.array(ids, dtype=object),
    }

    if metadata:
        import json

        save_dict["metadata"] = np.array([json.dumps(metadata)])

    np.savez_compressed(path, **save_dict)


def load_embeddings(path: Path) -> tuple[NDArray[np.float32], list[str], dict]:
    """
    Load embeddings from .npz file.

    Args:
        path: Input path

    Returns:
        Tuple of (embeddings, ids, metadata)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)

    embeddings = data["embeddings"].astype(np.float32)
    ids = list(data["ids"])

    metadata = {}
    if "metadata" in data:
        import json

        metadata = json.loads(str(data["metadata"][0]))

    return embeddings, ids, metadata
