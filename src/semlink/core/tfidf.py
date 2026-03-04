"""
TF-IDF Baseline Similarity Module.

This module provides TF-IDF (Term Frequency-Inverse Document Frequency)
based text representation and similarity computation as a non-neural baseline.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFEmbedder:
    """
    TF-IDF based text embedder.

    Provides a simple, fast baseline for text similarity
    that works well for keyword-heavy matching.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ) -> None:
        """
        Initialize TF-IDF embedder.

        Args:
            max_features: Maximum vocabulary size
            ngram_range: (min_n, max_n) for n-grams
            min_df: Minimum document frequency (ignore rare terms)
            max_df: Maximum document frequency (ignore common terms)
            sublinear_tf: Use 1 + log(tf) for better results
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            sublinear_tf=sublinear_tf,
        )
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if vectorizer has been fitted."""
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        """Return vocabulary size after fitting."""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

    def fit(self, corpus: list[str]) -> "TFIDFEmbedder":
        """
        Fit vectorizer on corpus.

        Args:
            corpus: List of documents

        Returns:
            Self for method chaining
        """
        self.vectorizer.fit(corpus)
        self._is_fitted = True
        return self

    def encode(self, texts: list[str], to_dense: bool = True) -> NDArray[np.float32]:
        """
        Transform texts to TF-IDF vectors.

        Args:
            texts: List of texts to encode
            to_dense: Convert sparse matrix to dense array

        Returns:
            TF-IDF matrix (n_texts, vocabulary_size)

        Raises:
            ValueError: If vectorizer not fitted
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before encoding")

        matrix = self.vectorizer.transform(texts)
        if to_dense:
            return matrix.toarray().astype(np.float32)
        return matrix.astype(np.float32)

    def fit_encode(
        self, texts: list[str], to_dense: bool = True
    ) -> NDArray[np.float32]:
        """
        Fit and transform in one step.

        Args:
            texts: List of texts
            to_dense: Convert to dense array

        Returns:
            TF-IDF matrix
        """
        matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        if to_dense:
            return matrix.toarray().astype(np.float32)
        return matrix.astype(np.float32)

    def compute_similarity_matrix(
        self, embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: TF-IDF matrix (n_docs, vocab_size)

        Returns:
            Similarity matrix (n_docs, n_docs)
        """
        return cosine_similarity(embeddings).astype(np.float32)

    def search(
        self,
        query: str,
        corpus_embeddings: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find most similar documents to query.

        Args:
            query: Query text
            corpus_embeddings: Pre-computed corpus embeddings
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before search")

        query_embedding = self.encode([query])
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

        # Get top-k indices (sorted by similarity descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_feature_names(self) -> list[str]:
        """Return list of feature (term) names."""
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return list(self.vectorizer.get_feature_names_out())

    def get_top_terms(
        self, doc_index: int, embeddings: NDArray[np.float32], top_n: int = 10
    ) -> list[tuple[str, float]]:
        """
        Get top TF-IDF terms for a document.

        Args:
            doc_index: Index of document
            embeddings: TF-IDF matrix
            top_n: Number of terms to return

        Returns:
            List of (term, tfidf_score) tuples
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted first")

        feature_names = self.get_feature_names()
        doc_vector = embeddings[doc_index]

        # Get indices of top terms by TF-IDF score
        top_indices = np.argsort(doc_vector)[::-1][:top_n]

        return [
            (feature_names[idx], float(doc_vector[idx]))
            for idx in top_indices
            if doc_vector[idx] > 0
        ]

    def save(self, path: Path) -> None:
        """
        Save fitted vectorizer to file.

        Args:
            path: Output file path (.pkl)
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbedder":
        """
        Load fitted vectorizer from file.

        Args:
            path: Input file path (.pkl)

        Returns:
            TFIDFEmbedder instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")

        with open(path, "rb") as f:
            vectorizer = pickle.load(f)

        embedder = cls()
        embedder.vectorizer = vectorizer
        embedder._is_fitted = True
        return embedder

    def estimate_memory(self, n_docs: int) -> dict[str, int | float | str]:
        """
        Estimate memory usage for corpus.

        Args:
            n_docs: Number of documents

        Returns:
            Dictionary with memory estimates in MB
        """
        vocab_size = self.vocabulary_size if self._is_fitted else 10000
        # Sparse: ~2% non-zero typically
        sparsity = 0.02
        dense_mb = n_docs * vocab_size * 4 / 1024 / 1024
        sparse_mb = dense_mb * sparsity

        return {
            "vocab_size": vocab_size,
            "dense_mb": round(dense_mb, 2),
            "sparse_mb": round(sparse_mb, 2),
            "recommended": "sparse" if n_docs > 1000 else "dense",
        }
