"""
SQLite Storage Module.

Provides persistent storage for notes, embeddings, and graph data
using SQLite3 for efficient querying and incremental updates.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class StoredNote:
    """Note stored in database."""

    id: str
    path: str
    title: str
    content: str
    clean_content: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class StoredEmbedding:
    """Embedding stored in database."""

    note_id: str
    method: str
    vector: NDArray[np.float32]
    created_at: datetime


@dataclass
class StoredEdge:
    """Edge stored in database."""

    source_id: str
    target_id: str
    weight: float
    method: str
    reason: str | None
    shared_terms: list[str]


class SemLinkDB:
    """
    SQLite-based storage for SemLink data.

    Provides persistent storage for:
    - Notes and their metadata
    - Embeddings (cached for incremental updates)
    - Edges and graph data
    - Processing state and history
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path | str = ".semlink.db") -> None:
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                -- Notes table
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    clean_content TEXT NOT NULL,
                    metadata JSON NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Embeddings table
                CREATE TABLE IF NOT EXISTS embeddings (
                    note_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (note_id, method),
                    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
                );

                -- Edges table
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    weight REAL NOT NULL,
                    method TEXT NOT NULL,
                    reason TEXT,
                    shared_terms JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, method),
                    FOREIGN KEY (source_id) REFERENCES notes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES notes(id) ON DELETE CASCADE
                );

                -- Processing state table
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Schema version
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
                CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at);
                CREATE INDEX IF NOT EXISTS idx_embeddings_method ON embeddings(method);
                CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight);
            """
            )

            # Check/update schema version
            cursor = conn.execute("SELECT version FROM schema_info LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_info (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with context management."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Note Operations
    # =========================================================================

    def upsert_note(
        self,
        note_id: str,
        path: str,
        title: str,
        content: str,
        clean_content: str,
        metadata: dict[str, Any],
        file_hash: str,
    ) -> bool:
        """
        Insert or update a note.

        Returns True if note was inserted/updated, False if unchanged.
        """
        with self._get_connection() as conn:
            # Check if note exists with same hash (unchanged)
            cursor = conn.execute(
                "SELECT file_hash FROM notes WHERE id = ?", (note_id,)
            )
            row = cursor.fetchone()

            if row and row["file_hash"] == file_hash:
                return False  # Unchanged

            conn.execute(
                """
                INSERT INTO notes (id, path, title, content, clean_content, metadata, file_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    content = excluded.content,
                    clean_content = excluded.clean_content,
                    metadata = excluded.metadata,
                    file_hash = excluded.file_hash,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (
                    note_id,
                    path,
                    title,
                    content,
                    clean_content,
                    json.dumps(metadata),
                    file_hash,
                ),
            )
            return True

    def get_note(self, note_id: str) -> StoredNote | None:
        """Get a note by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            if row is None:
                return None

            return StoredNote(
                id=row["id"],
                path=row["path"],
                title=row["title"],
                content=row["content"],
                clean_content=row["clean_content"],
                metadata=json.loads(row["metadata"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_all_notes(self) -> list[StoredNote]:
        """Get all notes."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM notes ORDER BY title")
            return [
                StoredNote(
                    id=row["id"],
                    path=row["path"],
                    title=row["title"],
                    content=row["content"],
                    clean_content=row["clean_content"],
                    metadata=json.loads(row["metadata"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in cursor
            ]

    def get_note_ids(self) -> list[str]:
        """Get all note IDs."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM notes ORDER BY id")
            return [row["id"] for row in cursor]

    def get_note_count(self) -> int:
        """Get total number of notes."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM notes")
            return cursor.fetchone()["cnt"]

    def delete_note(self, note_id: str) -> bool:
        """Delete a note and its embeddings/edges."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            return cursor.rowcount > 0

    def get_changed_notes(
        self, file_hashes: dict[str, str]
    ) -> tuple[list[str], list[str]]:
        """
        Find notes that have changed or been added.

        Args:
            file_hashes: Dict mapping note_id to current file hash

        Returns:
            Tuple of (new_or_changed_ids, deleted_ids)
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, file_hash FROM notes")
            stored = {row["id"]: row["file_hash"] for row in cursor}

        new_or_changed = []
        for note_id, current_hash in file_hashes.items():
            if note_id not in stored or stored[note_id] != current_hash:
                new_or_changed.append(note_id)

        deleted = [nid for nid in stored if nid not in file_hashes]

        return new_or_changed, deleted

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    def save_embedding(
        self,
        note_id: str,
        method: str,
        vector: NDArray[np.float32],
    ) -> None:
        """Save an embedding vector."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (note_id, method, vector)
                VALUES (?, ?, ?)
                ON CONFLICT(note_id, method) DO UPDATE SET
                    vector = excluded.vector,
                    created_at = CURRENT_TIMESTAMP
            """,
                (note_id, method, vector.tobytes()),
            )

    def get_embedding(self, note_id: str, method: str) -> NDArray[np.float32] | None:
        """Get an embedding vector."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT vector FROM embeddings WHERE note_id = ? AND method = ?",
                (note_id, method),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return np.frombuffer(row["vector"], dtype=np.float32)

    def get_all_embeddings(self, method: str) -> tuple[list[str], NDArray[np.float32]]:
        """
        Get all embeddings for a method.

        Returns:
            Tuple of (note_ids, embedding_matrix)
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT note_id, vector FROM embeddings
                WHERE method = ?
                ORDER BY note_id
            """,
                (method,),
            )
            rows = list(cursor)

        if not rows:
            return [], np.array([], dtype=np.float32)

        ids = [row["note_id"] for row in rows]
        vectors = [np.frombuffer(row["vector"], dtype=np.float32) for row in rows]
        return ids, np.vstack(vectors)

    def get_notes_without_embeddings(self, method: str) -> list[str]:
        """Get IDs of notes that don't have embeddings for the given method."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT n.id FROM notes n
                LEFT JOIN embeddings e ON n.id = e.note_id AND e.method = ?
                WHERE e.note_id IS NULL
            """,
                (method,),
            )
            return [row["id"] for row in cursor]

    def delete_embeddings(self, note_id: str, method: str | None = None) -> int:
        """Delete embeddings for a note."""
        with self._get_connection() as conn:
            if method:
                cursor = conn.execute(
                    "DELETE FROM embeddings WHERE note_id = ? AND method = ?",
                    (note_id, method),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM embeddings WHERE note_id = ?", (note_id,)
                )
            return cursor.rowcount

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def save_edges(self, edges: list[StoredEdge], method: str) -> int:
        """
        Save edges (replacing existing edges for the method).

        Returns number of edges saved.
        """
        with self._get_connection() as conn:
            # Delete existing edges for this method
            conn.execute("DELETE FROM edges WHERE method = ?", (method,))

            # Insert new edges
            conn.executemany(
                """
                INSERT INTO edges (source_id, target_id, weight, method, reason, shared_terms)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        e.source_id,
                        e.target_id,
                        e.weight,
                        e.method,
                        e.reason,
                        json.dumps(e.shared_terms) if e.shared_terms else None,
                    )
                    for e in edges
                ],
            )
            return len(edges)

    def get_edges(self, method: str | None = None) -> list[StoredEdge]:
        """Get edges, optionally filtered by method."""
        with self._get_connection() as conn:
            if method:
                cursor = conn.execute(
                    "SELECT * FROM edges WHERE method = ? ORDER BY weight DESC",
                    (method,),
                )
            else:
                cursor = conn.execute("SELECT * FROM edges ORDER BY weight DESC")

            return [
                StoredEdge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    weight=row["weight"],
                    method=row["method"],
                    reason=row["reason"],
                    shared_terms=json.loads(row["shared_terms"])
                    if row["shared_terms"]
                    else [],
                )
                for row in cursor
            ]

    def get_edge_count(self, method: str | None = None) -> int:
        """Get number of edges."""
        with self._get_connection() as conn:
            if method:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM edges WHERE method = ?", (method,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM edges")
            return cursor.fetchone()["cnt"]

    # =========================================================================
    # State Operations
    # =========================================================================

    def set_state(self, key: str, value: Any) -> None:
        """Set a state value."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO state (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """,
                (key, json.dumps(value)),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row is None:
                return default
            return json.loads(row["value"])

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            stats = {}

            cursor = conn.execute("SELECT COUNT(*) as cnt FROM notes")
            stats["notes"] = cursor.fetchone()["cnt"]

            cursor = conn.execute(
                "SELECT method, COUNT(*) as cnt FROM embeddings GROUP BY method"
            )
            stats["embeddings"] = {row["method"]: row["cnt"] for row in cursor}

            cursor = conn.execute(
                "SELECT method, COUNT(*) as cnt FROM edges GROUP BY method"
            )
            stats["edges"] = {row["method"]: row["cnt"] for row in cursor}

            return stats

    def vacuum(self) -> None:
        """Compact database file."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def clear(self) -> None:
        """Clear all data (keep schema)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM state")


def compute_file_hash(path: Path) -> str:
    """Compute hash of file for change detection."""
    import hashlib

    content = path.read_bytes()
    return hashlib.md5(content).hexdigest()
