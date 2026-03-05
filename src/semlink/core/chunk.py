"""
Note Segmentation (Chunking) Module.

This module provides strategies for segmenting notes into smaller chunks
to improve semantic resolution during embedding and linking.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A segment of a note with source tracking."""

    id: str
    content: str
    note_id: str
    start_offset: int
    end_offset: int
    chunk_type: str  # 'whole', 'paragraph', 'heading', 'window'

    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "note_id": self.note_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "chunk_type": self.chunk_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(**data)


class ChunkStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        ...

    @abstractmethod
    def chunk(self, note_id: str, content: str) -> list[Chunk]:
        """
        Split content into chunks.

        Args:
            note_id: ID of the source note
            content: Text content to chunk

        Returns:
            List of Chunk objects
        """
        ...


class WholeNoteStrategy(ChunkStrategy):
    """
    Treat entire note as single chunk.

    Best for short notes (<500 words).
    """

    @property
    def name(self) -> str:
        return "whole"

    def chunk(self, note_id: str, content: str) -> list[Chunk]:
        """Return entire note as single chunk."""
        chunk_id = f"{note_id}_whole"
        return [
            Chunk(
                id=chunk_id,
                content=content.strip(),
                note_id=note_id,
                start_offset=0,
                end_offset=len(content),
                chunk_type="whole",
            )
        ]


class ParagraphStrategy(ChunkStrategy):
    """
    Split note on paragraph boundaries (double newlines).

    Best for medium notes (500-2000 words).
    """

    def __init__(self, min_length: int = 50, merge_short: bool = True) -> None:
        """
        Initialize paragraph chunker.

        Args:
            min_length: Minimum characters per chunk
            merge_short: Whether to merge short paragraphs
        """
        self.min_length = min_length
        self.merge_short = merge_short

    @property
    def name(self) -> str:
        return "paragraph"

    def chunk(self, note_id: str, content: str) -> list[Chunk]:
        """Split content on paragraph boundaries."""
        # Split on double newlines
        paragraphs = re.split(r"\n\s*\n", content)

        chunks: list[Chunk] = []
        current_offset = 0
        pending_content = ""
        pending_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                current_offset += 2  # Account for the split
                continue

            # Find actual position in original content
            para_start = content.find(para, current_offset)
            if para_start == -1:
                para_start = current_offset
            para_end = para_start + len(para)

            if len(para) < self.min_length and self.merge_short:
                # Accumulate short paragraphs
                if pending_content:
                    pending_content += "\n\n" + para
                else:
                    pending_content = para
                    pending_start = para_start
            else:
                # Flush pending content first
                if pending_content:
                    chunk_id = f"{note_id}_p{len(chunks)}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            content=pending_content,
                            note_id=note_id,
                            start_offset=pending_start,
                            end_offset=para_start - 1,
                            chunk_type="paragraph",
                        )
                    )
                    pending_content = ""

                # Add current paragraph
                chunk_id = f"{note_id}_p{len(chunks)}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=para,
                        note_id=note_id,
                        start_offset=para_start,
                        end_offset=para_end,
                        chunk_type="paragraph",
                    )
                )

            current_offset = para_end

        # Flush any remaining pending content
        if pending_content:
            chunk_id = f"{note_id}_p{len(chunks)}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=pending_content,
                    note_id=note_id,
                    start_offset=pending_start,
                    end_offset=len(content),
                    chunk_type="paragraph",
                )
            )

        # If no chunks created, return whole note
        if not chunks:
            return WholeNoteStrategy().chunk(note_id, content)

        return chunks


class HeadingStrategy(ChunkStrategy):
    """
    Split note on markdown headings.

    Best for long structured documents.
    """

    def __init__(self, max_level: int = 3) -> None:
        """
        Initialize heading chunker.

        Args:
            max_level: Maximum heading level to split on (1-6)
        """
        self.max_level = max_level

    @property
    def name(self) -> str:
        return "heading"

    def chunk(self, note_id: str, content: str) -> list[Chunk]:
        """Split content on markdown headings."""
        # Pattern for headings up to max_level
        heading_pattern = re.compile(
            rf"^(#{1, {self.max_level}})\s+(.+)$", re.MULTILINE
        )

        # Find all headings
        matches = list(heading_pattern.finditer(content))

        if not matches:
            # No headings found, return whole note
            return WholeNoteStrategy().chunk(note_id, content)

        chunks: list[Chunk] = []

        # Content before first heading
        if matches[0].start() > 0:
            preamble = content[: matches[0].start()].strip()
            if preamble:
                chunks.append(
                    Chunk(
                        id=f"{note_id}_h0",
                        content=preamble,
                        note_id=note_id,
                        start_offset=0,
                        end_offset=matches[0].start(),
                        chunk_type="heading",
                    )
                )

        # Content for each heading section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            section_content = content[start:end].strip()
            if section_content:
                chunks.append(
                    Chunk(
                        id=f"{note_id}_h{i + 1}",
                        content=section_content,
                        note_id=note_id,
                        start_offset=start,
                        end_offset=end,
                        chunk_type="heading",
                    )
                )

        return chunks if chunks else WholeNoteStrategy().chunk(note_id, content)


class SlidingWindowStrategy(ChunkStrategy):
    """
    Create overlapping chunks using sliding window.

    Best for very long documents where context continuity matters.
    """

    def __init__(
        self, window_size: int = 512, overlap: int = 128, unit: str = "chars"
    ) -> None:
        """
        Initialize sliding window chunker.

        Args:
            window_size: Size of each window
            overlap: Overlap between consecutive windows
            unit: 'chars' or 'words'
        """
        self.window_size = window_size
        self.overlap = overlap
        self.unit = unit

    @property
    def name(self) -> str:
        return "window"

    def chunk(self, note_id: str, content: str) -> list[Chunk]:
        """Create overlapping chunks."""
        if self.unit == "words":
            return self._chunk_by_words(note_id, content)
        else:
            return self._chunk_by_chars(note_id, content)

    def _chunk_by_chars(self, note_id: str, content: str) -> list[Chunk]:
        """Create character-based sliding window chunks."""
        chunks: list[Chunk] = []
        step = self.window_size - self.overlap
        n = len(content)

        if n <= self.window_size:
            return WholeNoteStrategy().chunk(note_id, content)

        i = 0
        chunk_num = 0
        while i < n:
            end = min(i + self.window_size, n)
            chunk_content = content[i:end].strip()

            if chunk_content:
                chunks.append(
                    Chunk(
                        id=f"{note_id}_w{chunk_num}",
                        content=chunk_content,
                        note_id=note_id,
                        start_offset=i,
                        end_offset=end,
                        chunk_type="window",
                    )
                )
                chunk_num += 1

            i += step
            if i >= n and end < n:
                break

        return chunks if chunks else WholeNoteStrategy().chunk(note_id, content)

    def _chunk_by_words(self, note_id: str, content: str) -> list[Chunk]:
        """Create word-based sliding window chunks."""
        words = content.split()
        n = len(words)

        if n <= self.window_size:
            return WholeNoteStrategy().chunk(note_id, content)

        chunks: list[Chunk] = []
        step = self.window_size - self.overlap
        i = 0
        chunk_num = 0

        while i < n:
            end = min(i + self.window_size, n)
            window_words = words[i:end]
            chunk_content = " ".join(window_words)

            # Approximate character offsets
            start_offset = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_offset = start_offset + len(chunk_content)

            chunks.append(
                Chunk(
                    id=f"{note_id}_w{chunk_num}",
                    content=chunk_content,
                    note_id=note_id,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    chunk_type="window",
                )
            )
            chunk_num += 1

            i += step
            if i >= n:
                break

        return chunks if chunks else WholeNoteStrategy().chunk(note_id, content)


# Strategy registry
_STRATEGIES: dict[str, type[ChunkStrategy]] = {
    "whole": WholeNoteStrategy,
    "paragraph": ParagraphStrategy,
    "heading": HeadingStrategy,
    "window": SlidingWindowStrategy,
}


def get_strategy(name: str, **kwargs) -> ChunkStrategy:
    """
    Get chunking strategy by name.

    Args:
        name: Strategy name ('whole', 'paragraph', 'heading', 'window')
        **kwargs: Strategy-specific parameters

    Returns:
        ChunkStrategy instance

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


@dataclass
class ChunkIndex:
    """
    Index mapping chunks back to source notes.

    Enables tracing links back to original notes.
    """

    chunks: list[Chunk]
    note_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    chunk_to_note: dict[str, str] = field(default_factory=dict)

    def get_chunks_for_note(self, note_id: str) -> list[Chunk]:
        """Get all chunks belonging to a note."""
        chunk_ids = self.note_to_chunks.get(note_id, [])
        return [c for c in self.chunks if c.id in chunk_ids]

    def get_note_for_chunk(self, chunk_id: str) -> str | None:
        """Get source note ID for a chunk."""
        return self.chunk_to_note.get(chunk_id)

    def to_dict(self) -> dict:
        """Convert index to dictionary."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "note_to_chunks": self.note_to_chunks,
            "chunk_to_note": self.chunk_to_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkIndex":
        """Create index from dictionary."""
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        return cls(
            chunks=chunks,
            note_to_chunks=data["note_to_chunks"],
            chunk_to_note=data["chunk_to_note"],
        )


def build_chunk_index(
    notes: list[tuple[str, str]], strategy: ChunkStrategy
) -> ChunkIndex:
    """
    Build chunk index from notes using specified strategy.

    Args:
        notes: List of (note_id, content) tuples
        strategy: Chunking strategy to use

    Returns:
        ChunkIndex with all chunks and mappings
    """
    all_chunks: list[Chunk] = []
    note_to_chunks: dict[str, list[str]] = {}
    chunk_to_note: dict[str, str] = {}

    for note_id, content in notes:
        chunks = strategy.chunk(note_id, content)
        all_chunks.extend(chunks)

        note_to_chunks[note_id] = [c.id for c in chunks]
        for chunk in chunks:
            chunk_to_note[chunk.id] = note_id

    return ChunkIndex(
        chunks=all_chunks,
        note_to_chunks=note_to_chunks,
        chunk_to_note=chunk_to_note,
    )
