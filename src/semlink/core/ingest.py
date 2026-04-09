"""
Data Ingestion and Preprocessing Module.

This module handles loading plain-text and Markdown files,
stripping markup, normalizing text, and storing processed representations.
"""

from __future__ import annotations

import hashlib
import re
import os
import fnmatch
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import orjson
from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.token import Token

# High-priority directories to never traverse
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "obj",
    "bin",
}

# Specific filenames or patterns that add semantic noise
EXCLUDED_FILE_PATTERNS = {
    "LICENSE*",
    "COPYING*",
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "uv.lock",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.exe",
    "*.dll",
}


@dataclass
class NoteMetadata:
    """Metadata extracted from a note file."""

    filename: str
    path: str
    title: str
    headings: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    word_count: int = 0


@dataclass
class Note:
    """Represents a processed note with content and metadata."""

    id: str
    raw_content: str
    clean_content: str
    metadata: NoteMetadata

    def to_dict(self) -> dict:
        """Convert note to dictionary for serialization."""
        return {
            "id": self.id,
            "raw_content": self.raw_content,
            "clean_content": self.clean_content,
            "metadata": {
                "filename": self.metadata.filename,
                "path": self.metadata.path,
                "title": self.metadata.title,
                "headings": self.metadata.headings,
                "links": self.metadata.links,
                "word_count": self.metadata.word_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        """Create note from dictionary."""
        metadata = NoteMetadata(**data["metadata"])
        return cls(
            id=data["id"],
            raw_content=data["raw_content"],
            clean_content=data["clean_content"],
            metadata=metadata,
        )


def discover_notes(
    vault_path: Path, extensions: tuple[str, ...] = (".md", ".txt")
) -> Iterator[Path]:
    """
    Recursively discover note files in a vault directory.

    Args:
        vault_path: Root directory to search
        extensions: File extensions to include

    Yields:
        Path objects for each discovered note file
    """
    vault_path = Path(vault_path)
    if not vault_path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {vault_path}")

    for root, dirs, files in os.walk(vault_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for file in files:
            # Skip hidden files (e.g., .env)
            if file.startswith("."):
                continue

            # Skip noise files matching our exclusion patterns
            if any(
                fnmatch.fnmatch(file, pattern) for pattern in EXCLUDED_FILE_PATTERNS
            ):
                continue

            path = Path(root) / file
            if path.suffix.lower() in extensions:
                yield path


def load_note(path: Path) -> str:
    """
    Load note content from file with encoding detection.

    Args:
        path: Path to the note file

    Returns:
        Raw content of the note

    Raises:
        FileNotFoundError: If file does not exist
        UnicodeDecodeError: If encoding cannot be determined
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Note file not found: {path}")

    # Try common encodings in order of likelihood
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pypdf

        reader = pypdf.PdfReader(path)
        # Extract text from all pages and join with double newlines to simulate paragraphs
        text = [page.extract_text() for page in reader.pages]
        return "\n\n".join(filter(None, text))

    if suffix == ".docx":
        import docx

        doc = docx.Document(path)
        # Extract paragraphs that actually contain text
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    # Fallback for .md and .txt: Try common encodings
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    # Last resort: read with errors replaced
    return path.read_text(encoding="utf-8", errors="replace")


def strip_markdown(text: str) -> str:
    """
    Remove markdown syntax while preserving semantic structure.

    Uses markdown-it-py for robust parsing.

    Args:
        text: Raw markdown text

    Returns:
        Plain text with markdown syntax removed
    """
    md = MarkdownIt("commonmark")
    tokens: list[Token] = md.parse(text)

    def extract_text(tokens: list[Token]) -> str:
        """Recursively extract text from tokens."""
        result: list[str] = []

        for token in tokens:
            if token.type == "text" or token.type == "code_inline":
                result.append(token.content)
            elif token.type == "code_block" or token.type == "fence":
                # Preserve code content but mark it
                result.append(token.content)
            elif token.type == "softbreak" or token.type == "hardbreak":
                result.append("\n")
            elif token.type == "inline" and token.children:
                result.append(extract_text(token.children))
            elif token.type in ("paragraph_open", "heading_open"):
                pass  # Opening tags don't add content
            elif token.type in ("paragraph_close", "heading_close"):
                result.append("\n")
            elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
                pass
            elif token.type == "list_item_open":
                pass
            elif token.type == "list_item_close":
                result.append("\n")

        return "".join(result)

    plain_text = extract_text(tokens)

    # Clean up excessive whitespace while preserving paragraph breaks
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)
    return plain_text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Operations:
    - Lowercase conversion
    - Unicode normalization (NFKC)
    - Whitespace normalization
    - Remove excessive newlines

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Unicode normalization (NFKC decomposes and recomposes)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Normalize whitespace (tabs, multiple spaces -> single space)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Normalize newlines (collapse 3+ into 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def extract_metadata(path: Path, content: str) -> NoteMetadata:
    """
    Extract metadata from note file and content.

    Extracts:
    - Filename and path
    - Title (first heading or filename)
    - All headings
    - Wiki-style links [[link]]
    - Word count

    Args:
        path: Path to the note file
        content: Raw content of the note

    Returns:
        NoteMetadata object
    """
    path = Path(path)

    # Extract headings (# Heading format)
    heading_pattern = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    headings = heading_pattern.findall(content)

    # Title: first heading or filename without extension
    title = headings[0] if headings else path.stem

    # Extract wiki-style links [[link]] and [[link|alias]]
    link_pattern = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
    links = link_pattern.findall(content)

    # Word count (on stripped content)
    stripped = strip_markdown(content)
    words = stripped.split()
    word_count = len(words)

    return NoteMetadata(
        filename=path.name,
        path=str(path),
        title=title,
        headings=headings,
        links=links,
        word_count=word_count,
    )


def process_note(path: Path, normalize: bool = True) -> Note:
    """
    Process a single note file into a Note object.

    Args:
        path: Path to the note file
        normalize: Whether to normalize the clean content

    Returns:
        Processed Note object
    """
    path = Path(path)

    # Load raw content
    raw_content = load_note(path)

    # Strip markdown
    clean_content = strip_markdown(raw_content)

    # Optionally normalize
    if normalize:
        clean_content = normalize_text(clean_content)

    # Extract metadata
    metadata = extract_metadata(path, raw_content)

    # Generate ID from path (stable hash)
    note_id = hashlib.md5(str(path).encode()).hexdigest()[:12]

    return Note(
        id=note_id,
        raw_content=raw_content,
        clean_content=clean_content,
        metadata=metadata,
    )


class NoteStore:
    """
    Manages a collection of processed notes with persistence.

    Supports saving/loading to JSON format.
    """

    def __init__(self) -> None:
        """Initialize empty note store."""
        self._notes: dict[str, Note] = {}

    def add(self, note: Note) -> None:
        """Add a note to the store."""
        self._notes[note.id] = note

    def get(self, note_id: str) -> Note | None:
        """Get a note by ID."""
        return self._notes.get(note_id)

    def all(self) -> list[Note]:
        """Get all notes."""
        return list(self._notes.values())

    def __len__(self) -> int:
        """Return number of notes in store."""
        return len(self._notes)

    def save(self, path: Path) -> None:
        """
        Save note store to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"notes": [note.to_dict() for note in self._notes.values()]}
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, path: Path) -> "NoteStore":
        """
        Load note store from JSON file.

        Args:
            path: Input file path

        Returns:
            NoteStore instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Note store file not found: {path}")

        data = orjson.loads(path.read_bytes())
        store = cls()

        for note_data in data.get("notes", []):
            note = Note.from_dict(note_data)
            store.add(note)

        return store


def ingest_vault(
    vault_path: Path,
    extensions: tuple[str, ...] = (".md", ".txt", ".pdf", ".docx"),
    normalize: bool = True,
) -> NoteStore:
    """
    Ingest all notes from a vault directory.

    Args:
        vault_path: Path to vault directory
        extensions: File extensions to process
        normalize: Whether to normalize text

    Returns:
        NoteStore containing all processed notes
    """
    vault_path = Path(vault_path)
    store = NoteStore()

    for note_path in discover_notes(vault_path, extensions):
        note = process_note(note_path, normalize=normalize)
        store.add(note)

    return store
