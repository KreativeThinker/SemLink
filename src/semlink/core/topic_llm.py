"""
LLM-Based Topic Generation Module.

Uses large language models to generate descriptive topic labels
from clusters of semantically related notes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TopicLabel:
    """A generated topic label with metadata."""

    label: str
    confidence: float = 1.0
    reasoning: str | None = None
    method: str = "llm"


class TopicGenerator(ABC):
    """Abstract base class for topic label generation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return generator name."""
        ...

    @abstractmethod
    def generate_topic_label(
        self,
        note_titles: list[str],
        note_contents: list[str],
    ) -> TopicLabel:
        """
        Generate a topic label from a cluster of notes.

        Args:
            note_titles: List of note titles in the cluster
            note_contents: List of content excerpts from each note

        Returns:
            TopicLabel with the generated topic name
        """
        ...


class OpenAITopicGenerator(TopicGenerator):
    """
    OpenAI-based topic label generator.

    Uses GPT-4o or similar to analyze note clusters
    and produce descriptive topic labels.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        """
        Initialize OpenAI topic generator.

        Args:
            model_name: Model to use (default: gpt-4o-mini for cost)
            temperature: Sampling temperature (lower = more focused)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.temperature = temperature
        self._client = None

    @property
    def name(self) -> str:
        return f"openai:{self.model_name}"

    def _load_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for LLM topic generation. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI()
        return self._client

    def generate_topic_label(
        self,
        note_titles: list[str],
        note_contents: list[str],
    ) -> TopicLabel:
        """Generate topic label using OpenAI."""
        client = self._load_client()

        system_prompt = """You are an expert at analyzing clusters of related documents
and identifying their shared theme or topic.

Your task is to give this cluster of notes a short, descriptive label (2-4 words).

Guidelines:
- If notes are about a person, use the person's name (e.g., "John Doe", "Dr. Smith")
- If notes are about an academic subject, use the subject name (e.g., "Computer Networks")
- If notes are about a project, use the project name
- If notes are about a concept, use the concept name
- Avoid generic labels like "Notes", "Ideas", "Miscellaneous"
- Use title case (e.g., "Machine Learning", "Quantum Physics")

Respond ONLY with the topic label, nothing else."""

        user_content = self._build_user_content(note_titles, note_contents)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_tokens=50,
            )

            label = response.choices[0].message.content.strip()

            label = self._clean_label(label)

            if not label or len(label) < 2:
                return TopicLabel(
                    label="General Topic",
                    confidence=0.5,
                    reasoning="LLM returned empty response",
                    method=self.name,
                )

            return TopicLabel(
                label=label,
                confidence=0.9,
                reasoning=f"Generated from {len(note_titles)} notes",
                method=self.name,
            )

        except Exception as e:
            return TopicLabel(
                label="Topic",
                confidence=0.1,
                reasoning=f"Error: {str(e)[:100]}",
                method=self.name,
            )

    def _build_user_content(
        self,
        note_titles: list[str],
        note_contents: list[str],
    ) -> str:
        """Build user prompt with note information."""
        lines = [
            f"These {len(note_titles)} notes appear to be related:",
        ]

        for i, (title, content) in enumerate(zip(note_titles, note_contents)):
            truncated_title = title[:80] if title else f"Note {i + 1}"
            truncated_content = content[:300] if content else ""

            lines.append(f"## {truncated_title}")
            if truncated_content:
                lines.append(truncated_content[:500])
            lines.append("")

        lines.extend(
            [
                "What single topic or theme do these notes share?",
                "Respond with a 2-4 word topic label.",
            ]
        )

        return "\n".join(lines)

    def _clean_label(self, label: str) -> str:
        """Clean and validate the generated label."""
        label = label.strip()

        label = label.replace('"', "").replace("'", "").strip()

        bad_labels = {
            "miscellaneous",
            "notes",
            "misc",
            "various",
            "general",
            "mixed",
            "multiple",
            "topic",
            "none",
            "nothing",
        }
        if label.lower() in bad_labels:
            return ""

        return label[:50]


class FallbackTopicGenerator(TopicGenerator):
    """Fallback using keyword extraction when LLM is unavailable."""

    def __init__(self) -> None:
        self._delegate = None

    @property
    def name(self) -> str:
        return "fallback:keywords"

    def generate_topic_label(
        self,
        note_titles: list[str],
        note_contents: list[str],
    ) -> TopicLabel:
        """Generate label using keyword extraction."""
        if self._delegate is None:
            from semlink.core.aggregate import (
                generate_topic_label,
                extract_topic_keywords,
            )

            self._delegate = (generate_topic_label, extract_topic_keywords)

        gen_label, extract_kw = self._delegate

        combined = " ".join(note_contents)
        keywords = extract_kw([combined], n_keywords=5)

        label = gen_label(keywords, note_titles)

        return TopicLabel(
            label=label,
            confidence=0.5,
            reasoning="Generated from keywords (LLM unavailable)",
            method=self.name,
        )


def create_topic_generator(
    method: str = "openai",
    model: str | None = None,
    **kwargs,
) -> TopicGenerator:
    """
    Create a topic generator based on method.

    Args:
        method: Generator method ("openai" or "fallback")
        model: Model name (for OpenAI)
        **kwargs: Additional options

    Returns:
        TopicGenerator instance
    """
    if method == "openai":
        return OpenAITopicGenerator(model_name=model, **kwargs)

    return FallbackTopicGenerator()


def is_available(method: str = "openai") -> bool:
    """
    Check if a topic generator method is available.

    Args:
        method: Generator method to check

    Returns:
        True if method is available and usable
    """
    if method == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return False
        try:
            client = OpenAI()
            client.models.list()
            return True
        except Exception:
            return False
    return True


def batch_generate_topics(
    generator: TopicGenerator,
    clusters: list[dict[str, Any]],
    batch_size: int = 10,
) -> list[TopicLabel]:
    """
    Generate topic labels for multiple clusters.

    Args:
        generator: TopicGenerator instance
        clusters: List of cluster dicts with 'note_titles' and 'note_contents'
        batch_size: Limit concurrent calls if needed

    Returns:
        List of TopicLabel objects
    """
    results = []

    for cluster in clusters:
        titles = cluster.get("note_titles", [])
        contents = cluster.get("note_contents", [])

        label = generator.generate_topic_label(titles, contents)
        results.append(label)

    return results
