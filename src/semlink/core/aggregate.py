"""
Topic Aggregation Module.

Groups notes by detected communities and generates topic summaries.
"""

from __future__ import annotations
import re

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from semlink.core.analysis import detect_communities_louvain
from semlink.core.tfidf import TFIDFEmbedder


@dataclass
class Topic:
    """Represents an aggregated topic cluster."""

    id: int
    label: str
    keywords: list[str]
    note_ids: list[str]
    note_titles: list[str]
    size: int
    central_notes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    generation_method: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "keywords": self.keywords,
            "note_ids": self.note_ids,
            "note_titles": self.note_titles,
            "size": self.size,
            "central_notes": self.central_notes,
            "confidence": self.confidence,
            "generation_method": self.generation_method,
        }


@dataclass
class TopicAggregation:
    """Result of topic aggregation."""

    topics: list[Topic]
    note_to_topic: dict[str, int]
    orphan_notes: list[str]  # Notes not in any topic

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topics": [t.to_dict() for t in self.topics],
            "note_to_topic": self.note_to_topic,
            "orphan_notes": self.orphan_notes,
        }


CODE_KEYWORDS = {
    "int",
    "float",
    "double",
    "char",
    "void",
    "bool",
    "return",
    "printf",
    "scanf",
    "malloc",
    "free",
    "sizeof",
    "if",
    "else",
    "for",
    "while",
    "switch",
    "case",
    "break",
    "continue",
    "struct",
    "class",
    "public",
    "private",
    "protected",
    "def",
    "import",
    "from",
    "export",
    "async",
    "await",
    "let",
    "const",
    "var",
    "function",
    "module",
    "require",
    "include",
    "using",
    "namespace",
    "new",
    "delete",
    "this",
    "super",
    "null",
    "true",
    "false",
}


def extract_topic_keywords(
    texts: list[str],
    n_keywords: int = 5,
) -> list[str]:
    """
    Extract representative keywords from a collection of texts.

    Uses TF-IDF to find the most important terms,
    filtering out code keywords and short non-meaningful words.
    """
    if not texts:
        return []

    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    combined = " ".join(texts)
    if not combined.strip():
        return []

    try:
        embedder = TFIDFEmbedder(
            max_features=1000,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
        )
        embedder.fit_encode([combined])

        if embedder.vectorizer is None:
            return []

        feature_names = embedder.vectorizer.get_feature_names_out()
        tfidf_matrix = embedder.vectorizer.transform([combined])

        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-n_keywords * 2 :][::-1]

        keywords = []
        for i in top_indices:
            if scores[i] > 0:
                kw = str(feature_names[i])
                if is_meaningful_keyword(kw):
                    keywords.append(kw)
                if len(keywords) >= n_keywords:
                    break

        return keywords
    except Exception:
        words = combined.lower().split()
        word_counts: dict[str, int] = {}
        for word in words:
            if len(word) > 3 and word.isalpha() and is_meaningful_keyword(word):
                word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return []

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        return [w[0] for w in sorted_words[:n_keywords]]


def is_meaningful_keyword(keyword: str) -> bool:
    """Check if a keyword is meaningful for topic labels."""
    kw = keyword.lower()
    if len(kw) < 3:
        return False
    if kw in CODE_KEYWORDS:
        return False
    if kw.isdigit():
        return False
    if re.fullmatch(r"[a-z]{1,2}\d+", kw):
        return False
    if re.fullmatch(r"\d+[a-z]{1,2}", kw):
        return False
    return True


def generate_topic_label(
    keywords: list[str],
    note_titles: list[str],
    central_note_ids: list[str] | None = None,
    note_ids: list[str] | None = None,
) -> str:
    """
    Generate a human-readable topic label.

    Priority:
    1. Most central note's title (best for topic clarity)
    2. If no central notes, use keywords (but improved)
    3. Fall back to keywords + word cloud from titles

    Args:
        keywords: TF-IDF keywords for the topic
        note_titles: All note titles in the topic
        central_note_ids: IDs of most central notes (by PageRank)
        note_ids: All note IDs in order
    """
    if central_note_ids and note_ids and note_titles:
        if central_note_ids and len(central_note_ids) > 0:
            central_idx = note_ids.index(central_note_ids[0])
            if central_idx < len(note_titles):
                central_title = note_titles[central_idx]
                if central_title and len(central_title.strip()) > 0:
                    clean_title = clean_topic_label(central_title)
                    if clean_title:
                        return clean_title

    if keywords:
        meaningful_keywords = [
            k for k in keywords if len(k) > 2 and not is_common_word(k)
        ]
        if meaningful_keywords:
            return meaningful_keywords[0].title()

    if note_titles:
        common = find_common_words(note_titles, min_len=4, top_n=3)
        if common:
            return common[0].title() if common else "Topic"

    return "General Notes"


def clean_topic_label(title: str) -> str:
    """Clean a note title to use as topic label."""
    cleaned = re.sub(r"\d+Bai\d+", "", title)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_common_word(word: str) -> bool:
    """Check if word is too common to be meaningful."""
    common = {
        "note",
        "notes",
        "file",
        "files",
        "data",
        "information",
        "content",
        "document",
        "documents",
        "writing",
        "writing",
        "thought",
        "thoughts",
        "idea",
        "ideas",
        "summary",
        "review",
        "test",
        "question",
        "answer",
        "examples",
        "notes",
    }
    return word.lower() in common


def find_common_words(
    note_titles: list[str], min_len: int = 4, top_n: int = 3
) -> list[str]:
    """Find the most common words across note titles."""
    word_counts: dict[str, int] = defaultdict(int)
    for title in note_titles:
        words = re.findall(r"\b[a-zA-Z]{" + str(min_len) + r",}\b", title.lower())
        for word in words:
            if not is_common_word(word):
                word_counts[word] += 1
    if not word_counts:
        words = re.findall(
            r"\b[a-zA-Z]{" + str(min_len) + r",}\b", " ".join(note_titles).lower()
        )
        for word in set(words):
            word_counts[word] = 1
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    return [w[0] for w in sorted_words[:top_n]]


def aggregate_by_topic(
    graph: nx.Graph,
    notes: dict[str, dict[str, Any]],
    min_cluster_size: int = 2,
    resolution: float = 1.0,
    n_keywords: int = 5,
    topic_generator: Any = None,
) -> TopicAggregation:
    """
    Group notes by detected communities and generate topic summaries.

    Args:
        graph: NetworkX graph with notes as nodes
        notes: Dict mapping note_id to note data (must have 'title' and 'content' or 'clean_content')
        min_cluster_size: Minimum notes to form a topic (smaller clusters become orphans)
        resolution: Louvain resolution parameter (higher = more clusters)
        n_keywords: Number of keywords to extract per topic
        topic_generator: Optional TopicGenerator for LLM-based labels

    Returns:
        TopicAggregation with topics, mappings, and orphan notes
    """
    # Detect communities
    communities = detect_communities_louvain(graph, resolution=resolution)

    # Calculate centrality for finding representative notes
    try:
        pagerank = nx.pagerank(graph)
    except nx.NetworkXError:
        pagerank = {n: 1.0 / len(graph) for n in graph.nodes()}

    topics = []
    note_to_topic: dict[str, int] = {}
    orphan_notes: list[str] = []

    for topic_id, community in enumerate(communities):
        note_ids = list(community)

        # Skip small clusters
        if len(note_ids) < min_cluster_size:
            orphan_notes.extend(note_ids)
            continue

        # Get note data
        note_titles = []
        texts = []
        for nid in note_ids:
            if nid in notes:
                note_data = notes[nid]
                note_titles.append(note_data.get("title", nid))
                content = note_data.get("clean_content") or note_data.get("content", "")
                texts.append(content)
            else:
                # Use graph node attributes as fallback
                node_data = graph.nodes.get(nid, {})
                note_titles.append(node_data.get("title", nid))

        # Extract keywords from cluster content
        keywords = extract_topic_keywords(texts, n_keywords=n_keywords)

        # Find central notes FIRST (highest PageRank in cluster)
        central_notes = sorted(
            note_ids,
            key=lambda n: pagerank.get(n, 0),
            reverse=True,
        )[:3]

        # Generate label - use LLM if available, otherwise fallback
        confidence = 0.5
        generation_method = "keywords"
        if topic_generator is not None:
            try:
                llm_label = topic_generator.generate_topic_label(note_titles, texts)
                if llm_label and llm_label.label and llm_label.confidence > 0.5:
                    label = llm_label.label
                    confidence = llm_label.confidence
                    generation_method = llm_label.method or "llm"
                else:
                    label = generate_topic_label(
                        keywords, note_titles, central_notes, note_ids
                    )
            except Exception:
                label = generate_topic_label(
                    keywords, note_titles, central_notes, note_ids
                )
        else:
            label = generate_topic_label(keywords, note_titles, central_notes, note_ids)

        topic = Topic(
            id=topic_id,
            label=label,
            keywords=keywords,
            note_ids=note_ids,
            note_titles=note_titles,
            size=len(note_ids),
            central_notes=central_notes,
            confidence=confidence,
            generation_method=generation_method,
        )
        topics.append(topic)

        # Update mapping
        for nid in note_ids:
            note_to_topic[nid] = topic_id

    # Sort topics by size
    topics.sort(key=lambda t: -t.size)

    # Reassign IDs after sorting
    for i, topic in enumerate(topics):
        topic.id = i
        for nid in topic.note_ids:
            note_to_topic[nid] = i

    return TopicAggregation(
        topics=topics,
        note_to_topic=note_to_topic,
        orphan_notes=orphan_notes,
    )


def export_topics_markdown(
    aggregation: TopicAggregation,
    notes: dict[str, dict[str, Any]],
    output_dir: Path,
    include_content: bool = True,
) -> list[Path]:
    """
    Export topics as Markdown files.

    Creates one file per topic with links to all notes in the topic.

    Args:
        aggregation: Topic aggregation result
        notes: Dict mapping note_id to note data
        output_dir: Directory to write files to
        include_content: Whether to include note content in topic files

    Returns:
        List of created file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for topic in aggregation.topics:
        filename = f"topic_{topic.id:02d}_{topic.label.lower().replace(' & ', '_').replace(' ', '_')}.md"
        filepath = output_dir / filename

        lines = [
            f"# {topic.label}",
            "",
            f"**Keywords:** {', '.join(topic.keywords)}",
            f"**Notes:** {topic.size}",
            "",
            "## Notes in this topic",
            "",
        ]

        for note_id in topic.note_ids:
            note_data = notes.get(note_id, {})
            title = note_data.get("title", note_id)
            lines.append(f"### {title}")
            lines.append("")

            if include_content:
                content = note_data.get("clean_content") or note_data.get("content", "")
                if content:
                    # Truncate long content
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    lines.append(content)
                    lines.append("")

            lines.append("---")
            lines.append("")

        filepath.write_text("\n".join(lines))
        created_files.append(filepath)

    # Create index file
    index_path = output_dir / "index.md"
    index_lines = [
        "# Topic Index",
        "",
        f"Total topics: {len(aggregation.topics)}",
        f"Orphan notes: {len(aggregation.orphan_notes)}",
        "",
        "## Topics",
        "",
    ]

    for topic in aggregation.topics:
        filename = f"topic_{topic.id:02d}_{topic.label.lower().replace(' & ', '_').replace(' ', '_')}.md"
        index_lines.append(f"- [{topic.label}]({filename}) ({topic.size} notes)")

    if aggregation.orphan_notes:
        index_lines.extend(
            [
                "",
                "## Uncategorized Notes",
                "",
            ]
        )
        for nid in aggregation.orphan_notes:
            note_data = notes.get(nid, {})
            title = note_data.get("title", nid)
            index_lines.append(f"- {title}")

    index_path.write_text("\n".join(index_lines))
    created_files.append(index_path)

    return created_files


def export_topics_json(
    aggregation: TopicAggregation,
    output_path: Path,
) -> Path:
    """
    Export topics as JSON.

    Args:
        aggregation: Topic aggregation result
        output_path: Path to write JSON file

    Returns:
        Path to created file
    """
    import orjson

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        orjson.dumps(aggregation.to_dict(), option=orjson.OPT_INDENT_2)
    )

    return output_path


def export_topics_obsidian(
    aggregation: TopicAggregation,
    notes: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Export topics in Obsidian-compatible folder structure.

    Creates folders for each topic with note files inside.

    Args:
        aggregation: Topic aggregation result
        notes: Dict mapping note_id to note data
        output_dir: Root directory for Obsidian vault

    Returns:
        Path to root directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for topic in aggregation.topics:
        # Create topic folder
        folder_name = f"{topic.id:02d} - {topic.label}"
        topic_dir = output_dir / folder_name
        topic_dir.mkdir(exist_ok=True)

        # Create topic index
        index_lines = [
            f"# {topic.label}",
            "",
            f"Keywords: {', '.join(topic.keywords)}",
            "",
            "## Notes",
            "",
        ]

        for note_id in topic.note_ids:
            note_data = notes.get(note_id, {})
            title = note_data.get("title", note_id)
            # Link to note file
            safe_title = title.replace("/", "-").replace("\\", "-")
            index_lines.append(f"- [[{safe_title}]]")

            # Create note file
            content = note_data.get("content") or note_data.get("clean_content", "")
            note_content = [
                f"# {title}",
                "",
                f"Topic: [[{topic.label}]]",
                "",
                content,
            ]
            note_path = topic_dir / f"{safe_title}.md"
            note_path.write_text("\n".join(note_content))

        # Write topic index
        (topic_dir / f"{topic.label}.md").write_text("\n".join(index_lines))

    # Handle orphans
    if aggregation.orphan_notes:
        orphan_dir = output_dir / "Uncategorized"
        orphan_dir.mkdir(exist_ok=True)

        for note_id in aggregation.orphan_notes:
            note_data = notes.get(note_id, {})
            title = note_data.get("title", note_id)
            content = note_data.get("content") or note_data.get("clean_content", "")

            safe_title = title.replace("/", "-").replace("\\", "-")
            note_path = orphan_dir / f"{safe_title}.md"
            note_path.write_text(f"# {title}\n\n{content}")

    return output_dir
