"""
SemLink API Server.

Provides REST API for the React frontend and serves static files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import networkx as nx

from semlink.core.aggregate import aggregate_by_topic
from semlink.core.graph import load_json
from semlink.core.storage import SemLinkDB


class GraphResponse(BaseModel):
    """Graph data for frontend."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class TopicResponse(BaseModel):
    """Single topic data."""

    id: int
    label: str
    keywords: list[str]
    note_ids: list[str]
    note_titles: list[str]
    size: int
    central_notes: list[str]


class TopicsResponse(BaseModel):
    """All topics data for frontend."""

    topics: list[TopicResponse]
    note_to_topic: dict[str, int]
    orphan_notes: list[str]


class StatsResponse(BaseModel):
    """Database statistics."""

    notes: int
    embeddings: dict[str, int]
    edges: dict[str, int]


def _build_graph_from_db(db: SemLinkDB, method: str | None = None) -> nx.Graph:
    """Build a NetworkX graph from database."""
    graph = nx.Graph()
    notes = db.get_all_notes()
    edges = db.get_edges(method)

    # Add nodes
    for note in notes:
        graph.add_node(
            note.id,
            title=note.title,
            content=note.clean_content[:500] if note.clean_content else "",
        )

    # Add edges
    for edge in edges:
        graph.add_edge(
            edge.source_id,
            edge.target_id,
            weight=edge.weight,
            reason=edge.reason,
            shared_terms=edge.shared_terms,
        )

    return graph


def _get_topic_generator(state: Any) -> Any:
    """Get the topic generator from app state."""
    if hasattr(state, "topic_generator") and state.topic_generator is not None:
        return state.topic_generator
    return None


def _compute_topics_for_graph(
    graph: nx.Graph,
    notes_dict: dict[str, dict[str, Any]],
    resolution: float = 1.0,
    topic_generator: Any = None,
) -> dict[str, Any]:
    """Compute topic aggregation for a graph."""
    if graph.number_of_nodes() == 0:
        return {
            "topics": [],
            "note_to_topic": {},
            "orphan_notes": [],
            "topic_labels": {},
        }

    aggregation = aggregate_by_topic(
        graph=graph,
        notes=notes_dict,
        min_cluster_size=2,
        resolution=resolution,
        n_keywords=5,
        topic_generator=topic_generator,
    )

    # Build topic_id -> label mapping
    topic_labels = {t.id: t.label for t in aggregation.topics}

    return {
        "topics": aggregation.topics,
        "note_to_topic": aggregation.note_to_topic,
        "orphan_notes": aggregation.orphan_notes,
        "topic_labels": topic_labels,
    }


def _load_notes_from_file(notes_path: Path) -> dict[str, dict[str, Any]]:
    """Load notes from JSON file and return as dict."""
    import orjson

    if not notes_path.exists():
        return {}

    data = orjson.loads(notes_path.read_bytes())
    notes_list = data.get("notes", data) if isinstance(data, dict) else data

    return {
        n["id"]: {
            "title": n.get("metadata", {}).get("title", n.get("id", "")),
            "content": n.get("raw_content", ""),
            "clean_content": n.get("clean_content", ""),
        }
        for n in notes_list
        if isinstance(n, dict) and "id" in n
    }


def create_app(
    db_path: Path | None = None,
    graph_path: Path | None = None,
    notes_path: Path | None = None,
    static_dir: Path | None = None,
    topic_method: str = "llm",
    topic_model: str = "gpt-4o-mini",
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        db_path: Path to SQLite database (optional)
        graph_path: Path to graph JSON file (optional)
        notes_path: Path to notes JSON file (optional, for content with graph files)
        static_dir: Path to static files directory (frontend build)
        topic_method: Topic generation method ("llm" or "keywords")
        topic_model: LLM model for topic generation
    """
    app = FastAPI(
        title="SemLink API",
        description="REST API for semantic note linking",
        version="0.1.0",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state
    app.state.db_path = db_path
    app.state.graph_path = graph_path
    app.state.notes_path = notes_path
    app.state.topic_method = topic_method
    app.state.topic_model = topic_model

    # Initialize topic generator
    app.state.topic_generator = None
    if topic_method == "llm":
        try:
            from semlink.core.topic_llm import create_topic_generator, is_available

            if is_available("openai"):
                app.state.topic_generator = create_topic_generator(
                    "openai", topic_model
                )
        except Exception:
            pass

    # Pre-load notes if path provided
    app.state.notes_cache = {}
    if notes_path and notes_path.exists():
        app.state.notes_cache = _load_notes_from_file(notes_path)

    @app.get("/api/graph", response_model=GraphResponse)
    async def get_graph(
        method: Annotated[str | None, Query(description="Filter by method")] = None,
        min_weight: Annotated[
            float, Query(description="Minimum edge weight", ge=0, le=1)
        ] = 0.0,
        include_topics: Annotated[
            bool, Query(description="Include topic labels on nodes")
        ] = True,
    ) -> GraphResponse:
        """Get graph data for visualization."""
        # Try database first
        if app.state.db_path and Path(app.state.db_path).exists():
            db = SemLinkDB(app.state.db_path)
            db_notes = db.get_all_notes()
            edges = db.get_edges(method)

            # Build graph for topic detection
            graph = _build_graph_from_db(db, method)

            # Compute topics if requested
            note_to_topic: dict[str, int] = {}
            topic_labels: dict[int, str] = {}
            if include_topics and graph.number_of_nodes() > 0:
                notes_dict = {
                    n.id: {
                        "title": n.title,
                        "content": n.content,
                        "clean_content": n.clean_content,
                    }
                    for n in db_notes
                }
                topic_data = _compute_topics_for_graph(
                    graph, notes_dict, topic_generator=app.state.topic_generator
                )
                note_to_topic = topic_data["note_to_topic"]
                topic_labels = topic_data["topic_labels"]

            # Compute centrality
            try:
                centrality = nx.pagerank(graph)
            except Exception:
                centrality = {n: 0.0 for n in graph.nodes()}

            nodes = []
            for n in db_notes:
                topic_id = note_to_topic.get(n.id)
                node_data = {
                    "id": n.id,
                    "title": n.title,
                    "content": n.clean_content[:500] if n.clean_content else None,
                    "community": topic_id,
                    "centrality": centrality.get(n.id, 0.0),
                }
                if topic_id is not None:
                    node_data["topic_id"] = topic_id
                    node_data["topic_label"] = topic_labels.get(
                        topic_id, f"Topic {topic_id}"
                    )
                nodes.append(node_data)

            edge_list = [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "weight": e.weight,
                    "reason": e.reason,
                    "shared_terms": e.shared_terms,
                }
                for e in edges
                if e.weight >= min_weight
            ]

            return GraphResponse(nodes=nodes, edges=edge_list)

        # Fall back to graph file
        if app.state.graph_path and Path(app.state.graph_path).exists():
            graph = load_json(app.state.graph_path)

            # Compute topics for graph file
            note_to_topic: dict[str, int] = {}
            topic_labels: dict[int, str] = {}
            notes_cache = app.state.notes_cache

            if include_topics and graph.number_of_nodes() > 0:
                # Build notes dict - prefer cached notes for content
                notes_dict = {}
                for node, data in graph.nodes(data=True):
                    if node in notes_cache:
                        notes_dict[node] = notes_cache[node]
                    else:
                        notes_dict[node] = {
                            "title": data.get("title", node),
                            "content": data.get("content", ""),
                            "clean_content": data.get("content", ""),
                        }
                topic_data = _compute_topics_for_graph(
                    graph, notes_dict, topic_generator=app.state.topic_generator
                )
                note_to_topic = topic_data["note_to_topic"]
                topic_labels = topic_data["topic_labels"]

            # Compute centrality
            try:
                centrality = nx.pagerank(graph)
            except Exception:
                centrality = {n: 0.0 for n in graph.nodes()}

            nodes = []
            for node, data in graph.nodes(data=True):
                topic_id = note_to_topic.get(node)
                node_data = {
                    "id": node,
                    "title": data.get("title", node),
                    "community": topic_id
                    if topic_id is not None
                    else data.get("community"),
                    "centrality": centrality.get(node, data.get("centrality", 0.0)),
                }
                if topic_id is not None:
                    node_data["topic_id"] = topic_id
                    node_data["topic_label"] = topic_labels.get(
                        topic_id, f"Topic {topic_id}"
                    )
                nodes.append(node_data)

            edge_list = [
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                    "reason": data.get("reason"),
                    "shared_terms": data.get("shared_terms"),
                }
                for u, v, data in graph.edges(data=True)
                if data.get("weight", 1.0) >= min_weight
            ]

            return GraphResponse(nodes=nodes, edges=edge_list)

        raise HTTPException(status_code=404, detail="No graph data available")

    @app.get("/api/topics", response_model=TopicsResponse)
    async def get_topics(
        method: Annotated[str | None, Query(description="Filter by method")] = None,
        resolution: Annotated[
            float, Query(description="Topic detection resolution", ge=0.1, le=3.0)
        ] = 1.0,
        min_cluster_size: Annotated[
            int, Query(description="Minimum notes per topic", ge=1)
        ] = 2,
    ) -> TopicsResponse:
        """Get detected topics from the graph."""
        # Try database first
        if app.state.db_path and Path(app.state.db_path).exists():
            db = SemLinkDB(app.state.db_path)
            db_notes = db.get_all_notes()

            # Build graph
            graph = _build_graph_from_db(db, method)

            if graph.number_of_nodes() == 0:
                return TopicsResponse(topics=[], note_to_topic={}, orphan_notes=[])

            # Build notes dict for aggregation
            notes_dict = {
                n.id: {
                    "title": n.title,
                    "content": n.content,
                    "clean_content": n.clean_content,
                }
                for n in db_notes
            }

            # Compute topics
            aggregation = aggregate_by_topic(
                graph=graph,
                notes=notes_dict,
                min_cluster_size=min_cluster_size,
                resolution=resolution,
                n_keywords=5,
                topic_generator=_get_topic_generator(app.state),
            )

            topics = [
                TopicResponse(
                    id=t.id,
                    label=t.label,
                    keywords=t.keywords,
                    note_ids=t.note_ids,
                    note_titles=t.note_titles,
                    size=t.size,
                    central_notes=t.central_notes,
                )
                for t in aggregation.topics
            ]

            return TopicsResponse(
                topics=topics,
                note_to_topic=aggregation.note_to_topic,
                orphan_notes=aggregation.orphan_notes,
            )

        # Fall back to graph file
        if app.state.graph_path and Path(app.state.graph_path).exists():
            graph = load_json(app.state.graph_path)

            if graph.number_of_nodes() == 0:
                return TopicsResponse(topics=[], note_to_topic={}, orphan_notes=[])

            # Build notes dict - prefer cached notes for content
            notes_cache = app.state.notes_cache
            notes_dict = {}
            for node, data in graph.nodes(data=True):
                if node in notes_cache:
                    notes_dict[node] = notes_cache[node]
                else:
                    notes_dict[node] = {
                        "title": data.get("title", node),
                        "content": data.get("content", ""),
                        "clean_content": data.get("content", ""),
                    }

            # Compute topics
            aggregation = aggregate_by_topic(
                graph=graph,
                notes=notes_dict,
                min_cluster_size=min_cluster_size,
                resolution=resolution,
                n_keywords=5,
                topic_generator=_get_topic_generator(app.state),
            )

            topics = [
                TopicResponse(
                    id=t.id,
                    label=t.label,
                    keywords=t.keywords,
                    note_ids=t.note_ids,
                    note_titles=t.note_titles,
                    size=t.size,
                    central_notes=t.central_notes,
                )
                for t in aggregation.topics
            ]

            return TopicsResponse(
                topics=topics,
                note_to_topic=aggregation.note_to_topic,
                orphan_notes=aggregation.orphan_notes,
            )

        raise HTTPException(status_code=404, detail="No graph data available")

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """Get database statistics."""
        if not app.state.db_path or not Path(app.state.db_path).exists():
            raise HTTPException(status_code=404, detail="Database not found")

        db = SemLinkDB(app.state.db_path)
        stats = db.get_stats()

        return StatsResponse(
            notes=stats["notes"],
            embeddings=stats.get("embeddings", {}),
            edges=stats.get("edges", {}),
        )

    @app.get("/api/notes/{note_id}")
    async def get_note(note_id: str) -> dict[str, Any]:
        """Get a single note by ID."""
        if not app.state.db_path or not Path(app.state.db_path).exists():
            raise HTTPException(status_code=404, detail="Database not found")

        db = SemLinkDB(app.state.db_path)
        note = db.get_note(note_id)

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return {
            "id": note.id,
            "title": note.title,
            "path": note.path,
            "content": note.content,
            "clean_content": note.clean_content,
            "metadata": note.metadata,
        }

    @app.get("/api/notes")
    async def list_notes(
        search: Annotated[str | None, Query(description="Search query")] = None,
        limit: Annotated[int, Query(description="Max results", ge=1, le=1000)] = 100,
    ) -> list[dict[str, Any]]:
        """List all notes."""
        if not app.state.db_path or not Path(app.state.db_path).exists():
            raise HTTPException(status_code=404, detail="Database not found")

        db = SemLinkDB(app.state.db_path)
        notes = db.get_all_notes()

        if search:
            search_lower = search.lower()
            notes = [
                n
                for n in notes
                if search_lower in n.title.lower()
                or search_lower in n.clean_content.lower()
            ]

        return [
            {
                "id": n.id,
                "title": n.title,
                "path": n.path,
            }
            for n in notes[:limit]
        ]

    # Serve static files (frontend) if directory provided
    if static_dir and static_dir.exists():
        # Serve index.html for all non-API routes (SPA routing)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            # Check if it's a static file
            file_path = static_dir / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            # Otherwise serve index.html
            return FileResponse(static_dir / "index.html")

        # Mount static files
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app
