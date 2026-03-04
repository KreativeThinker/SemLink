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

from semlink.core.graph import load_json
from semlink.core.storage import SemLinkDB


class GraphResponse(BaseModel):
    """Graph data for frontend."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class StatsResponse(BaseModel):
    """Database statistics."""

    notes: int
    embeddings: dict[str, int]
    edges: dict[str, int]


def create_app(
    db_path: Path | None = None,
    graph_path: Path | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        db_path: Path to SQLite database (optional)
        graph_path: Path to graph JSON file (optional)
        static_dir: Path to static files directory (frontend build)
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

    @app.get("/api/graph", response_model=GraphResponse)
    async def get_graph(
        method: Annotated[str | None, Query(description="Filter by method")] = None,
        min_weight: Annotated[
            float, Query(description="Minimum edge weight", ge=0, le=1)
        ] = 0.0,
    ) -> GraphResponse:
        """Get graph data for visualization."""
        # Try database first
        if app.state.db_path and Path(app.state.db_path).exists():
            db = SemLinkDB(app.state.db_path)
            notes = db.get_all_notes()
            edges = db.get_edges(method)

            nodes = [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.clean_content[:500] if n.clean_content else None,
                }
                for n in notes
            ]

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

            nodes = [
                {
                    "id": node,
                    "title": data.get("title", node),
                    "community": data.get("community"),
                    "centrality": data.get("centrality"),
                }
                for node, data in graph.nodes(data=True)
            ]

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
