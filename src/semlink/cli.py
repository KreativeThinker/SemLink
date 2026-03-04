"""
SemLink CLI Interface.

Provides command-line access to semantic note linking functionality.
"""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from semlink.errors import AppError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Automatic semantic note linking for graph visualization and quick topic access",
)

console = Console()


def _parse_extensions(ext_str: str) -> tuple[str, ...]:
    """Parse comma-separated extensions string into tuple."""
    return tuple(e.strip() for e in ext_str.split(",") if e.strip())


# =============================================================================
# Ingest Command
# =============================================================================


@app.command()
def ingest(
    vault_path: Annotated[
        Path,
        typer.Argument(help="Path to vault directory containing notes"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = Path("notes.json"),
    chunk: Annotated[
        str,
        typer.Option(
            "--chunk", "-c", help="Chunking strategy: whole, paragraph, heading, window"
        ),
    ] = "whole",
    extensions: Annotated[
        str,
        typer.Option("--ext", help="File extensions to process (comma-separated)"),
    ] = ".md,.txt",
) -> None:
    """
    Ingest notes from a vault directory.

    Loads, preprocesses, and stores notes with metadata.
    """
    from semlink.core.chunk import build_chunk_index, get_strategy
    from semlink.core.ingest import ingest_vault

    try:
        ext_tuple = _parse_extensions(extensions)
        console.print(f"[bold]Ingesting notes from:[/bold] {vault_path}")
        console.print(f"[dim]Chunking strategy: {chunk}[/dim]")
        console.print(f"[dim]Extensions: {extensions}[/dim]")

        if not vault_path.is_dir():
            console.print(f"[red]Error:[/red] Not a directory: {vault_path}")
            raise typer.Exit(code=1)

        # Ingest notes
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading notes...", total=None)
            store = ingest_vault(vault_path, extensions=ext_tuple)
            progress.update(task, description=f"Loaded {len(store)} notes")

        console.print(f"[green]Loaded {len(store)} notes[/green]")

        # Apply chunking
        notes_for_chunking = [(note.id, note.clean_content) for note in store.all()]
        strategy = get_strategy(chunk)
        chunk_index = build_chunk_index(notes_for_chunking, strategy)

        console.print(f"[green]Created {len(chunk_index.chunks)} chunks[/green]")

        # Save both notes and chunks
        store.save(output)

        # Save chunk index alongside
        chunk_output = output.with_suffix(".chunks.json")
        import orjson

        chunk_output.write_bytes(
            orjson.dumps(chunk_index.to_dict(), option=orjson.OPT_INDENT_2)
        )

        console.print(f"[bold green]Notes saved to:[/bold green] {output}")
        console.print(f"[bold green]Chunks saved to:[/bold green] {chunk_output}")

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Embed Command
# =============================================================================


@app.command()
def embed(
    notes_path: Annotated[
        Path,
        typer.Argument(help="Path to notes JSON file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output embeddings file path (.npz)"),
    ] = Path("embeddings.npz"),
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Embedding method: tfidf, sbert, openai"),
    ] = "tfidf",
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="Specific model name"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for encoding"),
    ] = 32,
) -> None:
    """
    Generate embeddings for notes.

    Supports TF-IDF baseline and neural embedding methods.
    """
    from semlink.core.ingest import NoteStore

    try:
        console.print(f"[bold]Generating embeddings for:[/bold] {notes_path}")
        console.print(f"[dim]Method: {method}[/dim]")
        if model:
            console.print(f"[dim]Model: {model}[/dim]")

        if not notes_path.exists():
            console.print(f"[red]Error:[/red] File not found: {notes_path}")
            raise typer.Exit(code=1)

        # Load notes
        store = NoteStore.load(notes_path)
        notes = store.all()

        if not notes:
            console.print("[red]Error:[/red] No notes found in file")
            raise typer.Exit(code=1)

        ids = [note.id for note in notes]
        texts = [note.clean_content for note in notes]

        console.print(f"[dim]Processing {len(texts)} notes[/dim]")

        # Choose embedder based on method
        if method == "tfidf":
            from semlink.core.tfidf import TFIDFEmbedder

            embedder = TFIDFEmbedder()
            embeddings = embedder.fit_encode(texts)
            console.print(
                f"[dim]TF-IDF vocabulary size: {embedder.vocabulary_size}[/dim]"
            )

        elif method == "sbert":
            from semlink.core.embeddings import SBERTEmbedder

            model_name = model or "all-MiniLM-L6-v2"
            embedder = SBERTEmbedder(model_name=model_name)
            console.print(f"[dim]Using SBERT model: {model_name}[/dim]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Encoding texts...", total=None)
                embeddings = embedder.encode(texts, batch_size=batch_size)
                progress.update(task, description="Encoding complete")

        elif method == "openai":
            from semlink.core.embeddings import OpenAIEmbedder

            model_name = model or "text-embedding-3-small"
            embedder = OpenAIEmbedder(model_name=model_name)
            console.print(f"[dim]Using OpenAI model: {model_name}[/dim]")

            embeddings = embedder.encode(texts, batch_size=batch_size)

        else:
            console.print(
                f"[red]Error:[/red] Unknown method: {method}. Use tfidf, sbert, or openai"
            )
            raise typer.Exit(code=1)

        # Save embeddings with titles for visualization
        titles = [note.metadata.title for note in notes]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            embeddings=embeddings,
            ids=np.array(ids, dtype=object),
            titles=np.array(titles, dtype=object),
            method=method,
        )

        console.print(f"[bold green]Embeddings saved to:[/bold green] {output}")
        console.print(f"[dim]Shape: {embeddings.shape}[/dim]")

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print(
            "[dim]Install optional dependencies: pip install semlink[sbert] or semlink[openai][/dim]"
        )
        raise typer.Exit(code=1)
    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Link Command
# =============================================================================


@app.command()
def link(
    embeddings_path: Annotated[
        Path,
        typer.Argument(help="Path to embeddings file (.npz)"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output graph file path (.json)"),
    ] = Path("graph.json"),
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy", "-s", help="Link strategy: threshold, knn, mutual_knn, hybrid"
        ),
    ] = "hybrid",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of neighbors for kNN strategies"),
    ] = 7,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Similarity threshold"),
    ] = 0.5,
    min_weight: Annotated[
        float,
        typer.Option(
            "--min-weight",
            "-m",
            help="Minimum edge weight to keep (filters weak links)",
        ),
    ] = 0.0,
) -> None:
    """
    Infer links between notes based on similarity.

    Converts embeddings into a knowledge graph.
    """
    from semlink.core.graph import build_graph, export_json
    from semlink.core.linker import (
        HybridStrategy,
        KNNStrategy,
        MutualKNNStrategy,
        ThresholdStrategy,
        filter_edges,
    )

    try:
        console.print(f"[bold]Building graph from:[/bold] {embeddings_path}")
        console.print(f"[dim]Strategy: {strategy}, k={k}, threshold={threshold}[/dim]")
        if min_weight > 0:
            console.print(f"[dim]Minimum weight filter: {min_weight}[/dim]")

        if not embeddings_path.exists():
            console.print(f"[red]Error:[/red] File not found: {embeddings_path}")
            raise typer.Exit(code=1)

        # Load embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        ids = list(data["ids"])
        # Load titles if available (for visualization)
        titles = list(data["titles"]) if "titles" in data else None

        console.print(
            f"[dim]Loaded {len(ids)} embeddings with shape {embeddings.shape}[/dim]"
        )

        # Choose strategy
        if strategy == "threshold":
            linker = ThresholdStrategy(threshold=threshold)
        elif strategy == "knn":
            linker = KNNStrategy(k=k)
        elif strategy == "mutual_knn":
            linker = MutualKNNStrategy(k=k)
        elif strategy == "hybrid":
            linker = HybridStrategy(k=k, threshold=threshold)
        else:
            console.print(
                f"[red]Error:[/red] Unknown strategy: {strategy}. "
                "Use threshold, knn, mutual_knn, or hybrid"
            )
            raise typer.Exit(code=1)

        # Infer links
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Inferring links...", total=None)
            edges = linker.infer_links(embeddings, ids)
            progress.update(task, description=f"Found {len(edges)} links")

        # Filter weak links if min_weight specified
        if min_weight > 0:
            original_count = len(edges)
            edges = filter_edges(edges, min_weight=min_weight)
            filtered_count = original_count - len(edges)
            console.print(
                f"[dim]Filtered {filtered_count} weak links (weight < {min_weight})[/dim]"
            )

        console.print(f"[green]Inferred {len(edges)} links[/green]")

        # Build graph with titles for visualization
        if titles:
            nodes = [
                {"id": doc_id, "title": title} for doc_id, title in zip(ids, titles)
            ]
        else:
            nodes = [{"id": doc_id} for doc_id in ids]
        graph = build_graph(nodes, edges)

        # Export graph
        output.parent.mkdir(parents=True, exist_ok=True)
        export_json(graph, output)

        console.print(f"[bold green]Graph saved to:[/bold green] {output}")
        console.print(
            f"[dim]Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}[/dim]"
        )

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Analyze Command
# =============================================================================


@app.command()
def analyze(
    graph_path: Annotated[
        Path,
        typer.Argument(help="Path to graph file (.json)"),
    ],
    communities: Annotated[
        bool,
        typer.Option("--communities", "-c", help="Detect communities"),
    ] = True,
    centrality: Annotated[
        bool,
        typer.Option("--centrality", help="Compute centrality measures"),
    ] = True,
    resolution: Annotated[
        float,
        typer.Option("--resolution", "-r", help="Community detection resolution"),
    ] = 1.0,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Save updated graph with analysis"),
    ] = None,
) -> None:
    """
    Analyze graph structure and detect communities.

    Computes metrics, finds communities, and identifies central notes.
    """
    from semlink.core.analysis import (
        compute_centrality,
        compute_metrics,
        detect_communities_louvain,
    )
    from semlink.core.graph import export_json, load_json

    try:
        console.print(f"[bold]Analyzing graph:[/bold] {graph_path}")

        if not graph_path.exists():
            console.print(f"[red]Error:[/red] File not found: {graph_path}")
            raise typer.Exit(code=1)

        # Load graph
        graph = load_json(graph_path)
        console.print(
            f"[dim]Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges[/dim]"
        )

        # Compute basic metrics
        metrics = compute_metrics(graph)

        # Display metrics table
        table = Table(title="Graph Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Nodes", str(metrics["n_nodes"]))
        table.add_row("Edges", str(metrics["n_edges"]))
        table.add_row("Density", f"{metrics['density']:.4f}")
        table.add_row("Components", str(metrics["n_components"]))
        table.add_row("Avg Clustering", f"{metrics['avg_clustering']:.4f}")
        table.add_row("Transitivity", f"{metrics['transitivity']:.4f}")
        if "diameter" in metrics:
            table.add_row("Diameter", str(metrics["diameter"]))
        if "avg_path_length" in metrics:
            table.add_row("Avg Path Length", f"{metrics['avg_path_length']:.2f}")

        console.print(table)
        console.print()

        # Detect communities
        if communities:
            import networkx as nx

            console.print("[bold]Detecting communities...[/bold]")
            community_sets = detect_communities_louvain(graph, resolution=resolution)
            n_communities = len(community_sets)

            # Compute modularity
            modularity = nx.community.modularity(graph, community_sets)

            console.print(f"[green]Found {n_communities} communities[/green]")
            console.print(f"[dim]Modularity: {modularity:.4f}[/dim]")

            # Show community sizes
            sizes = sorted([len(c) for c in community_sets], reverse=True)
            console.print(
                "[dim]Community sizes: "
                + ", ".join(str(s) for s in sizes[:10])
                + "[/dim]"
            )

        # Compute centrality
        if centrality:
            console.print("[bold]Computing centrality measures...[/bold]")
            centrality_result = compute_centrality(
                graph, measures=["degree", "betweenness", "pagerank"]
            )

            # Show top nodes by PageRank
            pagerank = centrality_result.get("pagerank", {})
            if pagerank:
                top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]

                table2 = Table(title="Top Nodes by PageRank")
                table2.add_column("Node", style="cyan")
                table2.add_column("PageRank", justify="right")

                for node_id, score in top_nodes:
                    table2.add_row(str(node_id)[:40], f"{score:.4f}")

                console.print(table2)

        # Save updated graph if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            export_json(graph, output)
            console.print(f"[bold green]Updated graph saved to:[/bold green] {output}")

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Visualize Command
# =============================================================================


@app.command()
def visualize(
    graph_path: Annotated[
        Path,
        typer.Argument(help="Path to graph file (.json)"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("graph.html"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: html, png, d3, obsidian"),
    ] = "html",
) -> None:
    """
    Generate graph visualization.

    Creates interactive HTML or static image visualizations.
    """
    from semlink.core.graph import load_json
    from semlink.core.visualize import (
        to_d3_json,
        to_matplotlib,
        to_obsidian,
        to_pyvis,
    )

    try:
        console.print(f"[bold]Visualizing graph:[/bold] {graph_path}")
        console.print(f"[dim]Format: {format}[/dim]")

        if not graph_path.exists():
            console.print(f"[red]Error:[/red] File not found: {graph_path}")
            raise typer.Exit(code=1)

        # Load graph
        graph = load_json(graph_path)
        console.print(
            f"[dim]Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges[/dim]"
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            result_path = to_pyvis(graph, output)
            console.print(
                f"[bold green]Interactive visualization saved to:[/bold green] {result_path}"
            )

        elif format == "png":
            result_path = to_matplotlib(graph, output)
            console.print(
                f"[bold green]Static visualization saved to:[/bold green] {result_path}"
            )

        elif format == "d3":
            result_path = to_d3_json(graph, output)
            console.print(
                f"[bold green]D3.js JSON saved to:[/bold green] {result_path}"
            )

        elif format == "obsidian":
            result_path = to_obsidian(graph, output.parent)
            console.print(
                f"[bold green]Obsidian format saved to:[/bold green] {result_path}"
            )

        else:
            console.print(
                f"[red]Error:[/red] Unknown format: {format}. Use html, png, d3, or obsidian"
            )
            raise typer.Exit(code=1)

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print(
            "[dim]Install optional dependencies: pip install semlink[viz][/dim]"
        )
        raise typer.Exit(code=1)
    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Compare Command
# =============================================================================


@app.command()
def compare(
    embeddings: Annotated[
        list[Path],
        typer.Argument(help="Paths to embedding files to compare"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output report path"),
    ] = Path("comparison.md"),
) -> None:
    """
    Compare different embedding methods.

    Generates evaluation report with similarity distributions and graph metrics.
    """
    from semlink.core.evaluate import (
        compare_methods,
        generate_comparison_report,
        generate_plots,
    )

    try:
        console.print(f"[bold]Comparing {len(embeddings)} embedding files[/bold]")

        if len(embeddings) < 2:
            console.print(
                "[red]Error:[/red] Need at least 2 embedding files to compare"
            )
            raise typer.Exit(code=1)

        # Load embeddings
        emb_dict: dict[str, np.ndarray] = {}
        ids: list[str] = []

        for emb_path in embeddings:
            if not emb_path.exists():
                console.print(f"[red]Error:[/red] File not found: {emb_path}")
                raise typer.Exit(code=1)

            data = np.load(emb_path, allow_pickle=True)
            method_name = str(data.get("method", emb_path.stem))
            emb_dict[method_name] = data["embeddings"]

            if not ids:
                ids = [str(i) for i in data["ids"]]
            else:
                if [str(i) for i in data["ids"]] != ids:
                    console.print(
                        "[yellow]Warning:[/yellow] Embedding files have different IDs"
                    )

        console.print(f"[dim]Methods: {', '.join(emb_dict.keys())}[/dim]")
        console.print(f"[dim]Documents: {len(ids)}[/dim]")

        # Compare methods
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Comparing methods...", total=None)
            results = compare_methods(emb_dict, ids)
            progress.update(task, description="Comparison complete")

        # Generate report
        output.parent.mkdir(parents=True, exist_ok=True)
        report_format = "html" if output.suffix == ".html" else "markdown"
        report_path = generate_comparison_report(results, output, format=report_format)
        console.print(f"[bold green]Report saved to:[/bold green] {report_path}")

        # Generate plots if matplotlib available
        try:
            plots_dir = output.parent / "plots"
            plot_paths = generate_plots(results, plots_dir)
            if plot_paths:
                console.print(f"[bold green]Plots saved to:[/bold green] {plots_dir}")
        except ImportError:
            console.print("[dim]Skipping plots (matplotlib not installed)[/dim]")

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Run Command (Full Pipeline)
# =============================================================================


@app.command()
def run(
    vault_path: Annotated[
        Path,
        typer.Argument(help="Path to vault directory"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("output"),
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Embedding method"),
    ] = "tfidf",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of neighbors"),
    ] = 7,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Similarity threshold"),
    ] = 0.5,
    min_weight: Annotated[
        float,
        typer.Option("--min-weight", help="Minimum edge weight to keep"),
    ] = 0.0,
    visualize_output: Annotated[
        bool,
        typer.Option("--visualize/--no-visualize", help="Generate visualization"),
    ] = True,
) -> None:
    """
    Run full pipeline: ingest -> embed -> link -> analyze -> visualize.

    Convenience command for processing a vault end-to-end.
    """
    from semlink.core.analysis import (
        compute_metrics,
        detect_communities_louvain,
    )
    from semlink.core.graph import build_graph, export_json
    from semlink.core.ingest import ingest_vault
    from semlink.core.linker import HybridStrategy, filter_edges

    try:
        console.print(f"[bold]Running full pipeline on:[/bold] {vault_path}")
        console.print(f"[dim]Output directory: {output_dir}[/dim]")
        console.print(f"[dim]Method: {method}, k={k}, threshold={threshold}[/dim]")
        if min_weight > 0:
            console.print(f"[dim]Minimum weight filter: {min_weight}[/dim]")

        if not vault_path.is_dir():
            console.print(f"[red]Error:[/red] Not a directory: {vault_path}")
            raise typer.Exit(code=1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Ingest notes
        console.print("\n[bold cyan]Step 1/5: Ingesting notes...[/bold cyan]")
        store = ingest_vault(vault_path)
        notes = store.all()
        console.print(f"[green]Loaded {len(notes)} notes[/green]")

        if not notes:
            console.print("[red]Error:[/red] No notes found in vault")
            raise typer.Exit(code=1)

        # Save notes
        notes_path = output_dir / "notes.json"
        store.save(notes_path)

        # Step 2: Generate embeddings
        console.print("\n[bold cyan]Step 2/5: Generating embeddings...[/bold cyan]")
        ids = [note.id for note in notes]
        texts = [note.clean_content for note in notes]

        if method == "tfidf":
            from semlink.core.tfidf import TFIDFEmbedder

            embedder = TFIDFEmbedder()
            embeddings = embedder.fit_encode(texts)
        elif method == "sbert":
            from semlink.core.embeddings import SBERTEmbedder

            embedder = SBERTEmbedder()
            embeddings = embedder.encode(texts)
        elif method == "openai":
            from semlink.core.embeddings import OpenAIEmbedder

            embedder = OpenAIEmbedder()
            embeddings = embedder.encode(texts)
        else:
            console.print(f"[red]Error:[/red] Unknown method: {method}")
            raise typer.Exit(code=1)

        console.print(
            f"[green]Generated embeddings with shape {embeddings.shape}[/green]"
        )

        # Save embeddings with titles for visualization
        titles = [note.metadata.title for note in notes]
        emb_path = output_dir / "embeddings.npz"
        np.savez_compressed(
            emb_path,
            embeddings=embeddings,
            ids=np.array(ids, dtype=object),
            titles=np.array(titles, dtype=object),
            method=method,
        )

        # Step 3: Build graph with titles
        console.print("\n[bold cyan]Step 3/5: Building graph...[/bold cyan]")
        linker = HybridStrategy(k=k, threshold=threshold)
        edges = linker.infer_links(embeddings, ids)

        # Filter weak links if min_weight specified
        if min_weight > 0:
            original_count = len(edges)
            edges = filter_edges(edges, min_weight=min_weight)
            filtered_count = original_count - len(edges)
            console.print(
                f"[dim]Filtered {filtered_count} weak links (weight < {min_weight})[/dim]"
            )

        nodes = [{"id": doc_id, "title": title} for doc_id, title in zip(ids, titles)]
        graph = build_graph(nodes, edges)
        console.print(
            f"[green]Created graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges[/green]"
        )

        # Save graph
        graph_path = output_dir / "graph.json"
        export_json(graph, graph_path)

        # Step 4: Analyze graph
        console.print("\n[bold cyan]Step 4/5: Analyzing graph...[/bold cyan]")
        metrics = compute_metrics(graph)
        console.print(
            f"[dim]Density: {metrics['density']:.4f}, Clustering: {metrics['avg_clustering']:.4f}[/dim]"
        )

        communities = detect_communities_louvain(graph)
        console.print(f"[green]Detected {len(communities)} communities[/green]")

        # Step 5: Visualize (optional)
        if visualize_output:
            console.print(
                "\n[bold cyan]Step 5/5: Generating visualization...[/bold cyan]"
            )
            try:
                from semlink.core.visualize import to_pyvis

                viz_path = output_dir / "graph.html"
                to_pyvis(graph, viz_path)
                console.print(f"[green]Visualization saved to {viz_path}[/green]")
            except ImportError:
                console.print(
                    "[yellow]Skipping visualization (pyvis not installed)[/yellow]"
                )
        else:
            console.print("\n[dim]Skipping visualization (--no-visualize)[/dim]")

        # Summary
        console.print("\n[bold green]Pipeline complete![/bold green]")
        console.print(f"[dim]Output files in: {output_dir}[/dim]")

        table = Table(title="Output Files")
        table.add_column("File", style="cyan")
        table.add_column("Description")
        table.add_row("notes.json", f"{len(notes)} processed notes")
        table.add_row("embeddings.npz", f"{method} embeddings ({embeddings.shape})")
        table.add_row(
            "graph.json",
            f"Knowledge graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)",
        )
        if visualize_output:
            table.add_row("graph.html", "Interactive visualization")
        console.print(table)

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        console.print("[dim]Install optional dependencies as needed[/dim]")
        raise typer.Exit(code=1)
    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Info Command
# =============================================================================


# =============================================================================
# Sync Command (Incremental Updates with SQLite)
# =============================================================================


@app.command()
def sync(
    vault_path: Annotated[
        Path,
        typer.Argument(help="Path to vault directory"),
    ],
    db_path: Annotated[
        Path,
        typer.Option("--db", "-d", help="SQLite database path"),
    ] = Path(".semlink.db"),
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Embedding method"),
    ] = "tfidf",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of neighbors"),
    ] = 7,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Similarity threshold"),
    ] = 0.5,
    min_weight: Annotated[
        float,
        typer.Option("--min-weight", help="Minimum edge weight to keep"),
    ] = 0.0,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force full reprocessing"),
    ] = False,
) -> None:
    """
    Sync vault changes incrementally using SQLite storage.

    Only processes new or modified notes, reusing cached embeddings.
    """
    from semlink.core.ingest import discover_notes, process_note
    from semlink.core.linker import HybridStrategy, filter_edges
    from semlink.core.storage import SemLinkDB, StoredEdge, compute_file_hash

    try:
        console.print(f"[bold]Syncing vault:[/bold] {vault_path}")
        console.print(f"[dim]Database: {db_path}[/dim]")

        if not vault_path.is_dir():
            console.print(f"[red]Error:[/red] Not a directory: {vault_path}")
            raise typer.Exit(code=1)

        # Initialize database
        db = SemLinkDB(db_path)

        if force:
            console.print("[yellow]Force mode: clearing existing data[/yellow]")
            db.clear()

        # Discover notes and compute hashes
        console.print("[dim]Scanning for changes...[/dim]")
        note_paths = discover_notes(vault_path)
        file_hashes = {
            str(p.relative_to(vault_path)): compute_file_hash(p) for p in note_paths
        }

        # Find changed notes
        changed, deleted = db.get_changed_notes(file_hashes)

        if deleted:
            console.print(f"[dim]Removing {len(deleted)} deleted notes[/dim]")
            for note_id in deleted:
                db.delete_note(note_id)

        if not changed and not deleted and not force:
            stats = db.get_stats()
            console.print("[green]Vault is up to date![/green]")
            console.print(
                f"[dim]Notes: {stats['notes']}, Edges: {sum(stats['edges'].values()) if stats['edges'] else 0}[/dim]"
            )
            return

        # Process changed notes
        if changed:
            console.print(f"[bold]Processing {len(changed)} changed notes...[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing notes...", total=len(changed))

                for note_rel_path in changed:
                    note_path = vault_path / note_rel_path
                    note = process_note(note_path)

                    db.upsert_note(
                        note_id=note.id,
                        path=note_rel_path,
                        title=note.metadata.title,
                        content=note.raw_content,
                        clean_content=note.clean_content,
                        metadata={
                            "title": note.metadata.title,
                            "headings": note.metadata.headings,
                            "links": note.metadata.links,
                        },
                        file_hash=file_hashes[note_rel_path],
                    )
                    progress.advance(task)

        # Check for notes needing embeddings
        notes_needing_emb = db.get_notes_without_embeddings(method)

        if notes_needing_emb or changed:
            console.print(
                f"[bold]Generating embeddings for {len(notes_needing_emb) or len(changed)} notes...[/bold]"
            )

            # Get all notes for embedding
            all_notes = db.get_all_notes()
            ids = [n.id for n in all_notes]
            texts = [n.clean_content for n in all_notes]

            if method == "tfidf":
                from semlink.core.tfidf import TFIDFEmbedder

                embedder = TFIDFEmbedder()
                embeddings = embedder.fit_encode(texts)
            elif method == "sbert":
                from semlink.core.embeddings import SBERTEmbedder

                embedder = SBERTEmbedder()
                embeddings = embedder.encode(texts)
            elif method == "openai":
                from semlink.core.embeddings import OpenAIEmbedder

                embedder = OpenAIEmbedder()
                embeddings = embedder.encode(texts)
            else:
                console.print(f"[red]Error:[/red] Unknown method: {method}")
                raise typer.Exit(code=1)

            # Save embeddings to database
            for i, note_id in enumerate(ids):
                db.save_embedding(note_id, method, embeddings[i])

            console.print(f"[green]Embeddings cached ({embeddings.shape})[/green]")

            # Build graph
            console.print("[bold]Building graph...[/bold]")
            linker = HybridStrategy(k=k, threshold=threshold)
            edges = linker.infer_links(embeddings, ids)

            # Filter weak links
            if min_weight > 0:
                original_count = len(edges)
                edges = filter_edges(edges, min_weight=min_weight)
                console.print(
                    f"[dim]Filtered {original_count - len(edges)} weak links[/dim]"
                )

            # Save edges to database
            stored_edges = [
                StoredEdge(
                    source_id=e.source,
                    target_id=e.target,
                    weight=e.weight,
                    method=method,
                    reason=e.reason,
                    shared_terms=e.shared_terms or [],
                )
                for e in edges
            ]
            db.save_edges(stored_edges, method)

            # Save state
            db.set_state(
                "last_sync",
                {
                    "method": method,
                    "k": k,
                    "threshold": threshold,
                    "min_weight": min_weight,
                },
            )

        # Show summary
        stats = db.get_stats()
        console.print("\n[bold green]Sync complete![/bold green]")

        table = Table(title="Database Status")
        table.add_column("Item", style="cyan")
        table.add_column("Count", justify="right")

        table.add_row("Notes", str(stats["notes"]))
        for method_name, count in stats.get("embeddings", {}).items():
            table.add_row(f"Embeddings ({method_name})", str(count))
        for method_name, count in stats.get("edges", {}).items():
            table.add_row(f"Edges ({method_name})", str(count))

        console.print(table)

    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(code=1)
    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Export Command (Export from SQLite)
# =============================================================================


@app.command(name="export")
def export_db(
    db_path: Annotated[
        Path,
        typer.Option("--db", "-d", help="SQLite database path"),
    ] = Path(".semlink.db"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("graph.json"),
    method: Annotated[
        Optional[str],
        typer.Option("--method", "-m", help="Export edges for specific method"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json, graphml, gexf"),
    ] = "json",
) -> None:
    """
    Export graph from SQLite database.

    Exports stored notes and edges to various graph formats.
    """
    from semlink.core.graph import build_graph, export_gexf, export_graphml, export_json
    from semlink.core.storage import SemLinkDB

    try:
        if not db_path.exists():
            console.print(f"[red]Error:[/red] Database not found: {db_path}")
            console.print("[dim]Run 'semlink sync' first to create the database[/dim]")
            raise typer.Exit(code=1)

        db = SemLinkDB(db_path)
        stats = db.get_stats()

        if stats["notes"] == 0:
            console.print("[red]Error:[/red] Database is empty")
            raise typer.Exit(code=1)

        console.print(f"[bold]Exporting from:[/bold] {db_path}")

        # Get notes and edges
        notes = db.get_all_notes()
        edges = db.get_edges(method)

        if not edges:
            msg = "[yellow]Warning:[/yellow] No edges found"
            if method:
                msg += f" for method '{method}'"
            console.print(msg)

        # Build graph
        from semlink.core.linker import Edge

        node_list = [{"id": n.id, "title": n.title} for n in notes]
        edge_list = [
            Edge(
                source=e.source_id,
                target=e.target_id,
                weight=e.weight,
                method=e.method,
                reason=e.reason,
                shared_terms=e.shared_terms,
            )
            for e in edges
        ]

        graph = build_graph(node_list, edge_list)

        # Export
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            export_json(graph, output)
        elif format == "graphml":
            export_graphml(graph, output)
        elif format == "gexf":
            export_gexf(graph, output)
        else:
            console.print(f"[red]Error:[/red] Unknown format: {format}")
            raise typer.Exit(code=1)

        console.print(f"[bold green]Exported to:[/bold green] {output}")
        console.print(
            f"[dim]Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}[/dim]"
        )

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


# =============================================================================
# Status Command
# =============================================================================


@app.command()
def status(
    db_path: Annotated[
        Path,
        typer.Option("--db", "-d", help="SQLite database path"),
    ] = Path(".semlink.db"),
) -> None:
    """
    Show database status and statistics.
    """
    from semlink.core.storage import SemLinkDB

    if not db_path.exists():
        console.print(f"[yellow]No database found at:[/yellow] {db_path}")
        console.print("[dim]Run 'semlink sync <vault>' to create one[/dim]")
        return

    db = SemLinkDB(db_path)
    stats = db.get_stats()
    last_sync = db.get_state("last_sync", {})

    console.print(f"[bold]Database:[/bold] {db_path}")
    console.print()

    table = Table(title="Storage Statistics")
    table.add_column("Item", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Notes", str(stats["notes"]))

    if stats.get("embeddings"):
        for method, count in stats["embeddings"].items():
            table.add_row(f"Embeddings ({method})", str(count))
    else:
        table.add_row("Embeddings", "0")

    if stats.get("edges"):
        for method, count in stats["edges"].items():
            table.add_row(f"Edges ({method})", str(count))
    else:
        table.add_row("Edges", "0")

    console.print(table)

    if last_sync:
        console.print()
        console.print("[bold]Last sync settings:[/bold]")
        console.print(f"[dim]Method: {last_sync.get('method', 'N/A')}[/dim]")
        console.print(
            f"[dim]k: {last_sync.get('k', 'N/A')}, threshold: {last_sync.get('threshold', 'N/A')}[/dim]"
        )


# =============================================================================
# Aggregate Command
# =============================================================================


@app.command()
def aggregate(
    graph_path: Annotated[
        Path,
        typer.Argument(help="Path to graph JSON file"),
    ],
    notes_path: Annotated[
        Path,
        typer.Option("--notes", "-n", help="Path to notes JSON file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("topics"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: markdown, json, obsidian"),
    ] = "markdown",
    min_size: Annotated[
        int,
        typer.Option("--min-size", help="Minimum cluster size to form a topic"),
    ] = 2,
    resolution: Annotated[
        float,
        typer.Option(
            "--resolution", "-r", help="Louvain resolution (higher = more topics)"
        ),
    ] = 1.0,
    keywords: Annotated[
        int,
        typer.Option("--keywords", "-k", help="Number of keywords per topic"),
    ] = 5,
) -> None:
    """
    Aggregate notes into topics based on community detection.

    Groups semantically related notes and generates topic summaries.
    """
    from semlink.core.aggregate import (
        aggregate_by_topic,
        export_topics_json,
        export_topics_markdown,
        export_topics_obsidian,
    )
    from semlink.core.graph import load_json
    from semlink.core.ingest import NoteStore

    try:
        console.print("[bold]Aggregating notes into topics[/bold]")
        console.print(f"[dim]Graph: {graph_path}[/dim]")
        console.print(f"[dim]Notes: {notes_path}[/dim]")

        if not graph_path.exists():
            console.print(f"[red]Error:[/red] Graph file not found: {graph_path}")
            raise typer.Exit(code=1)

        if not notes_path.exists():
            console.print(f"[red]Error:[/red] Notes file not found: {notes_path}")
            raise typer.Exit(code=1)

        # Load graph and notes
        graph = load_json(graph_path)
        store = NoteStore.load(notes_path)

        # Convert notes to dict format
        notes_dict = {
            note.id: {
                "title": note.metadata.title,
                "content": note.raw_content,
                "clean_content": note.clean_content,
            }
            for note in store.all()
        }

        console.print(
            f"[dim]Loaded {graph.number_of_nodes()} nodes, {len(notes_dict)} notes[/dim]"
        )

        # Perform aggregation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting topics...", total=None)
            aggregation = aggregate_by_topic(
                graph,
                notes_dict,
                min_cluster_size=min_size,
                resolution=resolution,
                n_keywords=keywords,
            )
            progress.update(task, description="Topics detected")

        console.print(f"[green]Found {len(aggregation.topics)} topics[/green]")
        if aggregation.orphan_notes:
            console.print(
                f"[dim]{len(aggregation.orphan_notes)} notes not in any topic[/dim]"
            )

        # Show topic summary
        table = Table(title="Topics")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Label", style="white")
        table.add_column("Notes", justify="right")
        table.add_column("Keywords", style="dim")

        for topic in aggregation.topics[:10]:  # Show top 10
            table.add_row(
                str(topic.id),
                topic.label,
                str(topic.size),
                ", ".join(topic.keywords[:3]),
            )

        if len(aggregation.topics) > 10:
            table.add_row("...", f"({len(aggregation.topics) - 10} more)", "", "")

        console.print(table)

        # Export
        output.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            files = export_topics_markdown(aggregation, notes_dict, output)
            console.print(
                f"[bold green]Exported {len(files)} files to:[/bold green] {output}"
            )
        elif format == "json":
            json_path = output / "topics.json"
            export_topics_json(aggregation, json_path)
            console.print(f"[bold green]Exported to:[/bold green] {json_path}")
        elif format == "obsidian":
            export_topics_obsidian(aggregation, notes_dict, output)
            console.print(
                f"[bold green]Exported Obsidian vault to:[/bold green] {output}"
            )
        else:
            console.print(f"[red]Error:[/red] Unknown format: {format}")
            raise typer.Exit(code=1)

    except AppError as e:
        console.print(f"[red]Error:[/red] {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """
    Display information about available models and strategies.
    """
    console.print("\n[bold]SemLink - Semantic Note Linking[/bold]\n")

    # Embedding methods
    table = Table(title="Embedding Methods")
    table.add_column("Method", style="cyan")
    table.add_column("Description")
    table.add_column("Requirements")

    table.add_row("tfidf", "TF-IDF baseline (keyword matching)", "scikit-learn")
    table.add_row(
        "sbert", "Sentence-BERT (semantic similarity)", "sentence-transformers"
    )
    table.add_row("openai", "OpenAI embeddings (API)", "openai, tiktoken")

    console.print(table)
    console.print()

    # Link strategies
    table2 = Table(title="Link Strategies")
    table2.add_column("Strategy", style="cyan")
    table2.add_column("Description")

    table2.add_row("threshold", "Connect if similarity >= threshold")
    table2.add_row("knn", "Connect to k nearest neighbors")
    table2.add_row("mutual_knn", "Connect only if mutually nearest")
    table2.add_row("hybrid", "KNN + threshold (recommended)")

    console.print(table2)
    console.print()

    # Chunk strategies
    table3 = Table(title="Chunking Strategies")
    table3.add_column("Strategy", style="cyan")
    table3.add_column("Best For")

    table3.add_row("whole", "Short notes (<500 words)")
    table3.add_row("paragraph", "Medium notes (500-2000 words)")
    table3.add_row("heading", "Long structured documents")
    table3.add_row("window", "Very long documents")

    console.print(table3)


# =============================================================================
# Serve Command
# =============================================================================


@app.command()
def serve(
    db_path: Annotated[
        Optional[Path],
        typer.Option("--db", "-d", help="SQLite database path"),
    ] = None,
    graph_path: Annotated[
        Optional[Path],
        typer.Option("--graph", "-g", help="Graph JSON file path"),
    ] = None,
    notes_path: Annotated[
        Optional[Path],
        typer.Option("--notes", "-n", help="Notes JSON file path (for content)"),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8000,
    dev: Annotated[
        bool,
        typer.Option("--dev", help="Development mode (no static files)"),
    ] = False,
) -> None:
    """
    Start the web server for the React frontend.

    Serves the graph visualization UI and REST API.
    """
    try:
        import uvicorn

        from semlink.server import create_app
    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependency: {e}")
        console.print("[dim]Install with: pip install semlink[server][/dim]")
        raise typer.Exit(code=1)

    # Find static directory (frontend build)
    static_dir = None
    if not dev:
        # Check common locations for frontend build
        possible_paths = [
            Path(__file__).parent.parent.parent / "frontend" / "dist",
            Path(__file__).parent / "static",
            Path.cwd() / "frontend" / "dist",
        ]
        for p in possible_paths:
            if p.exists() and (p / "index.html").exists():
                static_dir = p
                break

    # Auto-detect db/graph/notes if not specified
    if not db_path and not graph_path:
        if Path(".semlink.db").exists():
            db_path = Path(".semlink.db")
        elif Path("graph.json").exists():
            graph_path = Path("graph.json")
        elif Path("output/graph.json").exists():
            graph_path = Path("output/graph.json")

    # Auto-detect notes.json for content (when using graph files)
    if not notes_path and graph_path:
        # Try to find notes.json near graph.json
        graph_dir = graph_path.parent
        possible_notes = [
            graph_dir / "notes.json",
            Path("notes.json"),
            Path("output/notes.json"),
        ]
        for p in possible_notes:
            if p.exists():
                notes_path = p
                break

    console.print("[bold]Starting SemLink server...[/bold]")
    if db_path:
        console.print(f"[dim]Database: {db_path}[/dim]")
    if graph_path:
        console.print(f"[dim]Graph: {graph_path}[/dim]")
    if notes_path:
        console.print(f"[dim]Notes: {notes_path}[/dim]")
    if static_dir:
        console.print(f"[dim]Static files: {static_dir}[/dim]")
    else:
        console.print("[yellow]No static files found. API-only mode.[/yellow]")
        console.print("[dim]Build frontend with: cd frontend && npm run build[/dim]")

    console.print(f"\n[bold green]Server running at:[/bold green] http://{host}:{port}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    app = create_app(
        db_path=db_path,
        graph_path=graph_path,
        notes_path=notes_path,
        static_dir=static_dir,
    )

    uvicorn.run(app, host=host, port=port, log_level="info")
