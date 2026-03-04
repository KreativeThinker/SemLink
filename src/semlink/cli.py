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
