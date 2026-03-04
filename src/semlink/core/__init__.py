"""
SemLink Core Module.

Automatic semantic note linking for graph visualization.
"""

# Data Ingestion
from semlink.core.ingest import (
    Note,
    NoteMetadata,
    NoteStore,
    discover_notes,
    extract_metadata,
    ingest_vault,
    load_note,
    normalize_text,
    process_note,
    strip_markdown,
)

# Chunking
from semlink.core.chunk import (
    Chunk,
    ChunkIndex,
    ChunkStrategy,
    HeadingStrategy,
    ParagraphStrategy,
    SlidingWindowStrategy,
    WholeNoteStrategy,
    build_chunk_index,
    get_strategy as get_chunk_strategy,
    list_strategies as list_chunk_strategies,
)

# TF-IDF Baseline
from semlink.core.tfidf import TFIDFEmbedder

# Neural Embeddings
from semlink.core.embeddings import (
    EmbedderBase,
    OpenAIEmbedder,
    SBERTEmbedder,
    create_embedder,
    load_embeddings,
    save_embeddings,
)

# Link Inference
from semlink.core.linker import (
    Edge,
    HybridStrategy,
    KNNStrategy,
    LinkStrategy,
    MutualKNNStrategy,
    ThresholdStrategy,
    add_reasoning_to_edges,
    compute_similarity_matrix,
    edge_list_stats,
    filter_edges,
    generate_link_explanation,
    get_strategy as get_link_strategy,
    list_strategies as list_link_strategies,
)

# Graph Construction
from semlink.core.graph import (
    add_node_attributes,
    build_graph,
    export_edge_list,
    export_gexf,
    export_graphml,
    export_json,
    get_neighbors,
    get_node_attributes,
    largest_component,
    load_graphml,
    load_json,
    merge_graphs,
    prune_graph,
    subgraph_around_node,
    validate_graph,
)

# Graph Analysis
from semlink.core.analysis import (
    add_centrality_attributes,
    add_community_labels,
    cluster_coefficient_by_node,
    community_summary,
    compute_centrality,
    compute_metrics,
    degree_distribution,
    detect_communities_label_propagation,
    detect_communities_leiden,
    detect_communities_louvain,
    find_articulation_points,
    find_bridges,
    generate_analysis_report,
    top_nodes_by_centrality,
    weight_distribution,
)

# Visualization
from semlink.core.visualize import (
    compute_layout,
    configure_pyvis_physics,
    generate_obsidian_summary,
    style_nodes_by_centrality,
    style_nodes_by_community,
    to_d3_json,
    to_matplotlib,
    to_obsidian,
    to_pyvis,
    visualize_community,
    visualize_neighborhood,
)

# Evaluation
from semlink.core.evaluate import (
    compare_graphs,
    compare_methods,
    compare_similarity_distributions,
    evaluate_graph_quality,
    generate_comparison_report,
    generate_plots,
    link_coherence_score,
    link_overlap,
    plot_similarity_histogram,
    sample_links,
    similarity_distribution,
)

# Storage
from semlink.core.storage import (
    SemLinkDB,
    StoredEdge,
    StoredEmbedding,
    StoredNote,
    compute_file_hash,
)

__all__ = [
    # Ingest
    "Note",
    "NoteMetadata",
    "NoteStore",
    "discover_notes",
    "load_note",
    "strip_markdown",
    "normalize_text",
    "extract_metadata",
    "process_note",
    "ingest_vault",
    # Chunk
    "Chunk",
    "ChunkIndex",
    "ChunkStrategy",
    "WholeNoteStrategy",
    "ParagraphStrategy",
    "HeadingStrategy",
    "SlidingWindowStrategy",
    "get_chunk_strategy",
    "list_chunk_strategies",
    "build_chunk_index",
    # TF-IDF
    "TFIDFEmbedder",
    # Embeddings
    "EmbedderBase",
    "SBERTEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "save_embeddings",
    "load_embeddings",
    # Linker
    "Edge",
    "LinkStrategy",
    "ThresholdStrategy",
    "KNNStrategy",
    "MutualKNNStrategy",
    "HybridStrategy",
    "get_link_strategy",
    "list_link_strategies",
    "compute_similarity_matrix",
    "filter_edges",
    "edge_list_stats",
    "add_reasoning_to_edges",
    "generate_link_explanation",
    # Graph
    "build_graph",
    "add_node_attributes",
    "get_node_attributes",
    "get_neighbors",
    "subgraph_around_node",
    "validate_graph",
    "export_json",
    "export_graphml",
    "export_gexf",
    "export_edge_list",
    "load_json",
    "load_graphml",
    "merge_graphs",
    "prune_graph",
    "largest_component",
    # Analysis
    "compute_metrics",
    "degree_distribution",
    "weight_distribution",
    "detect_communities_louvain",
    "detect_communities_leiden",
    "detect_communities_label_propagation",
    "add_community_labels",
    "community_summary",
    "compute_centrality",
    "add_centrality_attributes",
    "top_nodes_by_centrality",
    "find_bridges",
    "find_articulation_points",
    "cluster_coefficient_by_node",
    "generate_analysis_report",
    # Visualize
    "to_pyvis",
    "configure_pyvis_physics",
    "style_nodes_by_community",
    "style_nodes_by_centrality",
    "to_matplotlib",
    "compute_layout",
    "to_d3_json",
    "to_obsidian",
    "generate_obsidian_summary",
    "visualize_neighborhood",
    "visualize_community",
    # Evaluate
    "similarity_distribution",
    "compare_similarity_distributions",
    "plot_similarity_histogram",
    "evaluate_graph_quality",
    "compare_graphs",
    "sample_links",
    "link_coherence_score",
    "compare_methods",
    "link_overlap",
    "generate_comparison_report",
    "generate_plots",
    # Storage
    "SemLinkDB",
    "StoredNote",
    "StoredEmbedding",
    "StoredEdge",
    "compute_file_hash",
]
