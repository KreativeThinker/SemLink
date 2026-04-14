# SemLink - Automatic Semantic Note Linking

Transform unstructured markdown/text notes into an interconnected knowledge graph using semantic similarity analysis.

## Features

- **Automatic Link Discovery** - Infer semantic relationships between notes without manual linking
- **Multiple Embedding Methods** - TF-IDF baseline, Sentence-BERT, or OpenAI embeddings
- **Graph Visualization** - Interactive D3.js web interface or Pyvis HTML export
- **Community Detection** - Discover topic clusters using Louvain algorithm
- **Topic Aggregation** - Group related notes and generate topic summaries
- **Incremental Updates** - SQLite storage for efficient vault synchronization
- **Link Reasoning** - Understand why notes are connected with shared terms

Similar solutions around "Document linking" for human interpretablity fall short by focusing on purely key-word matching approaches like [this one](https://forum.obsidian.md/t/obsidian-note-linker-automatically-find-and-create-new-links-between-notes/41504)

Some ideas for how to visualize the semantic linking is given [here](https://forum.obsidian.md/t/have-we-finally-figuredout-semantic-links-aka-how2-cultivate-complex-link-properties-for-visualising-distinct-relationships/86732) even though some of those are a bit over-the-top and would only constitute UI complexity which is to be avoided.

## Objectives
- Infer semantic relationships between textual notes
- Compare keyword-based and embedding-based similarity methods
- Construct a sparse, interpretable note graph
- Analyze graph structure and semantic coherence
- Provide a CLI tool that allows for easy usage

## Installation

```bash
# Core installation (TF-IDF only, lightweight)
pip install semlink

# With neural embeddings (includes PyTorch)
pip install semlink[sbert]

# With OpenAI embeddings
pip install semlink[openai]

# With web server for React frontend
pip install semlink[server]

# With Pyvis visualization
pip install semlink[viz]

# Everything
pip install semlink[all]
```

## Quick Start

```bash
# Run full pipeline on a vault
semlink run ./my-notes --output ./output

# Or use incremental sync with SQLite
semlink sync ./my-notes --db .semlink.db

# Start the web interface
semlink serve --db .semlink.db

# Aggregate notes into topics
semlink aggregate output/graph.json --notes output/notes.json -o topics/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Load and preprocess notes from a vault directory |
| `embed` | Generate embeddings (TF-IDF, SBERT, or OpenAI) |
| `link` | Infer links between notes based on similarity |
| `analyze` | Compute graph metrics and detect communities |
| `visualize` | Generate HTML, PNG, D3 JSON, or Obsidian export |
| `compare` | Compare different embedding methods |
| `run` | Full pipeline: ingest → embed → link → analyze → visualize |
| `sync` | Incremental vault sync with SQLite storage |
| `export` | Export graph from SQLite database |
| `status` | Show database statistics |
| `aggregate` | Group notes into topics by community |
| `serve` | Start web server for React frontend |
| `info` | Display available models and strategies |

## Usage Examples

### Basic Pipeline

```bash
# Process a vault with TF-IDF (default, lightweight)
semlink run ./vault --output ./output

# Use Sentence-BERT for better semantic matching
semlink run ./vault --method sbert --output ./output

# Filter weak links (< 25% similarity)
semlink run ./vault --min-weight 0.25 --output ./output
```

### Incremental Workflow

```bash
# Initial sync (creates .semlink.db)
semlink sync ./vault

# Re-run after editing notes (only processes changes)
semlink sync ./vault

# Export graph for visualization
semlink export --db .semlink.db -o graph.json

# Check database status
semlink status
```

### Web Interface

```bash
# Build frontend (first time only)
cd frontend && npm install && npm run build && cd ..

# Start server
semlink serve --db .semlink.db

# Opens at http://localhost:8000
```

### Topic Aggregation

```bash
# Generate topic clusters from graph
semlink aggregate graph.json --notes notes.json --format markdown -o topics/

# Export as Obsidian vault structure
semlink aggregate graph.json --notes notes.json --format obsidian -o vault/

# More granular topics (higher resolution)
semlink aggregate graph.json --notes notes.json --resolution 1.5 -k 7
```

## Embedding Methods

| Method | Description | Requirements |
|--------|-------------|--------------|
| `tfidf` | TF-IDF keyword matching (default) | scikit-learn |
| `sbert` | Sentence-BERT semantic similarity | sentence-transformers |
| `openai` | OpenAI text embeddings API | openai, tiktoken |

## Link Strategies

| Strategy | Description |
|----------|-------------|
| `threshold` | Connect notes with similarity ≥ threshold |
| `knn` | Connect to k nearest neighbors |
| `mutual_knn` | Connect only if mutually nearest |
| `hybrid` | KNN + threshold (recommended) |

## Project Structure

```
semlink/
├── frontend/              # React + D3.js web interface
│   ├── src/
│   │   ├── components/    # ForceGraph, SidePanel, ControlPanel
│   │   ├── hooks/         # useGraphData
│   │   └── types/         # TypeScript definitions
│   └── package.json
├── src/semlink/
│   ├── core/
│   │   ├── ingest.py      # Note discovery and preprocessing
│   │   ├── chunk.py       # Chunking strategies
│   │   ├── tfidf.py       # TF-IDF embeddings
│   │   ├── embeddings.py  # SBERT, OpenAI embeddings
│   │   ├── linker.py      # Link inference strategies
│   │   ├── graph.py       # NetworkX graph building
│   │   ├── analysis.py    # Metrics, community detection
│   │   ├── visualize.py   # Pyvis, D3, Obsidian export
│   │   ├── evaluate.py    # Method comparison
│   │   ├── storage.py     # SQLite persistence
│   │   └── aggregate.py   # Topic aggregation
│   ├── server.py          # FastAPI backend
│   └── cli.py             # Typer CLI
└── pyproject.toml
```

## Development

```bash
# Clone and install
git clone https://github.com/KreativeThinker/SemLink.git
cd SemLink
uv sync

# Install pre-commit hooks
pre-commit install

# Run CLI
uv run semlink --help
```

## Abstract

The digital note-taking software market is experiencing significant growth, with projections reaching $1.35-1.5 billion by 2027-2028. Knowledge workers are increasingly adopting tools like Obsidian, Roam Research, and Logseq for personal knowledge management. However, **manual linking does not scale** beyond a few hundred notes, leaving valuable relationships unexpressed.

This project addresses the problem of automatically identifying and representing semantic relationships between notes without user intervention, using NLP techniques to infer contextual relationships and represent them as a knowledge graph.

## References

- Arnold, M., Goldschmitt, M., & Rigotti, T. (2023). Dealing with information overload: A comprehensive review. *Frontiers in Psychology*, 14.
- Karr-Wisniewski, P., & Lu, Y. (2010). When more is too much: Operationalizing technology overload. *Computers in Human Behavior*, 26(5), 1061-1072.
- Li, H., et al. (2023). Knowledge graphs in practice: characterizing users, challenges, and visualization opportunities. *IEEE TVCG*.

## Contributing

Please refer to the guidelines in [CONTRIBUTING.md](/CONTRIBUTING.md).

## License

MIT License - see [LICENSE](/LICENSE) for details.
