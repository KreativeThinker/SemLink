# Smart Semantic Note Linking and Visualizer

## Abstract

The digital note-taking software market is experiencing significant growth, with projections reaching $1.35-1.5 billion by 2027-2028 at a CAGR of 5-7%. Knowledge workers are increasingly adopting tools like Obsidian, Roam Research, and Logseq for personal knowledge management. However, research shows that **28% of the workday is lost to technology-induced interruptions and information overload** (Karr-Wisniewski & Lu, 2010), costing the U.S. economy approximately **$588 billion annually**. A comprehensive review of 87 studies found that information overload is positively correlated with burnout and negatively impacts job satisfaction (Arnold et al., 2023).

Graph-based visualization has emerged as a promising approach to knowledge management. Studies on knowledge graph visualization highlight the importance of intuitive interfaces for navigating complex information spaces (Li et al., 2023; Nararatwong et al., 2020). Tools like Obsidian leverage graph views to help users discover connections between notes, yet **manual linking does not scale** beyond a few hundred notes, leaving valuable relationships unexpressed.

As the volume of digital notes increases, manually creating meaningful links between related concepts becomes impractical. This project investigates automated semantic note linking using Natural Language Processing techniques to infer contextual relationships between unstructured notes and represent them as a knowledge graph.

## Problem Statement
Graph-based note systems rely heavily on manual linking, which does not scale with large or long-term note collections. As a result, relationships between conceptually related notes remain unexpressed, limiting the usefulness of graph visualizations.

This project addresses the problem of automatically identifying and representing semantic relationships between notes without user intervention.

## Objectives
- Infer semantic relationships between textual notes
- Compare keyword-based and embedding-based similarity methods
- Construct a sparse, interpretable note graph
- Analyze graph structure and semantic coherence
- Provide a CLI tool that allows for easy usage

## Evaluation Criteria
- Similarity score distributions
- Graph density and connectivity
- Qualitative inspection of inferred links
- Comparison between baseline and embedding-based methods

## Project Structure

```bash
semlink/
├── .github/
│   ├── ISSUE_TEMPLATE/       # Issue Templates
│   └── workflows/
│       └── package.yml       # Workflow to make installable package and publish to pypi.org
├── docs/                     # Installation and usage documentation
├── research/                 # Research conducted over the course of this project
├── src/
│   └── semlink/
│       ├── core/
│       │   ├── __init__.py   # Public API exports
│       │   ├── ingest.py     # Data ingestion and preprocessing
│       │   ├── chunk.py      # Note segmentation strategies
│       │   ├── tfidf.py      # TF-IDF baseline embeddings
│       │   ├── embeddings.py # Neural embedding backends
│       │   ├── linker.py     # Link inference strategies
│       │   ├── graph.py      # Graph construction and export
│       │   ├── analysis.py   # Graph metrics and communities
│       │   ├── visualize.py  # Visualization outputs
│       │   └── evaluate.py   # Method comparison
│       ├── cli.py            # CLI interface
│       ├── errors.py         # Central Error System
│       ├── __init__.py
│       └── __main__.py
├── .pre-commit-config.yaml   # pre-commit hooks
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock                   # Installation lockfile
```

This project follows the `src` package layout. Core logic resides in `src/semlink/core/` modules, exposed via `src/semlink/cli.py`.

## Installation

```bash
# Core installation (TF-IDF only, lightweight)
pip install semlink

# With neural embeddings (includes PyTorch)
pip install semlink[sbert]

# With OpenAI embeddings
pip install semlink[openai]

# With visualization
pip install semlink[viz]

# Everything
pip install semlink[all]
```

## Quick Start

```bash
# View available commands
semlink --help

# View available models and strategies
semlink info

# Full pipeline (when implemented)
semlink run ./my-vault --output ./output/
```

---

## Development Roadmap

All modules have been planned and tracked as GitHub issues. See the [Issues](https://github.com/KreativeThinker/SemLink/issues) page for detailed task breakdowns.

| Module | Description | Issue | Priority |
|--------|-------------|-------|----------|
| 1. Data Ingestion | Load and preprocess markdown/text files | [#4](https://github.com/KreativeThinker/SemLink/issues/4) | P0 |
| 2. Chunking | Note segmentation strategies | [#5](https://github.com/KreativeThinker/SemLink/issues/5) | P1 |
| 3. TF-IDF | Baseline keyword similarity | [#6](https://github.com/KreativeThinker/SemLink/issues/6) | P0 |
| 4. Embeddings | Neural embedding backends | [#7](https://github.com/KreativeThinker/SemLink/issues/7) | P0 |
| 5. Link Inference | Convert similarity to edges | [#8](https://github.com/KreativeThinker/SemLink/issues/8) | P0 |
| 6. Graph Construction | Build and export graphs | [#9](https://github.com/KreativeThinker/SemLink/issues/9) | P0 |
| 7. Graph Analysis | Metrics and community detection | [#10](https://github.com/KreativeThinker/SemLink/issues/10) | P1 |
| 8. Visualization | Interactive and static outputs | [#11](https://github.com/KreativeThinker/SemLink/issues/11) | P1 |
| 9. Evaluation | Method comparison and reporting | [#12](https://github.com/KreativeThinker/SemLink/issues/12) | P2 |

## References

- Arnold, M., Goldschmitt, M., & Rigotti, T. (2023). Dealing with information overload: A comprehensive review. *Frontiers in Psychology*, 14.
- Karr-Wisniewski, P., & Lu, Y. (2010). When more is too much: Operationalizing technology overload. *Computers in Human Behavior*, 26(5), 1061-1072.
- Li, H., et al. (2023). Knowledge graphs in practice: characterizing users, challenges, and visualization opportunities. *IEEE TVCG*.
- Nararatwong, R., et al. (2020). Knowledge graph visualization: Challenges, framework, and implementation.

---

## Contributing

Please refer the guidelines as outlined in the [CONTRIBUTING](/CONTRIBUTING.md) file
