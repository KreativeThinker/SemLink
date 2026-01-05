# Smart Semantic Note Linking and Visualizer

## Abstract
[RESEARCH THE INCREASE IN USAGE OF DIGITAL NOTE TAKING AND INSERT STATISTICS HERE]
[Talk about applications like obsidian and include research on the topics of graphical visualization being helpful in note taking]

As the volume of digital notes increases, manually creating meaningful links between related concepts becomes impractical. This project investigates automated semantic note linking using Natural Language Processing techniques to infer contextual relationships between unstructured notes and represent them as a knowledge graph.

## Problem Statement
Graph-based note systems rely heavily on manual linking, which does not scale with large or long-term note collections. As a result, relationships between conceptually related notes remain unexpressed, limiting the usefulness of graph visualizations.

This project addresses the problem of automatically identifying and representing semantic relationships between notes without user intervention.

Similar solutions around "Document linking" for human interpretablity fall short by focusing on purely key-word matching approaches like [this one](https://forum.obsidian.md/t/obsidian-note-linker-automatically-find-and-create-new-links-between-notes/41504)

Some ideas for how to visualize the semantic linking is given [here](https://forum.obsidian.md/t/have-we-finally-figuredout-semantic-links-aka-how2-cultivate-complex-link-properties-for-visualising-distinct-relationships/86732) even though some of those are a bit over-the-top and would only constitute UI complexity which is to be avoided.

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
│   ├── ISSUE_TEMPLATE/       # Issue Templates
│   └── workflows/
│       └── package.yml       # Workflow to make installable package and publish to pypi.org
├── docs/                     # Installation and usage documentation
├── research/                 # Research conducted over the course of this project
├── src/
│   └── semlink/
│       ├── core/             # Core logic
│       │   └── __init__.py
│       ├── cli.py            # CLI interface
│       ├── errors.py         # Central Error System
│       ├── __init__.py
│       └── __main__.py
├── .pre-commit-config.yaml   # pre-commit hooks
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock                   # Installation lockfile
```

This project will follow the legacy `src` package layout for all core development. All logic is to be written in the `src/semlink/core/<file>.py` and required functionality is to be exposed via the `src/semlink/cli.py`

---

> Following are placeholder issues that will be migrated to the github issues panel along with a clearer instructions

## 1. Data Ingestion and Preprocessing

**Goal:** Standardize note input for downstream processing.

Tasks:

* Load plain-text and Markdown files
* Strip markup and normalize text
* Handle file-level metadata (filename, path)
* Store processed text representations

Deliverables:

* Preprocessing module
* Sample input/output validation

---

## 2. Note Segmentation (Chunking)

**Goal:** Improve semantic resolution by operating on note segments.

Tasks:

* Implement whole-note representation
* Implement paragraph-based segmentation
* Store segment-to-note mappings

Deliverables:

* Chunking strategies with configurable parameters
* Comparison-ready data structures

---

## 3. Baseline Similarity: TF-IDF

**Goal:** Establish a non-neural baseline.

Tasks:

* Build TF-IDF representations
* Compute cosine similarity between notes or segments
* Generate similarity matrix

Deliverables:

* Baseline similarity scores
* Reproducible results for comparison

---

## 4. Embedding-Based Similarity

**Goal:** Capture semantic similarity beyond lexical overlap.

Tasks:

* Integrate pre-trained sentence/document embeddings
* Generate vector representations for notes or segments
* Compute cosine similarity

Deliverables:

* Embedding-based similarity matrices
* Direct comparison with TF-IDF baseline

---

## 5. Link Inference Strategy

**Goal:** Convert similarity scores into graph edges.

Tasks:

* Implement similarity thresholding
* Implement k-nearest-neighbor linking
* Control graph sparsity

Deliverables:

* Configurable link inference logic
* Edge list generation

---

## 6. Graph Construction

**Goal:** Represent inferred relationships formally.

Tasks:

* Construct graph from inferred links
* Assign nodes and weighted edges
* Export graph in standard format (e.g., NetworkX)

Deliverables:

* Graph object
* Serialized graph output

---

## 7. Graph Analysis

**Goal:** Analyze structural properties of the note graph.

Tasks:

* Compute graph density and degree distribution
* Identify connected components or clusters
* Analyze central nodes

Deliverables:

* Quantitative graph metrics
* Analysis scripts

---

## 8. Evaluation and Comparison

**Goal:** Evaluate semantic coherence and method effectiveness.

Tasks:

* Compare baseline vs embedding-based graphs
* Analyze differences in connectivity and sparsity
* Perform qualitative inspection on sampled links

Deliverables:

* Evaluation report
* Plots or tables summarizing results

---

## 9. Documentation and Reproducibility

**Goal:** Ensure the project can be understood and rerun.

Tasks:

* Document configuration parameters
* Describe experimental setup
* Provide example datasets and commands

Deliverables:

* Updated README
* Reproducibility notes

---

## Contributing

Please refer the guidelines as outlined in the [CONTRIBUTING](/CONTRIBUTING.md) file
