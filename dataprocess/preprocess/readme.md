# Heterogeneous Graph Semantic Preprocessing Pipeline

This repository provides a **multi-stage preprocessing system** for heterogeneous graph datasets, designed to automatically **discover, assign, and normalize fine-grained semantic subtypes** for nodes and edges. The resulting graphs are semantically richer and better suited for **diffusion-based graph generation, semantic-aware representation learning, and structured evaluation**.

---

## 1. What This Project Does

Most existing graph datasets (e.g., DBLP, OGB-MAG, Cora, PubMed) only expose **coarse node types** such as `Paper`, `Author`, and `Organization`. This project upgrades such datasets into **semantically enriched heterogeneous graphs** by:

- discovering meaningful subtype schemas (e.g., research nature of papers),
- assigning fine-grained subtype labels to every node,
- enforcing global semantic consistency across the dataset.

The output graph supports **semantic control, interpretability, and richer generation objectives**.

---

## 2. Pipeline Overview

The preprocessing pipeline is decomposed into **three roles**, executed sequentially:

```
Raw Records (JSONL)
   ↓
Graph Extraction
   ↓
Role 0: Schema Discovery
   ↓
Role 1: Subtype Assignment
   ↓
Role 2: Global Normalization
   ↓
Semantically Enriched Heterogeneous Graph
```

Each role is implemented as a dedicated module and uses a persistent LLM session with role-specific instructions.

---

## 3. Role Definitions

### Role 0 — Schema Discovery

**Purpose**: Decide whether a node category should be subdivided, and propose a **downstream-meaningful subtype schema**.

**Input**:
- A small random sample of nodes from the same coarse type (e.g., `Paper`).

**Output (JSON)**:
```json
{
  "should_classify": true,
  "schema": ["Theory", "Application", "System", "Survey", "Other"]
}
```

**Key Properties**:
- Schema is judged by **downstream usefulness**, not just frequency.
- Categories may include labels not observed in the current sample.
- If no meaningful axis exists, `should_classify` is set to `false`.

---

### Role 1 — Subtype Assignment

**Purpose**: Assign one or more subtype labels to **each individual node**.

**Input**:
- One node JSON.
- The schema discovered by Role 0 (if any).

**Output (JSON array)**:
```json
["Application"]
```

or multi-label:
```json
["System", "Application"]
```

**Key Properties**:
- Only labels from the provided schema may be used.
- All node attributes are considered (title, abstract, keywords, affiliation, etc.).
- This stage performs instance-level semantic inference.

---

### Role 2 — Global Normalization

**Purpose**: Normalize subtype labels into a **canonical, dataset-wide vocabulary**.

**Input**:
```json
{
  "node_id": "...",
  "type": "Paper",
  "raw_labels": ["Applied", "Application-Oriented"]
}
```

**Output**:
```json
{
  "canonical_map": {
    "Applied": "Application",
    "Application-Oriented": "Application"
  }
}
```

**Key Properties**:
- No new semantic categories are invented.
- Only labels appearing in the data may be merged.
- Prevents semantic fragmentation.

---

## 4. Repository Structure

```
preprocess/
├── config/
│   └── settings.py
├── scripts/
│   └── preprocess_all.py
├── src/
│   ├── graph/
│   │   └── extract_graph.py
│   ├── llm_client.py
│   ├── pipeline.py
│   └── roles/
│       ├── role0_schema_discovery.py
│       ├── role1_extractor.py
│       └── role2_normalizer.py
└── output/
    ├── stage1_nodes.jsonl
    ├── edges.jsonl
    └── stage2_refined.json
```

---

## 5. How to Run

### Step 1. Prepare Input

Place your raw dataset as a JSONL file (e.g., Aminer-style records):

```json
{"id": "...", "title": "...", "authors": ["..."], "references": ["..."]}
```

Set the path in `settings.RAW_INPUT`.

---

### Step 2. Configure API and Parameters

Edit `preprocess/config/settings.py`:

- `API_KEY`
- `ROLE0_MODEL`, `ROLE1_MODEL`, `ROLE2_MODEL`
- `CHUNK_SIZE`, `NUM_WORKERS`

---

### Step 3. Run Full Pipeline

```bash
python preprocess/scripts/preprocess_all.py
```

Outputs will be written to:

- `preprocess/output/stage1_nodes.jsonl`
- `preprocess/output/edges.jsonl`
- `preprocess/output/stage2_refined.json`

---

## 6. Semantic Richness Evaluation

The enriched graph can be quantitatively compared against baseline datasets using:

- subtype vocabulary size,
- average subtypes per node,
- subtype entropy,
- attribute density.

These metrics can be visualized using **radar charts** to demonstrate that the enriched dataset exhibits higher semantic expressiveness than standard homogeneous or heterogeneous graphs.

---

## 7. Design Notes on Agents

This system follows an **agent-decomposed design**:

- Role 0: global semantic reasoning agent,
- Role 1: instance-level semantic inference agent,
- Role 2: global consistency and quality-control agent.

While it does not use ReAct-style tool loops, the separation of responsibilities improves controllability, scalability, and reproducibility for offline preprocessing.

---

## 8. Intended Use Cases

- Diffusion-based heterogeneous graph generation
- Semantic-aware graph representation learning
- Dataset enrichment and benchmarking
- Controlled graph synthesis

---

## 9. Summary

This preprocessing pipeline transforms raw heterogeneous graphs into **semantically rich, structurally expressive datasets**, enabling downstream models to reason about **what nodes represent**, not just **how they are connected**.

It is designed to be:
- scalable,
- domain-adaptive,
- and suitable for modern graph generative modeling.

