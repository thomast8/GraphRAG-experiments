# GraphRAG Experiments

`neo4j-graphrag-first-rego.py` is the main entrypoint. It ingests the BRD example PDF (`BRD-examples/Business_Inspection_Policy_Document.pdf`), builds a policy graph in Neo4j, and emits both a Rego module and an intermediate IR JSONL file.

## Requirements
- Neo4j reachable at `neo4j://localhost:7687` (defaults baked into the script). The `graphrag-test` database is wiped on each run.
- `OPENAI_API_KEY` in your shell for embeddings and LLM calls.
- Python 3.10+ with dependencies from `pyproject.toml`.

## Install
- Base setup: `uv sync`
- Notebook extras (for `end-to-end-lupus.ipynb`): `uv sync --extra notebook`

Optional `.env.graphrag` (loaded automatically if present):
```
OPENAI_API_KEY=...
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=graphrag-test
```

## Run

```bash
OPENAI_API_KEY=<key> uv run python neo4j-graphrag-first-rego.py
```

Other assets:
- `neo4j-graphrag-quickstart.py` – minimal Neo4j GraphRAG bootstrap.
- `end-to-end-lupus.ipynb` – notebook experiment.
- `BRD-examples/Business_Inspection_Policy_Document.pdf` – sample BRD for the main script.
