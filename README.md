# GraphRAG Experiments

`neo4j-graphrag-first-rego.py` is the main entrypoint. It ingests the BRD example PDF (`BRD-examples/Business_Inspection_Policy_Document.pdf`), builds a policy graph in Neo4j, and emits both a Rego module and an intermediate IR JSONL file.

## Requirements
- Neo4j reachable at `neo4j://localhost:7687` (defaults baked into the script). The `graphrag-test` database is wiped on each run.
- `OPENAI_API_KEY` in your shell for embeddings and LLM calls.
- Python deps installed via `uv` (or `pip`) with the repo’s `pyproject.toml` once added; for now, install the same libraries you were using in the parent sandbox.

## Run

```bash
OPENAI_API_KEY=<key> uv run python neo4j-graphrag-first-rego.py
```

Other assets:
- `neo4j-graphrag-quickstart.py` – minimal Neo4j GraphRAG bootstrap.
- `end-to-end-lupus.ipynb` – notebook experiment.
- `BRD-examples/Business_Inspection_Policy_Document.pdf` – sample BRD for the main script.
