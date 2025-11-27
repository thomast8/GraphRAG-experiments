# GraphRAG Experiments

`neo4j-graphrag-first-rego.py` is the main entrypoint. It ingests the BRD example PDF (`BRD-examples/Business_Inspection_Policy_Document.pdf`), builds a policy graph in Neo4j, and emits both a Rego module and an intermediate IR JSONL file.

## Requirements
- Neo4j reachable at `neo4j://localhost:7687` (defaults baked into the script). The `graphrag-test` database is wiped on each run.
- `OPENAI_API_KEY` in your shell for embeddings and LLM calls.
- Python 3.10+ with dependencies from `pyproject.toml`.

## Install
- Base setup: `uv sync`
- Notebook extras (for `end-to-end-lupus.ipynb`): `uv sync --extra notebook`

Optional `.env` (loaded automatically if present):
```
OPENAI_API_KEY=...
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=graphrag-test
```

`cp .env.example .env` to get started.

## Run

```bash
OPENAI_API_KEY=<key> uv run python neo4j-graphrag-first-rego.py
```

## Reliability helpers
- Structured schema: all LLM output is parsed into a typed `Neo4jGraph` using Pydantic validation (`GraphSchema` / `Neo4jGraph`), so malformed or schema-breaking JSON never makes it into Neo4j as half-written nodes/relationships.
- Retrying extractor: the custom `RetryingLLMEntityRelationExtractor` wraps GraphRAG’s extractor with `fix_invalid_json` + validation and retries up to 4 times with exponential backoff when the LLM returns invalid JSON or schema violations. Final failures are logged and their raw content, source chunk, and prompt are written to `outputs/failed_json` for inspection instead of silently dropping chunks.
- Database hygiene: the `graphrag-test` database is wiped at startup so each run starts from a clean slate, and a missing database is automatically created when the server supports multi-DB (falling back to `neo4j` on single-DB servers).

Other assets:
- `neo4j-graphrag-quickstart.py` – minimal Neo4j GraphRAG bootstrap.
- `end-to-end-lupus.ipynb` – notebook experiment.
- `BRD-examples/Business_Inspection_Policy_Document.pdf` – sample BRD for the main script.
