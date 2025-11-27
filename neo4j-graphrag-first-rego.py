import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
    DocumentInfo,
    LexicalGraphConfig,
    GraphSchema,
    TextChunks,
    fix_invalid_json,
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import Neo4jGraph, TextChunk
from neo4j_graphrag.experimental.pipeline.config.object_config import ComponentType
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.simple_kg_builder import (
    SimpleKGPipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.exceptions import (
    InvalidJSONError,
    PipelineDefinitionError,
)
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.utils.rate_limit import RetryRateLimitHandler
from pydantic import ValidationError
from neo4j_graphrag.exceptions import LLMGenerationError

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> [<cyan>{name}</cyan>] <level>{message}</level>",
)
logging.getLogger("pypdf").setLevel(logging.ERROR)


class Neo4jNotificationHandler(logging.Handler):
    """Condense Neo4j notification blobs into short, readable messages."""

    @staticmethod
    def _extract_field(message: str, field: str) -> Optional[str]:
        token = f"{field}: "
        idx = message.find(token)
        if idx == -1:
            return None
        start = idx + len(token)
        end = message.find("}", start)
        if end == -1:
            end = len(message)
        value = message[start:end].strip()
        if value.startswith("`") and value.endswith("`"):
            value = value[1:-1]
        return value or None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Received notification from DBMS server" not in message:
            return

        severity = self._extract_field(message, "severity")
        code = self._extract_field(message, "code")
        category = self._extract_field(message, "category")
        title = self._extract_field(message, "title")
        description = self._extract_field(message, "description")

        # Drop very low-signal informational noise like "index already exists".
        if severity == "INFORMATION" and code and "IndexOrConstraintAlreadyExists" in code:
            return

        logger.warning(
            f"Neo4j notification [{severity or '?'} / {category or '?'}] {code or ''} - {title or ''}"
        )
        if description:
            logger.debug(f"Neo4j notification detail: {description}")


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
runner_logger = logging.getLogger("neo4j_graphrag.experimental.pipeline.config.runner")
runner_logger.setLevel(logging.WARNING)

neo4j_notif_logger = logging.getLogger("neo4j.notifications")
neo4j_notif_logger.setLevel(logging.INFO)
neo4j_notif_logger.handlers.clear()
neo4j_notif_logger.propagate = False
neo4j_notif_logger.addHandler(Neo4jNotificationHandler())

ENV_FILE = Path(__file__).resolve().parent / ".env"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
FAILED_JSON_DIR = OUTPUTS_DIR / "failed_json"


def load_env_from_file(path: Path) -> None:
    """Best-effort .env loader (no dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Always load the GraphRAG-local .env first; allow shell values to override.
load_env_from_file(ENV_FILE)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
# Configured target DB; may be overridden if server is single-DB or the DB is missing.
NEO4J_DATABASE_CONFIGURED = os.environ.get("NEO4J_DATABASE", "graphrag-test")

def ensure_database_exists(neo4j_driver: Driver, database: str) -> None:
    """Create the target database when the server supports multi-DB; otherwise no-op."""
    if not database:
        return
    # Aura Free and some single-DB servers do not expose system DB or multi-DB.
    if hasattr(neo4j_driver, "supports_multi_db") and not neo4j_driver.supports_multi_db():
        logger.info(
            "Neo4j server does not support multi-db; skipping database creation."
        )
        return
    if database == "neo4j":
        return
    try:
        with neo4j_driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{database}` IF NOT EXISTS")
            logger.info(f"Ensured Neo4j database exists: {database}")
    except Exception as exc:
        logger.warning(f"Database creation skipped for {database}: {exc}")


# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def resolve_database(neo4j_driver: Driver, configured_db: str) -> str:
    """Use configured DB when available; for single-DB servers or missing DB, fall back to 'neo4j'."""
    target = configured_db or "neo4j"
    if hasattr(neo4j_driver, "supports_multi_db") and not neo4j_driver.supports_multi_db():
        if target != "neo4j":
            logger.info(f"Server is single-DB; forcing database to 'neo4j' instead of {target}")
        return "neo4j"
    try:
        with neo4j_driver.session(database=target) as session:
            session.run("RETURN 1").consume()
        return target
    except Neo4jError as exc:
        logger.warning(f"Database {target} unavailable, falling back to 'neo4j': {exc}")
        return "neo4j"


NEO4J_DATABASE = resolve_database(driver, NEO4J_DATABASE_CONFIGURED)
ensure_database_exists(driver, NEO4J_DATABASE)
logger.info(f"Using Neo4j URI={NEO4J_URI} user={NEO4J_USERNAME} database={NEO4J_DATABASE}")


def reset_database(neo4j_driver: Driver) -> None:
    """Wipe all nodes and relationships so each run starts fresh."""
    logger.info(f"Resetting database {NEO4J_DATABASE} before ingestion")
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")


reset_database(driver)

# Rego-oriented schema for policy rules
node_types = [
    "Rule",
    "SubjectType",
    "Action",
    "ResourceType",
    "Condition",
]

relationship_types = [
    "APPLIES_TO_SUBJECT",
    "APPLIES_TO_ACTION",
    "APPLIES_TO_RESOURCE",
    "HAS_CONDITION",
]

patterns = [
    ("Rule", "APPLIES_TO_SUBJECT", "SubjectType"),
    ("Rule", "APPLIES_TO_ACTION", "Action"),
    ("Rule", "APPLIES_TO_RESOURCE", "ResourceType"),
    ("Rule", "HAS_CONDITION", "Condition"),
]

prompt_template = '''
You are a top-tier algorithm designed for extracting authorization and policy rules from business requirements and representing them as a knowledge graph suitable for Rego (policy-as-code).

Extract the entities (nodes) and specify their type from the following text.
Use the node types with the following meanings:
- "Rule": a single policy or constraint, often describing when an action is allowed, denied, or required.
- "SubjectType": who or what the rule is about (actors, roles, or regulated parties).
- "Action": what the subject is doing or must / must not / may do.
- "ResourceType": what the action applies to or affects (objects, systems, documents, locations, etc.).
- "Condition": any precondition, exception, threshold, time limit, or other contextual qualifier for the rule.

Also extract the relationships between these nodes. Each relationship should connect a "Rule" node to its subjects, actions, resources, and conditions.

For each node, use these properties where possible:
- Rule:
  - "name": short human-readable name summarizing the rule.
  - "effect": string such as "ALLOW", "DENY", or "SET_RISK_SCORE", based on the intent of the rule.
  - "effect_params": optional object for details such as score deltas or routing information.
- SubjectType:
  - "name": string, for example "Applicant" or "Company".
- Action:
  - "name": string, for example "start_compliance_registration".
- ResourceType:
  - "name": string, for example "ComplianceRegistrationService".
- Condition:
  - "cond_id": local identifier string within the rule (for example "C1", "C2").
  - "attribute": string, for example "auth.sop_level" or "applicant.id".
  - "operator": string, for example "==", "IN", "NOT_IN_LIST".
  - "value": optional scalar value (string or number) when there is a single value.
  - "values": optional list of values when there are multiple options.
  - "list_name": optional string for named lists such as "blacklist".
  - "group": optional string to group conditions into a logical clause, for example "g1".
  - "group_op": optional string, typically "AND" or "OR", describing how to combine conditions within that group.

Non-graph side constraints you must respect:
- Each Condition node MUST have "attribute" and "operator", and at least one of "value", "values", or "list_name".
- Each Rule node SHOULD have at least one subject, one action, one resource, and one condition linked via the relationships:
  - (:Rule)-[:APPLIES_TO_SUBJECT]->(:SubjectType)
  - (:Rule)-[:APPLIES_TO_ACTION]->(:Action)
  - (:Rule)-[:APPLIES_TO_RESOURCE]->(:ResourceType)
  - (:Rule)-[:HAS_CONDITION]->(:Condition)

Only use the five node types listed above. Do NOT create additional node labels such as Policy, Outcome, SourceSpan, ListRef, or WorkflowStep; instead, encode effects and logical structure using Rule properties and Condition properties (including "group" and "group_op").

For complex logic, use "group" and "group_op" on Condition nodes instead of additional node types:
- All Condition nodes with the same "group" belong to one logical clause.
- "group_op" tells you whether to combine conditions inside a group with AND or OR.
- Different groups for the same Rule can be interpreted as being combined with AND at the rule level.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for each relationship and the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON itself.
- Omit any backticks around the JSON – simply output the JSON on its own.
- The JSON object must not be wrapped in a list – it is its own JSON object.
- Property names must be enclosed in double quotes.

Examples:
{examples}

Input text:

{text}
'''

GRAPH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["id", "label", "properties"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "start_node_id": {"type": "string"},
                    "end_node_id": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["type", "start_node_id", "end_node_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nodes", "relationships"],
    "additionalProperties": False,
}


class RetryingLLMEntityRelationExtractor(LLMEntityRelationExtractor):
    """LLM extractor that retries JSON repair/validation before giving up."""

    def __init__(
        self,
        llm: OpenAILLM,
        prompt_template: str,
        create_lexical_graph: bool = True,
        on_error: OnError = OnError.IGNORE,
        max_concurrency: int = 5,
        max_attempts: int = 4,
        initial_backoff: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_backoff: float = 20.0,
        failure_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            create_lexical_graph=create_lexical_graph,
            on_error=on_error,
            max_concurrency=max_concurrency,
        )
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff
        self.failure_dir = failure_dir

    async def extract_for_chunk(
        self, schema: Any, examples: str, chunk: TextChunk
    ) -> Neo4jGraph:
        prompt = self.prompt_template.format(
            text=chunk.text,
            schema=schema.model_dump(exclude_none=True),
            examples=examples,
        )
        delay = self.initial_backoff
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            llm_result = await self.llm.ainvoke(prompt)
            try:
                llm_generated_json = fix_invalid_json(llm_result.content)
                result = json.loads(llm_generated_json)
                return Neo4jGraph.model_validate(result)
            except (json.JSONDecodeError, InvalidJSONError, ValidationError) as exc:
                last_exception = exc
                if attempt < self.max_attempts:
                    logger.warning(
                        f"Chunk {chunk.index}: invalid JSON/format on attempt {attempt}/{self.max_attempts}; retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_multiplier, self.max_backoff)
                    continue

                logger.error(
                    f"Chunk {chunk.index}: LLM response failed JSON/validation after {self.max_attempts} attempts"
                )
                logger.debug(f"Final invalid content: {llm_result.content}")

                if self.failure_dir:
                    try:
                        self.failure_dir.mkdir(parents=True, exist_ok=True)
                        failure_path = self.failure_dir / f"chunk-{chunk.index}.txt"
                        chunk_path = self.failure_dir / f"chunk-{chunk.index}.source.txt"
                        prompt_path = self.failure_dir / f"chunk-{chunk.index}.prompt.txt"
                        failure_path.write_text(
                            llm_result.content or "", encoding="utf-8"
                        )
                        chunk_path.write_text(chunk.text, encoding="utf-8")
                        prompt_path.write_text(prompt, encoding="utf-8")
                        logger.info(
                            f"Wrote failing chunk {chunk.index} artifacts to {self.failure_dir}"
                        )
                    except Exception as write_exc:
                        logger.warning(
                            f"Unable to write failing chunk {chunk.index} response: {write_exc}"
                        )

                if self.on_error == OnError.RAISE:
                    raise LLMGenerationError(
                        f"LLM response failed after {self.max_attempts} retries"
                    ) from last_exception
                return Neo4jGraph()

        if self.on_error == OnError.RAISE:
            raise LLMGenerationError("LLM retries exhausted") from last_exception
        return Neo4jGraph()

    async def run(
        self,
        chunks: TextChunks,
        document_info: Optional[DocumentInfo] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        schema: Optional[GraphSchema] = None,
        examples: str = "",
        **kwargs: Any,
    ) -> Neo4jGraph:
        # Delegate to parent run for orchestration/lexical graph while keeping retrying extract.
        return await super().run(
            chunks=chunks,
            document_info=document_info,
            lexical_graph_config=lexical_graph_config,
            schema=schema,
            examples=examples,
            **kwargs,
        )


class RetryingKGPipelineConfig(SimpleKGPipelineConfig):
    extractor_max_attempts: int = 4
    extractor_initial_backoff: float = 2.0
    extractor_backoff_multiplier: float = 2.0
    extractor_max_backoff: float = 20.0
    failure_dir: Optional[Path] = None

    def _get_extractor(self) -> LLMEntityRelationExtractor:
        failure_dir = self.failure_dir or FAILED_JSON_DIR
        return RetryingLLMEntityRelationExtractor(
            llm=self.get_default_llm(),
            prompt_template=self.prompt_template,
            on_error=self.on_error,
            max_attempts=self.extractor_max_attempts,
            initial_backoff=self.extractor_initial_backoff,
            backoff_multiplier=self.extractor_backoff_multiplier,
            max_backoff=self.extractor_max_backoff,
            failure_dir=failure_dir,
        )


class RetryingSimpleKGPipeline:
    """Wire a SimpleKGPipeline with a retrying extractor."""

    def __init__(
        self,
        llm: OpenAILLM,
        driver: Driver,
        embedder: OpenAIEmbeddings,
        entities: Optional[List[Any]] = None,
        relations: Optional[List[Any]] = None,
        potential_schema: Optional[List[tuple[str, str, str]]] = None,
        schema: Optional[Any] = None,
        from_pdf: bool = True,
        text_splitter: Optional[Any] = None,
        pdf_loader: Optional[Any] = None,
        kg_writer: Optional[Any] = None,
        on_error: OnError = OnError.IGNORE,
        prompt_template: str = prompt_template,
        perform_entity_resolution: bool = True,
        lexical_graph_config: Optional[Any] = None,
        neo4j_database: Optional[str] = None,
        max_attempts: int = 4,
        initial_backoff: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_backoff: float = 20.0,
        failure_dir: Optional[Path] = None,
    ):
        try:
            config = RetryingKGPipelineConfig.model_validate(
                dict(
                    llm_config=llm,
                    neo4j_config=driver,
                    embedder_config=embedder,
                    entities=entities or [],
                    relations=relations or [],
                    potential_schema=potential_schema,
                    schema=schema,
                    from_pdf=from_pdf,
                    pdf_loader=ComponentType(pdf_loader) if pdf_loader else None,
                    kg_writer=ComponentType(kg_writer) if kg_writer else None,
                    text_splitter=ComponentType(text_splitter)
                    if text_splitter
                    else None,
                    on_error=on_error,
                    prompt_template=prompt_template,
                    perform_entity_resolution=perform_entity_resolution,
                    lexical_graph_config=lexical_graph_config,
                    neo4j_database=neo4j_database,
                    extractor_max_attempts=max_attempts,
                    extractor_initial_backoff=initial_backoff,
                    extractor_backoff_multiplier=backoff_multiplier,
                    extractor_max_backoff=max_backoff,
                    failure_dir=failure_dir,
                )
            )
        except (ValidationError, ValueError) as e:
            raise PipelineDefinitionError() from e

        self.runner = PipelineRunner.from_config(config)

    async def run_async(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        document_metadata: Optional[dict[str, Any]] = None,
    ):
        return await self.runner.run(
            {
                "file_path": file_path,
                "text": text,
                "document_metadata": document_metadata,
            }
        )
# Create an Embedder object
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
    rate_limit_handler=RetryRateLimitHandler(
        max_attempts=8,
        min_wait=2.0,
        max_wait=120.0,
    ),
)
logger.info("Initialized OpenAI embedder model=text-embedding-3-large")

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-5.1",
    model_params={
        "max_completion_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "graph_extraction",
                "schema": GRAPH_JSON_SCHEMA,
            },
        },
        "temperature": 0,
    },
    api_key=OPENAI_API_KEY,
    rate_limit_handler=RetryRateLimitHandler(
        max_attempts=10,
        min_wait=2.0,
        max_wait=120.0,
        multiplier=3.0,
    ),
)
logger.info("Initialized LLM model=gpt-5.1 max_completion_tokens=2000")

# Instantiate the RetryingSimpleKGPipeline for PDF ingestion
kg_builder = RetryingSimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    schema={
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns,
    },
    prompt_template=prompt_template,
    neo4j_database=NEO4J_DATABASE,
    on_error=OnError.IGNORE,
    from_pdf=True,
    max_attempts=4,
    initial_backoff=2.0,
    backoff_multiplier=2.5,
    max_backoff=30.0,
)
logger.info(
    "KG builder configured with chunk_size=500 chunk_overlap=100 max_attempts=4 backoff=(2.0, 2.5, 30.0)"
)

pdf_path = (
    Path(__file__).resolve().parent
    / "BRD-examples"
    / "Business_Inspection_Policy_Document.pdf"
)
logger.info(f"Input PDF path resolved to {pdf_path}")


def slugify(value: str) -> str:
    """Create a simple, stable slug from a rule name."""
    value = value.strip().lower()
    parts: List[str] = []
    for ch in value:
        if ch.isalnum():
            parts.append(ch)
        elif ch in (" ", "-", "_", "/"):
            parts.append("_")
    slug = "".join(parts).strip("_")
    return slug or "rule"


def build_rule_ir(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a single Neo4j rule row into an intermediate representation."""
    rule_name: Optional[str] = record.get("rule_name")
    if not rule_name:
        return None

    subjects: List[str] = record.get("subjects") or []
    actions: List[str] = record.get("actions") or []
    resources: List[str] = record.get("resources") or []
    conditions: List[Dict[str, Any]] = record.get("conditions") or []

    # Synthesize condition IDs if missing
    for idx, cond in enumerate(conditions, start=1):
        if not cond.get("cond_id"):
            cond["cond_id"] = f"C{idx}"

    # Build a simple logic tree from condition groups.
    groups: Dict[str, Dict[str, Any]] = {}
    ungrouped_ids: List[str] = []
    for cond in conditions:
        cond_id = cond["cond_id"]
        group = cond.get("group")
        group_op = (cond.get("group_op") or "AND").upper()
        if group:
            group_node = groups.setdefault(
                group,
                {"op": group_op, "children": []},
            )
            group_node["children"].append(cond_id)
        else:
            ungrouped_ids.append(cond_id)

    logic_children: List[Any] = []
    # Each group becomes its own node
    for group_node in groups.values():
        logic_children.append(group_node)
    # Ungrouped conditions form an implicit AND group if there are multiple
    if ungrouped_ids:
        if len(ungrouped_ids) == 1 and not groups:
            logic_children.append({"op": "AND", "children": ungrouped_ids})
        else:
            logic_children.append({"op": "AND", "children": ungrouped_ids})

    if not logic_children:
        logic: Dict[str, Any] = {"op": "AND", "children": []}
    elif len(logic_children) == 1:
        logic = logic_children[0]
    else:
        logic = {"op": "AND", "children": logic_children}

    ir: Dict[str, Any] = {
        "rule_name": rule_name,
        "rule_id": slugify(rule_name),
        "effect": record.get("effect") or "ALLOW",
        "effect_params": record.get("effect_params") or {},
        "subjects": subjects,
        "actions": actions,
        "resources": resources,
        "primary_subject": subjects[0] if subjects else None,
        "primary_action": actions[0] if actions else None,
        "primary_resource": resources[0] if resources else None,
        "conditions": conditions,
        "logic": logic,
    }
    return ir


def emit_condition(cond: Dict[str, Any]) -> str:
    """Render a single condition dict into a Rego expression."""
    attribute: str = cond.get("attribute") or ""
    operator: str = (cond.get("operator") or "").upper()
    value = cond.get("value")
    values = cond.get("values")
    list_name = cond.get("list_name")

    # Attributes are treated as paths under input, e.g. "auth.sop_level" -> input.auth.sop_level
    attr_path = attribute.strip()
    if not attr_path:
        return "# SKIP: missing attribute"
    attr_expr = f"input.{attr_path}"

    if operator in ("==", "EQ"):
        return f"{attr_expr} == {repr(value)}"
    if operator in (">=", "GTE"):
        return f"{attr_expr} >= {repr(value)}"
    if operator in ("<=", "LTE"):
        return f"{attr_expr} <= {repr(value)}"
    if operator in (">", "GT"):
        return f"{attr_expr} > {repr(value)}"
    if operator in ("<", "LT"):
        return f"{attr_expr} < {repr(value)}"
    if operator == "IN":
        vals = values if isinstance(values, list) else []
        quoted_vals = ", ".join(repr(v) for v in vals)
        return f"some v; v := [{quoted_vals}][_] \n    {attr_expr} == v"
    if operator == "NOT_IN_LIST" and list_name:
        # Expect a data.lists.<list_name> object with keys for disallowed values
        return f"not data.lists.{list_name}[{attr_expr}]"

    return f"# TODO: unsupported operator {operator} for {attr_expr}"


def emit_logic(node: Any, conditions_by_id: Dict[str, Dict[str, Any]], indent: str = "    ") -> str:
    """Recursively render the logic tree into Rego body clauses."""
    if isinstance(node, str):
        cond = conditions_by_id.get(node)
        if not cond:
            return f"{indent}# SKIP: unknown condition {node}"
        expr = emit_condition(cond)
        # Indent the first line of the condition
        expr_lines = expr.splitlines()
        if not expr_lines:
            return ""
        expr_lines[0] = f"{indent}{expr_lines[0]}"
        for idx in range(1, len(expr_lines)):
            expr_lines[idx] = f"{indent}{expr_lines[idx]}"
        return "\n".join(expr_lines)

    op = (node.get("op") or "AND").upper()
    children = node.get("children") or []
    rendered_children: List[str] = []
    for child in children:
        rendered = emit_logic(child, conditions_by_id, indent=indent)
        if rendered:
            rendered_children.append(rendered)

    if not rendered_children:
        return ""

    if op == "AND":
        # In Rego, AND is just multiple lines in the body.
        return "\n".join(rendered_children)

    if op == "OR":
        # Use inline `or` between expressions.
        # Flatten single-line children if possible; otherwise fall back to comments.
        single_lines = [line.strip() for block in rendered_children for line in block.splitlines() if line.strip() and not line.strip().startswith("#")]
        if len(single_lines) >= 2:
            first = f"{indent}{single_lines[0]}"
            rest = [f"{indent}or {expr}" for expr in single_lines[1:]]
            return "\n".join([first] + rest)

    # Fallback: just stack the children; semantics may be stricter than intended but deterministic.
    return "\n".join(rendered_children)


def emit_rego_module(rules: List[Dict[str, Any]], package_name: str = "brd.business_inspection") -> str:
    """Emit a complete Rego module for a list of rule IR dicts."""
    lines: List[str] = []
    lines.append(f"package {package_name}")
    lines.append("")
    lines.append("# Auto-generated from Neo4j Rule graph; do not edit by hand.")
    lines.append("default allow = false")
    lines.append("")

    # Sort for deterministic output
    for rule in sorted(rules, key=lambda r: r.get("rule_id", "")):
        rule_id = rule["rule_id"]
        subject = rule.get("primary_subject") or ""
        action = rule.get("primary_action") or ""
        resource = rule.get("primary_resource") or ""
        effect = (rule.get("effect") or "ALLOW").upper()

        if effect != "ALLOW":
            lines.append(f"# TODO: effect {effect} not yet mapped; treating as ALLOW semantics.")

        lines.append(f"allow[\"{rule_id}\"] {{")
        if subject:
            lines.append(f"    input.subject.type == {repr(subject)}")
        if action:
            lines.append(f"    input.action == {repr(action)}")
        if resource:
            lines.append(f"    input.resource.type == {repr(resource)}")

        conditions_by_id = {c["cond_id"]: c for c in rule.get("conditions", [])}
        logic = rule.get("logic") or {"op": "AND", "children": []}
        body = emit_logic(logic, conditions_by_id, indent="    ")
        if body:
            lines.append(body)
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def fetch_rules_ir(neo4j_driver: Driver) -> List[Dict[str, Any]]:
    """Query Neo4j for all Rule subgraphs and normalize them into IR dicts."""
    query = """
    MATCH (r:Rule)
    OPTIONAL MATCH (r)-[:APPLIES_TO_SUBJECT]->(s:SubjectType)
    OPTIONAL MATCH (r)-[:APPLIES_TO_ACTION]->(a:Action)
    OPTIONAL MATCH (r)-[:APPLIES_TO_RESOURCE]->(res:ResourceType)
    OPTIONAL MATCH (r)-[:HAS_CONDITION]->(c:Condition)
    RETURN
      r.name AS rule_name,
      r.effect AS effect,
      r.effect_params AS effect_params,
      collect(DISTINCT s.name) AS subjects,
      collect(DISTINCT a.name) AS actions,
      collect(DISTINCT res.name) AS resources,
      collect(DISTINCT {
        cond_id: c.cond_id,
        attribute: c.attribute,
        operator: c.operator,
        value: c.value,
        values: c.values,
        list_name: c.list_name,
        group: c.group,
        group_op: c.group_op
      }) AS conditions
    ORDER BY rule_name
    """
    records: List[Dict[str, Any]] = []
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        for record in session.run(query):
            records.append(record.data())

    ir_rules: List[Dict[str, Any]] = []
    for record in records:
        ir = build_rule_ir(record)
        if ir:
            ir_rules.append(ir)
    logger.info(f"Fetched {len(ir_rules)} rules from Neo4j")
    return ir_rules


def log_graph_summary(neo4j_driver: Driver) -> None:
    """Log a concise summary of graph contents in Neo4j."""
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]

        node_counts: Dict[str, int] = {}
        for label in node_types:
            result = session.run(
                f"MATCH (n:`{label}`) RETURN count(n) AS c"
            ).single()
            node_counts[label] = result["c"]

        rel_counts: Dict[str, int] = {}
        for rel_type in relationship_types:
            result = session.run(
                f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c"
            ).single()
            rel_counts[rel_type] = result["c"]

    logger.info(f"Graph summary: total_nodes={total_nodes} total_relationships={total_rels}")
    logger.info(
        "Node counts: "
        + ", ".join(f"{label}={node_counts.get(label, 0)}" for label in node_types)
    )
    logger.info(
        "Relationship counts: "
        + ", ".join(f"{rel}={rel_counts.get(rel, 0)}" for rel in relationship_types)
    )


async def main() -> None:
    # Build the knowledge graph from the BRD PDF
    logger.info(f"Starting KG build from PDF {pdf_path}")
    await kg_builder.run_async(file_path=str(pdf_path))
    logger.info("KG build completed")
    log_graph_summary(driver)

    # Enumerate all Rule subgraphs and compile them to Rego.
    rules_ir = fetch_rules_ir(driver)
    total_conditions = sum(len(rule.get("conditions", [])) for rule in rules_ir)
    avg_conditions = (
        float(total_conditions) / len(rules_ir) if rules_ir else 0.0
    )
    logger.info(
        f"Rule IR summary: rules={len(rules_ir)} total_conditions={total_conditions} avg_conditions_per_rule={avg_conditions:.1f}"
    )

    # Write outputs next to this script: scripts/GraphRAG/outputs
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs directory ensured at {outputs_dir}")

    # Write an intermediate JSONL representation for inspection.
    ir_path = outputs_dir / "business_inspection.rules.jsonl"
    with ir_path.open("w", encoding="utf-8") as f:
        for rule in rules_ir:
            # Manual JSON encoding to avoid adding a new dependency.
            import json

            f.write(json.dumps(rule, ensure_ascii=False))
            f.write("\n")
    logger.info(
        f"Wrote IR JSONL to {ir_path} ({ir_path.stat().st_size} bytes, {len(rules_ir)} records)"
    )

    # Emit a Rego module for all rules.
    rego_module = emit_rego_module(rules_ir, package_name="brd.business_inspection")
    rego_path = outputs_dir / "business_inspection.rego"
    rego_path.write_text(rego_module, encoding="utf-8")
    logger.info(f"Wrote Rego module to {rego_path} ({rego_path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
    driver.close()
