import asyncio
import os

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_API_KEY = os.environ.get(OPENAI_API_KEY_ENV)

if not OPENAI_API_KEY:
    raise RuntimeError(f"{OPENAI_API_KEY_ENV} must be set before running GraphRAG.")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def reset_database(neo4j_driver: Driver) -> None:
    """Wipe all nodes and relationships so each run starts fresh."""
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


reset_database(driver)

# List the entities and relations the LLM should look for in the text
node_types = ["Person", "House", "Planet"]
relationship_types = ["PARENT_OF", "HEIR_OF", "RULES"]
patterns = [
    ("Person", "PARENT_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
]

# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large",api_key=OPENAI_API_KEY)

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-5.1",
    model_params={
        "max_completion_tokens": 2000,
        "response_format": {"type": "json_object"},

    },
    api_key=OPENAI_API_KEY
)

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    schema={
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns,
    },
    on_error="IGNORE",
    from_pdf=False,
)

# Run the pipeline on a piece of text
text = (
    "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
    "Atreides, an aristocratic family that rules the planet Caladan."
)
asyncio.run(kg_builder.run_async(text=text))
driver.close()
