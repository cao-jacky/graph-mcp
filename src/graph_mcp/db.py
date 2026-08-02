"""Neo4j driver management and schema definition.

The graph is an *index*, never a second copy of the corpus: nodes carry paths,
titles and hashes so an agent can navigate, then read real content through
knowledge-mcp. Chunk text is the one exception — it must be stored to be
embedded and returned as a search snippet.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from neo4j import Driver, GraphDatabase

from .config import Settings, settings

# Uniqueness constraints double as lookup indexes, so every MERGE below hits
# an index instead of scanning.
CONSTRAINTS = [
    "CREATE CONSTRAINT document_path IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.path IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS "
    "FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX document_category IF NOT EXISTS FOR (d:Document) ON (d.category)",
    "CREATE INDEX document_updated IF NOT EXISTS FOR (d:Document) ON (d.updated)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_path)",
]


def vector_indexes(dim: int) -> list[str]:
    """Vector index DDL. 4096 is Neo4j's per-vector ceiling — exactly the
    native width of Qwen3-Embedding-8B, so nothing is truncated."""
    return [
        f"""
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
        f"""
        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
        FOR (e:Entity) ON (e.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """,
    ]


def make_driver(cfg: Settings | None = None) -> Driver:
    cfg = cfg or settings
    if not cfg.neo4j_password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Export it (or put it in .env) before "
            "running graph-sync or graph-mcp."
        )
    return GraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_user, cfg.neo4j_password),
        # Before Stage 3 runs, :Entity and :MENTIONS legitimately do not exist,
        # and the server warns about them on every query that mentions them.
        # Silencing UNRECOGNIZED keeps that expected state off stderr without
        # hiding real warnings.
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


@contextmanager
def driver_session(cfg: Settings | None = None) -> Iterator:
    cfg = cfg or settings
    driver = make_driver(cfg)
    try:
        with driver.session(database=cfg.neo4j_database) as session:
            yield session
    finally:
        driver.close()


def apply_schema(cfg: Settings | None = None) -> list[str]:
    """Create constraints and indexes. Idempotent — safe to re-run."""
    cfg = cfg or settings
    applied: list[str] = []
    with driver_session(cfg) as session:
        for statement in CONSTRAINTS + INDEXES:
            session.run(statement)
            applied.append(statement.split("IF NOT EXISTS")[0].strip())
        for statement in vector_indexes(cfg.embed_dim):
            try:
                session.run(statement)
                applied.append(statement.split("IF NOT EXISTS")[0].strip())
            except Exception as exc:  # noqa: BLE001 - surfaced to the operator
                applied.append(f"VECTOR INDEX SKIPPED ({exc.__class__.__name__}: {exc})")
    return applied


def graph_stats(cfg: Settings | None = None) -> dict[str, int]:
    # COUNT {} expressions rather than CALL {} subqueries: Neo4j 5.26 deprecates
    # a CALL subquery without a variable scope clause, and the replacement
    # syntax (`CALL () { ... }`) is not accepted by earlier 5.x. COUNT {} works
    # across the range and keeps deprecation warnings off the MCP stderr.
    query = """
    RETURN
      COUNT { MATCH (d:Document) WHERE coalesce(d.stub, false) = false } AS documents,
      COUNT { MATCH (d:Document) WHERE d.stub = true }                   AS stubs,
      COUNT { MATCH (c:Chunk) }                                          AS chunks,
      COUNT { MATCH (c:Chunk) WHERE c.embedding IS NOT NULL }            AS embedded_chunks,
      COUNT { MATCH (t:Tag) }                                            AS tags,
      COUNT { MATCH (e:Entity) }                                         AS entities,
      COUNT { MATCH ()-[r:LINKS_TO]->() }                                AS links_to,
      COUNT { MATCH ()-[r:TAGGED]->() }                                  AS tagged,
      COUNT { MATCH ()-[r:MENTIONS]->() }                                AS mentions,
      COUNT { MATCH ()-[r:RELATES_TO]->() }                              AS relates_to
    """
    with driver_session(cfg) as session:
        record = session.run(query).single()
        return dict(record) if record else {}
