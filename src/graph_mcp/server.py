"""MCP server exposing the knowledge graph as navigation tools.

The graph is an index, not a content store. Every tool returns document paths;
actual note content is read through knowledge-mcp's `read_knowledge`. Tool
descriptions say so explicitly, because routing quality between the two servers
depends on the descriptions rather than on any standing prompt instruction.

Relationship tools (`find_related_entities`, `shortest_path`,
`recent_related_changes`) stay unregistered until GRAPH_MCP_SEMANTIC_TOOLS=1,
so they cannot be called before Stage 3 has populated the edges they traverse.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import settings
from .db import driver_session, graph_stats
from .embeddings import EmbeddingClient

mcp = MCPServer(
    "graph",
    instructions=(
        "Knowledge-graph index over the personal Knowledge/ markdown corpus "
        f"at {settings.kb_root}. Use it to navigate by structure and meaning: "
        "semantic search, tags, wikilinks, and entity relationships. It returns "
        "document paths and snippets, never full notes — read those with "
        "knowledge-mcp's read_knowledge."
    ),
)

_embedder = EmbeddingClient(settings)


def _rows(query: str, **params: Any) -> list[dict]:
    with driver_session(settings) as session:
        return [dict(record) for record in session.run(query, **params)]


def _fmt(rows: list[dict], empty: str) -> str:
    if not rows:
        return empty
    lines = []
    for row in rows:
        parts = [f"{k}={v!r}" if not isinstance(v, str) else f"{k}={v}"
                 for k, v in row.items()]
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


@mcp.tool()
def semantic_search(query: str, limit: int = 8) -> str:
    """Find passages in the knowledge base by meaning rather than keywords.

    Use when the question is conceptual or the user's wording is unlikely to
    match the note's wording verbatim — "how do I handle X", "what did I decide
    about Y". For exact strings, literal identifiers, or filenames, prefer
    knowledge-mcp's search_knowledge instead.

    Returns the best-matching chunks with their document path, section
    breadcrumb, and similarity score. Read the full note via knowledge-mcp's
    read_knowledge using the returned path.
    """
    vector = _embedder.embed_query(query)
    rows = _rows(
        """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.doc_path AS path, node.breadcrumb AS section,
               round(score, 4) AS score, node.text AS text
        ORDER BY score DESC
        """,
        k=max(1, min(limit, 50)),
        embedding=vector,
    )
    if not rows:
        return (
            "No results. If the graph was just built, confirm the vector stage "
            "ran: `graph-sync status`."
        )
    out = []
    for row in rows:
        snippet = " ".join(row["text"].split()[:60])
        out.append(
            f"{row['path']}  (score {row['score']})\n"
            f"  section: {row['section']}\n"
            f"  {snippet}…"
        )
    return "\n\n".join(out)


@mcp.tool()
def documents_by_tag(tag: str) -> str:
    """List documents carrying a frontmatter tag, e.g. "hermes" or "homelab".

    Use for tag-based navigation — "what notes are tagged X". Call list_tags
    first if unsure which tags exist. Returns paths; read content via
    knowledge-mcp's read_knowledge.
    """
    rows = _rows(
        """
        MATCH (d:Document)-[:TAGGED]->(t:Tag {name: $tag})
        RETURN d.path AS path, d.title AS title, d.updated AS updated
        ORDER BY d.updated DESC
        """,
        tag=tag.strip().lower().lstrip("#"),
    )
    return _fmt(rows, f"No documents tagged {tag!r}. Try list_tags.")


@mcp.tool()
def list_tags(min_documents: int = 1) -> str:
    """List every tag in the corpus with how many documents use it.

    Use to discover the corpus's own vocabulary before calling documents_by_tag.
    """
    rows = _rows(
        """
        MATCH (d:Document)-[:TAGGED]->(t:Tag)
        WITH t.name AS tag, count(d) AS documents
        WHERE documents >= $min
        RETURN tag, documents ORDER BY documents DESC, tag
        """,
        min=max(1, min_documents),
    )
    return _fmt(rows, "No tags in the graph — has Stage 1 been run?")


@mcp.tool()
def entities_in_document(path: str) -> str:
    """Show what a specific note is about: its tags, its wikilinks in and out,
    and (once the semantic pass has run) the entities it mentions.

    Use when you already know which note you care about and want its
    neighbourhood — "what does this note connect to". `path` is relative to the
    knowledge base root, e.g. "projects/my-project.md".
    """
    rows = _rows(
        """
        MATCH (d:Document {path: $path})
        OPTIONAL MATCH (d)-[:TAGGED]->(t:Tag)
        OPTIONAL MATCH (d)-[:LINKS_TO]->(out:Document)
        OPTIONAL MATCH (inb:Document)-[:LINKS_TO]->(d)
        OPTIONAL MATCH (d)-[m:MENTIONS]->(e:Entity)
        RETURN d.title AS title, d.category AS category,
               collect(DISTINCT t.name) AS tags,
               collect(DISTINCT out.path) AS links_out,
               collect(DISTINCT inb.path) AS links_in,
               collect(DISTINCT e.name + ' (' + e.type + ')') AS entities
        """,
        path=path,
    )
    if not rows:
        return f"No document node for {path!r}. Paths are relative to the KB root."
    row = rows[0]
    blocks = [f"{row['title']}  [{row['category']}]  {path}"]
    for label, key in (
        ("tags", "tags"),
        ("links out", "links_out"),
        ("links in", "links_in"),
        ("entities", "entities"),
    ):
        values = [v for v in row[key] if v]
        if values:
            blocks.append(f"  {label}: " + ", ".join(sorted(values)))
    if not row["entities"]:
        blocks.append("  entities: none yet (semantic pass has not covered this note)")
    return "\n".join(blocks)


@mcp.tool()
def similar_documents(path: str, limit: int = 8) -> str:
    """Find notes covering similar ground to a given note, by embedding
    similarity rather than by explicit links.

    Use to surface related material the wikilinks miss — near-duplicates,
    or an older note on the same topic. Complements entities_in_document,
    which only sees links you wrote by hand.
    """
    rows = _rows(
        """
        MATCH (:Document {path: $path})-[:HAS_CHUNK]->(c:Chunk)
        WITH collect(c.embedding) AS vectors
        UNWIND vectors AS vector
        CALL db.index.vector.queryNodes('chunk_embedding', 20, vector)
        YIELD node, score
        WITH node.doc_path AS path, max(score) AS score
        WHERE path <> $path
        MATCH (d:Document {path: path})
        RETURN path, d.title AS title, round(score, 4) AS score
        ORDER BY score DESC LIMIT $k
        """,
        path=path,
        k=max(1, min(limit, 25)),
    )
    return _fmt(rows, f"No similar documents for {path!r} (is it embedded yet?).")


@mcp.tool()
def graph_overview() -> str:
    """Report what the graph currently contains — node and edge counts per type.

    Use to check which extraction stages have run before trusting a query that
    depends on them.
    """
    stats = graph_stats(settings)
    if not stats:
        return "Graph is empty or unreachable."
    return "\n".join(f"  {k:17}: {v}" for k, v in stats.items())


if settings.semantic_tools_enabled:

    @mcp.tool()
    def find_related_entities(entity: str, max_hops: int = 2) -> str:
        """Find what connects to a person, project, tool, or concept.

        Use when the query is about relationships rather than keyword content —
        "what connects to X", "who worked on Y", "what depends on Z". Returns
        entity names, relation types, and the document paths that assert them;
        fetch the actual content via knowledge-mcp's read_knowledge.
        """
        hops = max(1, min(max_hops, 4))
        rows = _rows(
            f"""
            MATCH (start:Entity)
            WHERE toLower(start.name) = toLower($entity)
               OR any(a IN coalesce(start.aliases, [])
                      WHERE toLower(a) = toLower($entity))
            MATCH path = (start)-[:RELATES_TO*1..{hops}]-(other:Entity)
            WITH other, path, relationships(path) AS rels
            OPTIONAL MATCH (d:Document)-[:MENTIONS]->(other)
            RETURN other.name AS entity, other.type AS type,
                   length(path) AS hops,
                   [r IN rels | r.type] AS via,
                   collect(DISTINCT d.path)[0..4] AS documents
            ORDER BY hops, entity LIMIT 40
            """,
            entity=entity,
        )
        return _fmt(rows, f"No entity matching {entity!r}. Try graph_overview.")

    @mcp.tool()
    def shortest_path(entity_a: str, entity_b: str) -> str:
        """Show how two entities are connected, as a chain of relations.

        Use for "how are X and Y related" questions. Returns the node/edge path;
        read supporting notes via knowledge-mcp.
        """
        rows = _rows(
            """
            MATCH (a:Entity), (b:Entity)
            WHERE toLower(a.name) = toLower($a) AND toLower(b.name) = toLower($b)
            MATCH path = shortestPath((a)-[:RELATES_TO*..6]-(b))
            RETURN [n IN nodes(path) | n.name] AS nodes,
                   [r IN relationships(path) | r.type] AS relations,
                   length(path) AS hops
            """,
            a=entity_a,
            b=entity_b,
        )
        if not rows:
            return f"No path found between {entity_a!r} and {entity_b!r}."
        row = rows[0]
        chain = row["nodes"][0]
        for relation, node in zip(row["relations"], row["nodes"][1:]):
            chain += f" -[{relation}]- {node}"
        return f"{chain}\n  ({row['hops']} hops)"

    @mcp.tool()
    def recent_related_changes(entity: str, since: str = "") -> str:
        """List recently updated notes that mention an entity.

        Use for "what's changed lately about X". `since` is an ISO date such as
        "2026-07-01"; omit it for the most recent regardless of date.
        """
        rows = _rows(
            """
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) = toLower($entity)
              AND ($since = '' OR d.updated >= $since)
            RETURN d.path AS path, d.title AS title, d.updated AS updated
            ORDER BY d.updated DESC LIMIT 25
            """,
            entity=entity,
            since=since,
        )
        return _fmt(rows, f"No recent documents mentioning {entity!r}.")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
