# graph-mcp

A Neo4j knowledge-graph index over a personal `Knowledge/` markdown corpus,
exposed to LLM agents over MCP. Companion to
[`knowledge-mcp`](https://github.com/cao-jacky/knowledge-mcp): that server owns
the files, this one owns navigation.

**The graph is an index, never a second copy of the corpus.** Nodes carry
paths, titles and hashes; agents traverse here and then read real content
through `knowledge-mcp`'s `read_knowledge`. The only stored text is chunk
text, which has to exist to be embedded and returned as a search snippet.

## Architecture

```
Knowledge/*.md ──> graph-sync ──> Neo4j (Bolt, private interface)
                       │                    ▲
                       │                    │
                       └─> embedding model (OpenAI-compatible endpoint)
                                            │
    LLM agent ──MCP──> graph-mcp ───────────┘
```

The three pieces can live on one machine or three. The reference deployment
runs Neo4j on a home server and `graph-sync`/`graph-mcp` on a workstation
alongside a locally-served embedding model.

Two MCP servers, deliberately split: connection profiles differ (Neo4j
pooling vs plain file I/O), the graph can be added or removed independently,
and scoped tool descriptions keep the model routing to the right one.

### Schema

| Node | Key | Notes |
|---|---|---|
| `:Document` | `path` | Relative to the KB root. Also `title`, `category`, `subcategory`, `content_hash`, `created`, `updated`, `word_count`. `stub: true` marks a wikilink target with no file. |
| `:Chunk` | `id` (`path#ordinal`) | `text`, `breadcrumb`, `embedding` (4096-d). |
| `:Entity` | `key` (slug) | `name`, `type`, `aliases`, `embedding`. Stage 3 only. |
| `:Tag` | `name` | One node per unique frontmatter tag. |

| Edge | Meaning |
|---|---|
| `(:Document)-[:LINKS_TO]->(:Document)` | Resolved `[[wikilink]]`. |
| `(:Document)-[:TAGGED]->(:Tag)` | Frontmatter tag. |
| `(:Document)-[:HAS_CHUNK]->(:Chunk)` | Vector index membership. |
| `(:Document)-[:MENTIONS {count}]->(:Entity)` | Stage 3. |
| `(:Entity)-[:RELATES_TO {type, confidence}]->(:Entity)` | Stage 3. |
| `(:Document)-[:SUPERSEDES]->(:Document)` | From a `supersedes:` frontmatter key. |

## Tools

Always available:

| Tool | Purpose |
|---|---|
| `semantic_search(query, limit)` | Meaning-based retrieval over chunks. Use when wording won't match; use `knowledge-mcp`'s `search_knowledge` for exact strings. |
| `documents_by_tag(tag)` | Tag navigation. |
| `list_tags(min_documents)` | Discover the corpus's tag vocabulary. |
| `entities_in_document(path)` | One note's neighbourhood: tags, links in/out, entities. |
| `similar_documents(path, limit)` | Related notes by embedding, beyond hand-written links. |
| `graph_overview()` | Node/edge counts — check which stages have run. |

Registered only when `GRAPH_MCP_SEMANTIC_TOOLS=1` (after Stage 3 populates the
edges they traverse):

| Tool | Purpose |
|---|---|
| `find_related_entities(entity, max_hops)` | "What connects to X", "who worked on Y". |
| `shortest_path(entity_a, entity_b)` | How two entities are connected. |
| `recent_related_changes(entity, since)` | "What's changed lately about X". |

## Setup

Requires [uv](https://docs.astral.sh/uv/), a Neo4j 5.x instance, and an
OpenAI-compatible embeddings endpoint.

```bash
git clone https://github.com/cao-jacky/graph-mcp
cd graph-mcp
uv sync
cp .env.example .env    # set GRAPH_MCP_KB_ROOT and NEO4J_PASSWORD
```

Register with Claude Code:

```bash
claude mcp add --scope user graph \
  --env GRAPH_MCP_KB_ROOT=/path/to/your/Knowledge \
  --env NEO4J_URI=bolt://127.0.0.1:7687 \
  --env NEO4J_PASSWORD=... \
  -- uv run --directory /path/to/graph-mcp graph-mcp
```

## Build plan and validation gates

Each stage has a gate. **Don't start the next stage until the current one's
check passes** — the two risky points are Stage 3 (extraction quality and
dedup) and Stage 5 (automating before the pipeline is trustworthy).

### Stage 0 — Neo4j

```bash
echo "NEO4J_PASSWORD=$(openssl rand -base64 24)" > .env
docker compose up -d
docker compose ps                      # healthy
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1"
```

**Portainer:** Stacks → Add stack → Repository, point it at this repo. The
root `docker-compose.yml` is the stack file; set `NEO4J_PASSWORD` (and
`BOLT_BIND_ADDR`, if needed) in Portainer's environment-variables editor.

Bolt publishes on `127.0.0.1` by default, which is correct when `graph-sync`
and `graph-mcp` run on the same host as Neo4j. If they run elsewhere, set
`BOLT_BIND_ADDR` to a **private** address — a VPN/WireGuard/Tailscale address
or a LAN address. Never `0.0.0.0`, which would expose Bolt to every network
the host can reach.

**Gate:** container healthy, `RETURN 1` succeeds, and `uv run graph-sync
status` reports counts rather than a connection error.

**Rollback:** `docker compose down -v neo4j` — its own volume, no blast radius
on the rest of the stack.

### Stage 1 — Structural extraction

```bash
uv run graph-sync parse-check     # offline, no Neo4j needed
uv run graph-sync schema
uv run graph-sync structural
```

**Gate:** document count matches the file count
(`find "$GRAPH_MCP_KB_ROOT" -name '*.md' -not -path '*/.*' | grep -v '/index.md' | wc -l`),
and notes you know cross-reference each other show `LINKS_TO` edges.

**Rollback:** idempotent — fix the script and re-run rather than hand-cleaning.
Re-running also prunes documents, edges and orphan tags that no longer exist,
so renames and deletions self-correct.

### Stage 1b — Vector index

```bash
uv run graph-sync embed          # ~7 min for 1408 chunks; --force to redo all
```

Not in the original plan; added because the local embedding model makes
semantic retrieval and reliable entity dedup free. Skips unchanged documents
by content hash.

**Gate:** `graph_overview` shows `embedded_chunks == chunks`, and
`semantic_search` on a topic you know returns the right note in the top few.

### Stage 2 — Minimal graph-mcp

Register the server (above) and leave `GRAPH_MCP_SEMANTIC_TOOLS` unset.

**Gate:** ask an agent a tag-navigation question whose answer you know
("what notes are tagged hermes") and confirm the tool is called and returns
the right set — before trusting it with anything semantic.

### Stage 3 — Semantic extraction

```bash
uv run graph-sync semantic --limit 8      # validate on notes you know first
uv run graph-sync semantic                # then the full backfill
```

Runs against the local LM Studio model by default: no API cost, and no note
content leaves the machine. Entity dedup happens **before** insert — exact key
match, then Qwen3 embedding similarity above
`GRAPH_MCP_ENTITY_MERGE_THRESHOLD` (0.92) within the same entity type.

**Gate:** inspect the extraction for 5–10 notes you know well before running
at scale. Watch for systematic misses — bullet-heavy notes tend to yield fewer
relations than prose. Check `find_related_entities` on a familiar entity for
wrongly-merged or wrongly-split entities; tune the threshold and re-run rather
than cleaning up afterwards.

**Rollback:** additive and idempotent per document.

```cypher
MATCH ()-[r:RELATES_TO]->() DELETE r;
MATCH (e:Entity) DETACH DELETE e;
MATCH (d:Document) REMOVE d.extracted_hash;
```

Stage 1 data is untouched by this.

### Stage 4 — Relationship tools

Set `GRAPH_MCP_SEMANTIC_TOOLS=1` and restart the MCP client.

**Gate:** ask a genuinely multi-hop question you *don't* already know the
answer to and check the returned path is sane. This is the first point where
the graph does something plain search could not.

### Stage 5 — Automation

Only after Stages 1–4 have been run by hand enough times to trust their
behaviour on renames, deletions and malformed frontmatter.

```bash
cp deploy/graph-sync.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now graph-sync.timer
```

The timer runs `structural` + `embed` only; the semantic pass stays manual.

**Gate:** edit one note, wait for the trigger, confirm the graph updated
without a manual run. Keep the manual path working as an escape hatch.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GRAPH_MCP_KB_ROOT` | falls back to `KNOWLEDGE_MCP_ROOT`, then `~/Knowledge` | Corpus root. Must exist. |
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Bolt endpoint. |
| `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | `neo4j` / — / `neo4j` | Credentials. |
| `GRAPH_MCP_EMBED_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio OpenAI-compatible endpoint. |
| `GRAPH_MCP_EMBED_MODEL` | `text-embedding-qwen3-embedding-8b` | Embedding model. |
| `GRAPH_MCP_EMBED_DIM` | `4096` | Must match the model *and* the vector index. |
| `GRAPH_MCP_EMBED_BATCH` | `16` | Texts per embedding request. |
| `GRAPH_MCP_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | LM Studio / `qwen3.5-122b-a10b` / `lm-studio` | Stage 3 extraction. |
| `GRAPH_MCP_CHUNK_WORDS` / `_OVERLAP` | `350` / `60` | Chunk sizing. |
| `GRAPH_MCP_ENTITY_MERGE_THRESHOLD` | `0.92` | Cosine similarity above which two entities merge. |
| `GRAPH_MCP_SEMANTIC_TOOLS` | unset | `1` registers the Stage 4 tools. |

## Corpus quirks the parser handles

Discovered by running against the real 277-note corpus; the tests in
`tests/test_parse.py` pin each one:

- **Most notes have no frontmatter.** 163 of 277 (the imported `ai-systems/`
  tree). Absent frontmatter is the common case, not an error; dates fall back
  to file mtime.
- **Nested subdirectories.** `ai-systems/08-memory-and-state/*.md` sits two
  levels deep. The walk is fully recursive.
- **Wikilinks inside code must not become edges.** A `grep "^[[:space:]]*$"`
  snippet and prose about `` `[[wikilinks]]` `` would otherwise create bogus
  nodes. Fenced blocks and inline code are blanked before extraction.
- **The synced `*-SKILL.md` notes have malformed fencing** — a bare ` ``` `
  preview block containing further ` ``` ` fences — so by CommonMark their
  shell snippets are *not* code. A plausibility filter rejects targets
  containing `:` or `*` while keeping real names like `Cyberpunk 2077`.
- **All 56 skill notes open with `# SKILL.md`.** A heading that is merely a
  filename is rejected in favour of the `synced_from:` directory name.
- **Tags come in both inline (`[a, b]`) and block (`- a`) YAML form.**
- **Short sections are packed together.** One chunk per heading gave 4024
  chunks averaging 90 words; packing yields 1408 averaging 258.

## Tests

```bash
uv run python tests/test_parse.py     # 23 checks, no Neo4j or network needed
```

## Known gap in `knowledge-mcp`

`knowledge-mcp`'s `_entry_files()` walks only the root and one level of
category directories, so the ~135 notes nested deeper (`ai-systems/*/*.md`)
are invisible to `search_knowledge`, `list_knowledge` and the generated
`index.md`. `graph-mcp` indexes them, which means `semantic_search` can return
a path that `read_knowledge` will still happily read but that
`search_knowledge` would never have found. Worth fixing there separately.
