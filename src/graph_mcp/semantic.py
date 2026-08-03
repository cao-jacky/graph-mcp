"""Stage 3 — semantic extraction: entities and typed relations, via an LLM.

Defaults to a locally-served OpenAI-compatible chat model, so the
full-corpus backfill costs nothing and no note content leaves the machine.
Point GRAPH_MCP_LLM_BASE_URL at any OpenAI-compatible endpoint to change that.

Entity dedup runs *before* insert, as the plan requires: exact key match first,
then a vector-similarity check against existing entities using the same
Qwen3 embeddings as the chunk index. Untangling wrongly-merged entities after
the fact is far more painful than matching correctly up front.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import Settings, settings
from .db import driver_session
from .embeddings import EmbeddingClient
from .parse import Document, iter_documents

ENTITY_TYPES = ("person", "project", "tool", "concept", "organization")

# A closed vocabulary, enforced by the JSON schema rather than merely suggested
# in the prompt. Left free-text, the model emits `part_of` and `is_part_of` for
# the same relationship, plus invented types like `runs` — and each variant
# becomes a distinct edge type in Neo4j, so a query for one silently misses the
# others. The edge label *is* the answer for shortest_path, so this matters.
RELATION_TYPES = (
    "works_on", "depends_on", "part_of", "uses", "supersedes",
    "contradicts", "created_by", "related_to",
)

SYSTEM_PROMPT = """\
You extract a knowledge graph from personal technical notes.

Return ONLY a JSON object, no prose and no markdown fence, of the shape:
{
  "entities": [
    {"name": "...", "type": "person|project|tool|concept|organization",
     "aliases": ["..."]}
  ],
  "relations": [
    {"source": "...", "target": "...", "type": "<one of the allowed types>",
     "confidence": 0.0-1.0}
  ]
}

Rules:
- Extract at most 12 entities: the ones this note is genuinely ABOUT, that
  someone would search for by name. Fewer is better than more.
- Do NOT extract generic technology nouns (OS, VM, API, database, server,
  container, model) unless the note is specifically about that thing.
- Do NOT extract the knowledge base, the note itself, or note-taking tools.
- Use the most canonical name you can (e.g. "Neo4j", not "the neo4j database").
- `source` and `target` in relations MUST exactly match a `name` in entities.
- `type` MUST be exactly one of: works_on, depends_on, part_of, uses, supersedes, contradicts, created_by, related_to.
- Prefer a specific relation type over "related_to".
- confidence reflects how explicitly the note states the relation.
- If the note contains nothing worth extracting, return empty lists.\
"""

# Plain Cypher, no APOC, so this runs on neo4j:5-community unmodified.
# Aliases accumulate across runs; the list comprehension de-duplicates them.
UPSERT_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {key: row.key})
ON CREATE SET e.name = row.name, e.type = row.type
WITH e, row, coalesce(e.aliases, []) + row.aliases AS merged
SET e.embedding = row.embedding,
    e.aliases = reduce(acc = [], a IN merged |
        CASE WHEN a IS NULL OR a IN acc THEN acc ELSE acc + a END)
"""

UPSERT_MENTIONS = """
UNWIND $rows AS row
MATCH (d:Document {path: row.path})
MATCH (e:Entity {key: row.key})
MERGE (d)-[m:MENTIONS]->(e)
SET m.count = coalesce(m.count, 0) + row.count
"""

UPSERT_RELATIONS = """
UNWIND $rows AS row
MATCH (a:Entity {key: row.source})
MATCH (b:Entity {key: row.target})
MERGE (a)-[r:RELATES_TO {type: row.type}]->(b)
SET r.confidence = CASE
        WHEN r.confidence IS NULL OR row.confidence > r.confidence
        THEN row.confidence ELSE r.confidence END,
    r.source_path = row.path
"""

CLEAR_DOC_SEMANTICS = """
UNWIND $paths AS path
MATCH (:Document {path: path})-[m:MENTIONS]->()
DELETE m
"""

STALE_DOCUMENTS = """
MATCH (d:Document)
WHERE coalesce(d.stub, false) = false
  AND (d.extracted_hash IS NULL OR d.extracted_hash <> d.content_hash)
RETURN d.path AS path
"""

# Written per document as soon as its writes land, not batched at the end of
# the run: extraction is slow enough that the job is normally run in slices,
# and an interrupted run must lose at most the document in flight.
MARK_EXTRACTED = """
MATCH (d:Document {path: $path})
SET d.extracted_hash = $content_hash
"""

SIMILAR_ENTITY = """
CALL db.index.vector.queryNodes('entity_embedding', 5, $embedding)
YIELD node, score
WHERE score >= $threshold AND node.type = $type
RETURN node.key AS key, node.name AS name, score
ORDER BY score DESC LIMIT 1
"""


def entity_key(name: str) -> str:
    """Stable identity for an entity name — case and punctuation insensitive."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass
class SemanticReport:
    documents_considered: int = 0
    documents_extracted: int = 0
    skipped_unchanged: int = 0
    entities_new: int = 0
    entities_merged: int = 0
    mentions: int = 0
    relations: int = 0
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0
    stopped_early: bool = False

    def render(self) -> str:
        lines = [
            "Stage 3 — semantic extraction",
            f"  documents in corpus : {self.documents_considered}",
            f"  extracted           : {self.documents_extracted} "
            f"(skipped {self.skipped_unchanged} unchanged)",
            f"  entities            : {self.entities_new} new, "
            f"{self.entities_merged} merged into existing",
            f"  MENTIONS edges      : {self.mentions}",
            f"  RELATES_TO edges    : {self.relations}",
            f"  elapsed             : {self.seconds:.1f}s",
        ]
        if self.stopped_early:
            remaining = (
                self.documents_considered
                - self.skipped_unchanged
                - self.documents_extracted
            )
            lines.append(
                f"  time budget reached — ~{remaining} documents still pending; "
                "re-run to continue where this left off"
            )
        if self.failures:
            lines.append(f"  failures            : {len(self.failures)}")
            lines.extend(f"      {f}" for f in self.failures[:10])
        return "\n".join(lines)


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string", "enum": list(RELATION_TYPES)},
                    "confidence": {"type": "number"},
                },
                "required": ["source", "target", "type"],
            },
        },
    },
    "required": ["entities", "relations"],
}

# Endpoints disagree on how to ask for structured output: LM Studio accepts
# only `json_schema` or `text`, older OpenAI-compatible servers only
# `json_object`. Try the strictest first and degrade, rather than pinning one
# and breaking when the backend is swapped.
RESPONSE_FORMATS: tuple[dict | None, ...] = (
    {
        "type": "json_schema",
        "json_schema": {"name": "kg_extraction", "strict": True,
                        "schema": EXTRACTION_SCHEMA},
    },
    {"type": "json_object"},
    None,
)


class LLMClient:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self._format_index = 0
        self._thinking_kwarg = cfg.llm_disable_thinking if cfg else settings.llm_disable_thinking

    def _request(self, prompt: str, response_format: dict | None) -> dict:
        payload: dict = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.cfg.llm_max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self._thinking_kwarg:
            # Reasoning models emit a long chain of thought before the JSON.
            # Extraction needs none of it: measured on a 28-word note, the model
            # spent 2744 reasoning tokens to produce ~200 tokens of JSON — 93%
            # of the work, and 103s instead of 14s.
            #
            # `reasoning_effort` is the parameter that actually works here.
            # `chat_template_kwargs: {enable_thinking: false}` was accepted
            # without error and silently ignored, and appending `/no_think` to
            # the prompt made it worse — 4000 reasoning tokens instead of 2744.
            # Verify any change here against `usage.completion_tokens_details.
            # reasoning_tokens`, not against latency alone.
            payload["reasoning_effort"] = "none"
        request = urllib.request.Request(
            f"{self.cfg.llm_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.llm_api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.cfg.llm_timeout) as response:
            return json.load(response)

    def complete_json(self, prompt: str) -> dict:
        # The working format is remembered, so the negotiation costs at most
        # one wasted request per process rather than one per document.
        while self._format_index < len(RESPONSE_FORMATS):
            try:
                body = self._request(prompt, RESPONSE_FORMATS[self._format_index])
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
                # Not a payload problem, so degrading response_format wastes
                # three more slow requests and still fails. Say what is wrong.
                if "context size" in detail.lower() or "context length" in detail.lower():
                    raise RuntimeError(
                        "Model context exceeded. With GRAPH_MCP_LLM_CONCURRENCY="
                        f"{self.cfg.llm_concurrency}, the server's context is split "
                        "across that many slots, so each request gets a fraction of "
                        "it. Either raise the model's context length (n_ctx) to at "
                        "least concurrency x 8192, lower GRAPH_MCP_LLM_CONCURRENCY, "
                        "or lower GRAPH_MCP_LLM_MAX_TOKENS."
                    ) from exc
                if exc.code == 400 and self._thinking_kwarg:
                    # Endpoint rejects chat_template_kwargs; drop it and retry
                    # before blaming the response_format.
                    self._thinking_kwarg = False
                    continue
                if exc.code == 400 and self._format_index < len(RESPONSE_FORMATS) - 1:
                    self._format_index += 1
                    continue
                raise
            choice = body["choices"][0]
            # A truncated response is not parseable JSON, and the resulting
            # error is opaque. Name the cause instead.
            if choice.get("finish_reason") == "length":
                raise ValueError(
                    f"extraction hit max_tokens ({self.cfg.llm_max_tokens}); the "
                    "model is over-generating. Raise GRAPH_MCP_LLM_MAX_TOKENS or "
                    "use a model that does not emit reasoning traces."
                )
            return _parse_json(choice["message"]["content"])
        raise RuntimeError("No usable response_format for this endpoint")


def _parse_json(text: str) -> dict:
    """Tolerate a model that wraps JSON in a fence or adds a stray preamble."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _windows(doc: Document, max_words: int) -> list[str]:
    """Split a note into extraction-sized windows on paragraph boundaries."""
    words = doc.body.split()
    if len(words) <= max_words:
        return [doc.body] if doc.body.strip() else []
    return [
        " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
    ]


def extract_document(
    doc: Document, cfg: Settings, extract_words: int
) -> tuple[dict[str, dict], list[dict], str | None]:
    """Run the LLM over one document. Pure network + parsing, no Neo4j.

    Isolated so it can run in a worker thread: the Neo4j session is not
    thread-safe, so only this part is parallelised and every graph write stays
    on the main thread. Returns (entities, relations, error).
    """
    llm = LLMClient(cfg)  # per-thread: it caches endpoint negotiation state
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    try:
        for window in _windows(doc, extract_words):
            result = llm.complete_json(
                f"Note path: {doc.path}\nTitle: {doc.title}\n\n{window}"
            )
            for raw in result.get("entities", []) or []:
                name = str(raw.get("name", "")).strip()
                if not name:
                    continue
                etype = str(raw.get("type", "concept")).strip().lower()
                if etype not in ENTITY_TYPES:
                    etype = "concept"
                key = entity_key(name)
                if not key:
                    continue
                slot = entities.setdefault(
                    key, {"name": name, "type": etype, "aliases": set(), "count": 0}
                )
                slot["count"] += 1
                for alias in raw.get("aliases", []) or []:
                    if str(alias).strip():
                        slot["aliases"].add(str(alias).strip())
            for raw in result.get("relations", []) or []:
                source = entity_key(str(raw.get("source", "")))
                target = entity_key(str(raw.get("target", "")))
                if not source or not target or source == target:
                    continue
                rtype = str(raw.get("type", "related_to")).strip().lower()
                if rtype not in RELATION_TYPES:
                    rtype = "related_to"
                try:
                    confidence = float(raw.get("confidence", 0.5))
                except (TypeError, ValueError):
                    confidence = 0.5
                relations.append({
                    "source": source, "target": target, "type": rtype,
                    "confidence": max(0.0, min(1.0, confidence)), "path": doc.path,
                })
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            KeyError, json.JSONDecodeError) as exc:
        return {}, [], f"{doc.path}: {exc.__class__.__name__}: {exc}"
    return entities, relations, None


def sync_semantic(
    cfg: Settings | None = None,
    docs: list[Document] | None = None,
    limit: int | None = None,
    force: bool = False,
    extract_words: int = 1500,
    max_seconds: float | None = None,
    progress: bool = True,
) -> SemanticReport:
    cfg = cfg or settings
    started = time.time()
    docs = docs if docs is not None else iter_documents(cfg.kb_root)
    report = SemanticReport(documents_considered=len(docs))
    llm = LLMClient(cfg)
    embedder = EmbeddingClient(cfg)

    with driver_session(cfg) as session:
        if force:
            targets = {d.path for d in docs}
        else:
            targets = {r["path"] for r in session.run(STALE_DOCUMENTS)}

        pending = [d for d in docs if d.path in targets]
        report.skipped_unchanged = len(docs) - len(pending)
        # `limit` exists for the plan's validation gate: extract a handful of
        # notes you know well and inspect them before spending the full run.
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            report.seconds = time.time() - started
            return report

        session.run(CLEAR_DOC_SEMANTICS, paths=[d.path for d in pending])
        done: list[Document] = []

        workers = max(1, cfg.llm_concurrency)
        for offset in range(0, len(pending), workers):
            # Time-boxing is checked between batches, never mid-document, so a
            # slice always ends on cleanly-watermarked boundaries.
            if max_seconds is not None and time.time() - started >= max_seconds:
                report.stopped_early = True
                break
            batch = pending[offset : offset + workers]
            if progress:
                print(
                    f"  [{offset + 1}-{offset + len(batch)}/{len(pending)}] "
                    f"{batch[0].path[:52]}",
                    end="\r",
                    flush=True,
                )

            # Only the LLM calls are parallel. Everything below — embedding,
            # dedup lookups, writes, watermarking — stays on this thread,
            # because the Neo4j session is not thread-safe and dedup must see
            # entities written by earlier documents in this run.
            if workers == 1:
                results = [extract_document(batch[0], cfg, extract_words)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(
                        pool.map(
                            lambda d: extract_document(d, cfg, extract_words), batch
                        )
                    )

            for doc, (entities, relations, error) in zip(batch, results):
                if error:
                    report.failures.append(error)
                    continue

                if not entities:
                    session.run(
                        MARK_EXTRACTED, path=doc.path, content_hash=doc.content_hash
                    )
                    done.append(doc)
                    continue

                # Dedup before insert: exact key, then vector similarity.
                keys = list(entities)
                vectors = embedder.embed(
                    [f"{entities[k]['name']} ({entities[k]['type']})" for k in keys]
                )
                resolved: dict[str, str] = {}
                rows = []
                for key, vector in zip(keys, vectors):
                    slot = entities[key]
                    match = session.run(
                        SIMILAR_ENTITY,
                        embedding=vector,
                        threshold=cfg.entity_merge_threshold,
                        type=slot["type"],
                    ).single()
                    if match and match["key"] != key:
                        resolved[key] = match["key"]
                        report.entities_merged += 1
                        rows.append(
                            {
                                "key": match["key"],
                                "name": match["name"],
                                "type": slot["type"],
                                "aliases": sorted(slot["aliases"] | {slot["name"]}),
                                "embedding": vector,
                            }
                        )
                    else:
                        resolved[key] = key
                        report.entities_new += 1
                        rows.append(
                            {
                                "key": key,
                                "name": slot["name"],
                                "type": slot["type"],
                                "aliases": sorted(slot["aliases"]),
                                "embedding": vector,
                            }
                        )

                session.run(UPSERT_ENTITIES, rows=rows)
                mention_rows = [
                    {"path": doc.path, "key": resolved[k], "count": entities[k]["count"]}
                    for k in keys
                ]
                session.run(UPSERT_MENTIONS, rows=mention_rows)
                report.mentions += len(mention_rows)

                relation_rows = [
                    {**r, "source": resolved.get(r["source"]), "target": resolved.get(r["target"])}
                    for r in relations
                ]
                # Drop relations naming an entity the model never declared.
                relation_rows = [
                    r for r in relation_rows
                    if r["source"] and r["target"] and r["source"] != r["target"]
                ]
                if relation_rows:
                    session.run(UPSERT_RELATIONS, rows=relation_rows)
                    report.relations += len(relation_rows)

                session.run(MARK_EXTRACTED, path=doc.path, content_hash=doc.content_hash)
                done.append(doc)

        report.documents_extracted = len(done)

    if progress:
        print(" " * 80, end="\r")
    report.seconds = time.time() - started
    return report
