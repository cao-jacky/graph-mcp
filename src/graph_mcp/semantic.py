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
from dataclasses import dataclass, field

from .config import Settings, settings
from .db import driver_session
from .embeddings import EmbeddingClient
from .parse import Document, iter_documents

ENTITY_TYPES = ("person", "project", "tool", "concept", "organization")

SYSTEM_PROMPT = """\
You extract a knowledge graph from personal technical notes.

Return ONLY a JSON object, no prose and no markdown fence, of the shape:
{
  "entities": [
    {"name": "...", "type": "person|project|tool|concept|organization",
     "aliases": ["..."]}
  ],
  "relations": [
    {"source": "...", "target": "...", "type": "works_on|depends_on|
      supersedes|contradicts|part_of|uses|related_to", "confidence": 0.0-1.0}
  ]
}

Rules:
- Extract only entities that genuinely matter to this note, not every noun.
- Use the most canonical name you can (e.g. "Neo4j", not "the neo4j database").
- `source` and `target` in relations MUST exactly match a `name` in entities.
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
                    "type": {"type": "string"},
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

    def _request(self, prompt: str, response_format: dict | None) -> dict:
        payload: dict = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            f"{self.cfg.llm_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.llm_api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.cfg.embed_timeout) as response:
            return json.load(response)

    def complete_json(self, prompt: str) -> dict:
        # The working format is remembered, so the negotiation costs at most
        # one wasted request per process rather than one per document.
        while self._format_index < len(RESPONSE_FORMATS):
            try:
                body = self._request(prompt, RESPONSE_FORMATS[self._format_index])
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and self._format_index < len(RESPONSE_FORMATS) - 1:
                    self._format_index += 1
                    continue
                raise
            return _parse_json(body["choices"][0]["message"]["content"])
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

        for index, doc in enumerate(pending, 1):
            # Time-boxing is checked between documents, never mid-document, so
            # a slice always ends on a cleanly-watermarked boundary.
            if max_seconds is not None and time.time() - started >= max_seconds:
                report.stopped_early = True
                break
            if progress:
                print(
                    f"  [{index}/{len(pending)}] {doc.path[:60]}",
                    end="\r",
                    flush=True,
                )
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
                            key,
                            {"name": name, "type": etype, "aliases": set(), "count": 0},
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
                        rtype = re.sub(
                            r"[^a-z_]+", "_", str(raw.get("type", "related_to")).lower()
                        ).strip("_") or "related_to"
                        try:
                            confidence = float(raw.get("confidence", 0.5))
                        except (TypeError, ValueError):
                            confidence = 0.5
                        relations.append(
                            {
                                "source": source,
                                "target": target,
                                "type": rtype,
                                "confidence": max(0.0, min(1.0, confidence)),
                                "path": doc.path,
                            }
                        )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                    KeyError, json.JSONDecodeError) as exc:
                report.failures.append(f"{doc.path}: {exc.__class__.__name__}: {exc}")
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
