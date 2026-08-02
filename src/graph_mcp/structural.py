"""Stage 1 — structural extraction: documents, tags, wikilinks. No LLM calls.

Idempotent by design: everything MERGEs on a stable key (`path` for documents,
`name` for tags) so the job can be re-run over the whole corpus at any time.
Each run stamps nodes with a run id, then prunes anything not seen — that is
what makes renames and deletions self-correcting rather than leaving orphans.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config import Settings, settings
from .db import driver_session
from .parse import Document, iter_documents, resolve_links

UPSERT_DOCUMENTS = """
UNWIND $rows AS row
MERGE (d:Document {path: row.path})
SET d.title = row.title,
    d.category = row.category,
    d.subcategory = row.subcategory,
    d.content_hash = row.content_hash,
    d.created = row.created,
    d.updated = row.updated,
    d.word_count = row.word_count,
    d.stub = false,
    d.last_run = $run_id
"""

# Stub nodes keep a dangling wikilink queryable ("notes I reference but have
# not written") without pretending a file exists. They never claim a hash.
UPSERT_STUBS = """
UNWIND $rows AS row
MERGE (d:Document {path: row.path})
ON CREATE SET d.title = row.title, d.stub = true, d.category = 'unresolved'
SET d.last_run = $run_id
"""

# The TAGGED edge must carry last_run too, not just the Tag node: PRUNE_EDGES
# deletes any edge missing the current run id, which would otherwise strip
# every tag edge on each run and leave the Tag nodes orphaned.
UPSERT_TAGS = """
UNWIND $rows AS row
MERGE (t:Tag {name: row.tag})
SET t.last_run = $run_id
WITH t, row
MATCH (d:Document {path: row.path})
MERGE (d)-[r:TAGGED]->(t)
SET r.last_run = $run_id
"""

UPSERT_LINKS = """
UNWIND $rows AS row
MATCH (src:Document {path: row.src})
MATCH (dst:Document {path: row.dst})
MERGE (src)-[r:LINKS_TO]->(dst)
SET r.last_run = $run_id
"""

UPSERT_SUPERSEDES = """
UNWIND $rows AS row
MATCH (src:Document {path: row.src})
MATCH (dst:Document {path: row.dst})
MERGE (src)-[r:SUPERSEDES]->(dst)
SET r.last_run = $run_id
"""

# Order matters: drop this run's stale edges before pruning nodes, so an edge
# that merely moved is not mistaken for a deleted document.
PRUNE_EDGES = """
MATCH ()-[r:LINKS_TO|TAGGED|SUPERSEDES]->()
WHERE r.last_run IS NULL OR r.last_run <> $run_id
DELETE r
"""

PRUNE_DOCUMENTS = """
MATCH (d:Document)
WHERE d.last_run <> $run_id AND coalesce(d.stub, false) = false
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE d, c
RETURN count(DISTINCT d) AS removed
"""

# A stub only exists to be the target of a link; once nothing points at it,
# it is noise.
PRUNE_ORPHAN_STUBS = """
MATCH (d:Document {stub: true})
WHERE NOT ()-[:LINKS_TO]->(d)
DETACH DELETE d
RETURN count(d) AS removed
"""

PRUNE_ORPHAN_TAGS = """
MATCH (t:Tag) WHERE NOT ()-[:TAGGED]->(t)
DELETE t
RETURN count(t) AS removed
"""


@dataclass
class StructuralReport:
    documents: int = 0
    tags: int = 0
    tag_edges: int = 0
    link_edges: int = 0
    supersedes_edges: int = 0
    stubs: int = 0
    unresolved: list[str] = field(default_factory=list)
    removed_documents: int = 0
    removed_stubs: int = 0
    removed_tags: int = 0
    seconds: float = 0.0

    def render(self) -> str:
        lines = [
            "Stage 1 — structural extraction",
            f"  documents upserted : {self.documents}",
            f"  tags               : {self.tags} ({self.tag_edges} TAGGED edges)",
            f"  wikilinks          : {self.link_edges} LINKS_TO edges",
            f"  supersedes         : {self.supersedes_edges} SUPERSEDES edges",
            f"  stub targets       : {self.stubs}",
            f"  pruned             : {self.removed_documents} docs, "
            f"{self.removed_stubs} stubs, {self.removed_tags} tags",
            f"  elapsed            : {self.seconds:.1f}s",
        ]
        if self.unresolved:
            lines.append(f"  unresolved links   : {len(self.unresolved)}")
            lines.extend(f"      {item}" for item in self.unresolved[:15])
            if len(self.unresolved) > 15:
                lines.append(f"      ... and {len(self.unresolved) - 15} more")
        return "\n".join(lines)


def _stub_title(target: str) -> str:
    return target.rsplit("/", 1)[-1].removesuffix(".md").replace("-", " ")


def sync_structural(
    cfg: Settings | None = None, docs: list[Document] | None = None
) -> StructuralReport:
    cfg = cfg or settings
    started = time.time()
    docs = docs if docs is not None else iter_documents(cfg.kb_root)
    run_id = uuid.uuid4().hex
    report = StructuralReport(documents=len(docs))

    edges, unresolved = resolve_links(docs)
    report.unresolved = unresolved

    doc_rows = [
        {
            "path": d.path,
            "title": d.title,
            "category": d.category,
            "subcategory": d.subcategory,
            "content_hash": d.content_hash,
            "created": d.created,
            "updated": d.updated,
            "word_count": d.word_count,
        }
        for d in docs
    ]

    known = {d.path for d in docs}
    stub_paths = {
        f"unresolved:{item.split(' -> [[')[1].rstrip(']]')}" for item in unresolved
    }
    stub_rows = [
        {"path": p, "title": _stub_title(p.removeprefix("unresolved:"))}
        for p in sorted(stub_paths)
    ]
    report.stubs = len(stub_rows)

    tag_rows = [{"path": d.path, "tag": t} for d in docs for t in d.tags]
    report.tags = len({r["tag"] for r in tag_rows})
    report.tag_edges = len(tag_rows)

    link_rows = [{"src": src, "dst": dst} for src, targets in edges.items() for dst in targets]
    # Unresolved links become edges to the stub node so the dangling reference
    # is still traversable.
    for item in unresolved:
        src, raw = item.split(" -> [[", 1)
        link_rows.append({"src": src, "dst": f"unresolved:{raw.rstrip(']]')}"})
    report.link_edges = len(link_rows)

    supersede_rows = [
        {"src": d.path, "dst": target}
        for d in docs
        for target in d.supersedes
        if target in known
    ]
    report.supersedes_edges = len(supersede_rows)

    with driver_session(cfg) as session:
        for start in range(0, len(doc_rows), 500):
            session.run(
                UPSERT_DOCUMENTS, rows=doc_rows[start : start + 500], run_id=run_id
            )
        if stub_rows:
            session.run(UPSERT_STUBS, rows=stub_rows, run_id=run_id)
        if tag_rows:
            session.run(UPSERT_TAGS, rows=tag_rows, run_id=run_id)
        if link_rows:
            session.run(UPSERT_LINKS, rows=link_rows, run_id=run_id)
        if supersede_rows:
            session.run(UPSERT_SUPERSEDES, rows=supersede_rows, run_id=run_id)

        session.run(PRUNE_EDGES, run_id=run_id)
        record = session.run(PRUNE_DOCUMENTS, run_id=run_id).single()
        report.removed_documents = record["removed"] if record else 0
        record = session.run(PRUNE_ORPHAN_STUBS).single()
        report.removed_stubs = record["removed"] if record else 0
        record = session.run(PRUNE_ORPHAN_TAGS).single()
        report.removed_tags = record["removed"] if record else 0

    report.seconds = time.time() - started
    return report
