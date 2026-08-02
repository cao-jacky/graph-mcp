"""Vector stage — chunk every document and embed it.

A locally-served embedding model makes semantic retrieval and reliable entity
dedup essentially free, so this stage sits between Stage 1 (structure) and
Stage 3 (semantics); Stage 3 depends on it for entity matching.

Work is skipped per document by content hash, so re-runs after editing a
handful of notes cost seconds rather than minutes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings, settings
from .db import driver_session
from .embeddings import EmbeddingClient
from .parse import Document, iter_documents

# Documents whose body changed since their chunks were written, plus documents
# that never had chunks at all.
STALE_DOCUMENTS = """
MATCH (d:Document)
WHERE coalesce(d.stub, false) = false
  AND (d.chunked_hash IS NULL OR d.chunked_hash <> d.content_hash)
RETURN d.path AS path
"""

DELETE_CHUNKS = """
UNWIND $paths AS path
MATCH (:Document {path: path})-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c
"""

WRITE_CHUNKS = """
UNWIND $rows AS row
MATCH (d:Document {path: row.doc_path})
CREATE (c:Chunk {
    id: row.id,
    doc_path: row.doc_path,
    ordinal: row.ordinal,
    text: row.text,
    breadcrumb: row.breadcrumb,
    word_count: row.word_count
})
SET c.embedding = row.embedding
MERGE (d)-[:HAS_CHUNK]->(c)
"""

# Applied as soon as a document's last chunk lands, not at the end of the run:
# a full pass takes minutes over a network link, and an interrupted run must
# keep the documents it already finished rather than discarding all of them.
MARK_CHUNKED = """
UNWIND $rows AS row
MATCH (d:Document {path: row.path})
SET d.chunked_hash = row.content_hash
"""


@dataclass
class EmbedReport:
    documents_considered: int = 0
    documents_embedded: int = 0
    chunks_written: int = 0
    seconds: float = 0.0
    skipped_unchanged: int = 0

    def render(self) -> str:
        return "\n".join(
            [
                "Vector stage — chunk + embed",
                f"  documents in corpus : {self.documents_considered}",
                f"  re-embedded         : {self.documents_embedded} "
                f"(skipped {self.skipped_unchanged} unchanged)",
                f"  chunks written      : {self.chunks_written}",
                f"  elapsed             : {self.seconds:.1f}s",
            ]
        )


def sync_embeddings(
    cfg: Settings | None = None,
    docs: list[Document] | None = None,
    force: bool = False,
    progress: bool = True,
) -> EmbedReport:
    cfg = cfg or settings
    started = time.time()
    docs = docs if docs is not None else iter_documents(cfg.kb_root)
    report = EmbedReport(documents_considered=len(docs))
    client = EmbeddingClient(cfg)

    with driver_session(cfg) as session:
        if force:
            targets = {d.path for d in docs}
        else:
            targets = {r["path"] for r in session.run(STALE_DOCUMENTS)}

        pending = [d for d in docs if d.path in targets]
        report.skipped_unchanged = len(docs) - len(pending)
        if not pending:
            report.seconds = time.time() - started
            return report

        session.run(DELETE_CHUNKS, paths=[d.path for d in pending])

        # Chunks are embedded across document boundaries so every request
        # fills a whole batch, rather than sending a 1-chunk batch per note.
        queue: list[tuple[str, int, str, str]] = []
        outstanding: dict[str, int] = {}
        hashes = {d.path: d.content_hash for d in pending}
        for doc in pending:
            chunks = doc.chunks(cfg.chunk_words, cfg.chunk_overlap_words)
            outstanding[doc.path] = len(chunks)
            for chunk in chunks:
                queue.append((doc.path, chunk.ordinal, chunk.text, chunk.breadcrumb))

        batch_size = max(1, cfg.embed_batch)
        for start in range(0, len(queue), batch_size):
            batch = queue[start : start + batch_size]
            vectors = client.embed(
                [f"{crumb}\n\n{text}" if crumb else text for _, _, text, crumb in batch]
            )
            rows = [
                {
                    "id": f"{path}#{ordinal}",
                    "doc_path": path,
                    "ordinal": ordinal,
                    "text": text,
                    "breadcrumb": crumb,
                    "word_count": len(text.split()),
                    "embedding": vector,
                }
                for (path, ordinal, text, crumb), vector in zip(batch, vectors)
            ]
            # execute_write retries transient failures (leader switch, dropped
            # connection, restarted server) instead of losing the batch.
            session.execute_write(lambda tx, r=rows: tx.run(WRITE_CHUNKS, rows=r))
            report.chunks_written += len(rows)

            # Mark every document whose final chunk was in this batch.
            finished = []
            for path, _, _, _ in batch:
                outstanding[path] -= 1
                if outstanding[path] == 0:
                    finished.append({"path": path, "content_hash": hashes[path]})
            if finished:
                session.execute_write(
                    lambda tx, r=finished: tx.run(MARK_CHUNKED, rows=r)
                )
                report.documents_embedded += len(finished)

            if progress:
                done = min(start + batch_size, len(queue))
                print(
                    f"  embedded {done}/{len(queue)} chunks "
                    f"({report.documents_embedded}/{len(pending)} docs)",
                    end="\r",
                    flush=True,
                )

    if progress:
        print(" " * 60, end="\r")
    report.seconds = time.time() - started
    return report
