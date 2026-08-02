"""Embedding client.

Talks any OpenAI-compatible `/v1/embeddings` endpoint — LM Studio, Ollama,
vLLM, text-embeddings-inference, or a hosted provider.

Developed against Qwen3-Embedding-8B, whose vectors come back L2-normalised,
so cosine similarity and dot product agree and Neo4j's `cosine` vector index
is the right choice.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence

from .config import Settings, settings

# Qwen3-Embedding is instruction-tuned and asymmetric: queries carry an
# instruction prefix, documents are embedded bare. Skipping this measurably
# degrades retrieval, so the task string lives here rather than at call sites.
QUERY_INSTRUCTION = (
    "Given a search query, retrieve relevant passages from a personal "
    "knowledge base of technical notes"
)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.cfg.embed_base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.cfg.embed_timeout
                ) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise EmbeddingError(
            f"Embedding endpoint {self.cfg.embed_base_url} unreachable after 3 "
            f"attempts: {last}. Is the model loaded in LM Studio?"
        ) from last

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        if not texts:
            return []
        data = self._post({"model": self.cfg.embed_model, "input": list(texts)})
        if "data" not in data:
            raise EmbeddingError(f"Unexpected embeddings response: {str(data)[:300]}")
        rows = sorted(data["data"], key=lambda r: r.get("index", 0))
        vectors = [r["embedding"] for r in rows]

        for vector in vectors:
            if len(vector) != self.cfg.embed_dim:
                raise EmbeddingError(
                    f"Model returned {len(vector)}-dim vectors but the graph is "
                    f"built for {self.cfg.embed_dim}. Set GRAPH_MCP_EMBED_DIM to "
                    f"{len(vector)} and rebuild the vector index, or switch model."
                )
        return vectors

    def embed_batched(self, texts: Sequence[str]) -> Iterator[list[list[float]]]:
        """Yield vectors batch by batch so callers can stream writes to Neo4j."""
        size = max(1, self.cfg.embed_batch)
        for start in range(0, len(texts), size):
            yield self.embed(texts[start : start + size])

    def embed_query(self, query: str, instruction: str = QUERY_INSTRUCTION) -> list[float]:
        """Embed a search query with the retrieval instruction prefix."""
        return self.embed([f"Instruct: {instruction}\nQuery: {query}"])[0]

    def health(self) -> str:
        vector = self.embed(["health check"])[0]
        return f"{self.cfg.embed_model} ok ({len(vector)} dims)"


def flatten(batches: Iterable[list[list[float]]]) -> list[list[float]]:
    return [vector for batch in batches for vector in batch]
