"""Environment-driven configuration for every graph-mcp entry point.

Both the sync CLI and the MCP server read the same settings, so a graph built
by `graph-sync` is always queried with the same KB root, database, and vector
dimension by `graph-mcp`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB_ROOT = Path.home() / "Knowledge"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    kb_root: Path
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    embed_base_url: str
    embed_model: str
    embed_dim: int
    embed_batch: int
    embed_timeout: int

    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_max_tokens: int
    llm_timeout: int
    llm_disable_thinking: bool
    llm_concurrency: int
    max_entities_per_doc: int

    chunk_words: int
    chunk_overlap_words: int
    entity_merge_threshold: float

    semantic_tools_enabled: bool

    transport: str
    http_host: str
    http_port: int
    http_path: str
    auth_token: str
    allowed_hosts: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        # KNOWLEDGE_MCP_ROOT is honoured so graph-mcp indexes exactly the corpus
        # knowledge-mcp serves, without configuring the same path twice.
        root = os.environ.get("GRAPH_MCP_KB_ROOT") or os.environ.get(
            "KNOWLEDGE_MCP_ROOT"
        )
        kb_root = (
            Path(root).expanduser().resolve() if root else DEFAULT_KB_ROOT.resolve()
        )

        return cls(
            kb_root=kb_root,
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", ""),
            neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            embed_base_url=os.environ.get(
                "GRAPH_MCP_EMBED_BASE_URL", "http://127.0.0.1:1234/v1"
            ),
            embed_model=os.environ.get(
                "GRAPH_MCP_EMBED_MODEL", "text-embedding-qwen3-embedding-8b"
            ),
            embed_dim=int(os.environ.get("GRAPH_MCP_EMBED_DIM", "4096")),
            embed_batch=int(os.environ.get("GRAPH_MCP_EMBED_BATCH", "16")),
            embed_timeout=int(os.environ.get("GRAPH_MCP_EMBED_TIMEOUT", "300")),
            llm_base_url=os.environ.get(
                "GRAPH_MCP_LLM_BASE_URL", "http://127.0.0.1:1234/v1"
            ),
            llm_model=os.environ.get("GRAPH_MCP_LLM_MODEL", "qwen3.5-122b-a10b"),
            llm_api_key=os.environ.get("GRAPH_MCP_LLM_API_KEY", "lm-studio"),
            # Sized for the largest real extraction, not the typical one. A
            # 606-word note already needs ~875 tokens, and a content-dense note
            # exceeds 1200 — at which point the response truncates, the
            # document is recorded as a failure, and it fails identically on
            # every later run. The cap exists to bound runaway generation, not
            # to be tight.
            llm_max_tokens=int(os.environ.get("GRAPH_MCP_LLM_MAX_TOKENS", "4000")),
            # Must exceed the worst-case generation time. A timeout shorter than
            # generation is worse than useless: the client abandons the request
            # but the server keeps generating it, so orphaned work steals
            # capacity from the requests that follow.
            llm_timeout=int(os.environ.get("GRAPH_MCP_LLM_TIMEOUT", "900")),
            llm_disable_thinking=_env_flag("GRAPH_MCP_LLM_DISABLE_THINKING", True),
            # Parallel extraction requests. Serial by default because raising
            # it is only safe once the server has the context to match.
            #
            # Local servers commonly split one context budget across slots
            # (llama.cpp with kv_unified), so N concurrent requests each get
            # n_ctx/N tokens. At n_ctx=8192, 4-way concurrency leaves ~2048
            # per request and any real note fails with "Context size has been
            # exceeded" — while the same note succeeds serially. Budget
            # roughly 8192 tokens per concurrent request.
            #
            # Measured: 1.79x throughput at 4, 1.4x at 2. Note that batched
            # decoding also changes floating-point accumulation order, so
            # extraction stops being reproducible even at temperature 0 — the
            # same note can yield a different entity set between runs.
            llm_concurrency=int(os.environ.get("GRAPH_MCP_LLM_CONCURRENCY", "1")),
            # The prompt caps entities per *window*, which does nothing for a
            # long note: a 25-window document merged to 167 entities, enough to
            # dominate the graph with one note's incidental mentions. This caps
            # per *document*, after merging, keeping the entities that recur
            # across windows — recurrence is the signal that something is
            # central to the note rather than mentioned once in passing.
            max_entities_per_doc=int(
                os.environ.get("GRAPH_MCP_MAX_ENTITIES_PER_DOC", "20")
            ),
            chunk_words=int(os.environ.get("GRAPH_MCP_CHUNK_WORDS", "350")),
            chunk_overlap_words=int(os.environ.get("GRAPH_MCP_CHUNK_OVERLAP", "60")),
            entity_merge_threshold=float(
                os.environ.get("GRAPH_MCP_ENTITY_MERGE_THRESHOLD", "0.92")
            ),
            # Stage gate: the relationship-aware tools stay hidden until the
            # semantic pass has actually populated RELATES_TO/MENTIONS edges.
            semantic_tools_enabled=_env_flag("GRAPH_MCP_SEMANTIC_TOOLS", False),
            # stdio stays the default: it is what MCP clients spawn directly.
            # streamable-http is for running as a shared service other hosts
            # or containers connect to.
            transport=os.environ.get("GRAPH_MCP_TRANSPORT", "stdio").strip(),
            http_host=os.environ.get("GRAPH_MCP_HTTP_HOST", "127.0.0.1"),
            http_port=int(os.environ.get("GRAPH_MCP_HTTP_PORT", "8000")),
            http_path=os.environ.get("GRAPH_MCP_HTTP_PATH", "/mcp"),
            auth_token=os.environ.get("GRAPH_MCP_AUTH_TOKEN", ""),
            allowed_hosts=tuple(
                h.strip()
                for h in os.environ.get("GRAPH_MCP_ALLOWED_HOSTS", "").split(",")
                if h.strip()
            ),
        )


settings = Settings.from_env()


def require_kb_root(cfg: Settings | None = None) -> Path:
    """Fail loudly when the corpus root is missing.

    Without this an unset GRAPH_MCP_KB_ROOT silently indexes nothing, which
    looks like a broken extractor rather than a missing setting.
    """
    cfg = cfg or settings
    if not cfg.kb_root.is_dir():
        raise SystemExit(
            f"Knowledge base root does not exist: {cfg.kb_root}\n"
            "Set GRAPH_MCP_KB_ROOT (or KNOWLEDGE_MCP_ROOT) to your Knowledge/ folder."
        )
    return cfg.kb_root
