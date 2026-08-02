"""`graph-sync` — build and inspect the graph from the command line.

Each stage is a separate subcommand on purpose: the build plan gates every
stage behind its own validation step, and running them independently is what
makes those gates enforceable.
"""

from __future__ import annotations

import argparse
import sys

from .config import require_kb_root, settings
from .db import apply_schema, graph_stats
from .parse import iter_documents, resolve_links


def cmd_schema(args: argparse.Namespace) -> int:
    for statement in apply_schema(settings):
        print(f"  {statement}")
    print("Schema applied.")
    return 0


def cmd_structural(args: argparse.Namespace) -> int:
    from .structural import sync_structural

    print(sync_structural(settings).render())
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .embed import sync_embeddings

    print(sync_embeddings(settings, force=args.force).render())
    return 0


def cmd_semantic(args: argparse.Namespace) -> int:
    from .semantic import sync_semantic

    report = sync_semantic(
        settings,
        limit=args.limit,
        force=args.force,
        extract_words=args.extract_words,
        max_seconds=args.max_seconds,
    )
    print(report.render())
    return 1 if report.failures and not report.documents_extracted else 0


def cmd_all(args: argparse.Namespace) -> int:
    from .embed import sync_embeddings
    from .structural import sync_structural

    apply_schema(settings)
    print(sync_structural(settings).render())
    print()
    print(sync_embeddings(settings, force=args.force).render())
    if args.semantic:
        from .semantic import sync_semantic

        print()
        print(sync_semantic(settings, force=args.force).render())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(f"KB root   : {settings.kb_root}")
    print(f"Neo4j     : {settings.neo4j_uri} (db {settings.neo4j_database})")
    print(f"Embeddings: {settings.embed_model} @ {settings.embed_base_url}")
    print(f"LLM       : {settings.llm_model} @ {settings.llm_base_url}")
    print(
        "Stage-4 tools: "
        + ("enabled" if settings.semantic_tools_enabled else "disabled "
           "(set GRAPH_MCP_SEMANTIC_TOOLS=1 after Stage 3)")
    )
    print()
    try:
        for key, value in graph_stats(settings).items():
            print(f"  {key:17}: {value}")
    except Exception as exc:  # noqa: BLE001 - status must never hard-fail
        print(f"  graph unreachable: {exc}")
    return 0


def cmd_parse_check(args: argparse.Namespace) -> int:
    """Parse the corpus without touching Neo4j — the offline sanity check."""
    docs = iter_documents(settings.kb_root)
    edges, unresolved = resolve_links(docs)
    chunks = sum(
        len(d.chunks(settings.chunk_words, settings.chunk_overlap_words)) for d in docs
    )
    print(f"KB root            : {settings.kb_root}")
    print(f"documents          : {len(docs)}")
    print(f"  with frontmatter : {sum(1 for d in docs if d.tags)}")
    print(f"unique tags        : {len({t for d in docs for t in d.tags})}")
    print(f"resolved wikilinks : {sum(len(v) for v in edges.values())}")
    print(f"unresolved links   : {len(unresolved)}")
    for item in unresolved:
        print(f"    {item}")
    print(f"chunks             : {chunks}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-sync", description="Build the knowledge graph from the KB corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "schema", help="create constraints, indexes and vector indexes"
    ).set_defaults(func=cmd_schema)

    subparsers.add_parser(
        "structural", help="Stage 1: documents, tags, wikilinks (no LLM)"
    ).set_defaults(func=cmd_structural)

    embed = subparsers.add_parser("embed", help="chunk and embed documents")
    embed.add_argument("--force", action="store_true", help="re-embed everything")
    embed.set_defaults(func=cmd_embed)

    semantic = subparsers.add_parser(
        "semantic", help="Stage 3: LLM entity/relation extraction"
    )
    semantic.add_argument(
        "--limit", type=int, default=None,
        help="only extract N documents — use this to validate before a full run",
    )
    semantic.add_argument("--force", action="store_true", help="re-extract everything")
    semantic.add_argument(
        "--extract-words", type=int, default=1500,
        help="words per LLM call for long notes (default 1500)",
    )
    semantic.add_argument(
        "--max-seconds", type=float, default=None,
        help="stop after roughly this long, on a document boundary. Progress is "
             "watermarked per document, so re-running resumes where it stopped — "
             "use this to run the backfill in slices.",
    )
    semantic.set_defaults(func=cmd_semantic)

    run_all = subparsers.add_parser("all", help="schema + structural + embed")
    run_all.add_argument("--force", action="store_true")
    run_all.add_argument(
        "--semantic", action="store_true", help="also run the Stage 3 LLM pass"
    )
    run_all.set_defaults(func=cmd_all)

    subparsers.add_parser("status", help="show configuration and graph counts").set_defaults(
        func=cmd_status
    )
    subparsers.add_parser(
        "parse-check", help="parse the corpus offline, without Neo4j"
    ).set_defaults(func=cmd_parse_check)

    return parser


# `status` is the one command that must work before anything is configured.
NEEDS_CORPUS = {"structural", "embed", "semantic", "all", "parse-check"}


def main() -> None:
    args = build_parser().parse_args()
    if args.command in NEEDS_CORPUS:
        require_kb_root(settings)
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
