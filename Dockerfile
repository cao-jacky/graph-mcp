# graph-mcp served over HTTP, for MCP clients that cannot spawn a local
# stdio process — Hermes Agent in another container, for instance.
#
# stdio remains the default for desktop clients; this image sets
# GRAPH_MCP_TRANSPORT=streamable-http because a container has no client
# attached to its stdin. Running the stdio entrypoint in a container just
# blocks forever.

FROM python:3.13-slim

# uv gives the same locked dependency resolution as local development.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so edits to source do not invalidate the layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src/ ./src/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH" \
    GRAPH_MCP_TRANSPORT=streamable-http \
    GRAPH_MCP_HTTP_HOST=0.0.0.0 \
    GRAPH_MCP_HTTP_PORT=8000

# No corpus mount: every tool queries Neo4j and none read the notes. Only
# graph-sync needs the markdown, and it runs alongside the embedding model.
RUN useradd --uid 10001 --create-home graphmcp
USER 10001

EXPOSE 8000
CMD ["graph-mcp"]
