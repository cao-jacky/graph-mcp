"""Pure markdown parsing for the corpus — no Neo4j, no network, fully testable.

Everything that reads the vault lives here so the extraction stages stay thin
and the parsing rules can be unit-tested against the real notes offline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
WIKILINK_RE = re.compile(r"(!?)\[\[([^\]\n]+)\]\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code, preserving line structure.

    Wikilink and heading extraction must not see code. The corpus contains
    `[[:space:]]` inside a shell snippet and `` `[[wikilinks]]` `` in prose
    about wikilinks; both would otherwise become bogus graph edges.
    Replacing with same-length blanks keeps line numbers and offsets intact.
    """
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is None:
            m = re.match(r"(`{3,}|~{3,})", stripped)
            if m:
                fence = m.group(1)[0] * 3
                out.append("\n" if line.endswith("\n") else "")
                continue
        else:
            # A closing fence must use the same character as the opener.
            if stripped.startswith(fence):
                fence = None
            out.append("\n" if line.endswith("\n") else "")
            continue
        out.append(INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line))
    return "".join(out)


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split YAML frontmatter from the body.

    Returns ({}, text) when there is no frontmatter — 163 of the 278 notes in
    the corpus (the imported ai-systems tree) have none, so this is the common
    case, not an error. Only the flat subset of YAML the corpus actually uses
    is understood: scalars, inline lists, and block lists.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    data: dict[str, object] = {}
    key: str | None = None
    for raw in m.group(1).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        item = re.match(r"^\s*-\s+(.*)$", raw)
        if item and key:
            data.setdefault(key, [])
            if isinstance(data[key], list):
                data[key].append(_scalar(item.group(1)))  # type: ignore[union-attr]
            continue

        kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", raw)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        if not value:
            # Either a block list/mapping follows, or the key is empty.
            data[key] = []
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [_scalar(p) for p in inner.split(",") if p.strip()] if inner else []
            key = None
        else:
            data[key] = _scalar(value)
            key = None

    return data, text[m.end():]


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def looks_like_note_target(target: str) -> bool:
    """Whether a wikilink target could plausibly name a note.

    Code stripping removes almost all false positives, but not all: the synced
    `*-SKILL.md` notes wrap their preview in a bare ``` fence whose contents
    include further ``` fences, so by CommonMark the inner shell snippets are
    *not* code, and `[[:space:]]` from a `grep` pattern survives. Requiring a
    letter and rejecting `:` (POSIX character classes) filters those without
    discarding real names like `Cyberpunk 2077` or `COVER_LETTER_PROMPT`.
    """
    if ":" in target or "*" in target:
        return False
    return any(ch.isalpha() for ch in target)


def extract_wikilinks(body: str) -> list[str]:
    """Wikilink targets in document order, deduplicated, code excluded.

    `![[x.png]]` embeds and bare `[[#anchor]]` same-page links are skipped —
    neither refers to another note. `[[target#Heading|Alias]]` reduces to
    `target`.
    """
    seen: dict[str, None] = {}
    for embed, inner in WIKILINK_RE.findall(strip_code(body)):
        if embed:
            continue
        target = inner.split("|", 1)[0].split("#", 1)[0].strip()
        if not target or not looks_like_note_target(target):
            continue
        seen.setdefault(target, None)
    return list(seen)


def _string_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def extract_tags(front: dict[str, object]) -> list[str]:
    raw = front.get("tags")
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = re.split(r"[,\s]+", raw)
    else:
        return []
    tags: list[str] = []
    for tag in values:
        tag = str(tag).strip().strip("#").lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


FILENAME_TITLE_RE = re.compile(r"^[\w.-]+\.(md|txt|yaml|yml|json)$", re.IGNORECASE)


def _title_from(body: str, front: dict[str, object], path: Path) -> str:
    """Best available human title, in descending order of trustworthiness.

    The 56 synced skill notes all open with a literal `# SKILL.md` heading,
    which would title every one of them identically and poison both display
    and embeddings. A heading that is just a filename is therefore rejected in
    favour of the `synced_from:` source directory, which carries the real name
    (`/opt/data/skills/research/web-research/SKILL.md` -> `web-research`).
    """
    front_title = front.get("title") or front.get("name")
    if isinstance(front_title, str) and front_title.strip():
        return front_title.strip()

    m = re.search(r"^#\s+(.+?)\s*$", strip_code(body), re.MULTILINE)
    if m and not FILENAME_TITLE_RE.match(m.group(1).strip()):
        return m.group(1).strip()

    synced = front.get("synced_from")
    if isinstance(synced, str) and synced.strip():
        parent = PurePosixPath(synced.strip()).parent.name
        if parent:
            return parent.replace("-", " ").replace("_", " ").strip()

    return path.stem.replace("-", " ").replace("_", " ").strip()


@dataclass
class Chunk:
    ordinal: int
    text: str
    breadcrumb: str

    @property
    def embed_text(self) -> str:
        """Heading breadcrumb prepended so each chunk carries its context."""
        return f"{self.breadcrumb}\n\n{self.text}" if self.breadcrumb else self.text


@dataclass
class Document:
    path: str  # POSIX, relative to the KB root — the node's identity
    title: str
    category: str
    subcategory: str
    content_hash: str
    created: str
    updated: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    body: str = ""
    word_count: int = 0

    def chunks(self, max_words: int, overlap_words: int) -> list[Chunk]:
        return chunk_markdown(self.body, max_words, overlap_words, self.title)


def chunk_markdown(
    body: str,
    max_words: int = 350,
    overlap_words: int = 60,
    doc_title: str = "",
) -> list[Chunk]:
    """Split a note into heading-aware, word-bounded chunks with overlap.

    Splitting on headings first keeps chunks topically coherent; the running
    heading stack becomes a breadcrumb so an embedded chunk still knows which
    document and section it came from.
    """
    lines = body.splitlines()
    coded = strip_code(body).splitlines()

    sections: list[tuple[str, list[str]]] = []
    stack: list[str] = []
    current: list[str] = []
    breadcrumb = doc_title

    for i, line in enumerate(lines):
        # Consult the code-stripped copy so `# comments` in shell blocks
        # are not mistaken for markdown headings.
        m = HEADING_RE.match(coded[i]) if i < len(coded) else None
        if m:
            if current:
                sections.append((breadcrumb, current))
                current = []
            level = len(m.group(1))
            del stack[level - 1:]
            stack.append(m.group(2).strip())
            # A lone H1 usually restates the note title; don't say it twice.
            trail = [h for h in stack if h != doc_title]
            breadcrumb = " > ".join(filter(None, [doc_title, *trail]))
        else:
            current.append(line)
    if current:
        sections.append((breadcrumb, current))

    chunks: list[Chunk] = []

    def emit(crumb: str, words: list[str]) -> None:
        if words:
            chunks.append(Chunk(len(chunks), " ".join(words), crumb))

    # Pack consecutive short sections together instead of emitting one chunk
    # per heading — the corpus is full of two-line sections, and 90-word
    # chunks both embed poorly and multiply the index for no recall gain.
    buf: list[str] = []
    buf_crumb = doc_title
    for crumb, section_lines in sections:
        words = "\n".join(section_lines).split()
        if not words:
            continue
        if len(words) > max_words:
            emit(buf_crumb, buf)
            buf, buf_crumb = [], crumb
            step = max(1, max_words - overlap_words)
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                if not window:
                    break
                emit(crumb, window)
                if start + max_words >= len(words):
                    break
            continue
        if buf and len(buf) + len(words) > max_words:
            emit(buf_crumb, buf)
            buf, buf_crumb = [], crumb
        if not buf:
            buf_crumb = crumb
        buf.extend(words)
    emit(buf_crumb, buf)

    if not chunks and body.strip():
        chunks.append(Chunk(0, body.strip(), doc_title))
    return chunks


def load_document(path: Path, kb_root: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    front, body = parse_frontmatter(text)
    rel = path.relative_to(kb_root).as_posix()
    parts = rel.split("/")

    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    created = front.get("created")
    updated = front.get("updated") or front.get("sync_time")

    return Document(
        path=rel,
        title=_title_from(body, front, path),
        category=parts[0] if len(parts) > 1 else "(root)",
        subcategory=parts[1] if len(parts) > 2 else "",
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        created=str(created) if isinstance(created, str) and created else mtime,
        updated=str(updated) if isinstance(updated, str) and updated else mtime,
        tags=extract_tags(front),
        links=extract_wikilinks(body),
        # No note declares `supersedes:` yet; wiring it now means revision
        # chains become queryable the moment one does.
        supersedes=_string_list(front.get("supersedes")),
        body=body,
        word_count=len(body.split()),
    )


def iter_documents(kb_root: Path) -> list[Document]:
    """Every markdown note under the KB root, recursively.

    Walks the full tree rather than root+1 level, so the nested `ai-systems/`
    subtrees are indexed too. Dot-directories (`.claude/`, `.git/`) and the
    generated `index.md` are skipped.
    """
    docs: list[Document] = []
    for path in sorted(kb_root.rglob("*.md")):
        rel_parts = path.relative_to(kb_root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if path.name == "index.md" and path.parent == kb_root:
            continue
        docs.append(load_document(path, kb_root))
    return docs


def resolve_links(docs: list[Document]) -> tuple[dict[str, list[str]], list[str]]:
    """Map each document's wikilink targets onto real document paths.

    Obsidian links by note name, not path, so targets resolve against file
    stems first, then full/partial paths. Returns (edges, unresolved) where
    unresolved targets are reported rather than silently dropped.
    """
    by_stem: dict[str, list[str]] = {}
    by_path: dict[str, str] = {}
    for doc in docs:
        stem = doc.path.rsplit("/", 1)[-1].removesuffix(".md")
        by_stem.setdefault(stem.lower(), []).append(doc.path)
        by_path[doc.path.lower()] = doc.path
        by_path[doc.path.lower().removesuffix(".md")] = doc.path

    edges: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for doc in docs:
        targets: list[str] = []
        for raw in doc.links:
            key = raw.lower().strip("/")
            hit = by_path.get(key) or by_path.get(key + ".md")
            if not hit:
                candidates = by_stem.get(key.rsplit("/", 1)[-1].removesuffix(".md"), [])
                # An ambiguous stem (same note name in two folders) is left
                # unresolved rather than guessed at.
                hit = candidates[0] if len(candidates) == 1 else None
            if hit and hit != doc.path:
                if hit not in targets:
                    targets.append(hit)
            elif not hit:
                unresolved.append(f"{doc.path} -> [[{raw}]]")
        if targets:
            edges[doc.path] = targets
    return edges, unresolved
