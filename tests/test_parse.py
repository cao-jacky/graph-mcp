"""Parser tests — no Neo4j, no network.

The fixtures encode the specific corpus quirks that broke a naive parser:
POSIX character classes inside shell snippets, notes with no frontmatter at
all, and 56 synced skill files that all open with `# SKILL.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_mcp.parse import (  # noqa: E402
    chunk_markdown,
    extract_tags,
    extract_wikilinks,
    load_document,
    parse_frontmatter,
    resolve_links,
    strip_code,
)


def check(name: str, actual, expected) -> bool:
    ok = actual == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        expected: {expected!r}")
        print(f"        actual:   {actual!r}")
    return ok


def test_frontmatter_inline_list():
    front, body = parse_frontmatter(
        "---\ncreated: 2026-07-19\ntags: [hermes, neo4j, mcp]\n---\n\n# Title\n"
    )
    return [
        check("inline tags parse", extract_tags(front), ["hermes", "neo4j", "mcp"]),
        check("frontmatter stripped from body", body.strip(), "# Title"),
        check("scalar retained", front["created"], "2026-07-19"),
    ]


def test_frontmatter_block_list():
    front, _ = parse_frontmatter("---\ntags:\n- hermes\n- docker\ncategory: x\n---\nbody")
    return [
        check("block tags parse", extract_tags(front), ["hermes", "docker"]),
        check("key after block list", front["category"], "x"),
    ]


def test_no_frontmatter():
    front, body = parse_frontmatter("# Just a heading\n\nSome text.\n")
    return [
        check("absent frontmatter is not an error", front, {}),
        check("body untouched", body.startswith("# Just a heading"), True),
    ]


def test_quoted_and_empty():
    front, _ = parse_frontmatter('---\nname: "web-research"\ntags: []\n---\nx')
    return [
        check("quotes stripped", front["name"], "web-research"),
        check("empty inline list", extract_tags(front), []),
    ]


def test_code_stripping():
    text = 'Prose `[[inline]]` here.\n\n```bash\ngrep "^[[:space:]]*$"\n```\n\n[[real-note]]\n'
    links = extract_wikilinks(text)
    return [
        check("fenced + inline code excluded", links, ["real-note"]),
        check("line count preserved", len(strip_code(text).splitlines()),
              len(text.splitlines())),
    ]


def test_wikilink_forms():
    text = (
        "[[plain]] [[target|Alias]] [[doc#Heading]] [[doc2#H|A]] "
        "![[embed.png]] [[#same-page]] [[:space:]] [[Cyberpunk 2077]]"
    )
    return [
        check(
            "alias/anchor/embed/POSIX handled",
            extract_wikilinks(text),
            ["plain", "target", "doc", "doc2", "Cyberpunk 2077"],
        )
    ]


def test_link_resolution(tmp: Path):
    (tmp / "projects").mkdir(parents=True, exist_ok=True)
    (tmp / "home-lab").mkdir(parents=True, exist_ok=True)
    (tmp / "projects" / "alpha.md").write_text(
        "---\ntags: [x]\n---\n# Alpha\n[[beta]] and [[home-lab/gamma]] and [[ghost]]\n"
    )
    (tmp / "projects" / "beta.md").write_text("# Beta\n")
    (tmp / "home-lab" / "gamma.md").write_text("# Gamma\n")

    docs = [
        load_document(tmp / "projects" / "alpha.md", tmp),
        load_document(tmp / "projects" / "beta.md", tmp),
        load_document(tmp / "home-lab" / "gamma.md", tmp),
    ]
    edges, unresolved = resolve_links(docs)
    return [
        check(
            "stem and path links resolve",
            sorted(edges["projects/alpha.md"]),
            ["home-lab/gamma.md", "projects/beta.md"],
        ),
        check("dangling link reported", unresolved, ["projects/alpha.md -> [[ghost]]"]),
        check("category from top dir", docs[0].category, "projects"),
        check("subcategory empty at depth 1", docs[0].subcategory, ""),
    ]


def test_title_precedence(tmp: Path):
    skill = tmp / "skill.md"
    skill.write_text(
        "---\nsynced_from: /opt/data/skills/research/web-research/SKILL.md\n---\n"
        "# SKILL.md\ncontent\n"
    )
    plain = tmp / "plain.md"
    plain.write_text("# Real Heading\ntext\n")
    return [
        check("filename heading rejected for synced_from",
              load_document(skill, tmp).title, "web research"),
        check("normal heading kept", load_document(plain, tmp).title, "Real Heading"),
    ]


def test_chunking():
    body = "# A\n" + ("alpha " * 500) + "\n## B\nshort tail\n"
    chunks = chunk_markdown(body, max_words=100, overlap_words=20, doc_title="Doc")
    sizes = [len(c.text.split()) for c in chunks]
    return [
        check("long section is windowed", all(s <= 100 for s in sizes), True),
        check("ordinals are sequential",
              [c.ordinal for c in chunks], list(range(len(chunks)))),
        check("title not repeated in breadcrumb",
              chunks[0].breadcrumb.count("Doc"), 1),
        check("breadcrumb prefixes embed text",
              chunks[-1].embed_text.startswith(chunks[-1].breadcrumb), True),
    ]


def test_small_sections_packed():
    body = "".join(f"## H{i}\nline {i}\n" for i in range(30))
    chunks = chunk_markdown(body, max_words=200, overlap_words=20, doc_title="D")
    return [
        check("tiny sections merge into few chunks", len(chunks) <= 2, True),
    ]


def main() -> int:
    import tempfile

    results: list[bool] = []
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        suites = [
            ("frontmatter: inline list", test_frontmatter_inline_list, ()),
            ("frontmatter: block list", test_frontmatter_block_list, ()),
            ("frontmatter: absent", test_no_frontmatter, ()),
            ("frontmatter: quotes/empty", test_quoted_and_empty, ()),
            ("wikilinks: code stripping", test_code_stripping, ()),
            ("wikilinks: forms", test_wikilink_forms, ()),
            ("wikilinks: resolution", test_link_resolution, (tmp,)),
            ("titles: precedence", test_title_precedence, (tmp,)),
            ("chunking: windowing", test_chunking, ()),
            ("chunking: packing", test_small_sections_packed, ()),
        ]
        for name, fn, args in suites:
            print(name)
            results.extend(fn(*args))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
