"""Load the knowledge base and split it into retrievable chunks.

Each KB article is a Markdown file with a small front-matter block (id,
title, category). We chunk on H2 sections so a chunk like "KB-113 >
Retention offers" stays together with its rules, and we prefix every
chunk with its article title so a short section still carries the
context it needs to be retrieved ("Retention offers" alone is ambiguous;
"Cancellation and retention offers > Retention offers" is not).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Article:
    kb_id: str
    title: str
    category: str
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    kb_id: str
    title: str
    section: str
    text: str


def parse_article(path: Path) -> Article:
    raw = path.read_text(encoding="utf-8")
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path.name} is missing front matter")
    meta = dict(
        line.split(":", 1) for line in m.group(1).splitlines() if ":" in line
    )
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    body = raw[m.end():].strip()
    return Article(
        kb_id=meta["id"], title=meta["title"], category=meta["category"], body=body
    )


def load_articles(kb_dir: Path) -> list[Article]:
    return [parse_article(p) for p in sorted(kb_dir.glob("KB-*.md"))]


def chunk_article(article: Article) -> list[Chunk]:
    # Drop the H1 (it duplicates the title) and split on H2 headings.
    body = re.sub(r"^# .*\n", "", article.body, count=1).strip()
    parts = re.split(r"^## (.+)$", body, flags=re.MULTILINE)

    sections: list[tuple[str, str]] = []
    intro = parts[0].strip()
    if intro:
        sections.append(("Overview", intro))
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))

    chunks = []
    for idx, (section, text) in enumerate(sections):
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{article.kb_id}#{idx}",
            kb_id=article.kb_id,
            title=article.title,
            section=section,
            text=f"{article.title} ({article.category}) > {section}\n{text}",
        ))
    return chunks


def load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in load_articles(kb_dir):
        chunks.extend(chunk_article(article))
    return chunks
