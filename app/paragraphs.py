"""Immutable source paragraph references, without linguistic sentence heuristics.

A paragraph boundary in plain text is one or more blank lines. A single
line-break stays inside its paragraph (PDF soft wrapping and Chinese writing
styles must not force an artificial sentence boundary). Keep Python offsets
against the ORIGINAL source: no NFC, case folding, character replacements, or
whitespace editing is performed.
"""
from __future__ import annotations

import re

from app.models import SourceParagraph

_BLANK = re.compile(r"\n[ \t\f\v]*\n(?:[ \t\f\v]*\n)*")


def build_paragraphs(source: str, prefix: str) -> list[SourceParagraph]:
    if not source.strip():
        return []
    segments: list[SourceParagraph] = []
    starts = [0] + [m.end() for m in _BLANK.finditer(source)]
    ends = [m.start() for m in _BLANK.finditer(source)] + [len(source)]
    for start, end in zip(starts, ends, strict=True):
        # Trim exterior whitespace by moving offsets, never by changing chars.
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        segments.append(SourceParagraph(id=f"{prefix}_p{len(segments) + 1}",
                                        start=start, end=end, text=source[start:end]))
    return segments


def verify_ids(ids: list[str], paragraphs: list[SourceParagraph], *, allow_empty: bool = False) -> None:
    known = {p.id for p in paragraphs}
    if not ids and not allow_empty:
        raise ValueError("缺少原文 Paragraph ID")
    if len(ids) != len(set(ids)):
        raise ValueError("重复 Paragraph ID")
    if any(pid not in known for pid in ids):
        raise ValueError(f"引用不存在的 Paragraph ID: {ids!r}")


def paragraph_evidence(ids: list[str], paragraphs: list[SourceParagraph]) -> str:
    verify_ids(ids, paragraphs, allow_empty=True)
    index = {p.id: p.text for p in paragraphs}
    return "\n\n".join(index[pid] for pid in ids)


def compact_paragraphs(paragraphs: list[SourceParagraph]) -> list[dict]:
    return [{"id": p.id, "text": p.text} for p in paragraphs]
