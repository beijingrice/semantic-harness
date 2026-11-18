"""Conservative, reversible grounding of LLM evidence to ORIGINAL source text.

Only whitespace at a Chinese-character <-> ASCII-letter/digit boundary may
be omitted/inserted. Never normalize case, punctuation, entity names,
internal English word spaces, or numbers. The result is always an actual
continuous substring copied from the source, not a model-edited quotation.
"""

from __future__ import annotations


def _cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2FA1F


def _ascii_word(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _strip_boundary_whitespace(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            left = i - 1
            right = i + 1
            while left >= 0 and text[left].isspace():
                left -= 1
            while right < len(text) and text[right].isspace():
                right += 1
            if left >= 0 and right < len(text):
                a, b = text[left], text[right]
                if (_cjk(a) and _ascii_word(b)) or (_ascii_word(a) and _cjk(b)):
                    continue
        chars.append(ch)
        offsets.append(i)
    return "".join(chars), offsets


def resolve_evidence(source: str, quote: str) -> str | None:
    """Return the source's literal contiguous quote, or None if ungrounded.

    A normalized hit NEVER becomes a synthesized output: indices are mapped
    back to the exact original source. The quote may not add words, delete
    numbers, alter relations or change Unicode punctuation.
    """
    if not quote or not quote.strip():
        return None
    if quote in source:
        return quote
    compact_source, source_offsets = _strip_boundary_whitespace(source)
    compact_quote, _ = _strip_boundary_whitespace(quote)
    if not compact_quote:
        return None
    pos = compact_source.find(compact_quote)
    if pos < 0:
        return None
    start = source_offsets[pos]
    end = source_offsets[pos + len(compact_quote) - 1] + 1
    return source[start:end]
