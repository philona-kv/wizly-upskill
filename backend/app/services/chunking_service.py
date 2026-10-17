"""Token-aware chunking with page provenance.

Algorithm:
1. Each page is split into paragraphs (blank lines) and paragraphs into sentences.
   Every resulting *unit* remembers the page it came from.
2. Units longer than the chunk size are split on word boundaries (and, only for a
   single pathological "word", on token boundaries).
3. Units are packed greedily into chunks of at most ``chunk_size`` tokens.
4. The next chunk starts with whole trailing units of the previous chunk totalling at
   most ``overlap`` tokens (falling back to a token tail when the last unit alone is
   larger than the overlap).
5. ``page_start``/``page_end`` are the min/max pages of the units in a chunk, and
   ``token_count`` is measured on the final chunk text.

Pure and synchronous; call via ``asyncio.to_thread`` for large documents.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from app.services.pdf_extractor import PageText

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    token_count: int


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page: int
    tokens: int
    starts_paragraph: bool


@lru_cache(maxsize=8)
def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class ChunkingService:
    def __init__(self, chunk_size: int, overlap: int, model: str = "text-embedding-3-small"):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = get_encoding(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    # -- unit construction -------------------------------------------------

    def _split_oversized(self, text: str) -> list[str]:
        """Split text that exceeds chunk_size on word boundaries."""
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for word in text.split(" "):
            word_tokens = self.count_tokens(" " + word)
            if word_tokens > self.chunk_size:
                if current:
                    pieces.append(" ".join(current))
                    current, current_tokens = [], 0
                tokens = self._enc.encode(word, disallowed_special=())
                for start in range(0, len(tokens), self.chunk_size):
                    pieces.append(self._enc.decode(tokens[start : start + self.chunk_size]))
                continue
            if current and current_tokens + word_tokens > self.chunk_size:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(word)
            current_tokens += word_tokens
        if current:
            pieces.append(" ".join(current))
        return [p for p in pieces if p.strip()]

    def _units(self, pages: Iterable[PageText]) -> list[_Unit]:
        units: list[_Unit] = []
        for page in pages:
            text = _HYPHEN_LINEBREAK.sub(r"\1\2", page.text)
            for paragraph in _PARAGRAPH_SPLIT.split(text):
                paragraph = " ".join(paragraph.split())
                if not paragraph:
                    continue
                first = True
                for sentence in _SENTENCE_SPLIT.split(paragraph):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    tokens = self.count_tokens(sentence)
                    parts = (
                        [(sentence, tokens)]
                        if tokens <= self.chunk_size
                        else [(p, self.count_tokens(p)) for p in self._split_oversized(sentence)]
                    )
                    for part, part_tokens in parts:
                        units.append(_Unit(part, page.page_number, part_tokens, first))
                        first = False
        return units

    # -- packing -----------------------------------------------------------

    @staticmethod
    def _join(units: Sequence[_Unit]) -> str:
        out: list[str] = []
        for i, unit in enumerate(units):
            if i:
                out.append("\n\n" if unit.starts_paragraph else " ")
            out.append(unit.text)
        return "".join(out)

    @staticmethod
    def _estimate(units: Sequence[_Unit]) -> int:
        # +1 per separator; the exact count is re-measured on the joined text.
        return sum(u.tokens for u in units) + max(len(units) - 1, 0)

    def _overlap_units(self, units: Sequence[_Unit]) -> list[_Unit]:
        if self.overlap == 0 or not units:
            return []
        tail: list[_Unit] = []
        total = 0
        for unit in reversed(units):
            cost = unit.tokens + (1 if tail else 0)
            if total + cost > self.overlap:
                break
            tail.insert(0, unit)
            total += cost
        if tail:
            return tail
        # Last unit alone exceeds the overlap: carry its final `overlap` tokens.
        last = units[-1]
        tokens = self._enc.encode(last.text, disallowed_special=())[-self.overlap :]
        text = self._enc.decode(tokens).strip()
        return [_Unit(text, last.page, len(tokens), True)] if text else []

    def chunk(self, pages: Iterable[PageText]) -> list[TextChunk]:
        units = self._units(pages)
        if not units:
            return []

        # Each group is (units, number of leading units carried over as overlap).
        groups: list[tuple[list[_Unit], int]] = []
        current: list[_Unit] = []
        carried = 0
        has_new_content = False
        for unit in units:
            if has_new_content and self._estimate([*current, unit]) > self.chunk_size:
                groups.append((current, carried))
                current = self._overlap_units(current)
                carried = len(current)
                has_new_content = False
            current.append(unit)
            has_new_content = True
            # Keep overlap from pushing a chunk past the limit.
            while carried and self._estimate(current) > self.chunk_size:
                current.pop(0)
                carried -= 1
        if has_new_content:
            groups.append((current, carried))

        chunks: list[TextChunk] = []
        for group, carried in groups:
            content = self._join(group)
            token_count = self.count_tokens(content)
            # Separator tokenisation can exceed the estimate; only ever drop overlap.
            while token_count > self.chunk_size and carried:
                group = group[1:]
                carried -= 1
                content = self._join(group)
                token_count = self.count_tokens(content)
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    content=content,
                    page_start=min(u.page for u in group),
                    page_end=max(u.page for u in group),
                    token_count=token_count,
                )
            )
        return chunks
