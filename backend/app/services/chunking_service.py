"""Semantic chunking with page provenance and token bounds.

Algorithm:
1. Each page is split into paragraphs and sentences (structure is used only to form
   *units*, not to choose chunk boundaries).
2. Sentence embeddings are computed for every unit; consecutive cosine similarities
   define semantic cohesion.
3. Breakpoints are placed where similarity is a local minimum and at or below a
   configurable percentile across the document (topic shifts).
4. Adjacent semantic segments are merged up to ``chunk_size`` tokens; oversized
   segments are split on word/token boundaries.
5. Chunks are emitted with trailing-unit overlap (at most ``overlap`` tokens).
6. ``page_start`` / ``page_end`` and ``token_count`` match the final chunk text.

Ingestion calls ``chunk`` asynchronously (requires an ``Embedder``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from app.services.embedding_service import Embedder
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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class ChunkingService:
    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        embedder: Embedder,
        *,
        breakpoint_percentile: float = 0.25,
        encoding_model: str = "text-embedding-3-small",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        if not 0.0 <= breakpoint_percentile <= 1.0:
            raise ValueError("breakpoint_percentile must be between 0 and 1")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._embedder = embedder
        self._breakpoint_percentile = breakpoint_percentile
        self._enc = get_encoding(encoding_model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def _split_oversized(self, text: str) -> list[str]:
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
        last = units[-1]
        tokens = self._enc.encode(last.text, disallowed_special=())[-self.overlap :]
        text = self._enc.decode(tokens).strip()
        return [_Unit(text, last.page, len(tokens), True)] if text else []

    def _similarity_threshold(self, similarities: list[float]) -> float:
        if not similarities:
            return 0.0
        ordered = sorted(similarities)
        idx = int(self._breakpoint_percentile * (len(ordered) - 1))
        return ordered[idx]

    def _semantic_break_after(self, similarities: list[float]) -> set[int]:
        """Indices i: prefer a break between unit i and unit i+1."""
        if not similarities:
            return set()
        threshold = min(self._similarity_threshold(similarities), 0.35)
        breaks: set[int] = set()
        for i, sim in enumerate(similarities):
            left = similarities[i - 1] if i > 0 else sim + 1.0
            right = similarities[i + 1] if i < len(similarities) - 1 else sim + 1.0
            is_local_min = sim < left and sim < right
            if is_local_min and sim <= threshold:
                breaks.add(i)
        return breaks

    def _segments_from_units(
        self, units: list[_Unit], break_after: set[int]
    ) -> list[list[_Unit]]:
        segments: list[list[_Unit]] = []
        current: list[_Unit] = []
        for idx, unit in enumerate(units):
            current.append(unit)
            if idx in break_after:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments or [units]

    def _split_oversized_segments(self, segments: list[list[_Unit]]) -> list[list[_Unit]]:
        out: list[list[_Unit]] = []
        for segment in segments:
            if self._estimate(segment) <= self.chunk_size:
                out.append(segment)
                continue
            batch: list[_Unit] = []
            for unit in segment:
                if batch and self._estimate(batch + [unit]) > self.chunk_size:
                    out.append(batch)
                    batch = self._overlap_units(batch)
                batch.append(unit)
                while batch and self._estimate(batch) > self.chunk_size and len(batch) > 1:
                    batch.pop(0)
            if batch:
                out.append(batch)
        return out

    def _apply_overlap_between_segments(
        self, segments: list[list[_Unit]]
    ) -> list[list[_Unit]]:
        if not segments:
            return []
        packed: list[list[_Unit]] = []
        for segment in segments:
            group = list(segment)
            if packed and self.overlap:
                prefix = self._overlap_units(packed[-1])
                group = prefix + group
                while (
                    len(group) > len(segment)
                    and self._estimate(group) > self.chunk_size
                ):
                    group.pop(0)
            packed.append(group)
        return packed

    async def chunk(self, pages: Iterable[PageText]) -> list[TextChunk]:
        units = self._units(pages)
        if not units:
            return []

        if self._estimate(units) <= self.chunk_size:
            content = self._join(units)
            return [
                TextChunk(
                    chunk_index=0,
                    content=content,
                    page_start=min(u.page for u in units),
                    page_end=max(u.page for u in units),
                    token_count=self.count_tokens(content),
                )
            ]

        vectors = await self._embedder.embed([u.text for u in units])
        similarities = [
            _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
        ]
        break_after = self._semantic_break_after(similarities)
        segments = self._segments_from_units(units, break_after)
        segments = self._split_oversized_segments(segments)
        groups = self._apply_overlap_between_segments(segments)

        chunks: list[TextChunk] = []
        for group in groups:
            content = self._join(group)
            token_count = self.count_tokens(content)
            while token_count > self.chunk_size and len(group) > 1:
                group = group[1:]
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
