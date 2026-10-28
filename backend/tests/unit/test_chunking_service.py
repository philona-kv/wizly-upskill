from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.services.chunking_service import ChunkingService
from app.services.pdf_extractor import PageText


class FakeEmbedder:
    """Topic-like vectors from the first word of each sentence (for unit tests)."""

    model = "test-embed"
    dimensions = 16

    async def embed(
        self, texts: Sequence[str], token_counts: Sequence[int] | None = None
    ) -> list[list[float]]:
        del token_counts
        out: list[list[float]] = []
        for text in texts:
            word = text.split()[0] if text.split() else text
            vec = [0.0] * self.dimensions
            vec[hash(word) % self.dimensions] = 1.0
            out.append(vec)
        return out


def sentences(prefix: str, count: int) -> str:
    return " ".join(
        f"{prefix} sentence number {i} talks about retrieval and grounding." for i in range(count)
    )


@pytest.fixture
def chunker() -> ChunkingService:
    return ChunkingService(80, 20, FakeEmbedder())


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        ChunkingService(100, 100, FakeEmbedder())
    with pytest.raises(ValueError):
        ChunkingService(0, 0, FakeEmbedder())


@pytest.mark.asyncio
async def test_empty_input(chunker: ChunkingService) -> None:
    assert await chunker.chunk([]) == []
    assert await chunker.chunk([PageText(1, ""), PageText(2, "   \n\n  ")]) == []


@pytest.mark.asyncio
async def test_short_document_is_single_chunk(chunker: ChunkingService) -> None:
    chunks = await chunker.chunk([PageText(1, "A short page. Only two sentences.")])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.content == "A short page. Only two sentences."
    assert (chunk.page_start, chunk.page_end) == (1, 1)
    assert chunk.token_count == chunker.count_tokens(chunk.content)


@pytest.mark.asyncio
async def test_long_text_respects_chunk_size(chunker: ChunkingService) -> None:
    chunks = await chunker.chunk([PageText(1, sentences("Alpha", 60))])
    assert len(chunks) > 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.content.strip()
        assert 0 < chunk.token_count <= chunker.chunk_size
        assert chunk.token_count == chunker.count_tokens(chunk.content)


@pytest.mark.asyncio
async def test_all_source_sentences_are_covered(chunker: ChunkingService) -> None:
    text = sentences("Alpha", 40)
    chunks = await chunker.chunk([PageText(1, text)])
    combined = " ".join(c.content for c in chunks)
    for i in range(40):
        assert f"Alpha sentence number {i} " in combined


@pytest.mark.asyncio
async def test_overlap_between_adjacent_chunks(chunker: ChunkingService) -> None:
    chunks = await chunker.chunk([PageText(1, sentences("Alpha", 40))])
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_sentence = nxt.content.split(". ")[0] + "."
        assert first_sentence in prev.content
        overlap_tokens = chunker.count_tokens(first_sentence)
        assert overlap_tokens <= chunker.overlap


@pytest.mark.asyncio
async def test_zero_overlap_has_no_repeated_sentences() -> None:
    chunker = ChunkingService(80, 0, FakeEmbedder())
    chunks = await chunker.chunk([PageText(1, sentences("Alpha", 40))])
    seen: list[str] = []
    for chunk in chunks:
        for sentence in chunk.content.split(". "):
            if not sentence.strip():
                continue
            assert sentence not in seen
            seen.append(sentence)


@pytest.mark.asyncio
async def test_page_boundary_metadata() -> None:
    chunker = ChunkingService(200, 20, FakeEmbedder())
    pages = [
        PageText(1, "Page one text. It is short."),
        PageText(2, "Page two text. Also short."),
        PageText(3, sentences("Gamma", 30)),
    ]
    chunks = await chunker.chunk(pages)
    assert chunks[0].page_start == 1
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 3
        if "Page one" in chunk.content:
            assert chunk.page_start == 1
        if "Gamma" in chunk.content:
            assert chunk.page_end == 3
    assert chunks[-1].page_end == 3


@pytest.mark.asyncio
async def test_pages_keep_paragraph_separation() -> None:
    chunker = ChunkingService(200, 10, FakeEmbedder())
    chunks = await chunker.chunk(
        [PageText(1, "First para.\n\nSecond para."), PageText(2, "Third.")]
    )
    assert chunks[0].content == "First para.\n\nSecond para.\n\nThird."


@pytest.mark.asyncio
async def test_oversized_sentence_is_split_on_words(chunker: ChunkingService) -> None:
    giant = " ".join(f"word{i}" for i in range(400))
    chunks = await chunker.chunk([PageText(1, giant)])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunker.chunk_size
    assert "word0" in chunks[0].content
    assert "word399" in chunks[-1].content


@pytest.mark.asyncio
async def test_hyphenated_line_breaks_are_joined(chunker: ChunkingService) -> None:
    chunks = await chunker.chunk([PageText(1, "Infor-\nmation retrieval works.")])
    assert chunks[0].content == "Information retrieval works."


@pytest.mark.asyncio
async def test_default_configuration_targets_650_tokens() -> None:
    chunker = ChunkingService(650, 100, FakeEmbedder())
    chunks = await chunker.chunk([PageText(p, sentences("Topic", 60)) for p in range(1, 6)])
    assert all(c.token_count <= 650 for c in chunks)
    assert max(c.token_count for c in chunks) > 500


@pytest.mark.asyncio
async def test_semantic_break_between_topics() -> None:
    chunker = ChunkingService(100, 0, FakeEmbedder(), breakpoint_percentile=0.5)
    text = sentences("Alpha", 8) + " " + sentences("Beta", 8)
    chunks = await chunker.chunk([PageText(1, text)])
    assert len(chunks) >= 2
    first_has_alpha = any("Alpha sentence" in c.content for c in chunks)
    last_has_beta = any("Beta sentence" in c.content for c in chunks)
    assert first_has_alpha and last_has_beta
    if len(chunks) >= 2:
        assert not ("Alpha sentence" in chunks[-1].content and "Beta sentence" in chunks[0].content)
