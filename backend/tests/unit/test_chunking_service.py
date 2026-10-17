import pytest

from app.services.chunking_service import ChunkingService
from app.services.pdf_extractor import PageText


def sentences(prefix: str, count: int) -> str:
    return " ".join(
        f"{prefix} sentence number {i} talks about retrieval and grounding." for i in range(count)
    )


@pytest.fixture
def chunker() -> ChunkingService:
    return ChunkingService(chunk_size=80, overlap=20)


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=0, overlap=0)


def test_empty_input(chunker: ChunkingService) -> None:
    assert chunker.chunk([]) == []
    assert chunker.chunk([PageText(1, ""), PageText(2, "   \n\n  ")]) == []


def test_short_document_is_single_chunk(chunker: ChunkingService) -> None:
    chunks = chunker.chunk([PageText(1, "A short page. Only two sentences.")])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.content == "A short page. Only two sentences."
    assert (chunk.page_start, chunk.page_end) == (1, 1)
    assert chunk.token_count == chunker.count_tokens(chunk.content)


def test_long_text_respects_chunk_size(chunker: ChunkingService) -> None:
    chunks = chunker.chunk([PageText(1, sentences("Alpha", 60))])
    assert len(chunks) > 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.content.strip()
        assert 0 < chunk.token_count <= chunker.chunk_size
        assert chunk.token_count == chunker.count_tokens(chunk.content)


def test_all_source_sentences_are_covered(chunker: ChunkingService) -> None:
    text = sentences("Alpha", 40)
    chunks = chunker.chunk([PageText(1, text)])
    combined = " ".join(c.content for c in chunks)
    for i in range(40):
        assert f"Alpha sentence number {i} " in combined


def test_overlap_between_adjacent_chunks(chunker: ChunkingService) -> None:
    chunks = chunker.chunk([PageText(1, sentences("Alpha", 40))])
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        # The next chunk begins with whole sentences taken from the end of the previous one.
        first_sentence = nxt.content.split(". ")[0] + "."
        assert first_sentence in prev.content
        assert prev.content.endswith(first_sentence) or first_sentence + " " in prev.content
        overlap_tokens = chunker.count_tokens(first_sentence)
        assert overlap_tokens <= chunker.overlap


def test_zero_overlap_has_no_repeated_sentences() -> None:
    chunker = ChunkingService(chunk_size=80, overlap=0)
    chunks = chunker.chunk([PageText(1, sentences("Alpha", 40))])
    seen: list[str] = []
    for chunk in chunks:
        for sentence in chunk.content.split(". "):
            assert sentence not in seen
            seen.append(sentence)


def test_page_boundary_metadata() -> None:
    chunker = ChunkingService(chunk_size=200, overlap=20)
    pages = [
        PageText(1, "Page one text. It is short."),
        PageText(2, "Page two text. Also short."),
        PageText(3, sentences("Gamma", 30)),
    ]
    chunks = chunker.chunk(pages)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end >= 2  # short pages 1 and 2 merge into the first chunk
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 3
        if "Page one" in chunk.content:
            assert chunk.page_start == 1
        if "Gamma" in chunk.content:
            assert chunk.page_end == 3
    assert chunks[-1].page_end == 3
    assert all(c.page_start == 3 for c in chunks[1:])


def test_pages_keep_paragraph_separation() -> None:
    chunker = ChunkingService(chunk_size=200, overlap=10)
    chunks = chunker.chunk([PageText(1, "First para.\n\nSecond para."), PageText(2, "Third.")])
    assert chunks[0].content == "First para.\n\nSecond para.\n\nThird."


def test_oversized_sentence_is_split_on_words(chunker: ChunkingService) -> None:
    giant = " ".join(f"word{i}" for i in range(400))  # one "sentence", no punctuation
    chunks = chunker.chunk([PageText(1, giant)])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= chunker.chunk_size
    assert "word0" in chunks[0].content
    assert "word399" in chunks[-1].content


def test_hyphenated_line_breaks_are_joined(chunker: ChunkingService) -> None:
    chunks = chunker.chunk([PageText(1, "Infor-\nmation retrieval works.")])
    assert chunks[0].content == "Information retrieval works."


def test_default_configuration_targets_650_tokens() -> None:
    chunker = ChunkingService(chunk_size=650, overlap=100)
    chunks = chunker.chunk([PageText(p, sentences(f"P{p}", 60)) for p in range(1, 6)])
    assert all(c.token_count <= 650 for c in chunks)
    assert max(c.token_count for c in chunks) > 500
