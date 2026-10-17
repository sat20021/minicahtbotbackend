"""
Tests for text chunker with configurable size and overlap.
"""
import pytest
from app.chunker import split_text_into_chunks


def test_empty_and_whitespace():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\n  ") == []


def test_text_shorter_than_chunk_size():
    text = "This is a short sentence."
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0]["index"] == 0
    assert chunks[0]["text"] == text


def test_chunking_with_overlap():
    text = "First paragraph content that has multiple words.\n\nSecond paragraph content that continues the text."
    chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=15)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c["index"] == i
        assert len(c["text"]) <= 70  # Within boundary extension limits


def test_max_chunks_cap():
    text = "word " * 500
    chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10, max_chunks=5)
    assert len(chunks) == 5


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text_into_chunks("Hello", chunk_size=0)
    with pytest.raises(ValueError):
        split_text_into_chunks("Hello", chunk_size=100, chunk_overlap=-5)
    with pytest.raises(ValueError):
        split_text_into_chunks("Hello", chunk_size=100, chunk_overlap=100)
