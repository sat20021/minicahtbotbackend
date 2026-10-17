"""
Tests for embedding generation, parsing, and cosine similarity ranking.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.embedding_service import (
    cosine_similarity,
    rank_chunks_by_similarity,
    generate_embeddings,
    generate_single_embedding,
    EmbeddingError,
)


def test_cosine_similarity_cases():
    # Identical
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    # Orthogonal
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    # Opposite
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0])) == -1.0
    # Zero norm
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    # Empty or mismatched
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_rank_chunks_by_similarity():
    query_vec = [1.0, 0.0]
    chunks = [
        {"id": "1", "chunk_index": 0, "text": "orthogonal", "embedding": [0.0, 1.0]},
        {"id": "2", "chunk_index": 1, "text": "identical", "embedding": [1.0, 0.0]},
        {"id": "3", "chunk_index": 2, "text": "semi-aligned", "embedding": [0.8, 0.6]},
    ]
    ranked = rank_chunks_by_similarity(query_vec, chunks, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["id"] == "2"
    assert pytest.approx(ranked[0]["score"]) == 1.0
    assert ranked[1]["id"] == "3"
    assert pytest.approx(ranked[1]["score"]) == 0.8


def test_generate_embeddings_parsing_mock():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.1, 0.4]},
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        res = generate_embeddings(["first", "second"])
        assert len(res) == 2
        # Must be ordered by index
        assert res[0] == [0.1, 0.4]
        assert res[1] == [0.2, 0.3]


def test_generate_single_embedding_mock():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.5, 0.5]}]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        vec = generate_single_embedding("test question")
        assert vec == [0.5, 0.5]


def test_generate_embeddings_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized: Invalid API Key"

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(EmbeddingError, match="HTTP 401"):
            generate_embeddings(["test text"])
