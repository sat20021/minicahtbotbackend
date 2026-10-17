"""
Tests for LLM service context formatting and grounded answer generation.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.llm_service import (
    format_context_block,
    generate_grounded_answer,
    LLMError,
)


def test_format_context_block():
    chunks = [
        {"chunk_index": 0, "text": "This is chunk 0."},
        {"chunk_index": 1, "text": "This is chunk 1."},
    ]
    block = format_context_block(chunks)
    assert "--- [Chunk 0] ---" in block
    assert "This is chunk 0." in block
    assert "--- [Chunk 1] ---" in block
    assert "This is chunk 1." in block


def test_generate_grounded_answer_empty_chunks():
    res = generate_grounded_answer("What is this?", [])
    assert "does not contain information" in res["answer"]
    assert res["usage"] is None


def test_generate_grounded_answer_success_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"content": "The system processes documents in batches."}}
        ],
        "usage": {"total_tokens": 42},
    }

    chunks = [{"chunk_index": 0, "text": "The system processes documents in batches."}]

    with patch("httpx.Client.post", return_value=mock_resp):
        res = generate_grounded_answer("How are documents processed?", chunks)
        assert res["answer"] == "The system processes documents in batches."
        assert res["usage"]["total_tokens"] == 42


def test_generate_grounded_answer_http_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error at LLM Provider"

    chunks = [{"chunk_index": 0, "text": "Some text."}]

    with patch("httpx.Client.post", return_value=mock_resp):
        with patch("time.sleep"):  # Avoid waiting in tests
            with pytest.raises(LLMError, match="HTTP 500"):
                generate_grounded_answer("Question?", chunks)
