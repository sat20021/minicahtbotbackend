"""
Integration and API endpoint tests using FastAPI TestClient.
Covers upload, Q&A, listing, deletion, ownership enforcement, error handling, and mock APIs.
"""
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import fitz

from app.main import app
from app.database import in_memory_store, check_db_health


@pytest.fixture(autouse=True)
def clean_in_memory_store():
    """Resets the in-memory database store before each test."""
    in_memory_store.reset()


@pytest.fixture
def client():
    return TestClient(app)


def create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    b = doc.write()
    doc.close()
    return b


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "service" in data
    assert "version" in data
    assert "database" in data


def test_upload_txt_success(client):
    mock_emb = [[0.1, 0.2, 0.3]]

    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        file_content = b"This is a company policy manual document with important guidelines."
        files = {"file": ("manual.txt", io.BytesIO(file_content), "text/plain")}
        headers = {"X-User-Id": "user_alpha"}

        response = client.post("/documents/upload", files=files, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "manual.txt"
        assert data["chunk_count"] == 1
        assert data["status"] == "ready"


def test_upload_pdf_success(client):
    mock_emb = [[0.1, 0.2, 0.3]]
    pdf_bytes = create_sample_pdf("Employee benefits guide: 25 days annual leave.")

    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        files = {"file": ("benefits.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        headers = {"X-User-Id": "user_alpha"}

        response = client.post("/documents/upload", files=files, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "benefits.pdf"
        assert data["chunk_count"] == 1


def test_upload_empty_file_fails(client):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_invalid_extension_fails(client):
    files = {"file": ("document.csv", io.BytesIO(b"a,b,c"), "text/csv")}
    response = client.post("/documents/upload", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_list_documents_by_owner(client):
    mock_emb = [[0.1, 0.2]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        # User Alpha uploads 2 docs
        client.post(
            "/documents/upload",
            files={"file": ("doc1.txt", io.BytesIO(b"Content doc 1"), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        client.post(
            "/documents/upload",
            files={"file": ("doc2.txt", io.BytesIO(b"Content doc 2"), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        # User Beta uploads 1 doc
        client.post(
            "/documents/upload",
            files={"file": ("beta.txt", io.BytesIO(b"Content beta"), "text/plain")},
            headers={"X-User-Id": "user_beta"},
        )

    # User Alpha listing
    res_alpha = client.get("/documents", headers={"X-User-Id": "user_alpha"})
    assert res_alpha.status_code == 200
    docs_alpha = res_alpha.json()
    assert len(docs_alpha) == 2
    assert all(d["owner_id"] == "user_alpha" for d in docs_alpha)

    # User Beta listing
    res_beta = client.get("/documents", headers={"X-User-Id": "user_beta"})
    assert res_beta.status_code == 200
    docs_beta = res_beta.json()
    assert len(docs_beta) == 1
    assert docs_beta[0]["owner_id"] == "user_beta"


def test_ask_document_grounded_answer(client):
    mock_emb = [[0.1, 0.2, 0.3]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        up_res = client.post(
            "/documents/upload",
            files={"file": ("secret.txt", io.BytesIO(b"The vault code is 4321."), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        doc_id = up_res.json()["document_id"]

    mock_llm_res = {
        "answer": "The vault code is 4321.",
        "usage": {"total_tokens": 15},
    }

    with patch("app.main.generate_single_embedding", return_value=[0.1, 0.2, 0.3]):
        with patch("app.main.generate_grounded_answer", return_value=mock_llm_res):
            ask_res = client.post(
                f"/documents/{doc_id}/ask",
                json={"question": "What is the vault code?"},
                headers={"X-User-Id": "user_alpha"},
            )
            assert ask_res.status_code == 200
            data = ask_res.json()
            assert data["answer"] == "The vault code is 4321."
            assert data["document_id"] == doc_id
            assert len(data["sources"]) > 0


def test_document_ownership_enforcement(client):
    mock_emb = [[0.5, 0.5]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        up_res = client.post(
            "/documents/upload",
            files={"file": ("alpha_doc.txt", io.BytesIO(b"Alpha secret document content"), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        doc_id = up_res.json()["document_id"]

    # User Beta attempts to query User Alpha's document -> 403 Forbidden
    ask_res = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "What is in this document?"},
        headers={"X-User-Id": "user_beta"},
    )
    assert ask_res.status_code == 403
    assert "permission" in ask_res.json()["detail"].lower()

    # User Beta attempts to delete User Alpha's document -> 403 Forbidden
    del_res = client.delete(
        f"/documents/{doc_id}",
        headers={"X-User-Id": "user_beta"},
    )
    assert del_res.status_code == 403
    assert "permission" in del_res.json()["detail"].lower()


def test_delete_document_success(client):
    mock_emb = [[0.1, 0.1]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        up_res = client.post(
            "/documents/upload",
            files={"file": ("temp.txt", io.BytesIO(b"Temporary file to delete"), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        doc_id = up_res.json()["document_id"]

    # Delete as owner
    del_res = client.delete(f"/documents/{doc_id}", headers={"X-User-Id": "user_alpha"})
    assert del_res.status_code == 200
    assert del_res.json()["document_id"] == doc_id

    # Verify document no longer exists
    get_res = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "Is anyone here?"},
        headers={"X-User-Id": "user_alpha"},
    )
    assert get_res.status_code == 404


def test_llm_failure_handling(client):
    from app.llm_service import LLMError
    mock_emb = [[0.1, 0.2]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        up_res = client.post(
            "/documents/upload",
            files={"file": ("doc.txt", io.BytesIO(b"Sample data"), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        doc_id = up_res.json()["document_id"]

    with patch("app.main.generate_single_embedding", return_value=[0.1, 0.2]):
        with patch("app.main.generate_grounded_answer", side_effect=LLMError("Upstream timeout")):
            ask_res = client.post(
                f"/documents/{doc_id}/ask",
                json={"question": "What is this?"},
                headers={"X-User-Id": "user_alpha"},
            )
            assert ask_res.status_code == 502
            assert "LLM API error" in ask_res.json()["detail"]


def test_mongodb_connection_error_health():
    with patch("app.database.get_mongo_client", side_effect=Exception("Database unreachable")):
        with patch("app.config.settings.MONGODB_URI", "mongodb+srv://invalid:uri@cluster"):
            status_data = check_db_health()
            assert status_data["status"] in ("degraded", "error")


def test_chat_cross_documents_success(client):
    """Verifies /chat queries across all uploaded documents for a user."""
    mock_emb = [[0.9, 0.1]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        # Upload doc 1
        client.post(
            "/documents/upload",
            files={"file": ("handbook.txt", io.BytesIO(b"Company policy specifies 20 annual leave days."), "text/plain")},
            headers={"X-User-Id": "user_chat"},
        )
        # Upload doc 2
        client.post(
            "/documents/upload",
            files={"file": ("benefits.txt", io.BytesIO(b"Health insurance covers dental and vision."), "text/plain")},
            headers={"X-User-Id": "user_chat"},
        )

    with patch("app.main.generate_single_embedding", return_value=[0.9, 0.1]):
        with patch("app.main.generate_grounded_answer", return_value={"answer": "You get 20 days.", "usage": {"total_tokens": 42}}):
            chat_res = client.post(
                "/chat",
                json={"message": "How many leave days do I have?"},
                headers={"X-User-Id": "user_chat"},
            )
            assert chat_res.status_code == 200
            data = chat_res.json()
            assert data["answer"] == "You get 20 days."
            assert len(data["sources"]) > 0
            assert data["sources"][0]["filename"] in ("handbook.txt", "benefits.txt")
            assert data["retrieval_metadata"]["scope"] == "all_documents"


def test_chat_with_conversation_history(client):
    """Verifies /chat correctly extracts query from messages history array."""
    mock_emb = [[0.9, 0.1]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        client.post(
            "/documents/upload",
            files={"file": ("guide.txt", io.BytesIO(b"The password reset link expires in 15 minutes."), "text/plain")},
            headers={"X-User-Id": "user_chat_hist"},
        )

    with patch("app.main.generate_single_embedding", return_value=[0.9, 0.1]):
        with patch("app.main.generate_grounded_answer", return_value={"answer": "It expires in 15 minutes.", "usage": None}) as mock_llm:
            chat_res = client.post(
                "/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Hello there"},
                        {"role": "assistant", "content": "Hello! How can I help you?"},
                        {"role": "user", "content": "When does the link expire?"},
                    ]
                },
                headers={"X-User-Id": "user_chat_hist"},
            )
            assert chat_res.status_code == 200
            assert chat_res.json()["answer"] == "It expires in 15 minutes."
            # Verify history passed to LLM
            mock_llm.assert_called_once()
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["question"] == "When does the link expire?"
            assert len(call_kwargs["history"]) == 2


def test_chat_document_scope_and_ownership(client):
    """Verifies document_id filter and ownership enforcement in /chat."""
    mock_emb = [[0.9, 0.1]]
    with patch("app.document_service.generate_embeddings", return_value=mock_emb):
        up_res = client.post(
            "/documents/upload",
            files={"file": ("secret.txt", io.BytesIO(b"Confidential project Apollo notes."), "text/plain")},
            headers={"X-User-Id": "user_alpha"},
        )
        doc_id = up_res.json()["document_id"]

    # Forbidden for user_beta
    forbidden_res = client.post(
        "/chat",
        json={"message": "What is Apollo?", "document_id": doc_id},
        headers={"X-User-Id": "user_beta"},
    )
    assert forbidden_res.status_code == 403

    # Success for owner
    with patch("app.main.generate_single_embedding", return_value=[0.9, 0.1]):
        with patch("app.main.generate_grounded_answer", return_value={"answer": "Project Apollo notes.", "usage": None}):
            owner_res = client.post(
                "/chat",
                json={"message": "What is Apollo?", "document_id": doc_id},
                headers={"X-User-Id": "user_alpha"},
            )
            assert owner_res.status_code == 200
            assert owner_res.json()["retrieval_metadata"]["scope"] == doc_id

