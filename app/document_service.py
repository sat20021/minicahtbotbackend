"""
Document processing and MongoDB persistence service.
Handles PDF/TXT text extraction, chunking, and database CRUD.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from bson import ObjectId
import fitz  # PyMuPDF

from app.config import settings
from app.chunker import split_text_into_chunks
from app.embedding_service import generate_embeddings
from app.database import get_database, in_memory_store, DatabaseError


class DocumentProcessingError(Exception):
    """Raised when document parsing or validation fails."""
    pass


# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------
def extract_text_from_pdf(content: bytes) -> str:
    """Extracts text from PDF byte stream using PyMuPDF."""
    if not content or len(content) == 0:
        raise DocumentProcessingError("PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise DocumentProcessingError(f"Corrupt or invalid PDF file: {str(e)}")

    try:
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise DocumentProcessingError("Encrypted or password-protected PDFs are not supported.")

        if doc.page_count == 0:
            raise DocumentProcessingError("PDF contains no pages.")

        pages_text = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            if text and text.strip():
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise DocumentProcessingError(
                "PDF contains no readable text. Scanned images or empty documents are not supported."
            )
        return full_text
    finally:
        doc.close()


def extract_text_from_txt(content: bytes) -> str:
    """Extracts text from TXT byte stream using Python built-in decoders."""
    if not content or len(content) == 0:
        raise DocumentProcessingError("Text file is empty (0 bytes).")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            decoded = content.decode(enc).strip()
            if not decoded:
                raise DocumentProcessingError("Text file is empty or contains only whitespace.")
            return decoded
        except UnicodeDecodeError:
            continue

    raise DocumentProcessingError("Unable to decode text file. Ensure it is encoded in valid UTF-8.")


def extract_text(filename: str, content: bytes) -> Tuple[str, str]:
    """Routes extraction based on filename extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(content), "pdf"
    elif lower.endswith(".txt"):
        return extract_text_from_txt(content), "txt"
    else:
        raise DocumentProcessingError(f"Unsupported file format for '{filename}'. Only .pdf and .txt are accepted.")


# -----------------------------------------------------------------------------
# Document Upload and Storage Pipeline
# -----------------------------------------------------------------------------
def process_and_store_document(
    filename: str,
    content: bytes,
    owner_id: str,
) -> Dict[str, Any]:
    """
    Validates, extracts text, chunks, embeds, and stores in MongoDB.
    Frees raw file bytes immediately to minimize RAM consumption.
    """
    # 1. Extraction
    extracted_text, file_type = extract_text(filename, content)
    del content  # Discard raw file bytes immediately

    # 2. Chunking
    chunks = split_text_into_chunks(
        text=extracted_text,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        max_chunks=settings.MAX_CHUNKS,
    )
    del extracted_text  # Discard full raw text string

    if not chunks:
        raise DocumentProcessingError("Could not split document into readable chunks.")

    # 3. Generate embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # 4. Save metadata & chunks to MongoDB
    db = get_database()
    now = datetime.now(timezone.utc)

    if db is None:
        # In-memory store fallback
        doc_data = {
            "owner_id": owner_id,
            "filename": filename,
            "file_type": file_type,
            "status": "ready",
            "created_at": now,
            "chunk_count": len(chunks),
        }
        doc_id = in_memory_store.insert_document(doc_data)
        doc_oid = ObjectId(doc_id)

        chunks_data = [
            {
                "document_id": doc_oid,
                "owner_id": owner_id,
                "chunk_index": c["index"],
                "text": c["text"],
                "embedding": embeddings[i],
                "created_at": now,
            }
            for i, c in enumerate(chunks)
        ]
        in_memory_store.insert_chunks(chunks_data)

        return {
            "document_id": doc_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "status": "ready",
        }

    # Persistent MongoDB Atlas write
    doc_record = {
        "owner_id": owner_id,
        "filename": filename,
        "file_type": file_type,
        "status": "ready",
        "created_at": now,
        "chunk_count": len(chunks),
    }

    try:
        doc_res = db.documents.insert_one(doc_record)
        doc_id = doc_res.inserted_id

        chunks_to_insert = [
            {
                "document_id": doc_id,
                "owner_id": owner_id,
                "chunk_index": c["index"],
                "text": c["text"],
                "embedding": embeddings[i],
                "created_at": now,
            }
            for i, c in enumerate(chunks)
        ]

        db.document_chunks.insert_many(chunks_to_insert)
        return {
            "document_id": str(doc_id),
            "filename": filename,
            "chunk_count": len(chunks),
            "status": "ready",
        }
    except Exception as e:
        raise DatabaseError(f"Failed to persist document to MongoDB: {str(e)}")


# -----------------------------------------------------------------------------
# Document Read / List / Delete Operations
# -----------------------------------------------------------------------------
def get_document_by_id(document_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves document metadata if owner matches."""
    db = get_database()
    if db is None:
        return in_memory_store.find_document(document_id, owner_id)

    try:
        oid = ObjectId(document_id)
    except Exception:
        return None

    doc = db.documents.find_one({"_id": oid, "owner_id": owner_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def get_document_any_owner(document_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves document metadata by ID regardless of owner to distinguish 403 from 404."""
    db = get_database()
    if db is None:
        return in_memory_store.find_document(document_id)

    try:
        oid = ObjectId(document_id)
    except Exception:
        return None

    doc = db.documents.find_one({"_id": oid})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_documents(owner_id: str) -> List[Dict[str, Any]]:
    """Lists all documents belonging to owner."""
    db = get_database()
    if db is None:
        docs = in_memory_store.list_documents(owner_id)
        return [
            {
                "id": str(d["_id"]),
                "owner_id": d["owner_id"],
                "filename": d["filename"],
                "file_type": d["file_type"],
                "status": d["status"],
                "chunk_count": d.get("chunk_count", 0),
                "created_at": d["created_at"].isoformat() if isinstance(d["created_at"], datetime) else str(d["created_at"]),
            }
            for d in docs
        ]

    cursor = db.documents.find({"owner_id": owner_id}).sort("created_at", -1)
    results = []
    for d in cursor:
        results.append({
            "id": str(d["_id"]),
            "owner_id": d["owner_id"],
            "filename": d["filename"],
            "file_type": d["file_type"],
            "status": d["status"],
            "chunk_count": d.get("chunk_count", 0),
            "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else str(d.get("created_at")),
        })
    return results


def delete_document(document_id: str, owner_id: str) -> bool:
    """Deletes document and cascades deletion to all associated chunks."""
    db = get_database()
    if db is None:
        return in_memory_store.delete_document(document_id, owner_id)

    try:
        oid = ObjectId(document_id)
    except Exception:
        return False

    # Delete document record
    res = db.documents.delete_one({"_id": oid, "owner_id": owner_id})
    if res.deleted_count == 0:
        return False

    # Cascade delete all chunks belonging to this document and owner
    db.document_chunks.delete_many({"document_id": oid, "owner_id": owner_id})
    return True
