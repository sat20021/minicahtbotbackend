"""
FastAPI application for the Ultra-Lightweight Document Q&A RAG Backend Using MongoDB.
Uses synchronous route handlers with PyMongo for simplicity and minimal RAM footprint.
"""
from contextlib import asynccontextmanager
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database import (
    ensure_indexes,
    check_db_health,
    close_mongo_connection,
    DatabaseError,
)
from app.document_service import (
    process_and_store_document,
    get_document_any_owner,
    list_documents,
    delete_document,
    DocumentProcessingError,
)
from app.embedding_service import generate_single_embedding, EmbeddingError
from app.retrieval_service import retrieve_relevant_chunks
from app.llm_service import generate_grounded_answer, LLMError

# Minimal and safe logging (never logs API keys, passwords, or document text)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mini_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes indexes on startup and closes connections on shutdown."""
    logger.info("Starting up %s (v%s)...", settings.PROJECT_NAME, settings.VERSION)
    ensure_indexes()
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)
    close_mongo_connection()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Ultra-lightweight document Q&A RAG backend using FastAPI, MongoDB Atlas, and external AI APIs.",
    lifespan=lifespan,
)

# Explicit CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler (prevents leaking internal traces or secret data)
@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception at %s %s: %s", request.method, request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# -----------------------------------------------------------------------------
# Authentication Dependency
# -----------------------------------------------------------------------------
class User(BaseModel):
    id: str


def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> User:
    """
    Validates user authentication credentials and extracts verified user identity.
    Enforces that client cannot query or delete another user's documents.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    # 1. API Key mode
    if settings.AUTH_API_KEY:
        if token and token == settings.AUTH_API_KEY:
            owner = x_user_id.strip() if x_user_id else "authenticated_user"
            return User(id=owner)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Dev mode / test mode (when AUTH_API_KEY is not configured)
    if settings.ALLOW_DEV_AUTH:
        if x_user_id and x_user_id.strip():
            return User(id=x_user_id.strip())
        if token:
            return User(id=token)
        return User(id="dev_user_default")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )


# -----------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# -----------------------------------------------------------------------------
class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask.")


class ChunkSource(BaseModel):
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    document_id: str
    filename: str
    sources: List[ChunkSource]
    retrieval_metadata: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender: 'user', 'assistant', or 'system'")
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Your direct message or question (e.g. 'summarize document'). Automatically connects across all uploaded documents internally.",
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional: specific document ID. Leave empty or omit to search all documents.",
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Optional: number of chunks to retrieve (defaults to 5).",
    )
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional: conversational history for multi-turn chat.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "summarize document"
            }
        }
    }



class ChatSource(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
    retrieval_metadata: Dict[str, Any]


class DocumentItem(BaseModel):
    id: str
    owner_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    created_at: str


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """Returns API health status without exposing credentials."""
    db_status = check_db_health()
    return {
        "status": "healthy" if db_status["status"] == "ok" else "degraded",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status["mode"],
    }


@app.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Accepts PDF or TXT files, validates size/type, extracts text, chunks it,
    generates embeddings, and stores everything in MongoDB.
    """
    filename = file.filename or "uploaded_file"
    lower_name = filename.lower()
    if not (lower_name.endswith(".pdf") or lower_name.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Only .pdf and .txt files are accepted.",
        )

    # Enforce upload size limit safely
    max_bytes = settings.max_upload_bytes
    contents = bytearray()
    chunk_buffer_size = 64 * 1024

    try:
        while True:
            chunk = file.file.read(chunk_buffer_size)
            if not chunk:
                break
            contents.extend(chunk)
            if len(contents) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_MB}MB.",
                )
    finally:
        file.file.close()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        raw_bytes = bytes(contents)
        del contents  # Free buffer memory
        res = process_and_store_document(
            filename=filename,
            content=raw_bytes,
            owner_id=current_user.id,
        )
        return UploadResponse(**res)
    except DocumentProcessingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EmbeddingError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding API error: {str(e)}")
    except DatabaseError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")


@app.post("/documents/{document_id}/ask", response_model=AskResponse, tags=["Documents"])
def ask_document(
    document_id: str,
    payload: AskRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Answers a question grounded strictly in the retrieved context chunks of the document.
    Enforces document ownership.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    # 1. Verify existence and ownership
    doc = get_document_any_owner(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc["owner_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to query this document.",
        )

    # 2. Generate question embedding
    try:
        query_vector = generate_single_embedding(question)
    except EmbeddingError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding API error: {str(e)}")

    # 3. Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(
        document_id=document_id,
        owner_id=current_user.id,
        query_vector=query_vector,
        top_k=settings.TOP_K,
    )

    if not chunks:
        return AskResponse(
            answer="The provided document does not contain information to answer this question.",
            document_id=document_id,
            filename=doc["filename"],
            sources=[],
            retrieval_metadata={"chunks_retrieved": 0, "top_k": settings.TOP_K},
        )

    # 4. Generate answer with external LLM
    try:
        llm_response = generate_grounded_answer(question=question, chunks=chunks)
    except LLMError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM API error: {str(e)}")

    sources = [
        ChunkSource(
            chunk_index=c.get("chunk_index", 0),
            score=c.get("score", 0.0),
            excerpt=(c.get("text", "")[:200] + "...") if len(c.get("text", "")) > 200 else c.get("text", ""),
        )
        for c in chunks
    ]

    return AskResponse(
        answer=llm_response["answer"],
        document_id=document_id,
        filename=doc["filename"],
        sources=sources,
        retrieval_metadata={
            "chunks_retrieved": len(chunks),
            "top_k": settings.TOP_K,
            "token_usage": llm_response.get("usage"),
        },
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_documents(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Direct Conversational RAG endpoint.
    - Connects across all user documents internally.
    - Users can simply send {"message": "summarize document"}.
    - Automatically recovers and cites all relevant documents.
    """
    # 1. Resolve query text
    query = payload.message.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty.",
        )

    # Clean optional history (filter out Swagger dummy "string" entries)
    history: List[Dict[str, str]] = []
    if payload.messages:
        for m in payload.messages:
            r = m.role.strip().lower()
            c = m.content.strip()
            if r in ("user", "assistant", "system") and c and c.lower() != "string":
                history.append({"role": r, "content": c})

    # 2. Clean document_id (ignore empty, "string", "null", "none", "all", "undefined")
    doc_id = payload.document_id.strip() if payload.document_id else None
    if doc_id and doc_id.lower() in ("string", "null", "none", "all", "undefined"):
        doc_id = None

    target_doc = None
    if doc_id:
        target_doc = get_document_any_owner(doc_id)
        if not target_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_id}' not found.")
        if target_doc["owner_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to query this document.",
            )

    # 3. Check if user has uploaded any documents
    user_docs = list_documents(current_user.id)
    if not user_docs:
        return ChatResponse(
            answer="You have not uploaded any documents yet. Please upload a PDF or TXT document using POST /documents/upload first.",
            sources=[],
            retrieval_metadata={
                "chunks_retrieved": 0,
                "top_k": payload.top_k or settings.TOP_K,
                "scope": doc_id or "all_documents",
            },
        )

    # 4. Determine top_k (if 1 or small, or summarizing, ensure adequate chunks)
    k = payload.top_k or settings.TOP_K
    if k == 1 and any(w in query.lower() for w in ("summar", "overview", "all", "about", "what is")):
        k = settings.TOP_K

    # 5. Generate query vector
    try:
        query_vector = generate_single_embedding(query)
    except EmbeddingError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding API error: {str(e)}")

    # 6. Retrieve relevant chunks (across all documents or within doc_id)
    chunks = retrieve_relevant_chunks(
        document_id=doc_id,
        owner_id=current_user.id,
        query_vector=query_vector,
        top_k=k,
    )

    # Fallback to document chunks if semantic similarity produced no hits (e.g. generic "summarize document")
    if not chunks:
        from app.database import get_database, in_memory_store
        db = get_database()
        if db is None:
            raw_chunks = in_memory_store.find_chunks_for_owner(current_user.id, doc_id=doc_id)[:k]
            chunks = [
                {
                    "chunk_id": str(c["_id"]),
                    "document_id": str(c.get("document_id", "")),
                    "chunk_index": c.get("chunk_index", 0),
                    "text": c.get("text", ""),
                    "score": 1.0,
                }
                for c in raw_chunks
            ]
        else:
            filt = {"owner_id": current_user.id}
            if doc_id:
                try:
                    filt["document_id"] = ObjectId(doc_id)
                except Exception:
                    pass
            cursor = db.document_chunks.find(filt).limit(k)
            chunks = [
                {
                    "chunk_id": str(c["_id"]),
                    "document_id": str(c.get("document_id", "")),
                    "chunk_index": c.get("chunk_index", 0),
                    "text": c.get("text", ""),
                    "score": 1.0,
                }
                for c in cursor
            ]

    # 7. Populate filenames for sources and context prompt
    doc_name_map = {d["id"]: d["filename"] for d in user_docs}
    if target_doc:
        doc_name_map[str(target_doc["_id"])] = target_doc["filename"]

    for c in chunks:
        cid = c.get("document_id") or ""
        c["filename"] = doc_name_map.get(cid, "Document")

    if not chunks:
        return ChatResponse(
            answer="No relevant content found in your documents to answer this question.",
            sources=[],
            retrieval_metadata={
                "chunks_retrieved": 0,
                "top_k": k,
                "scope": doc_id or "all_documents",
            },
        )

    # 8. Generate grounded answer
    try:
        llm_response = generate_grounded_answer(
            question=query,
            chunks=chunks,
            history=history,
        )
    except LLMError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM API error: {str(e)}")

    sources = [
        ChatSource(
            document_id=c.get("document_id", doc_id or ""),
            filename=c.get("filename", "Document"),
            chunk_index=c.get("chunk_index", 0),
            score=c.get("score", 0.0),
            excerpt=(c.get("text", "")[:200] + "...") if len(c.get("text", "")) > 200 else c.get("text", ""),
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=llm_response["answer"],
        sources=sources,
        retrieval_metadata={
            "chunks_retrieved": len(chunks),
            "top_k": k,
            "scope": doc_id or "all_documents",
            "token_usage": llm_response.get("usage"),
        },
    )



@app.get("/documents", response_model=List[DocumentItem], tags=["Documents"])
def list_user_documents(
    current_user: User = Depends(get_current_user),
):
    """Lists all documents belonging to the authenticated user."""
    docs = list_documents(current_user.id)
    return [DocumentItem(**d) for d in docs]


@app.delete("/documents/{document_id}", tags=["Documents"])
def delete_user_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """Deletes a document and all its associated chunks. Enforces ownership."""
    doc = get_document_any_owner(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if doc["owner_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this document.",
        )

    success = delete_document(document_id, current_user.id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found or could not be deleted.")

    return {
        "message": "Document and all associated chunks successfully deleted.",
        "document_id": document_id,
    }


@app.get("/", tags=["Root"])
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
    }
