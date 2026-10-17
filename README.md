# Ultra-Lightweight Document Q&A RAG Backend Using MongoDB

A production-conscious, ultra-lightweight document-based question-answering (RAG) backend built strictly with **FastAPI**, **MongoDB Atlas**, **PyMongo**, and direct HTTP calls via **HTTPX** to external AI providers.

---

## 1. Architectural Overview & Philosophy

Most modern RAG tutorials mandate heavy frameworks (such as LangChain, LlamaIndex, PyTorch, Transformers, ChromaDB, or FAISS) that bloat Docker images to gigabytes and demand 2GB+ of idle RAM.

This backend adheres strictly to **ultra-lightweight, production-grade engineering principles**:

- **No heavy ML dependencies**: Zero PyTorch, Transformers, LangChain, or local models. Runs comfortably on free-tier containers with as little as 256MB RAM.
- **Direct HTTP with HTTPX**: Calls external embedding and LLM APIs directly over HTTP with bounded timeouts and automatic exponential backoff retries.
- **MongoDB Atlas Vector Search with Fallback**: Leverages MongoDB Atlas `$vectorSearch` when configured. If unavailable or when running on standard MongoDB, it seamlessly falls back to a vectorized Python cosine-similarity search filtered by document and owner.
- **Synchronous PyMongo Simplicity**: Uses PyMongo's battle-tested synchronous driver with FastAPI synchronous route handlers, avoiding async event-loop deadlocks and thread pool contention.
- **Strict Grounding**: The LLM system prompt mandates that answers must be derived solely from retrieved document excerpts. If information is absent, it explicitly states so rather than hallucinating.
- **Multi-Tenant Ownership Security**: Every document, chunk, read, query, and delete operation is strictly scoped to the authenticated user ID (`owner_id`).

---

## 2. Technology Stack & Dependencies

All dependencies are strictly limited to the lightweight essentials in `requirements.txt`:

| Package | Purpose |
| :--- | :--- |
| `fastapi` | High-performance, clean web API framework |
| `uvicorn` | Lightweight ASGI production web server |
| `python-multipart` | Memory-bounded streaming file upload handling |
| `pymupdf` | High-speed, lightweight PDF text extraction |
| `pymongo` | Official synchronous MongoDB driver |
| `httpx` | Robust HTTP client for external AI API calls |
| `python-dotenv` | Clean environment variable loading |

---

## 3. MongoDB Database Architecture

Database Name: `mini_rag`

### `documents` Collection
Stores metadata for uploaded documents:
```json
{
  "_id": ObjectId("671..."),
  "owner_id": "user_123",
  "filename": "annual_report.pdf",
  "file_type": "pdf",
  "status": "ready",
  "created_at": ISODate("2026-10-16T10:00:00Z"),
  "chunk_count": 12
}
```

### `document_chunks` Collection
Stores text chunks and their corresponding embedding vectors:
```json
{
  "_id": ObjectId("671..."),
  "document_id": ObjectId("671..."),
  "owner_id": "user_123",
  "chunk_index": 0,
  "text": "The company's annual revenue grew by 18% in Q4...",
  "embedding": [0.0123, -0.0456, 0.0891, ...],
  "created_at": ISODate("2026-10-16T10:00:00Z")
}
```

### Standard Indexes
Automatically ensured at application startup:
- `documents`: `owner_id` (ASC), `created_at` (DESC)
- `document_chunks`: `document_id` (ASC), `owner_id` (ASC), compound `(document_id, chunk_index)`

### MongoDB Atlas Vector Search Index Configuration
To enable native hardware-accelerated vector similarity search in MongoDB Atlas:

1. Open your cluster in [MongoDB Atlas](https://cloud.mongodb.com/).
2. Navigate to **Atlas Search** -> **Create Search Index** -> **JSON Editor**.
3. Select database `mini_rag` and collection `document_chunks`.
4. Name the index `vector_index`.
5. Paste the following JSON definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "document_id"
    },
    {
      "type": "filter",
      "path": "owner_id"
    }
  ]
}
```

> **Note**: If `ATLAS_VECTOR_INDEX_NAME` is not configured or the index is still building, the system automatically falls back to pure Python cosine-similarity search filtered by `document_id` and `owner_id`.

---

## 4. API Endpoints

### 1. `GET /health`
Returns service and database connectivity health.
```json
{
  "status": "healthy",
  "service": "MongoDB Ultra-Lightweight Document Q&A RAG Backend",
  "version": "1.0.0",
  "environment": "production",
  "database": "mongodb_atlas (connected)"
}
```

### 2. `POST /documents/upload`
Accepts `multipart/form-data` with `file` (`.pdf` or `.txt`).
- Validates file type and bounds upload size to `MAX_UPLOAD_MB`.
- Extracts text via PyMuPDF or Python native decoders.
- Splits into overlapping chunks and generates embeddings.
- Immediately releases raw bytes from memory.

### 3. `POST /documents/{document_id}/ask`
Accepts JSON:
```json
{
  "question": "What is the primary topic discussed in the executive summary?"
}
```
- Verifies document ownership (returns `403` if unauthorized, `404` if not found).
- Embeds question and retrieves top relevant chunks.
- Asks external LLM to answer strictly from the retrieved chunks.
- Returns answer, source chunk indices, similarity scores, and excerpts.

### 4. `POST /chat` (Conversational RAG Across Documents)
Unified conversational Q&A endpoint supporting cross-document search and multi-turn history.
Accepts JSON:
```json
{
  "message": "What is the return policy?",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "What is the return policy?"}
  ],
  "document_id": "optional_specific_document_id",
  "top_k": 4
}
```
- If `document_id` is omitted, queries across **all documents** uploaded by the user.
- If `messages` is provided, includes conversational history for multi-turn dialogue while strictly grounding answers in document excerpts.
- Returns answer, cited sources with document filenames, and retrieval metadata.

### 5. `GET /documents`
Lists all documents belonging to the authenticated user.

### 6. `DELETE /documents/{document_id}`
Deletes the document record and cascades deletion to all associated chunks in MongoDB.

---

## 5. Security & Multi-Tenancy

- **Verified Ownership**: Every query and deletion explicitly checks `owner_id == current_user.id`.
- **API Key Authentication**: When `AUTH_API_KEY` is configured, incoming requests must provide `Authorization: Bearer <KEY>` or `X-API-Key: <KEY>`.
- **Safe Development Mode**: When `ALLOW_DEV_AUTH=true` (default in dev), clients can provide `X-User-Id` or Bearer tokens to simulate multiple users in development and automated testing.
- **Safe Error Reporting**: Database passwords, connection strings, API keys, and full document contents are never logged or exposed in HTTP responses.
- **Streaming Upload Guard**: Uploaded files are read in small 64KB buffers. If total bytes exceed `MAX_UPLOAD_MB`, the request is aborted immediately with `413 Request Entity Too Large`.

---

## 6. Environment Variables

Create a `.env` file in the root directory (see `.env.example`):

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=mini_rag
ATLAS_VECTOR_INDEX_NAME=vector_index
MONGODB_TIMEOUT_MS=5000

# External Embedding API (OpenAI-compatible)
EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_TIMEOUT_SECONDS=30.0

# External LLM API (OpenAI-compatible)
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=your-llm-api-key
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=45.0

# RAG Configuration
MAX_UPLOAD_MB=10
CHUNK_SIZE=800
CHUNK_OVERLAP=120
MAX_CHUNKS=500
TOP_K=5

# Security & CORS
AUTH_API_KEY=
ALLOW_DEV_AUTH=true
ALLOWED_ORIGINS=*
ENVIRONMENT=production
```

---

## 7. Local Run Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at: `http://localhost:8000/docs`

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

---

## 8. Example cURL Commands

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Upload a Document (TXT or PDF)
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "X-User-Id: user_alice" \
  -F "file=@sample_policy.txt"
```
Response:
```json
{
  "document_id": "67104b281cfc84bc2d398e01",
  "filename": "sample_policy.txt",
  "chunk_count": 3,
  "status": "ready"
}
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/documents/67104b281cfc84bc2d398e01/ask" \
  -H "X-User-Id: user_alice" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the policy regarding remote work?"}'
```
Response:
```json
{
  "answer": "Employees may work remotely up to two days per week with manager approval.",
  "document_id": "67104b281cfc84bc2d398e01",
  "filename": "sample_policy.txt",
  "sources": [
    {
      "chunk_index": 1,
      "score": 0.8842,
      "excerpt": "Section 4: Remote Work Policy. Employees may work remotely up to two days..."
    }
  ],
  "retrieval_metadata": {
    "chunks_retrieved": 2,
    "top_k": 5
  }
}
```

### List Documents
```bash
curl -X GET "http://localhost:8000/documents" \
  -H "X-User-Id: user_alice"
```

### Delete a Document
```bash
curl -X DELETE "http://localhost:8000/documents/67104b281cfc84bc2d398e01" \
  -H "X-User-Id: user_alice"
```

---

## 9. Deployment Instructions

### A. Deploying to Render (Recommended)

1. Push your repository to GitHub / GitLab.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New** -> **Web Service**.
3. Connect your repository.
4. Render automatically detects `render.yaml` or you can manually configure:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
5. In **Environment Variables**, add:
   - `MONGODB_URI`
   - `EMBEDDING_API_KEY`
   - `LLM_API_KEY`
   - `AUTH_API_KEY`
6. Click **Deploy Web Service**. Render performs health checks on `/health`.

#### MongoDB Atlas Network Access on Render:
In MongoDB Atlas -> **Network Access** -> **IP Access List**:
- Because Render instances on starter/free tiers use dynamic IPs, add `0.0.0.0/0` (Allow Access from Anywhere) or configure a static outbound IP with a paid Render service. Protect your cluster with a strong, auto-generated MongoDB user password.

---

### B. Deploying to Vercel (Serverless Python Functions)

The repository includes `vercel.json` and `api/index.py` configured for Vercel's Python runtime.

#### Important Vercel Serverless Limitations:
1. **Execution Time Limits**: Vercel Hobby tier caps function execution to 10 seconds (Pro tier up to 60s). Large documents with dozens of chunks may exceed this window during embedding generation. Keep `MAX_CHUNKS` bounded.
2. **Request Body Size**: Vercel limits serverless function request bodies to 4.5MB. For documents larger than 4.5MB, implement a direct pre-signed upload to cloud storage (e.g. S3 / Cloudflare R2).
3. **Ephemeral Filesystem**: The local filesystem on Vercel is strictly read-only except for temporary files in `/tmp` (which are purged between invocations). The backend stores all metadata and vectors in MongoDB Atlas, ensuring 100% data persistence.
4. **Cold Starts**: PyMongo maintains connection pools. In a serverless environment, connections can be recreated on cold starts. Keep `MONGODB_TIMEOUT_MS` low (e.g., 5000ms).

To deploy to Vercel:
```bash
npm i -g vercel
vercel
```

---

## 10. Remaining Limitations & Operating Guidance

1. **Scanned Images**: PDFs containing scanned images without embedded text layers require OCR (e.g., Tesseract). The backend purposefully rejects image-only PDFs to prevent massive memory and CPU consumption.
2. **Provider Pricing & Quotas**: This backend does not host local LLM inference engines. External API usage (OpenAI, Groq, Together, DeepSeek) is subject to each provider's billing rates and rate limits.
3. **Batch Embedding Size**: Very large documents split into hundreds of chunks should be uploaded in moderate file sizes (recommended <= 10MB) to stay within embedding API batch payload constraints.
