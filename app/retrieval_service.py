"""
Document chunk retrieval service.
Supports MongoDB Atlas Vector Search ($vectorSearch) with automatic fallback
to pure Python cosine similarity filtering by owner and document.
"""
import logging
from typing import List, Dict, Any, Optional
from bson import ObjectId
from pymongo.errors import PyMongoError, OperationFailure

from app.config import settings
from app.database import get_database, in_memory_store
from app.embedding_service import rank_chunks_by_similarity

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    owner_id: str,
    query_vector: List[float],
    document_id: Optional[str] = None,
    top_k: int = settings.TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant chunks for a specific owner, optionally filtered by document_id.
    If document_id is omitted or None, retrieves across all documents owned by owner_id.
    Attempts MongoDB Atlas Vector Search if configured; falls back to Python cosine similarity.
    """
    db = get_database()

    # 1. In-memory store fallback (used when MONGODB_URI is absent, e.g. tests)
    if db is None:
        chunks = in_memory_store.find_chunks_for_owner(owner_id=owner_id, doc_id=document_id)
        ranked = rank_chunks_by_similarity(query_vector, chunks, top_k=top_k)
        return [
            {
                "chunk_id": str(r["_id"]),
                "document_id": str(r.get("document_id", "")),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("text", ""),
                "score": round(float(r.get("score", 0.0)), 4),
            }
            for r in ranked
        ]

    doc_oid: Optional[ObjectId] = None
    if document_id:
        try:
            doc_oid = ObjectId(document_id)
        except Exception:
            return []

    # 2. Try Atlas Vector Search if index name is configured
    if settings.ATLAS_VECTOR_INDEX_NAME:
        try:
            search_filter: Dict[str, Any] = {"owner_id": owner_id}
            if doc_oid:
                search_filter["document_id"] = doc_oid

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.ATLAS_VECTOR_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": max(top_k * 10, 20),
                        "limit": top_k,
                        "filter": search_filter,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "document_id": 1,
                        "chunk_index": 1,
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            cursor = db.document_chunks.aggregate(pipeline)
            results = list(cursor)
            if results:
                logger.info("Retrieved %d chunks via Atlas Vector Search.", len(results))
                return [
                    {
                        "chunk_id": str(r["_id"]),
                        "document_id": str(r.get("document_id", "")),
                        "chunk_index": r.get("chunk_index", 0),
                        "text": r.get("text", ""),
                        "score": round(float(r.get("score", 0.0)), 4),
                    }
                    for r in results
                ]
        except OperationFailure as e:
            logger.warning("Atlas Vector Search failed or index not built yet (%s). Using Python fallback.", str(e))
        except Exception as e:
            logger.warning("Unexpected error during Atlas Vector Search: %s. Using Python fallback.", str(e))

    # 3. Pure Python Cosine-Similarity Fallback
    # Filter by owner_id (and document_id if specified), avoiding scanning entire database
    try:
        query_filter: Dict[str, Any] = {"owner_id": owner_id}
        if doc_oid:
            query_filter["document_id"] = doc_oid

        projection = {
            "_id": 1,
            "document_id": 1,
            "chunk_index": 1,
            "text": 1,
            "embedding": 1,
        }
        cursor = db.document_chunks.find(query_filter, projection)
        chunks = list(cursor)

        ranked = rank_chunks_by_similarity(query_vector, chunks, top_k=top_k)
        return [
            {
                "chunk_id": str(r["_id"]),
                "document_id": str(r.get("document_id", "")),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("text", ""),
                "score": round(float(r.get("score", 0.0)), 4),
            }
            for r in ranked
        ]
    except PyMongoError as e:
        logger.error("MongoDB query error while retrieving chunks: %s", str(e))
        return []

