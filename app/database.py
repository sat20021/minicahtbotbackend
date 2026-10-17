"""
Synchronous MongoDB database integration using PyMongo.
Manages connections to MongoDB Atlas, collection access, and index creation.
Provides an in-memory fallback for local unit tests and environments without live MongoDB.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from bson import ObjectId
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

from app.config import settings

logger = logging.getLogger(__name__)

_mongo_client: Optional[MongoClient] = None
_db = None


class DatabaseError(Exception):
    """Raised for database connectivity or execution errors."""
    pass



# -----------------------------------------------------------------------------
# In-Memory Fallback Store (for unit testing and dev when MONGODB_URI is absent)
# -----------------------------------------------------------------------------
class InMemoryStore:
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []

    def reset(self):
        self.documents.clear()
        self.chunks.clear()

    def insert_document(self, doc_data: Dict[str, Any]) -> str:
        doc_id = str(ObjectId())
        doc_data["_id"] = ObjectId(doc_id)
        self.documents[doc_id] = doc_data
        return doc_id

    def insert_chunks(self, chunks_data: List[Dict[str, Any]]) -> int:
        for c in chunks_data:
            c["_id"] = ObjectId()
            self.chunks.append(c)
        return len(chunks_data)

    def find_document(self, doc_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        if owner_id and doc["owner_id"] != owner_id:
            return None
        return doc

    def list_documents(self, owner_id: str) -> List[Dict[str, Any]]:
        docs = [d for d in self.documents.values() if d["owner_id"] == owner_id]
        return sorted(docs, key=lambda x: x["created_at"], reverse=True)

    def delete_document(self, doc_id: str, owner_id: str) -> bool:
        doc = self.documents.get(doc_id)
        if not doc or doc["owner_id"] != owner_id:
            return False
        del self.documents[doc_id]
        doc_oid = ObjectId(doc_id)
        self.chunks = [c for c in self.chunks if c["document_id"] != doc_oid]
        return True

    def find_chunks_for_document(self, doc_id: str, owner_id: str) -> List[Dict[str, Any]]:
        doc_oid = ObjectId(doc_id)
        return [
            c for c in self.chunks
            if c["document_id"] == doc_oid and c["owner_id"] == owner_id
        ]

    def find_chunks_for_owner(self, owner_id: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if doc_id:
            try:
                doc_oid = ObjectId(doc_id)
                return [
                    c for c in self.chunks
                    if c["document_id"] == doc_oid and c["owner_id"] == owner_id
                ]
            except Exception:
                return []
        return [
            c for c in self.chunks
            if c["owner_id"] == owner_id
        ]



in_memory_store = InMemoryStore()


# -----------------------------------------------------------------------------
# PyMongo Connection Management
# -----------------------------------------------------------------------------
def get_mongo_client() -> Optional[MongoClient]:
    """Returns the singleton PyMongo client."""
    global _mongo_client
    if _mongo_client is None and settings.MONGODB_URI:
        try:
            _mongo_client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                appname="mini_rag_backend",
            )
            # Test connectivity
            _mongo_client.admin.command("ping")
            logger.info("Connected successfully to MongoDB database.")
        except (PyMongoError, Exception) as e:
            logger.warning("Could not connect to MongoDB database: %s. Using in-memory fallback.", str(e))
            _mongo_client = None
    return _mongo_client


def get_database():
    """Returns the database instance or in-memory fallback indicator."""
    client = get_mongo_client()
    if client is not None:
        return client[settings.MONGODB_DATABASE]
    return None


def ensure_indexes():
    """Creates required indexes on documents and document_chunks collections."""
    db = get_database()
    if db is None:
        return

    try:
        # documents indexes: owner_id, created_at
        db.documents.create_index([("owner_id", pymongo.ASCENDING)])
        db.documents.create_index([("created_at", pymongo.DESCENDING)])

        # document_chunks indexes: document_id, owner_id, (document_id, chunk_index)
        db.document_chunks.create_index([("document_id", pymongo.ASCENDING)])
        db.document_chunks.create_index([("owner_id", pymongo.ASCENDING)])
        db.document_chunks.create_index([
            ("document_id", pymongo.ASCENDING),
            ("chunk_index", pymongo.ASCENDING),
        ])
        logger.info("MongoDB standard indexes ensured on 'documents' and 'document_chunks'.")
    except Exception as e:
        logger.warning("Failed to create MongoDB indexes: %s", str(e))


def check_db_health() -> Dict[str, str]:
    """Lightweight health check without revealing sensitive connection strings."""
    if not settings.MONGODB_URI:
        return {"status": "ok", "mode": "in-memory (MONGODB_URI not configured)"}

    try:
        client = get_mongo_client()
        if client is None:
            return {"status": "degraded", "mode": "disconnected (using in-memory fallback)"}
        client.admin.command("ping")
        return {"status": "ok", "mode": "mongodb (connected)"}
    except Exception as e:
        return {"status": "error", "mode": f"connection_failure: {type(e).__name__}"}


def close_mongo_connection():
    """Closes the MongoDB connection safely."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed.")
