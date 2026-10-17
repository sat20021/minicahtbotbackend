"""
Application configuration for the MongoDB-based mini RAG backend.
Loads settings from environment variables using python-dotenv.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Settings:
    # Project info
    PROJECT_NAME: str = "MongoDB Ultra-Lightweight Document Q&A RAG Backend"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # CORS
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

    # MongoDB Atlas
    MONGODB_URI: Optional[str] = os.getenv("MONGODB_URI")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "mini_rag")
    ATLAS_VECTOR_INDEX_NAME: Optional[str] = os.getenv("ATLAS_VECTOR_INDEX_NAME", None)
    MONGODB_TIMEOUT_MS: int = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))

    # Upload & Chunking configuration
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "500"))

    # External AI Provider Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Determine default provider (Gemini if GEMINI_API_KEY is present or specified)
    _use_gemini: bool = bool(
        os.getenv("GEMINI_API_KEY")
        or "generativelanguage.googleapis.com" in os.getenv("LLM_API_URL", "")
        or "generativelanguage.googleapis.com" in os.getenv("EMBEDDING_API_URL", "")
    )

    # Embedding API (HTTPX direct)
    EMBEDDING_API_URL: str = os.getenv(
        "EMBEDDING_API_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/v1/embeddings"
        if _use_gemini
        else "https://api.openai.com/v1/embeddings",
    )
    EMBEDDING_API_KEY: Optional[str] = (
        os.getenv("EMBEDDING_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-004" if _use_gemini else "text-embedding-3-small",
    )
    EMBEDDING_DIM: int = int(
        os.getenv("EMBEDDING_DIM", "768" if _use_gemini else "1536")
    )
    EMBEDDING_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30.0"))

    # LLM API (HTTPX direct)
    LLM_API_URL: str = os.getenv(
        "LLM_API_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
        if _use_gemini
        else "https://api.openai.com/v1/chat/completions",
    )
    LLM_API_KEY: Optional[str] = (
        os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "gemini-1.5-flash" if _use_gemini else "gpt-4o-mini",
    )
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Authentication
    AUTH_API_KEY: Optional[str] = os.getenv("AUTH_API_KEY")
    ALLOW_DEV_AUTH: bool = os.getenv("ALLOW_DEV_AUTH", "true").lower() in ("true", "1", "yes")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


settings = Settings()
