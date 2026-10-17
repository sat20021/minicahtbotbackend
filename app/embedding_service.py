"""
External embedding API integration using HTTPX and pure Python cosine similarity.
No heavy ML frameworks or local models.
"""
import math
import time
from typing import List, Dict, Any, Optional
import httpx

from app.config import settings


class EmbeddingError(Exception):
    """Raised when external embedding API call fails or response is invalid."""
    pass


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors using Python standard library.
    Returns float in range [-1.0, 1.0].
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def rank_chunks_by_similarity(
    query_vector: List[float],
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Ranks chunks by cosine similarity in memory.
    Each chunk dict must contain an 'embedding' key (list of floats).
    """
    scored = []
    for c in chunks:
        emb = c.get("embedding")
        if isinstance(emb, list) and emb:
            sim = cosine_similarity(query_vector, emb)
            item = dict(c)
            item["score"] = sim
            scored.append(item)

    # Sort descending by similarity score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_embeddings(
    texts: List[str],
    api_url: str = settings.EMBEDDING_API_URL,
    api_key: Optional[str] = settings.EMBEDDING_API_KEY,
    model: str = settings.EMBEDDING_MODEL,
    timeout: float = settings.EMBEDDING_TIMEOUT_SECONDS,
) -> List[List[float]]:
    """
    Calls external embedding API via HTTPX.
    Parses OpenAI-compatible response: {"data": [{"index": 0, "embedding": [...]}]}.
    """
    if not texts:
        return []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # Google endpoints also accept x-goog-api-key
        if "googleapis.com" in api_url or api_key.startswith("AIza"):
            headers["x-goog-api-key"] = api_key

    payload = {
        "input": texts,
        "model": model,
    }

    max_retries = 2
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                # 1. Standard OpenAI-compatible format
                if "data" in data and isinstance(data["data"], list):
                    sorted_items = sorted(data["data"], key=lambda x: x.get("index", 0))
                    return [item["embedding"] for item in sorted_items]

                # 2. Gemini native batchEmbedContents format
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    return [item["values"] for item in data["embeddings"]]

                # 3. Gemini native single embedContent format
                if "embedding" in data and isinstance(data["embedding"], dict) and "values" in data["embedding"]:
                    return [data["embedding"]["values"]]

                raise EmbeddingError(f"Unexpected embedding API response format: {data}")

            elif response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            else:
                raise EmbeddingError(
                    f"Embedding API returned HTTP {response.status_code}: {response.text[:200]}"
                )

        except httpx.TimeoutException:
            last_err = EmbeddingError(f"Embedding request timed out after {timeout} seconds.")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise last_err

        except httpx.RequestError as e:
            last_err = EmbeddingError(f"Embedding network connection error: {str(e)}")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise last_err

    if last_err:
        raise last_err
    raise EmbeddingError("Failed to obtain embeddings after retries.")


def generate_single_embedding(
    text: str,
    api_url: str = settings.EMBEDDING_API_URL,
    api_key: Optional[str] = settings.EMBEDDING_API_KEY,
    model: str = settings.EMBEDDING_MODEL,
) -> List[float]:
    """Generates embedding for a single text string (e.g. question)."""
    embeddings = generate_embeddings([text], api_url=api_url, api_key=api_key, model=model)
    if not embeddings:
        raise EmbeddingError("No embedding vector returned.")
    return embeddings[0]
