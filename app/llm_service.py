"""
External LLM API integration using HTTPX.
Strictly grounds answers in retrieved document context chunks.
"""
import time
from typing import List, Dict, Any, Optional
import httpx

from app.config import settings


class LLMError(Exception):
    """Raised when external LLM API fails or response format is unexpected."""
    pass


SYSTEM_PROMPT = """You are a strictly grounded document question-answering assistant.
Answer the user's question using ONLY the provided document excerpts below.

Rules:
1. Ground your answer solely and strictly in the provided Context excerpts.
2. If the excerpts do not contain sufficient information to answer the question, state:
   "The provided document does not contain information to answer this question."
3. Do not assume, speculate, extrapolate, or use outside knowledge.
4. Keep the response concise, factual, and accurate."""


def format_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with index tags and document names for grounding."""
    lines = []
    for c in chunks:
        idx = c.get("chunk_index", 0)
        filename = c.get("filename", "")
        header = f"--- [Document: {filename} | Chunk {idx}] ---" if filename else f"--- [Chunk {idx}] ---"
        text = c.get("text", "").strip()
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def generate_grounded_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    api_url: str = settings.LLM_API_URL,
    api_key: Optional[str] = settings.LLM_API_KEY,
    model: str = settings.LLM_MODEL,
    timeout: float = settings.LLM_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Calls the external LLM via HTTPX to generate a strictly grounded answer,
    optionally incorporating prior conversation history.
    """
    if not chunks:
        return {
            "answer": "The provided document does not contain information to answer this question.",
            "usage": None,
        }

    context_str = format_context_block(chunks)
    prompt = f"Context excerpts:\n{context_str}\n\nQuestion: {question}"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if "googleapis.com" in api_url or api_key.startswith("AIza"):
            headers["x-goog-api-key"] = api_key

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }


    max_retries = 2
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                try:
                    # 1. Standard OpenAI-compatible format
                    if "choices" in data and len(data["choices"]) > 0:
                        answer_text = data["choices"][0]["message"]["content"].strip()
                        usage = data.get("usage")
                        return {
                            "answer": answer_text,
                            "usage": usage,
                        }

                    # 2. Gemini native generateContent format
                    if "candidates" in data and len(data["candidates"]) > 0:
                        candidate = data["candidates"][0]
                        parts = candidate.get("content", {}).get("parts", [])
                        answer_text = "".join(p.get("text", "") for p in parts).strip()
                        usage = data.get("usageMetadata")
                        return {
                            "answer": answer_text,
                            "usage": usage,
                        }

                    raise LLMError(f"Malformed LLM API response: {data}")
                except (KeyError, IndexError) as err:
                    raise LLMError(f"Malformed LLM API response: {data}") from err

            elif response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            else:
                raise LLMError(f"LLM API returned HTTP {response.status_code}: {response.text[:200]}")

        except httpx.TimeoutException:
            last_err = LLMError(f"LLM request timed out after {timeout} seconds.")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise last_err

        except httpx.RequestError as e:
            last_err = LLMError(f"LLM network connection error: {str(e)}")
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise last_err

    if last_err:
        raise last_err
    raise LLMError("Failed to obtain answer from LLM API after retries.")
