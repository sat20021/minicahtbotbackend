"""
Custom text chunking utility using Python standard library.
No heavy frameworks like LangChain or LlamaIndex.
"""
from typing import List, Dict, Any


def split_text_into_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    max_chunks: int = 500,
) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks of approximately chunk_size characters,
    prioritizing natural paragraph and sentence boundaries.

    Args:
        text: Input raw text string.
        chunk_size: Target characters per chunk (> 0).
        chunk_overlap: Characters of overlap between consecutive chunks (0 <= chunk_overlap < chunk_size).
        max_chunks: Upper safety limit for chunks to bound memory and API cost.

    Returns:
        List of dicts: [{'index': int, 'text': str, 'char_count': int}]
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    # Normalize newlines while preserving paragraphs
    lines = [line.strip() for line in text.splitlines()]
    normalized_text = "\n".join(line for line in lines if line)

    if len(normalized_text) <= chunk_size:
        return [{"index": 0, "text": normalized_text, "char_count": len(normalized_text)}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    total_len = len(normalized_text)
    step = chunk_size - chunk_overlap

    while start < total_len and len(chunks) < max_chunks:
        end = min(start + chunk_size, total_len)

        # If not at the end of text, find the best natural split boundary
        if end < total_len:
            window = normalized_text[start:end]
            best_boundary = -1

            # Priority 1: Double newline (paragraph)
            p_break = window.rfind("\n\n")
            if p_break != -1 and p_break > (chunk_size // 3):
                best_boundary = p_break + 2

            # Priority 2: Sentence terminal (. ! ? followed by space or newline)
            if best_boundary == -1:
                for punct in [". ", "? ", "! ", ".\n", "!\n", "?\n"]:
                    s_break = window.rfind(punct)
                    if s_break != -1 and s_break > (chunk_size // 3):
                        best_boundary = max(best_boundary, s_break + len(punct))

            # Priority 3: Word boundary (space or newline)
            if best_boundary == -1:
                w_break = window.rfind(" ")
                if w_break != -1 and w_break > (chunk_size // 3):
                    best_boundary = w_break + 1

            if best_boundary != -1:
                end = start + best_boundary

        chunk_str = normalized_text[start:end].strip()
        if chunk_str:
            chunks.append({
                "index": len(chunks),
                "text": chunk_str,
                "char_count": len(chunk_str),
            })

        # Slide window forward
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks
