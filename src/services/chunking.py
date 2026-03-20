"""
Text chunking utilities for the RAG pipeline.

Splits RecipeNLG and FlavorDB metadata into retrieval-ready passages.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_frac: float = 0.10,
) -> list[str]:
    """
    Split text into overlapping chunks (word-level tokenisation).

    A word-level approximation is used for speed; the resulting chunks
    average ~400 sub-word tokens which stays well within context limits.
    """
    words = text.split()
    if not words:
        return []

    stride = max(1, int(max_tokens * (1 - overlap_frac)))
    chunks: list[str] = []
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + max_tokens])
        if chunk:
            chunks.append(chunk)
        if start + max_tokens >= len(words):
            break
    return chunks
