"""LLM-powered query rewriting for improved board-game retrieval."""

from __future__ import annotations


def rewrite_query(
    original_query: str,
    latent_dims: list[dict],
    api_key: str,
) -> str:
    """
    Ask the LLM to rewrite a natural-language query so it retrieves better
    results from the TF-IDF/SVD board-game index.

    Returns the rewritten query string, or the original on any failure.
    """
    from infosci_spark_client import LLMClient

    # Sort by activation strength so the most relevant dims appear first
    sorted_dims = sorted(latent_dims, key=lambda d: abs(d.get("activation", 0)), reverse=True)
    dims_text = "\n".join(
        f"- [{d['activation']:+.3f}] {d['label']}: {', '.join(t['term'] for t in d.get('terms', [])[:5])}"
        for d in sorted_dims
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a board game search query optimizer. "
                "Rewrite the user's query so it retrieves more relevant board games "
                "from a TF-IDF/SVD index. Use specific mechanics, themes, and genre "
                "vocabulary drawn from the dimensions this query already activates. "
                "The activation score shows how strongly the query hits each dimension — "
                "focus your rewrite on the highest-scoring ones. "
                "Output ONLY the rewritten query — no explanation, no quotes, under 25 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query: {original_query}\n\n"
                f"Dimensions this query activates (activation score + key terms):\n{dims_text}\n\n"
                "Rewritten query:"
            ),
        },
    ]

    client = LLMClient(api_key=api_key)
    text = ""
    try:
        for chunk in client.chat(messages, stream=True):
            if chunk.get("content"):
                text += chunk["content"]
    except Exception:
        return original_query

    return text.strip() or original_query
