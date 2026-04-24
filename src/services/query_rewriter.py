"""LLM-powered query rewriting for improved board-game retrieval."""

from __future__ import annotations


def generate_summary(
    original_query: str,
    original_dims: list[dict],
    original_results: list[dict],
    rewritten_query: str,
    rewritten_dims: list[dict],
    rag_results: list[dict],
    api_key: str,
) -> str:
    """
    Ask the LLM to write a unified, user-friendly narrative summarising both
    result sets and the latent themes they activated.
    """
    from infosci_spark_client import LLMClient

    def fmt_dims(dims: list[dict]) -> str:
        top = sorted(dims, key=lambda d: abs(d.get("activation", 0)), reverse=True)[:3]
        return "\n".join(
            f"- {d['label']} ({d['activation']:+.3f}): "
            f"{', '.join(t['term'] for t in d.get('terms', [])[:4])}"
            for d in top
        )

    def fmt_games(results: list[dict]) -> str:
        return "\n".join(
            f"- {r['name']} ({r.get('year_published') or '?'}): "
            f"{(r.get('snippet') or '')[:120].rstrip()}…"
            for r in results[:5]
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly board-game expert writing a concise recommendation summary. "
                "You have run two searches: a standard IR search and an AI-enhanced search with "
                "a rewritten query. Write the summary in this exact structure:\n\n"
                "One opening sentence explaining which key themes the query activated (plain English, no jargon).\n\n"
                "Standard picks:\n"
                "- **Game Name** — one sentence reason why it fits\n"
                "(repeat for top 3 standard results)\n\n"
                "One bridge sentence explaining what the AI noticed that the original query hinted at.\n\n"
                "AI-enhanced picks:\n"
                "- **Game Name** — one sentence reason why it fits\n"
                "(repeat for top 3 AI results)\n\n"
                "One closing sentence with a single top overall recommendation and why.\n\n"
                "Rules: Use exactly the format above. Bold game names with **Name**. "
                "Bullet lines start with '- '. No markdown headers. Be warm and specific."
            ),
        },
        {
            "role": "user",
            "content": (
                f'User query: "{original_query}"\n\n'
                f"Top latent themes activated:\n{fmt_dims(original_dims)}\n\n"
                f"Standard search — top results:\n{fmt_games(original_results)}\n\n"
                f'AI rewrote query to: "{rewritten_query}"\n'
                f"New themes activated:\n{fmt_dims(rewritten_dims)}\n\n"
                f"AI-enhanced search — top results:\n{fmt_games(rag_results)}\n\n"
                "Write the summary:"
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
        return ""

    return text.strip()


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
