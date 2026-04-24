"""LLM-powered query rewriting and summary generation using Gemini Flash."""

from __future__ import annotations

from google import genai


def _call(api_key: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )
    return response.text.strip()


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
    Ask Gemini to write a unified, user-friendly narrative summarising both
    result sets and the latent themes they activated.
    """
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

    prompt = (
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
        "Bullet lines start with '- '. No markdown headers. Be warm and specific.\n\n"
        f'User query: "{original_query}"\n\n'
        f"Top latent themes activated:\n{fmt_dims(original_dims)}\n\n"
        f"Standard search — top results:\n{fmt_games(original_results)}\n\n"
        f'AI rewrote query to: "{rewritten_query}"\n'
        f"New themes activated:\n{fmt_dims(rewritten_dims)}\n\n"
        f"AI-enhanced search — top results:\n{fmt_games(rag_results)}\n\n"
        "Write the summary:"
    )

    try:
        return _call(api_key, prompt)
    except Exception:
        return ""


def rewrite_query(
    original_query: str,
    latent_dims: list[dict],
    api_key: str,
) -> str:
    """
    Ask Gemini to rewrite a natural-language query so it retrieves better
    results from the TF-IDF/SVD board-game index.

    Returns the rewritten query string, or the original on any failure.
    """
    sorted_dims = sorted(latent_dims, key=lambda d: abs(d.get("activation", 0)), reverse=True)
    dims_text = "\n".join(
        f"- [{d['activation']:+.3f}] {d['label']}: {', '.join(t['term'] for t in d.get('terms', [])[:5])}"
        for d in sorted_dims
    )

    prompt = (
        "You are a board game search query optimizer. "
        "Rewrite the user's query so it retrieves more relevant board games "
        "from a TF-IDF/SVD index. Use specific mechanics, themes, and genre "
        "vocabulary drawn from the dimensions this query already activates. "
        "The activation score shows how strongly the query hits each dimension — "
        "focus your rewrite on the highest-scoring ones. "
        "Output ONLY the rewritten query — no explanation, no quotes, under 25 words.\n\n"
        f"Original query: {original_query}\n\n"
        f"Dimensions this query activates (activation score + key terms):\n{dims_text}\n\n"
        "Rewritten query:"
    )

    try:
        return _call(api_key, prompt) or original_query
    except Exception:
        return original_query
