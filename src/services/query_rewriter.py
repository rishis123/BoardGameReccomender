"""LLM-powered query rewriting and summary generation using Gemini Flash."""

from __future__ import annotations

from google import genai
from google.genai import types


def _call(api_key: str, system: str, user: str) -> str:
    print(f"[_call] api_key present: {bool(api_key)}, model: gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system),
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

    system = (
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
    )

    user = (
        f'User query: "{original_query}"\n\n'
        f"Top latent themes activated:\n{fmt_dims(original_dims)}\n\n"
        f"Standard search — top results:\n{fmt_games(original_results)}\n\n"
        f'AI rewrote query to: "{rewritten_query}"\n'
        f"New themes activated:\n{fmt_dims(rewritten_dims)}\n\n"
        f"AI-enhanced search — top results:\n{fmt_games(rag_results)}"
    )

    try:
        return _call(api_key, system, user)
    except Exception as e:
        print(f"[generate_summary error] {e}")
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

    system = (
        "You are a board game search query optimizer. Your job is to rewrite a user's query "
        "into a short list of precise board game vocabulary terms that will retrieve better results "
        "from a TF-IDF/SVD index. Focus on the latent dimensions with the highest activation scores — "
        "pull the most relevant terms from those dimensions and combine them with any strong theme words "
        "from the original query. Output ONLY the rewritten query. No explanation, no quotes, no preamble. "
        "Under 25 words. Think: mechanic names, theme words, genre terms — not natural language sentences."
    )

    user = (
        f"Original query: {original_query}\n\n"
        f"Latent dimensions this query activates (sorted by strength):\n{dims_text}\n\n"
        "Example of good output: 'resource trading settlement building hex tile strategy'\n"
        "Example of bad output: 'Games similar to Catan but with shorter play time'\n\n"
        "Rewritten query (space-separated terms only, under 25 words, NO sentences, NO 'games similar to'):"
    )

    try:
        return _call(api_key, system, user) or original_query
    except Exception as e:
        print(f"[rewrite_query error] {e}")
        return original_query
