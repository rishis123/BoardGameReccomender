"""
RAG pipeline – retrieval-augmented generation for the Flavor Chemist chat.

Retrieves relevant recipe chunks and molecular profiles, then constructs
a strictly grounded prompt for the LLM.
"""

from __future__ import annotations

import duckdb
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.index_store import IndexStore
from services.ir import find_substitutes, get_ingredient_profile


def retrieve_chunks(
    store: IndexStore,
    query: str,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k recipe/metadata chunks relevant to the query
    using sparse TF-IDF cosine similarity.
    """
    if store.chunk_tfidf_matrix is None or not store.chunk_features:
        return []

    vectorizer = TfidfVectorizer(
        vocabulary={f: i for i, f in enumerate(store.chunk_features)},
        stop_words="english",
        sublinear_tf=True,
    )
    vectorizer.fit([""])  # no-op fit; vocabulary is already set
    q_vec = vectorizer.transform([query])

    sims = cosine_similarity(q_vec, store.chunk_tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_indices:
        if sims[idx] <= 0:
            break
        results.append({
            "chunk_id": store.chunk_ids[idx],
            "score": round(float(sims[idx]), 4),
        })

    return results


def fetch_chunk_texts(db_path: str, chunk_ids: list[int]) -> dict[int, str]:
    """Load chunk texts from DuckDB by ID."""
    if not chunk_ids:
        return {}
    con = duckdb.connect(db_path, read_only=True)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT id, chunk_text FROM recipe_chunks WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    con.close()
    return {r[0]: r[1] for r in rows}


def build_grounded_prompt(
    user_message: str,
    molecular_context: str,
    recipe_context: str,
) -> list[dict]:
    """
    Construct the chat messages list with strict grounding instructions.

    The system prompt enforces citation and uncertainty handling.
    """
    system_prompt = (
        "You are the AI Flavor Chemist, a molecular gastronomy assistant. "
        "Answer questions using ONLY the provided molecular profiles and recipe data. "
        "If scientific evidence for a pairing or claim is missing from the context, "
        "state clearly that you do not have sufficient evidence.\n\n"
        "RULES:\n"
        "1. Every factual claim must cite specific molecules or recipe sources from the context.\n"
        "2. Use chemical compound names when explaining flavor pairings.\n"
        "3. If the user asks about an ingredient not in the context, say so.\n"
        "4. Keep answers concise and scientifically grounded.\n"
    )

    context_block = ""
    if molecular_context:
        context_block += f"=== Molecular Profiles ===\n{molecular_context}\n\n"
    if recipe_context:
        context_block += f"=== Recipe Data ===\n{recipe_context}\n\n"

    if not context_block:
        context_block = "(No relevant molecular or recipe data found in the corpus.)\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{context_block}User question: {user_message}",
        },
    ]
    return messages


def gather_context(
    store: IndexStore,
    db_path: str,
    user_message: str,
) -> tuple[str, str, list[str]]:
    """
    Gather molecular and recipe context for the RAG prompt.

    Returns (molecular_context, recipe_context, citation_sources).
    """
    citations: list[str] = []
    molecular_parts: list[str] = []

    # Try to find ingredients mentioned in the query
    words = user_message.lower().split()
    matched_ids: list[int] = []
    for iid, name in store.ingredient_names.items():
        if name.lower() in user_message.lower():
            matched_ids.append(iid)

    for iid in matched_ids[:3]:
        profile = get_ingredient_profile(db_path, iid)
        if profile:
            mol_summary = ", ".join(
                f"{m['common_name']} ({m['flavor_profile']})"
                for m in profile.get("molecules", [])[:10]
                if m.get("common_name")
            )
            molecular_parts.append(
                f"Ingredient: {profile['name']} (Category: {profile.get('category', 'N/A')})\n"
                f"Molecules: {mol_summary}"
            )
            citations.append(f"FlavorDB:{profile['name']}")

        subs = find_substitutes(store, iid, k=5)
        if subs:
            sub_text = ", ".join(f"{s['name']} ({s['similarity']})" for s in subs)
            molecular_parts.append(f"Top substitutes for {store.ingredient_names.get(iid, '')}: {sub_text}")

    molecular_context = "\n\n".join(molecular_parts)

    # Retrieve recipe chunks
    chunk_results = retrieve_chunks(store, user_message, k=5)
    chunk_ids = [c["chunk_id"] for c in chunk_results]
    chunk_texts = fetch_chunk_texts(db_path, chunk_ids)
    recipe_parts = []
    for cr in chunk_results:
        text = chunk_texts.get(cr["chunk_id"], "")
        if text:
            recipe_parts.append(text)
            citations.append(f"RecipeNLG:chunk_{cr['chunk_id']}")
    recipe_context = "\n---\n".join(recipe_parts)

    return molecular_context, recipe_context, citations
