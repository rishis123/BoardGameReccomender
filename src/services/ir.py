"""Core board-game retrieval and recommendation utilities."""

from __future__ import annotations

import html
import re
from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.index_store import IndexStore

Method = Literal["svd", "tfidf"]


def _clean_query(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _empty_result() -> dict:
    return {
        "query": {},
        "recommendations": [],
        "latent_dimensions": [],
    }


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def _latent_dims(store: IndexStore, limit: int = 10) -> list[dict]:
    dims = []
    for d in store.svd_top_terms[:limit]:
        idx = d.get("index", 0)
        dims.append(
            {
                "index": idx,
                "label": d.get("label", f"Dimension {idx + 1}"),
                "explained_variance": float(store.svd_explained[idx]) if idx < len(store.svd_explained) else 0.0,
                "terms": d.get("terms", []),
            }
        )
    return dims


def _why_tags(store: IndexStore, query_svd: np.ndarray, row_idx: int, top_n: int = 3) -> list[dict]:
    game_vec = store.svd_embeddings[row_idx]
    contrib = query_svd.flatten() * game_vec
    pos_dims = np.argsort(contrib)[::-1]

    tags: list[dict] = []
    for dim_idx in pos_dims:
        score = float(contrib[dim_idx])
        if score <= 0:
            continue
        top = store.svd_top_terms[dim_idx] if dim_idx < len(store.svd_top_terms) else {}
        tags.append(
            {
                "index": int(dim_idx),
                "label": top.get("label", f"Dimension {dim_idx + 1}"),
                "activation": round(score, 4),
            }
        )
        if len(tags) >= top_n:
            break
    return tags


def recommend_games(
    store: IndexStore,
    query_text: str | None = None,
    seed_id: str | int | None = None,
    k: int = 10,
    method: Method = "svd",
) -> dict:
    if store.tfidf_matrix is None or store.svd_embeddings is None or store.vectorizer is None or store.svd_model is None:
        return _empty_result()

    source = None
    query_tfidf = None
    query_svd = None
    seed_row = None

    if seed_id is not None:
        seed_row = store.game_row(seed_id)
        if seed_row is None:
            return _empty_result()
        query_tfidf = store.tfidf_matrix[seed_row]
        query_svd = store.svd_embeddings[seed_row : seed_row + 1]
        source = store.games[seed_row]
    elif query_text and query_text.strip():
        cleaned = _clean_query(query_text)
        query_tfidf = store.vectorizer.transform([cleaned])
        query_svd = store.svd_model.transform(query_tfidf)
    else:
        return _empty_result()

    sims_tfidf = cosine_similarity(query_tfidf, store.tfidf_matrix).flatten()
    sims_svd = cosine_similarity(query_svd, store.svd_embeddings).flatten()

    if seed_row is not None:
        sims_tfidf[seed_row] = -1
        sims_svd[seed_row] = -1

    primary = sims_svd if method == "svd" else sims_tfidf
    rank_primary = np.argsort(primary)[::-1]
    rank_tfidf = _rank_positions(sims_tfidf)
    rank_svd = _rank_positions(sims_svd)

    recs: list[dict] = []
    for idx in rank_primary:
        if primary[idx] <= 0:
            continue
        game = store.games[idx]
        recs.append(
            {
                "id": game["id"],
                "name": game["name"],
                "snippet": game.get("snippet", ""),
                "year_published": game.get("year_published"),
                "average_rating": game.get("average_rating"),
                "users_rated": game.get("users_rated", 0),
                "category": game.get("category", ""),
                "mechanic": game.get("mechanic", ""),
                "score_svd": round(float(sims_svd[idx]), 4),
                "score_tfidf": round(float(sims_tfidf[idx]), 4),
                "rank_svd": int(rank_svd[idx]),
                "rank_tfidf": int(rank_tfidf[idx]),
                "why_tags": _why_tags(store, query_svd, idx),
            }
        )
        if len(recs) >= k:
            break

    query_payload = {
        "method": method,
        "text": query_text if query_text else None,
        "seed": source,
    }

    return {
        "query": query_payload,
        "recommendations": recs,
        "latent_dimensions": _latent_dims(store, limit=10),
    }


def get_latent_dimensions(store: IndexStore, limit: int = 10) -> list[dict]:
    return _latent_dims(store, limit=limit)
