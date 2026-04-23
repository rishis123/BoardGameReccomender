"""
Build board-game recommendation artifacts from data/database.sqlite.

Artifacts written to data/artifacts/:
  - tfidf_matrix.npz
  - tfidf_vectorizer.pkl
  - svd_model.pkl
  - svd_embeddings.npy
  - svd_components.npy
  - svd_explained.json
  - svd_top_terms.json
  - game_ids.json
  - games.json

Usage:
  python scripts/build_indices.py
"""

from __future__ import annotations

import html
import json
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "database.sqlite"
ART_DIR = PROJECT_ROOT / "data" / "artifacts"

RATINGS_THRESHOLD = 50
MAX_FEATURES = 50000
SVD_COMPONENTS = 80


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\\r", " ").replace("\\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def clean_multi(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = text.replace("|", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return " ".join(parts)


def load_games(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    query = """
    SELECT
      "game.id" AS game_id,
      "details.name" AS name,
      "details.description" AS description,
      "attributes.boardgamecategory" AS category,
      "attributes.boardgamemechanic" AS mechanic,
      "details.yearpublished" AS year_published,
      "stats.average" AS average_rating,
      "stats.usersrated" AS users_rated,
      "game.type" AS game_type,
      "details.thumbnail" AS thumbnail
    FROM BoardGames
    WHERE "game.type" = 'boardgame'
      AND COALESCE("stats.usersrated", 0) > ?
      AND "details.name" IS NOT NULL
      AND "details.description" IS NOT NULL
    """
    cur = con.execute(query, [RATINGS_THRESHOLD])
    rows = cur.fetchall()
    con.close()

    games: list[dict] = []
    for row in rows:
        game_id, name, description, category, mechanic, year_published, average_rating, users_rated, _, thumbnail = row
        name = clean_text(name)
        description = clean_text(description)
        category = clean_multi(category)
        mechanic = clean_multi(mechanic)

        if not name or not description:
            continue

        combined_text = f"{description} {category} {mechanic}".strip()
        if not combined_text:
            continue

        games.append(
            {
                "id": str(game_id),
                "name": name,
                "description": description,
                "category": category,
                "mechanic": mechanic,
                "year_published": int(year_published) if year_published is not None else None,
                "average_rating": float(average_rating) if average_rating is not None else None,
                "users_rated": int(users_rated) if users_rated is not None else 0,
                "combined_text": combined_text,
                "thumbnail": thumbnail or None,
            }
        )

    return games


def fit_models(games: list[dict]):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform([g["combined_text"] for g in games])

    k = min(SVD_COMPONENTS, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    if k < 2:
        raise RuntimeError("Not enough data to build SVD model. Need at least 3 games.")

    svd_model = TruncatedSVD(n_components=k, random_state=42)
    svd_embeddings = svd_model.fit_transform(tfidf_matrix)

    return vectorizer, tfidf_matrix, svd_model, svd_embeddings


def build_top_terms(svd_model: TruncatedSVD, feature_names: np.ndarray, per_dim: int = 10):
    top_terms: list[dict] = []
    for dim_idx, component in enumerate(svd_model.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:per_dim]
        terms = []
        for idx in top_idx:
            terms.append(
                {
                    "term": str(feature_names[idx]),
                    "loading": round(float(component[idx]), 6),
                }
            )

        positive_terms = [
            str(feature_names[i])
            for i in np.argsort(component)[::-1]
            if component[i] > 0
        ][:3]
        label = " / ".join(positive_terms) if positive_terms else f"Dimension {dim_idx + 1}"

        top_terms.append(
            {
                "index": dim_idx,
                "label": label,
                "terms": terms,
            }
        )
    return top_terms


def to_game_records(games: list[dict]) -> list[dict]:
    records: list[dict] = []
    for game in games:
        description = game["description"] or ""
        snippet = (description[:220] + "...") if len(description) > 220 else description

        records.append(
            {
                "id": game["id"],
                "name": game["name"],
                "description": description,
                "snippet": snippet,
                "category": game["category"],
                "mechanic": game["mechanic"],
                "year_published": game["year_published"],
                "average_rating": game["average_rating"],
                "users_rated": game["users_rated"],
                "thumbnail": game.get("thumbnail"),
            }
        )
    return records


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    ART_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading board games from SQLite...")
    games = load_games(DB_PATH)
    if not games:
        print("No boardgame rows found after filters. Aborting.")
        sys.exit(1)
    print(f"Loaded {len(games):,} games (type=boardgame, usersrated>{RATINGS_THRESHOLD})")

    print("[2/4] Building TF-IDF and SVD models...")
    vectorizer, tfidf_matrix, svd_model, svd_embeddings = fit_models(games)
    feature_names = vectorizer.get_feature_names_out()
    top_terms = build_top_terms(svd_model, feature_names)

    print("[3/4] Writing matrices/models...")
    sparse.save_npz(ART_DIR / "tfidf_matrix.npz", tfidf_matrix)
    np.save(ART_DIR / "svd_embeddings.npy", svd_embeddings)
    np.save(ART_DIR / "svd_components.npy", svd_model.components_)

    with open(ART_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ART_DIR / "svd_model.pkl", "wb") as f:
        pickle.dump(svd_model, f)

    with open(ART_DIR / "svd_explained.json", "w", encoding="utf-8") as f:
        json.dump([float(x) for x in svd_model.explained_variance_ratio_], f)
    with open(ART_DIR / "svd_top_terms.json", "w", encoding="utf-8") as f:
        json.dump(top_terms, f, ensure_ascii=False)

    print("[4/4] Writing metadata...")
    game_records = to_game_records(games)
    game_ids = [g["id"] for g in game_records]

    with open(ART_DIR / "game_ids.json", "w", encoding="utf-8") as f:
        json.dump(game_ids, f)
    with open(ART_DIR / "games.json", "w", encoding="utf-8") as f:
        json.dump(game_records, f, ensure_ascii=False)

    print(f"Done. Artifacts written to {ART_DIR}")


if __name__ == "__main__":
    main()
