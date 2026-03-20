"""
Core IR module – TF-IDF cosine similarity search for molecular substitutes.

Given a seed ingredient, returns ranked substitutes based on cosine similarity
of their TF-IDF molecule profiles.
"""

from __future__ import annotations

import duckdb
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from services.index_store import IndexStore


def find_substitutes(
    store: IndexStore,
    seed_id: int,
    k: int = 20,
    category: str | None = None,
    exclude_ids: set[int] | None = None,
) -> list[dict]:
    """
    Return top-k molecular substitutes for the seed ingredient.

    Each result contains: id, name, category, similarity, shared_molecules.
    """
    if store.tfidf_matrix is None:
        return []

    row = store.id_to_row(seed_id)
    if row is None:
        return []

    seed_vec = store.tfidf_matrix[row]
    sims = cosine_similarity(seed_vec, store.tfidf_matrix).flatten()

    exclude = exclude_ids or set()
    exclude.add(seed_id)

    ranked: list[tuple[int, float]] = []
    for idx in np.argsort(sims)[::-1]:
        iid = store.ingredient_ids[idx]
        if iid in exclude:
            continue
        if category and store.ingredient_categories.get(iid, "").lower() != category.lower():
            continue
        ranked.append((iid, float(sims[idx])))
        if len(ranked) >= k:
            break

    return [
        {
            "id": iid,
            "name": store.ingredient_names.get(iid, str(iid)),
            "category": store.ingredient_categories.get(iid, "Unknown"),
            "similarity": round(sim, 4),
        }
        for iid, sim in ranked
    ]


def get_shared_molecules(
    db_path: str,
    ingredient_a: int,
    ingredient_b: int,
    limit: int = 10,
) -> list[dict]:
    """Return molecules shared by two ingredients from the DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        """
        SELECT m.id, m.pubchem_id, m.common_name, m.flavor_profile
        FROM ingredient_molecules a
        JOIN ingredient_molecules b ON a.molecule_id = b.molecule_id
        JOIN molecules m ON m.id = a.molecule_id
        WHERE a.ingredient_id = ? AND b.ingredient_id = ?
        LIMIT ?
        """,
        [ingredient_a, ingredient_b, limit],
    ).fetchall()
    con.close()
    return [
        {
            "id": r[0],
            "pubchem_id": r[1],
            "common_name": r[2],
            "flavor_profile": r[3],
        }
        for r in rows
    ]


def get_ingredient_profile(
    db_path: str,
    ingredient_id: int,
) -> dict:
    """Return full molecular profile for an ingredient."""
    con = duckdb.connect(db_path, read_only=True)

    ing = con.execute(
        "SELECT id, name, category, scientific_name FROM ingredients WHERE id = ?",
        [ingredient_id],
    ).fetchone()
    if not ing:
        con.close()
        return {}

    mols = con.execute(
        """
        SELECT m.id, m.pubchem_id, m.common_name, m.flavor_profile
        FROM ingredient_molecules im
        JOIN molecules m ON m.id = im.molecule_id
        WHERE im.ingredient_id = ?
        ORDER BY m.common_name
        """,
        [ingredient_id],
    ).fetchall()
    con.close()

    return {
        "id": ing[0],
        "name": ing[1],
        "category": ing[2],
        "scientific_name": ing[3],
        "molecule_count": len(mols),
        "molecules": [
            {
                "id": m[0],
                "pubchem_id": m[1],
                "common_name": m[2],
                "flavor_profile": m[3],
            }
            for m in mols
        ],
    }


def precision_at_k(
    store: IndexStore,
    db_path: str,
    k: int = 10,
    sample_size: int = 100,
) -> float:
    """
    Evaluate Precision@k using TasteTrios compatibility labels.

    For each compatibility pair (ingredient_a, ingredient_b) marked as
    "Highly Compatible", check whether ingredient_b appears in the
    top-k results when searching for ingredient_a.
    """
    con = duckdb.connect(db_path, read_only=True)
    pairs = con.execute(
        """
        SELECT ingredient_a, ingredient_b
        FROM compatibility_pairs
        WHERE LOWER(compatibility_level) LIKE '%highly%'
        LIMIT ?
        """,
        [sample_size],
    ).fetchall()
    con.close()

    if not pairs:
        return 0.0

    hits = 0
    evaluated = 0
    for ing_a_name, ing_b_name in pairs:
        a_id = store.name_to_id(ing_a_name)
        b_id = store.name_to_id(ing_b_name)
        if a_id is None or b_id is None:
            continue
        results = find_substitutes(store, a_id, k=k)
        result_ids = {r["id"] for r in results}
        if b_id in result_ids:
            hits += 1
        evaluated += 1

    return hits / evaluated if evaluated > 0 else 0.0
