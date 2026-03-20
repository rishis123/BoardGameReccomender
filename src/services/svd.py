"""
Explainability layer – Truncated SVD latent-dimension projections.

Provides 3D scatter data and latent-dimension metadata for the Sensory Map.
"""

from __future__ import annotations

import numpy as np

from services.index_store import IndexStore

# Human-readable labels assigned to latent dimensions based on
# the dominant molecules that load onto each axis.
DIMENSION_LABELS = [
    "Fruity / Ester",
    "Sulfuric / Pungent",
    "Roasted / Nutty",
    "Green / Fresh",
    "Floral / Terpene",
    "Citrus / Aldehyde",
    "Dairy / Buttery",
    "Smoky / Phenolic",
    "Sweet / Caramel",
    "Earthy / Mushroom",
    "Herbal / Minty",
    "Spicy / Warm",
    "Umami / Savory",
    "Oceanic / Briny",
    "Woody / Resinous",
    "Tropical / Lactone",
    "Maillard / Toasty",
    "Balsamic / Vinegar",
    "Waxy / Fatty",
    "Fermented / Yeasty",
]


def get_sensory_map(
    store: IndexStore,
    dims: tuple[int, int, int] = (0, 1, 2),
    category: str | None = None,
) -> dict:
    """
    Return 3D projection data for the sensory scatter plot.

    Returns {points: [...], dimensions: [...]}.
    """
    if store.svd_embeddings is None:
        return {"points": [], "dimensions": []}

    n_dims = store.svd_embeddings.shape[1]
    d0, d1, d2 = [min(d, n_dims - 1) for d in dims]

    points = []
    for idx, iid in enumerate(store.ingredient_ids):
        cat = store.ingredient_categories.get(iid, "Unknown")
        if category and cat.lower() != category.lower():
            continue
        emb = store.svd_embeddings[idx]
        points.append({
            "id": iid,
            "name": store.ingredient_names.get(iid, str(iid)),
            "category": cat,
            "x": round(float(emb[d0]), 6),
            "y": round(float(emb[d1]), 6),
            "z": round(float(emb[d2]), 6),
        })

    dim_info = []
    for d in (d0, d1, d2):
        label = DIMENSION_LABELS[d] if d < len(DIMENSION_LABELS) else f"Dimension {d}"
        top_mols = store.svd_top_molecules[d] if d < len(store.svd_top_molecules) else []
        explained = store.svd_explained[d] if d < len(store.svd_explained) else 0.0
        dim_info.append({
            "index": d,
            "label": label,
            "explained_variance": round(explained, 6),
            "top_molecules": top_mols,
        })

    return {"points": points, "dimensions": dim_info}


def get_latent_neighbours(
    store: IndexStore,
    ingredient_id: int,
    k: int = 10,
) -> list[dict]:
    """Find nearest neighbours in SVD latent space (Euclidean distance)."""
    if store.svd_embeddings is None:
        return []

    row = store.id_to_row(ingredient_id)
    if row is None:
        return []

    vec = store.svd_embeddings[row]
    dists = np.linalg.norm(store.svd_embeddings - vec, axis=1)

    ranked = np.argsort(dists)
    results = []
    for idx in ranked:
        iid = store.ingredient_ids[idx]
        if iid == ingredient_id:
            continue
        results.append({
            "id": iid,
            "name": store.ingredient_names.get(iid, str(iid)),
            "category": store.ingredient_categories.get(iid, "Unknown"),
            "distance": round(float(dists[idx]), 6),
        })
        if len(results) >= k:
            break

    return results
