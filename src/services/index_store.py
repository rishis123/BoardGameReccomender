"""
Artifact loader – reads pre-computed indices from data/artifacts/ into memory.

Loaded once at app startup and shared across all request handlers.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse


class IndexStore:
    """Singleton-style container for all pre-computed artifacts."""

    def __init__(self, artifacts_dir: str | Path | None = None):
        if artifacts_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            artifacts_dir = project_root / "data" / "artifacts"
        self.dir = Path(artifacts_dir)
        self._loaded = False

        # Populated by load()
        self.tfidf_matrix: sparse.csr_matrix | None = None
        self.ingredient_ids: list[int] = []
        self.ingredient_names: dict[int, str] = {}
        self.ingredient_categories: dict[int, str] = {}
        self.mol_vocab: dict[str, int] = {}
        self.pmi_edges: list[dict] = []
        self.svd_embeddings: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None
        self.svd_explained: list[float] = []
        self.svd_top_molecules: list[list[dict]] = []
        self.chunk_tfidf_matrix: sparse.csr_matrix | None = None
        self.chunk_ids: list[int] = []
        self.chunk_vocab: dict[str, int] = {}
        self.chunk_features: list[str] = []

    # ------------------------------------------------------------------

    def load(self):
        if self._loaded:
            return self
        if not self.dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: {self.dir}\n"
                "Run: python scripts/build_indices.py"
            )

        def _json(name):
            p = self.dir / name
            if p.exists():
                with open(p) as f:
                    return json.load(f)
            return None

        def _npy(name):
            p = self.dir / name
            return np.load(p) if p.exists() else None

        def _npz(name):
            p = self.dir / name
            return sparse.load_npz(p) if p.exists() else None

        self.tfidf_matrix = _npz("tfidf_matrix.npz")
        self.ingredient_ids = _json("ingredient_ids.json") or []
        raw_names = _json("ingredient_names.json") or {}
        self.ingredient_names = {int(k): v for k, v in raw_names.items()}
        raw_cats = _json("ingredient_categories.json") or {}
        self.ingredient_categories = {int(k): v for k, v in raw_cats.items()}
        self.mol_vocab = _json("tfidf_vocab.json") or {}
        self.pmi_edges = _json("pmi_edges.json") or []
        self.svd_embeddings = _npy("svd_embeddings.npy")
        self.svd_components = _npy("svd_components.npy")
        self.svd_explained = _json("svd_explained.json") or []
        self.svd_top_molecules = _json("svd_top_molecules.json") or []

        self.chunk_tfidf_matrix = _npz("chunk_tfidf_matrix.npz")
        self.chunk_ids = _json("chunk_ids.json") or []
        self.chunk_vocab = _json("chunk_tfidf_vocab.json") or {}
        self.chunk_features = _json("chunk_tfidf_features.json") or []

        self._loaded = True
        n_ing = len(self.ingredient_ids)
        n_edges = len(self.pmi_edges)
        n_chunks = len(self.chunk_ids)
        print(f"IndexStore loaded: {n_ing} ingredients, {n_edges} PMI edges, {n_chunks} RAG chunks")
        return self

    # convenience ----------------------------------------------------------

    def id_to_row(self, ingredient_id: int) -> int | None:
        try:
            return self.ingredient_ids.index(ingredient_id)
        except ValueError:
            return None

    def name_to_id(self, name: str) -> int | None:
        name_lower = name.lower()
        for iid, n in self.ingredient_names.items():
            if n.lower() == name_lower:
                return iid
        return None

    def search_names(self, query: str, limit: int = 20) -> list[dict]:
        q = query.lower()
        results = []
        for iid, name in self.ingredient_names.items():
            if q in name.lower():
                results.append({
                    "id": iid,
                    "name": name,
                    "category": self.ingredient_categories.get(iid, "Unknown"),
                })
        results.sort(key=lambda x: (not x["name"].lower().startswith(q), x["name"]))
        return results[:limit]
