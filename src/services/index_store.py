"""Load board-game recommendation artifacts into memory."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse


class IndexStore:
    def __init__(self, artifacts_dir: str | Path | None = None):
        if artifacts_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            artifacts_dir = project_root / "data" / "artifacts"
        self.dir = Path(artifacts_dir)
        self._loaded = False

        self.tfidf_matrix: sparse.csr_matrix | None = None
        self.vectorizer = None
        self.svd_model = None
        self.svd_embeddings: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None
        self.svd_explained: list[float] = []
        self.svd_top_terms: list[dict] = []

        self.game_ids: list[str] = []
        self.games: list[dict] = []
        self.id_to_index: dict[str, int] = {}

    def load(self):
        if self._loaded:
            return self
        if not self.dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: {self.dir}\\n"
                "Run: python scripts/build_indices.py"
            )

        def _json(name):
            path = self.dir / name
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            return None

        self.tfidf_matrix = sparse.load_npz(self.dir / "tfidf_matrix.npz")
        self.svd_embeddings = np.load(self.dir / "svd_embeddings.npy")
        self.svd_components = np.load(self.dir / "svd_components.npy")

        with open(self.dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self.dir / "svd_model.pkl", "rb") as f:
            self.svd_model = pickle.load(f)

        self.svd_explained = _json("svd_explained.json") or []
        self.svd_top_terms = _json("svd_top_terms.json") or []
        self.game_ids = [str(x) for x in (_json("game_ids.json") or [])]
        self.games = _json("games.json") or []
        self.id_to_index = {gid: idx for idx, gid in enumerate(self.game_ids)}

        self._loaded = True
        print(f"IndexStore loaded: {len(self.game_ids)} games")
        return self

    def game_row(self, game_id: str | int) -> int | None:
        return self.id_to_index.get(str(game_id))

    def game_by_id(self, game_id: str | int) -> dict | None:
        row = self.game_row(game_id)
        if row is None:
            return None
        return self.games[row]

    def search_names(self, query: str, limit: int = 20) -> list[dict]:
        q = query.lower().strip()
        if not q:
            return []

        matches: list[dict] = []
        for game in self.games:
            name = (game.get("name") or "")
            if q in name.lower():
                matches.append(
                    {
                        "id": game["id"],
                        "name": name,
                        "year_published": game.get("year_published"),
                        "users_rated": game.get("users_rated", 0),
                    }
                )

        matches.sort(
            key=lambda x: (
                not x["name"].lower().startswith(q),
                -int(x.get("users_rated") or 0),
                x["name"],
            )
        )
        return matches[:limit]
