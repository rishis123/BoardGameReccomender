"""
Pre-compute IR artifacts from the DuckDB database.

Produces data/artifacts/:
  - tfidf_matrix.npz       (sparse TF-IDF ingredient×molecule matrix)
  - tfidf_vocab.json        (molecule_id → column index mapping)
  - ingredient_ids.json     (row index → ingredient_id mapping)
  - ingredient_names.json   (ingredient_id → name mapping)
  - ingredient_categories.json
  - svd_components.npy      (V^T from TruncatedSVD)
  - svd_embeddings.npy      (U·Σ projections for all ingredients)
  - svd_explained.json      (explained-variance ratios)
  - svd_top_molecules.json  (top molecules per latent dimension)
  - pmi_edges.json          (pre-computed PMI edges)
  - chunk_tfidf_matrix.npz  (sparse TF-IDF for recipe chunks – RAG)
  - chunk_tfidf_vocab.json
  - chunk_ids.json

Usage:
    python scripts/build_indices.py
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "flavormatrix.duckdb"
ART_DIR = PROJECT_ROOT / "data" / "artifacts"


def _load_ingredient_molecules(con):
    """Return {ingredient_id: [molecule_id, …]}, plus metadata dicts."""
    rows = con.execute(
        "SELECT ingredient_id, molecule_id FROM ingredient_molecules ORDER BY ingredient_id"
    ).fetchall()

    ing_mols: dict[int, list[int]] = defaultdict(list)
    for iid, mid in rows:
        ing_mols[iid].append(mid)

    ing_rows = con.execute("SELECT id, name, category FROM ingredients").fetchall()
    id2name = {r[0]: r[1] for r in ing_rows}
    id2cat = {r[0]: (r[2] or "Unknown") for r in ing_rows}

    mol_rows = con.execute("SELECT id, common_name FROM molecules").fetchall()
    mid2name = {r[0]: r[1] for r in mol_rows}

    return ing_mols, id2name, id2cat, mid2name


# ---------------------------------------------------------------------------
# TF-IDF over molecules
# ---------------------------------------------------------------------------

def build_tfidf(ing_mols, id2name):
    ingredient_ids = sorted(ing_mols.keys())
    all_mol_ids = sorted({m for mols in ing_mols.values() for m in mols})
    mol2col = {m: i for i, m in enumerate(all_mol_ids)}

    n_ing = len(ingredient_ids)
    n_mol = len(all_mol_ids)
    print(f"  Building TF-IDF matrix: {n_ing} ingredients × {n_mol} molecules")

    # Document frequency
    df = np.zeros(n_mol)
    for mols in ing_mols.values():
        seen = set()
        for m in mols:
            c = mol2col[m]
            if c not in seen:
                df[c] += 1
                seen.add(c)

    idf = np.log((n_ing) / (1 + df))

    rows, cols, data = [], [], []
    for idx, iid in enumerate(ingredient_ids):
        mols = ing_mols[iid]
        tf_counts: dict[int, int] = defaultdict(int)
        for m in mols:
            tf_counts[mol2col[m]] += 1
        max_tf = max(tf_counts.values()) if tf_counts else 1
        for c, cnt in tf_counts.items():
            tf = 0.5 + 0.5 * (cnt / max_tf)
            rows.append(idx)
            cols.append(c)
            data.append(tf * idf[c])

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_ing, n_mol))
    mat = normalize(mat, norm="l2", axis=1)

    return mat, ingredient_ids, all_mol_ids, mol2col


# ---------------------------------------------------------------------------
# PMI
# ---------------------------------------------------------------------------

def build_pmi(ing_mols, mid2name, min_pmi: float = 1.0, max_edges: int = 5000):
    ingredient_ids = sorted(ing_mols.keys())
    n = len(ingredient_ids)
    mol_to_ings: dict[int, set[int]] = defaultdict(set)
    for iid, mols in ing_mols.items():
        for m in mols:
            mol_to_ings[m].add(iid)

    # P(ingredient) = fraction of molecules it appears with
    total_pairs = sum(len(mols) for mols in ing_mols.values())
    p_ing: dict[int, float] = {}
    for iid, mols in ing_mols.items():
        p_ing[iid] = len(mols) / total_pairs

    # Co-occurrence: two ingredients share a molecule
    cooccur: dict[tuple[int, int], int] = defaultdict(int)
    for mid, ings in mol_to_ings.items():
        ings_list = sorted(ings)
        for i in range(len(ings_list)):
            for j in range(i + 1, len(ings_list)):
                cooccur[(ings_list[i], ings_list[j])] += 1

    total_cooccur = sum(cooccur.values()) or 1

    edges = []
    for (a, b), count in cooccur.items():
        p_ab = count / total_cooccur
        p_a = p_ing.get(a, 1e-10)
        p_b = p_ing.get(b, 1e-10)
        pmi = math.log2(p_ab / (p_a * p_b)) if p_ab > 0 else 0
        if pmi >= min_pmi:
            shared = set(ing_mols.get(a, [])) & set(ing_mols.get(b, []))
            shared_names = [mid2name.get(m, str(m)) for m in sorted(shared)[:5]]
            edges.append({
                "source": a,
                "target": b,
                "pmi": round(pmi, 4),
                "shared_molecules": shared_names,
            })

    edges.sort(key=lambda e: e["pmi"], reverse=True)
    edges = edges[:max_edges]
    print(f"  PMI: {len(edges)} edges (min_pmi={min_pmi})")
    return edges


# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

def build_svd(tfidf_mat, ingredient_ids, all_mol_ids, mid2name, n_components: int = 20):
    k = min(n_components, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
    print(f"  Running TruncatedSVD with k={k}")
    svd = TruncatedSVD(n_components=k, random_state=42)
    embeddings = svd.fit_transform(tfidf_mat)

    explained = svd.explained_variance_ratio_.tolist()

    top_molecules = []
    for dim_idx in range(k):
        component = svd.components_[dim_idx]
        top_indices = np.argsort(np.abs(component))[::-1][:10]
        top_mols = []
        for ci in top_indices:
            mid = all_mol_ids[ci]
            top_mols.append({
                "molecule_id": mid,
                "name": mid2name.get(mid, str(mid)),
                "loading": round(float(component[ci]), 6),
            })
        top_molecules.append(top_mols)

    return svd.components_, embeddings, explained, top_molecules


# ---------------------------------------------------------------------------
# Chunk TF-IDF for RAG retrieval
# ---------------------------------------------------------------------------

def build_chunk_tfidf(con):
    rows = con.execute("SELECT id, chunk_text FROM recipe_chunks").fetchall()
    if not rows:
        print("  [skip] No recipe chunks to index")
        return

    chunk_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    print(f"  Building chunk TF-IDF index for {len(texts)} chunks")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform(texts)

    sparse.save_npz(ART_DIR / "chunk_tfidf_matrix.npz", mat)
    with open(ART_DIR / "chunk_tfidf_vocab.json", "w") as f:
        json.dump(vectorizer.vocabulary_, f)
    with open(ART_DIR / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    # Save the vectorizer feature names for query-time transform
    with open(ART_DIR / "chunk_tfidf_features.json", "w") as f:
        json.dump(vectorizer.get_feature_names_out().tolist(), f)

    print(f"  Chunk index: {mat.shape[0]} chunks × {mat.shape[1]} features")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run: python scripts/build_duckdb.py  first")
        sys.exit(1)

    ART_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    ing_mols, id2name, id2cat, mid2name = _load_ingredient_molecules(con)

    if not ing_mols:
        print("No ingredient-molecule data found in database. Aborting.")
        sys.exit(1)

    print("\n[1/4] Building TF-IDF matrix …")
    tfidf_mat, ingredient_ids, all_mol_ids, mol2col = build_tfidf(ing_mols, id2name)
    sparse.save_npz(ART_DIR / "tfidf_matrix.npz", tfidf_mat)
    with open(ART_DIR / "tfidf_vocab.json", "w") as f:
        json.dump({str(k): v for k, v in mol2col.items()}, f)
    with open(ART_DIR / "ingredient_ids.json", "w") as f:
        json.dump(ingredient_ids, f)
    with open(ART_DIR / "ingredient_names.json", "w") as f:
        json.dump({str(k): v for k, v in id2name.items()}, f)
    with open(ART_DIR / "ingredient_categories.json", "w") as f:
        json.dump({str(k): v for k, v in id2cat.items()}, f)

    print("\n[2/4] Computing PMI edges …")
    edges = build_pmi(ing_mols, mid2name)
    with open(ART_DIR / "pmi_edges.json", "w") as f:
        json.dump(edges, f)

    print("\n[3/4] Running Truncated SVD …")
    components, embeddings, explained, top_molecules = build_svd(
        tfidf_mat, ingredient_ids, all_mol_ids, mid2name
    )
    np.save(ART_DIR / "svd_components.npy", components)
    np.save(ART_DIR / "svd_embeddings.npy", embeddings)
    with open(ART_DIR / "svd_explained.json", "w") as f:
        json.dump(explained, f)
    with open(ART_DIR / "svd_top_molecules.json", "w") as f:
        json.dump(top_molecules, f)

    print("\n[4/4] Building chunk TF-IDF for RAG …")
    build_chunk_tfidf(con)

    con.close()
    print(f"\nAll artifacts written to {ART_DIR}/")
    print("Backend is ready to start: python src/app.py")


if __name__ == "__main__":
    main()
