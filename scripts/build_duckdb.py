"""
Build the canonical DuckDB database from raw dataset files.

Reads data/raw/ and writes data/flavormatrix.duckdb with normalised tables:
  - ingredients      (id, name, category, scientific_name)
  - molecules        (id, pubchem_id, common_name, flavor_profile)
  - ingredient_molecules (ingredient_id, molecule_id)
  - compatibility_pairs  (id, ingredient_a, ingredient_b, compatibility_level)
  - recipe_chunks        (id, recipe_id, chunk_text, chunk_index)
  - benchmark_items      (id, question, expected_answer, source)

Usage:
    python scripts/build_duckdb.py
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "flavormatrix.duckdb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            category  TEXT,
            scientific_name TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            id            INTEGER PRIMARY KEY,
            pubchem_id    TEXT,
            common_name   TEXT,
            flavor_profile TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingredient_molecules (
            ingredient_id INTEGER,
            molecule_id   INTEGER,
            PRIMARY KEY (ingredient_id, molecule_id)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS compatibility_pairs (
            id                  INTEGER PRIMARY KEY,
            ingredient_a        TEXT,
            ingredient_b        TEXT,
            compatibility_level TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS recipe_chunks (
            id          INTEGER PRIMARY KEY,
            recipe_id   INTEGER,
            chunk_text  TEXT,
            chunk_index INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_items (
            id              INTEGER PRIMARY KEY,
            question        TEXT,
            expected_answer TEXT,
            source          TEXT
        )
    """)


# ---------------------------------------------------------------------------
# FlavorDB ingestion
# ---------------------------------------------------------------------------

def _ingest_flavordb(con: duckdb.DuckDBPyConnection):
    details_path = RAW_DIR / "flavordb" / "entity_details.json"
    entities_path = RAW_DIR / "flavordb" / "entities.json"

    if not details_path.exists():
        print("  [skip] FlavorDB entity_details.json not found")
        return

    with open(details_path) as f:
        details = json.load(f)

    mol_map: dict[str, int] = {}
    mol_id_seq = 0
    ing_count = 0

    for ent in details:
        eid = ent.get("entity_id") or ent.get("id")
        if eid is None:
            continue
        eid = int(eid)
        name = (
            ent.get("entity_alias_readable")
            or ent.get("alias")
            or ent.get("name")
            or f"entity_{eid}"
        )
        category = ent.get("category", None)
        sci = ent.get("natural_source_name", None)

        con.execute(
            "INSERT OR IGNORE INTO ingredients VALUES (?, ?, ?, ?)",
            [eid, name, category, sci],
        )
        ing_count += 1

        molecules = ent.get("molecules", [])
        if isinstance(molecules, str):
            try:
                molecules = json.loads(molecules)
            except json.JSONDecodeError:
                molecules = [m.strip() for m in molecules.split(",") if m.strip()]

        for mol in molecules:
            if isinstance(mol, dict):
                pubchem = str(mol.get("pubchem_id", ""))
                cname = mol.get("common_name", "") or mol.get("name", "")
                fprofile = mol.get("flavor_profile", "") or mol.get("fooddb_flavor_profile", "")
                key = pubchem or cname
            else:
                key = str(mol)
                pubchem = key
                cname = ""
                fprofile = ""

            if key not in mol_map:
                mol_id_seq += 1
                mol_map[key] = mol_id_seq
                con.execute(
                    "INSERT OR IGNORE INTO molecules VALUES (?, ?, ?, ?)",
                    [mol_id_seq, pubchem, cname, fprofile],
                )

            con.execute(
                "INSERT OR IGNORE INTO ingredient_molecules VALUES (?, ?)",
                [eid, mol_map[key]],
            )

    print(f"  FlavorDB: {ing_count} ingredients, {len(mol_map)} molecules loaded")


# ---------------------------------------------------------------------------
# TasteTrios ingestion
# ---------------------------------------------------------------------------

def _ingest_tastetrios(con: duckdb.DuckDBPyConnection):
    tt_dir = RAW_DIR / "tastetrios"
    if not tt_dir.exists():
        print("  [skip] TasteTrios directory not found")
        return

    csv_files = list(tt_dir.glob("*.csv"))
    if not csv_files:
        print("  [skip] No CSV files in TasteTrios directory")
        return

    pair_id = 0
    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.lower().strip() for c in (reader.fieldnames or [])]
            for row in reader:
                row_lower = {k.lower().strip(): v for k, v in row.items()}
                ing_a = (
                    row_lower.get("ingredient_a")
                    or row_lower.get("ingredient1")
                    or row_lower.get("food_a")
                    or ""
                ).strip()
                ing_b = (
                    row_lower.get("ingredient_b")
                    or row_lower.get("ingredient2")
                    or row_lower.get("food_b")
                    or ""
                ).strip()
                compat = (
                    row_lower.get("compatibility_level")
                    or row_lower.get("compatibility")
                    or row_lower.get("label")
                    or ""
                ).strip()
                if ing_a and ing_b:
                    pair_id += 1
                    con.execute(
                        "INSERT OR IGNORE INTO compatibility_pairs VALUES (?, ?, ?, ?)",
                        [pair_id, ing_a, ing_b, compat],
                    )

    print(f"  TasteTrios: {pair_id} compatibility pairs loaded")


# ---------------------------------------------------------------------------
# RecipeNLG chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_tokens: int = 512, overlap_frac: float = 0.10):
    """Simple whitespace-based chunking (token ≈ word for speed)."""
    words = text.split()
    stride = max(1, int(max_tokens * (1 - overlap_frac)))
    chunks = []
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + max_tokens])
        if chunk:
            chunks.append(chunk)
        if start + max_tokens >= len(words):
            break
    return chunks


def _ingest_recipenlg(con: duckdb.DuckDBPyConnection, max_recipes: int = 50_000):
    """Ingest RecipeNLG recipes, chunked for RAG retrieval."""
    rn_dir = RAW_DIR / "recipenlg"
    if not rn_dir.exists():
        print("  [skip] RecipeNLG directory not found")
        return

    csv_files = list(rn_dir.glob("*.csv"))
    if not csv_files:
        print("  [skip] No CSV files in RecipeNLG directory")
        return

    chunk_id = 0
    recipe_count = 0

    for csv_path in csv_files:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if recipe_count >= max_recipes:
                    break
                title = row.get("title", "") or row.get("Title", "")
                ingredients = row.get("ingredients", "") or row.get("NER", "")
                directions = row.get("directions", "") or row.get("steps", "")

                full_text = f"Recipe: {title}\nIngredients: {ingredients}\nDirections: {directions}"
                chunks = _chunk_text(full_text)
                for ci, chunk in enumerate(chunks):
                    chunk_id += 1
                    con.execute(
                        "INSERT INTO recipe_chunks VALUES (?, ?, ?, ?)",
                        [chunk_id, recipe_count, chunk, ci],
                    )
                recipe_count += 1

    print(f"  RecipeNLG: {recipe_count} recipes → {chunk_id} chunks loaded")


# ---------------------------------------------------------------------------
# FOODPUZZLE ingestion
# ---------------------------------------------------------------------------

def _ingest_foodpuzzle(con: duckdb.DuckDBPyConnection):
    fp_dir = RAW_DIR / "foodpuzzle"
    if not fp_dir.exists():
        print("  [skip] FOODPUZZLE directory not found")
        return

    item_id = 0
    for fp in fp_dir.glob("*.json"):
        with open(fp) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("items", [])
        for item in items:
            item_id += 1
            con.execute(
                "INSERT INTO benchmark_items VALUES (?, ?, ?, ?)",
                [
                    item_id,
                    item.get("question", ""),
                    item.get("expected_answer", ""),
                    item.get("source", ""),
                ],
            )

    if item_id:
        print(f"  FOODPUZZLE: {item_id} benchmark items loaded")
    else:
        print("  [skip] No FOODPUZZLE JSON files found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed old database: {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    _create_schema(con)

    print("\nIngesting FlavorDB …")
    _ingest_flavordb(con)

    print("\nIngesting TasteTrios …")
    _ingest_tastetrios(con)

    print("\nIngesting RecipeNLG …")
    _ingest_recipenlg(con)

    print("\nIngesting FOODPUZZLE …")
    _ingest_foodpuzzle(con)

    row_counts = {}
    for table in ["ingredients", "molecules", "ingredient_molecules",
                   "compatibility_pairs", "recipe_chunks", "benchmark_items"]:
        (cnt,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        row_counts[table] = cnt

    con.close()
    print(f"\nDatabase written to {DB_PATH}")
    print("Row counts:")
    for t, c in row_counts.items():
        print(f"  {t}: {c:,}")

    print("\nNext step: python scripts/build_indices.py")


if __name__ == "__main__":
    main()
