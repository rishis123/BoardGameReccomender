"""
Download and prepare raw datasets for FlavorMatrix.

Sources:
  - FlavorDB:   cosylab.iiitd.edu.in/flavordb  (scraped via REST API)
  - TasteTrios: Kaggle dataset mbsssb/tastetrios
  - RecipeNLG:  Kaggle dataset paultimothymooney/recipenlg  (large ~2 GB)
  - FOODPUZZLE: HuggingFace dataset (placeholder – manual download)

Usage:
    python scripts/download_datasets.py          # download all available
    python scripts/download_datasets.py flavordb  # download only FlavorDB
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# FlavorDB scraper (public REST API)
# ---------------------------------------------------------------------------

FLAVORDB_BASE = "https://cosylab.iiitd.edu.in/flavordb"


def _get_json(url: str, retries: int = 3, delay: float = 1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to fetch {url}: {e}")


def download_flavordb():
    """Scrape FlavorDB entity list and per-entity molecule details."""
    out_dir = RAW_DIR / "flavordb"
    out_dir.mkdir(parents=True, exist_ok=True)

    entities_file = out_dir / "entities.json"
    details_file = out_dir / "entity_details.json"

    # Step 1: entity list
    if entities_file.exists():
        print(f"  [skip] {entities_file} already exists")
        with open(entities_file) as f:
            entities = json.load(f)
    else:
        print("  Fetching entity list from FlavorDB …")
        entities = _get_json(f"{FLAVORDB_BASE}/food_entities")
        with open(entities_file, "w") as f:
            json.dump(entities, f)
        print(f"  Saved {len(entities)} entities → {entities_file}")

    # Step 2: per-entity details (molecules)
    if details_file.exists():
        print(f"  [skip] {details_file} already exists")
        return

    details = []
    total = len(entities)
    print(f"  Fetching details for {total} entities (this may take a few minutes) …")
    for i, ent in enumerate(entities):
        eid = ent.get("entity_id") or ent.get("id")
        if eid is None:
            continue
        try:
            d = _get_json(f"{FLAVORDB_BASE}/entities_details?id={eid}")
            details.append(d)
        except RuntimeError:
            print(f"    [warn] Could not fetch entity {eid}, skipping")
        if (i + 1) % 50 == 0 or i + 1 == total:
            print(f"    {i + 1}/{total}")
        time.sleep(0.25)

    with open(details_file, "w") as f:
        json.dump(details, f)
    print(f"  Saved {len(details)} entity details → {details_file}")


# ---------------------------------------------------------------------------
# TasteTrios (Kaggle)
# ---------------------------------------------------------------------------

def download_tastetrios():
    """Download TasteTrios ingredient-compatibility dataset from Kaggle."""
    out_dir = RAW_DIR / "tastetrios"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        print(f"  [skip] TasteTrios already downloaded")
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download("mbsssb/tastetrios")
        print(f"  Downloaded TasteTrios to {path}")
        # Copy files into our raw dir
        import shutil
        for f in Path(path).rglob("*"):
            if f.is_file():
                dest = out_dir / f.name
                shutil.copy2(f, dest)
                print(f"    → {dest.name}")
        marker.touch()
    except ImportError:
        print("  [warn] kagglehub not installed. Install with: pip install kagglehub")
        print("         Or manually download mbsssb/tastetrios to data/raw/tastetrios/")
    except Exception as e:
        print(f"  [warn] TasteTrios download failed: {e}")
        print("         Manually download mbsssb/tastetrios to data/raw/tastetrios/")


# ---------------------------------------------------------------------------
# RecipeNLG (Kaggle – large dataset)
# ---------------------------------------------------------------------------

def download_recipenlg():
    """Download RecipeNLG recipe corpus from Kaggle."""
    out_dir = RAW_DIR / "recipenlg"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        print(f"  [skip] RecipeNLG already downloaded")
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download("paultimothymooney/recipenlg")
        print(f"  Downloaded RecipeNLG to {path}")
        import shutil
        for f in Path(path).rglob("*"):
            if f.is_file():
                dest = out_dir / f.name
                shutil.copy2(f, dest)
                print(f"    → {dest.name}")
        marker.touch()
    except ImportError:
        print("  [warn] kagglehub not installed. Install with: pip install kagglehub")
        print("         Or manually download paultimothymooney/recipenlg to data/raw/recipenlg/")
    except Exception as e:
        print(f"  [warn] RecipeNLG download failed: {e}")
        print("         Manually download paultimothymooney/recipenlg to data/raw/recipenlg/")


# ---------------------------------------------------------------------------
# FOODPUZZLE benchmark (placeholder)
# ---------------------------------------------------------------------------

def download_foodpuzzle():
    """Placeholder for FOODPUZZLE benchmark data."""
    out_dir = RAW_DIR / "foodpuzzle"
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".downloaded"
    if marker.exists():
        print(f"  [skip] FOODPUZZLE already downloaded")
        return

    print("  FOODPUZZLE requires manual download.")
    print("  Place benchmark JSON/CSV files in data/raw/foodpuzzle/")
    print("  Expected format: JSON with fields 'question', 'expected_answer', 'source'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "flavordb": download_flavordb,
    "tastetrios": download_tastetrios,
    "recipenlg": download_recipenlg,
    "foodpuzzle": download_foodpuzzle,
}


def main():
    parser = argparse.ArgumentParser(description="Download FlavorMatrix datasets")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DOWNLOADERS.keys()),
        choices=list(DOWNLOADERS.keys()),
        help="Datasets to download (default: all)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        print(f"\n{'='*60}")
        print(f" Downloading: {name}")
        print(f"{'='*60}")
        DOWNLOADERS[name]()

    print("\nDone. Next step: python scripts/build_duckdb.py")


if __name__ == "__main__":
    main()
