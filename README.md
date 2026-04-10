# Board Game Recommender

A Flask + React recommender app that suggests board games using:
- **TF-IDF cosine similarity** (baseline IR)
- **Truncated SVD latent space** (LSA-style semantic retrieval)

The app supports both:
- **Free-text queries** (e.g., `strategic medieval war game with no dice`)
- **Seed-title recommendations** (pick a game and get similar games)

## What It Uses

- Backend: Flask, Flask-SQLAlchemy
- Frontend: React + TypeScript + Vite
- Data source: `data/database.sqlite` (`BoardGames` table)
- ML/IR: scikit-learn (`TfidfVectorizer`, `TruncatedSVD`)
- Artifacts: generated under `data/artifacts/`

## Data Filters and Features

The index pipeline applies:
- `game.type = 'boardgame'`
- `stats.usersrated > 50`

Combined text field for retrieval:
- `details.description`
- `attributes.boardgamecategory`
- `attributes.boardgamemechanic`

Text cleaning includes HTML unescape and whitespace normalization.

## Project Structure

```text
FlavorMatrix/
├── src/
│   ├── app.py
│   ├── routes.py
│   ├── models.py
│   └── services/
│       ├── index_store.py
│       └── ir.py
├── scripts/
│   └── build_indices.py
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── App.css
│       └── types.ts
├── data/
│   ├── database.sqlite
│   └── artifacts/
├── requirements.txt
└── README.md
```

## Setup

### 1. Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build recommendation artifacts

```bash
python3 scripts/build_indices.py
```

This creates required files like:
- `tfidf_matrix.npz`
- `tfidf_vectorizer.pkl`
- `svd_model.pkl`
- `svd_embeddings.npy`
- `games.json`
- `game_ids.json`
- `svd_top_terms.json`

### 3. Frontend dependencies/build

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run server

```bash
python3 src/app.py
```

Open: `http://localhost:5001`

## API Endpoints

- `GET /api/config`
- `GET /api/games/search?q=...`
- `GET /api/recommendations?q=...&method=svd|tfidf&k=...`
- `GET /api/recommendations?seed=<game_id>&method=svd|tfidf&k=...`
- `GET /api/latent-dimensions?limit=...`
- `GET /api/metrics`
- `POST /api/feedback`

## Explainability Output

Each recommendation includes:
- `score_svd`, `score_tfidf`
- `rank_svd`, `rank_tfidf`
- `why_tags` (top activated latent dimensions)

Latent dimension endpoint includes:
- auto-generated labels from top-weighted terms
- explained variance per dimension

## Notes

- `data/database.sqlite` is intentionally ignored in git (too large for GitHub limits).
- If queries return no results in deployment, ensure `data/artifacts/` was rebuilt and deployed from the current SQLite dataset.

## NPM Scripts

From repo root:

```bash
npm run build       # builds frontend
npm run start       # starts Flask app
npm run prepare-data
```

`prepare-data` currently runs:

```bash
python3 scripts/build_indices.py
```
