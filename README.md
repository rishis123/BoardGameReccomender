# FlavorMatrix

A molecular gastronomy information retrieval and generative dashboard built with **Flask + React + TypeScript**.

FlavorMatrix helps chefs and food enthusiasts discover ingredient substitutions, explore flavor networks, and understand the molecular science behind great pairings.

## Features

| Module | Description |
|--------|-------------|
| **Substitution Engine** | TF-IDF cosine similarity over molecular profiles to find ingredient replacements |
| **Flavor Universe** | PMI-based association graph revealing non-obvious ingredient connections |
| **Sensory Map** | 3D SVD projection of ingredients into latent flavor dimensions |
| **AI Flavor Chemist** | RAG-powered chat grounded in FlavorDB and RecipeNLG data |

## Architecture

```
RecallRadar/
├── scripts/
│   ├── download_datasets.py   # Fetch FlavorDB, TasteTrios, RecipeNLG
│   ├── build_duckdb.py        # Normalise raw data → DuckDB
│   └── build_indices.py       # Pre-compute TF-IDF, PMI, SVD artifacts
├── src/
│   ├── app.py                 # Flask entry point
│   ├── models.py              # SQLAlchemy models (feedback, metrics)
│   ├── routes.py              # All /api/* endpoints
│   ├── llm_routes.py          # RAG chat streaming endpoint
│   └── services/
│       ├── index_store.py     # Artifact loader (singleton)
│       ├── ir.py              # TF-IDF cosine search + Precision@k
│       ├── pmi.py             # PMI graph generation
│       ├── svd.py             # Truncated SVD projections
│       ├── chunking.py        # Text chunking for RAG
│       └── rag.py             # Retrieval + prompt grounding
├── frontend/
│   └── src/
│       ├── App.tsx            # Dashboard shell with sidebar + views
│       ├── Chat.tsx           # AI Flavor Chemist drawer
│       └── components/
│           ├── SubstitutionEngine.tsx
│           ├── SubstitutionCard.tsx
│           ├── FlavorNetwork.tsx
│           ├── SensoryMap3D.tsx
│           ├── IngredientProfileTable.tsx
│           └── FeedbackPanel.tsx
├── data/                      # Generated (not committed)
│   ├── raw/                   # Downloaded dataset files
│   ├── flavormatrix.duckdb    # Canonical database
│   └── artifacts/             # Pre-computed matrices and indices
├── requirements.txt
├── Dockerfile
└── .env                       # API_KEY (not committed)
```

## Quick Start

### 1. Set up Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download and prepare data

```bash
# Download datasets (FlavorDB via API, others via Kaggle)
python scripts/download_datasets.py

# Build the DuckDB database
python scripts/build_duckdb.py

# Pre-compute TF-IDF, PMI, SVD artifacts
python scripts/build_indices.py
```

> **Note:** FlavorDB is scraped from the public API at cosylab.iiitd.edu.in/flavordb.
> TasteTrios and RecipeNLG require `kagglehub` (`pip install kagglehub`) or manual download.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API_KEY for the LLM chat feature
```

### 4. Start the app

**Terminal 1 — Flask backend:**
```bash
python src/app.py
```

**Terminal 2 — React frontend (dev mode):**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Production build

```bash
cd frontend && npm install && npm run build && cd ..
python src/app.py
```

Open `http://localhost:5001`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/config` | Feature flags (`use_llm`) |
| GET | `/api/ingredients/search?q=` | Autocomplete ingredient names |
| GET | `/api/substitutions?seed=&k=&category=` | Molecular substitutes ranked by cosine similarity |
| GET | `/api/network?ingredient=&min_pmi=&limit=` | PMI-based flavor network graph |
| GET | `/api/sensory-map?dims=0,1,2&category=` | 3D SVD projection data |
| GET | `/api/ingredient-profile?id=` | Full molecular profile for an ingredient |
| GET | `/api/metrics` | Retrieval relevance and feedback metrics |
| POST | `/api/feedback` | Submit +1/-1 rating (`{score, context_type, context_id}`) |
| POST | `/api/chat` | RAG-grounded streaming chat (SSE) |

## Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| FlavorDB | 936 ingredients, 25k+ molecules | [cosylab.iiitd.edu.in/flavordb](https://cosylab.iiitd.edu.in/flavordb/) |
| TasteTrios | Ingredient compatibility ground truth | Kaggle `mbsssb/tastetrios` |
| RecipeNLG | Recipe text corpus for RAG | Kaggle `paultimothymooney/recipenlg` |
| FOODPUZZLE | Flavor-science QA benchmark | Manual download |

## Deploying on the Server

1. Run the data pipeline locally (steps 2 above) to generate `data/flavormatrix.duckdb` and `data/artifacts/`
2. Push your code **and** the `data/` directory to GitHub
3. Deploy via the [4300 Showcase Dashboard](https://4300showcase.infosci.cornell.edu/login)

The Dockerfile copies `data/` into the container image, so artifacts persist across restarts.

## Technical Details

### IR Module
- Each ingredient is a "document"; its flavor molecules (PubChem IDs) are "terms"
- TF uses augmented frequency: `0.5 + 0.5 * (count / max_count)`
- IDF: `log(N / (1 + df))`
- Retrieval ranked by cosine similarity of L2-normalised TF-IDF vectors

### Text Mining (PMI)
- Co-occurrence defined by shared molecules between ingredient pairs
- PMI: `log2(P(x,y) / (P(x) * P(y)))` with configurable threshold

### SVD Explainability
- Truncated SVD (k=20) on the TF-IDF ingredient-molecule matrix
- Top-10 molecule loadings per dimension used for human-readable labels
- 3D scatter plot via Plotly for interactive exploration

### RAG Pipeline
- Recipe and metadata corpus chunked at ~512 words with 10% overlap
- Sparse TF-IDF retrieval over chunks
- Strict grounding prompt: "Answer ONLY using provided data; cite sources"
- Streaming SSE responses with inline citation metadata
