"""
API routes for the FlavorMatrix dashboard.

Endpoints:
  GET  /api/config                   – feature flags
  GET  /api/ingredients/search?q=    – autocomplete ingredient names
  GET  /api/substitutions?seed=&k=&category=
  GET  /api/network?ingredient=&min_pmi=&limit=
  GET  /api/sensory-map?dims=0,1,2&category=
  GET  /api/ingredient-profile?id=
  GET  /api/metrics
  POST /api/feedback
  POST /api/chat                     – RAG streaming (when USE_LLM=True)
"""

import json
import os
from flask import send_from_directory, request, jsonify
from models import db, Feedback, MetricsCache

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


def register_routes(app):
    # ── SPA static serving ────────────────────────────────────────────────
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, "index.html")

    # ── Config ────────────────────────────────────────────────────────────
    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    # ── Ingredient search (autocomplete) ──────────────────────────────────
    @app.route("/api/ingredients/search")
    def ingredients_search():
        q = request.args.get("q", "").strip()
        store = app.config.get("INDEX_STORE")
        if not store or not q:
            return jsonify([])
        results = store.search_names(q, limit=20)
        return jsonify(results)

    # ── Substitutions (core IR) ───────────────────────────────────────────
    @app.route("/api/substitutions")
    def substitutions():
        from services.ir import find_substitutes, get_shared_molecules

        store = app.config.get("INDEX_STORE")
        db_path = app.config.get("DB_PATH")
        seed = request.args.get("seed", type=int)
        k = request.args.get("k", 20, type=int)
        category = request.args.get("category", None)

        if not store or seed is None:
            return jsonify([])

        results = find_substitutes(store, seed, k=k, category=category)

        for r in results:
            if db_path:
                shared = get_shared_molecules(db_path, seed, r["id"], limit=5)
                r["shared_molecules"] = shared
            else:
                r["shared_molecules"] = []

        return jsonify(results)

    # ── Flavor Network (PMI graph) ────────────────────────────────────────
    @app.route("/api/network")
    def network():
        from services.pmi import get_network

        store = app.config.get("INDEX_STORE")
        ingredient = request.args.get("ingredient", type=int)
        min_pmi = request.args.get("min_pmi", 1.0, type=float)
        limit = request.args.get("limit", 200, type=int)

        if not store:
            return jsonify({"nodes": [], "edges": []})

        data = get_network(store, ingredient_id=ingredient, min_pmi=min_pmi, limit=limit)
        return jsonify(data)

    # ── Sensory Map (SVD 3D projection) ───────────────────────────────────
    @app.route("/api/sensory-map")
    def sensory_map():
        from services.svd import get_sensory_map

        store = app.config.get("INDEX_STORE")
        dims_str = request.args.get("dims", "0,1,2")
        category = request.args.get("category", None)

        if not store:
            return jsonify({"points": [], "dimensions": []})

        try:
            dims = tuple(int(d) for d in dims_str.split(","))[:3]
        except ValueError:
            dims = (0, 1, 2)
        while len(dims) < 3:
            dims = dims + (len(dims),)

        data = get_sensory_map(store, dims=dims, category=category)
        return jsonify(data)

    # ── Ingredient profile ────────────────────────────────────────────────
    @app.route("/api/ingredient-profile")
    def ingredient_profile():
        from services.ir import get_ingredient_profile

        db_path = app.config.get("DB_PATH")
        iid = request.args.get("id", type=int)
        if not db_path or iid is None:
            return jsonify({})

        data = get_ingredient_profile(db_path, iid)
        return jsonify(data)

    # ── Metrics ───────────────────────────────────────────────────────────
    @app.route("/api/metrics")
    def metrics():
        cached = MetricsCache.query.all()
        metrics_dict = {m.key: m.value for m in cached}

        feedback_rows = Feedback.query.all()
        if feedback_rows:
            avg = sum(f.score for f in feedback_rows) / len(feedback_rows)
            metrics_dict["avg_feedback"] = round(avg, 4)
            metrics_dict["total_feedback"] = len(feedback_rows)
        else:
            metrics_dict["avg_feedback"] = 0
            metrics_dict["total_feedback"] = 0

        return jsonify(metrics_dict)

    # ── Feedback ──────────────────────────────────────────────────────────
    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        data = request.get_json() or {}
        score = data.get("score")
        if score not in (1, -1):
            return jsonify({"error": "score must be 1 or -1"}), 400

        fb = Feedback(
            context_type=data.get("context_type", "chat"),
            context_id=data.get("context_id", ""),
            score=score,
        )
        db.session.add(fb)
        db.session.commit()
        return jsonify({"status": "ok", "id": fb.id})

    # ── Chat (RAG pipeline) ───────────────────────────────────────────────
    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app)
