"""API routes for the board-game recommender."""

from __future__ import annotations

import os

from flask import jsonify, request, send_from_directory

from models import Feedback, MetricsCache, db

USE_LLM = False


def register_routes(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/games/search")
    def games_search():
        q = request.args.get("q", "").strip()
        store = app.config.get("INDEX_STORE")
        if not store or not q:
            return jsonify([])
        return jsonify(store.search_names(q, limit=20))

    @app.route("/api/recommendations")
    def recommendations():
        from services.ir import recommend_games

        store = app.config.get("INDEX_STORE")
        if not store:
            return jsonify({"query": {}, "recommendations": [], "latent_dimensions": []})

        query_text = request.args.get("q", "").strip() or None
        seed_id = request.args.get("seed", "").strip() or None
        k = request.args.get("k", 10, type=int)
        method = request.args.get("method", "svd").strip().lower()
        if method not in {"svd", "tfidf"}:
            method = "svd"

        payload = recommend_games(
            store=store,
            query_text=query_text,
            seed_id=seed_id,
            k=max(1, min(k, 25)),
            method=method,
        )
        return jsonify(payload)

    @app.route("/api/latent-dimensions")
    def latent_dimensions():
        from services.ir import get_latent_dimensions

        store = app.config.get("INDEX_STORE")
        limit = request.args.get("limit", 10, type=int)
        if not store:
            return jsonify([])
        return jsonify(get_latent_dimensions(store, limit=max(1, min(limit, 20))))

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

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        data = request.get_json() or {}
        score = data.get("score")
        if score not in (1, -1):
            return jsonify({"error": "score must be 1 or -1"}), 400

        fb = Feedback(
            context_type=data.get("context_type", "recommendation"),
            context_id=data.get("context_id", ""),
            score=score,
        )
        db.session.add(fb)
        db.session.commit()
        return jsonify({"status": "ok", "id": fb.id})

    if USE_LLM:
        from llm_routes import register_chat_route

        register_chat_route(app)
