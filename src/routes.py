"""API routes for the board-game recommender."""

from __future__ import annotations

import os

from flask import jsonify, request, send_from_directory

from models import Feedback, MetricsCache, db


def register_routes(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": bool(os.getenv("SPARK_API_KEY"))})

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

    @app.route("/api/games/dimensions")
    def game_dimensions():
        from services.ir import get_game_dimensions
        store = app.config.get("INDEX_STORE")
        game_id = request.args.get("id", "").strip()
        if not store or not game_id:
            return jsonify([])
        return jsonify(get_game_dimensions(store, game_id))

    @app.route("/api/rag")
    def rag():
        from services.ir import recommend_games, get_query_dimensions, get_game_dimensions
        from services.query_rewriter import rewrite_query

        store = app.config.get("INDEX_STORE")
        if not store:
            return jsonify({"error": "Index not loaded"}), 503

        seed_id = request.args.get("seed", "").strip() or None
        query_text = request.args.get("q", "").strip() or None

        if not seed_id and not query_text:
            return jsonify({"error": "seed or q is required"}), 400

        k = max(1, min(request.args.get("k", 10, type=int), 25))
        method = request.args.get("method", "svd").strip().lower()
        if method not in {"svd", "tfidf"}:
            method = "svd"

        # Standard IR — seed takes priority; details text is for LLM context only
        if seed_id:
            original_payload = recommend_games(store=store, seed_id=seed_id, k=k, method=method)
            original_dims = get_game_dimensions(store, seed_id)
            seed_game = store.game_by_id(seed_id)
            seed_name = seed_game["name"] if seed_game else ""
        else:
            original_payload = recommend_games(store=store, query_text=query_text, k=k, method=method)
            original_dims = get_query_dimensions(store, query_text)
            seed_name = None

        # Build a combined description for the LLM to rewrite
        if seed_name and query_text:
            llm_input = f"Games similar to {seed_name}, but {query_text}"
        elif seed_name:
            llm_input = f"Games similar to {seed_name}"
        else:
            llm_input = query_text

        # Human-readable label for the left column header
        if seed_name and query_text:
            original_label = f"Similar to {seed_name} + details"
        elif seed_name:
            original_label = f"Similar to {seed_name}"
        else:
            original_label = query_text

        # LLM query rewriting
        api_key = os.getenv("SPARK_API_KEY")
        rewritten_query = llm_input
        rag_results = original_payload.get("recommendations", [])
        rewritten_dims = original_dims
        error = None

        llm_summary = None
        if api_key:
            try:
                rewritten_query = rewrite_query(llm_input, original_dims, api_key)
                rag_payload = recommend_games(store=store, query_text=rewritten_query, k=k, method=method)
                rag_results = rag_payload.get("recommendations", [])
                rewritten_dims = get_query_dimensions(store, rewritten_query)
                from services.query_rewriter import generate_summary
                llm_summary = generate_summary(
                    original_query=llm_input,
                    original_dims=original_dims,
                    original_results=original_payload.get("recommendations", []),
                    rewritten_query=rewritten_query,
                    rewritten_dims=rewritten_dims,
                    rag_results=rag_results,
                    api_key=api_key,
                )
            except Exception as e:
                error = str(e)
        else:
            error = "SPARK_API_KEY not configured"

        return jsonify({
            "original_label": original_label,
            "original_dims": original_dims,
            "rewritten_query": rewritten_query,
            "rewritten_dims": rewritten_dims,
            "original_results": original_payload.get("recommendations", []),
            "rag_results": rag_results,
            "llm_summary": llm_summary,
            "error": error,
        })
