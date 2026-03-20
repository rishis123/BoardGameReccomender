"""
LLM chat route – FlavorMatrix RAG pipeline.

Adds a POST /api/chat endpoint that performs grounded generation using
molecular profiles and recipe data retrieved from the FlavorMatrix corpus.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py (default)
"""

import json
import os
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

from services.rag import gather_context, build_grounded_prompt

logger = logging.getLogger(__name__)


def register_chat_route(app):
    """Register the /api/chat SSE endpoint."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        store = app.config.get("INDEX_STORE")
        db_path = app.config.get("DB_PATH")

        # Gather molecular + recipe context for RAG
        molecular_ctx = ""
        recipe_ctx = ""
        citations: list[str] = []
        if store and db_path:
            molecular_ctx, recipe_ctx, citations = gather_context(
                store, db_path, user_message
            )

        messages = build_grounded_prompt(user_message, molecular_ctx, recipe_ctx)
        client = LLMClient(api_key=api_key)

        def generate():
            if citations:
                yield f"data: {json.dumps({'citations': citations})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
