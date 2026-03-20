"""
Text-mining module – PMI-based association discovery.

Provides graph data for the Flavor Universe network visualization.
"""

from __future__ import annotations

from services.index_store import IndexStore


def get_network(
    store: IndexStore,
    ingredient_id: int | None = None,
    min_pmi: float = 1.0,
    limit: int = 200,
) -> dict:
    """
    Return a graph payload {nodes, edges} for the flavor network.

    If ingredient_id is given, return only edges connected to that ingredient
    (ego-network). Otherwise return the global top edges.
    """
    edges = store.pmi_edges
    if not edges:
        return {"nodes": [], "edges": []}

    if ingredient_id is not None:
        edges = [
            e for e in edges
            if e["source"] == ingredient_id or e["target"] == ingredient_id
        ]

    edges = [e for e in edges if e["pmi"] >= min_pmi]
    edges = edges[:limit]

    node_ids: set[int] = set()
    for e in edges:
        node_ids.add(e["source"])
        node_ids.add(e["target"])

    nodes = []
    for nid in sorted(node_ids):
        nodes.append({
            "id": nid,
            "name": store.ingredient_names.get(nid, str(nid)),
            "category": store.ingredient_categories.get(nid, "Unknown"),
        })

    formatted_edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "pmi": e["pmi"],
            "shared_molecules": e.get("shared_molecules", []),
        }
        for e in edges
    ]

    return {"nodes": nodes, "edges": formatted_edges}
