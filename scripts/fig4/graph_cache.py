"""Save/load Fig.4 spatial hypergraphs to skip expensive rebuilds."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import scipy.sparse as sp


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "graph_meta.json")


def _graph_path(cache_dir: str, slice_idx: int) -> str:
    return os.path.join(cache_dir, f"graph{slice_idx}_spatial.npz")


def graph_cache_meta(
    n_obs1: int,
    n_obs2: int,
    num_neighbors: int,
    graph_kind: str = "spatial",
) -> dict[str, Any]:
    return {
        "n_obs1": int(n_obs1),
        "n_obs2": int(n_obs2),
        "num_neighbors": int(num_neighbors),
        "graph_kind": graph_kind,
    }


def graphs_cached(cache_dir: str, meta: dict[str, Any]) -> bool:
    if not os.path.isdir(cache_dir):
        return False
    paths = [_meta_path(cache_dir), _graph_path(cache_dir, 1), _graph_path(cache_dir, 2)]
    if not all(os.path.isfile(p) for p in paths):
        return False
    with open(_meta_path(cache_dir)) as f:
        saved = json.load(f)
    return saved == meta


def save_graphs(cache_dir: str, graph1, graph2, meta: dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    sp.save_npz(_graph_path(cache_dir, 1), graph1.tocsr())
    sp.save_npz(_graph_path(cache_dir, 2), graph2.tocsr())
    with open(_meta_path(cache_dir), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[graph_cache] saved graphs -> {cache_dir}")


def load_graphs(cache_dir: str, meta: dict[str, Any], return_type: str = "csr"):
    if not graphs_cached(cache_dir, meta):
        raise FileNotFoundError(f"No matching graph cache in {cache_dir}")
    graph1 = sp.load_npz(_graph_path(cache_dir, 1))
    graph2 = sp.load_npz(_graph_path(cache_dir, 2))
    if return_type == "csr":
        graph1 = graph1.tocsr()
        graph2 = graph2.tocsr()
    elif return_type == "coo":
        graph1 = graph1.tocoo()
        graph2 = graph2.tocoo()
    print(f"[graph_cache] loaded graphs from {cache_dir}")
    return graph1, graph2


def get_or_build_graphs(
    build_fn,
    cache_dir: str,
    meta: dict[str, Any],
    rebuild: bool = False,
    return_type: str = "csr",
):
    if not rebuild and graphs_cached(cache_dir, meta):
        return load_graphs(cache_dir, meta, return_type=return_type)
    graph1, graph2 = build_fn()
    save_graphs(cache_dir, graph1, graph2, meta)
    return graph1, graph2
