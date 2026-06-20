"""Load Fig.4 million-cell big slice data from data/."""

from __future__ import annotations

import glob
import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import SpatialEx as se

DEFAULT_EXPR_H5AD = [
    "/root/autodl-tmp/novae/data/all_h5ad/human/breast/Xenium_V1_FFPE_Human_Breast_IDC_Big_{i}_outs.h5ad",
]


def _first_existing(root: str, names: Sequence[str]) -> Optional[str]:
    for name in names:
        path = os.path.join(root, name) if root else name
        if os.path.isfile(path):
            return path
    return None


def _as_array(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _build_adata(he, expr, spatial, gene_names, obs_names=None) -> AnnData:
    adata = AnnData(X=np.asarray(expr, dtype=np.float32))
    adata.var_names = pd.Index([str(g) for g in gene_names])
    if obs_names is not None:
        adata.obs_names = pd.Index([str(o) for o in obs_names])
    adata.obsm["he"] = np.asarray(he, dtype=np.float32)
    adata.obsm["spatial"] = np.asarray(spatial, dtype=np.float32)
    return adata


def _expr_h5ad_candidates(data_root: str, slice_idx: int) -> list[str]:
    i = int(slice_idx)
    names = [
        os.path.join(data_root, f"Human_Breast_IDC_Big{i}_uni_resolution64_full.h5ad"),
        os.path.join(data_root, f"Human_Breast_IDC_Big{i}.h5ad"),
        os.path.join(data_root, f"Human_breast_cancer_big_{i}.h5ad"),
        os.path.join(data_root, f"Big{i}_full.h5ad"),
        os.path.join(data_root, f"Big{i}.h5ad"),
        os.path.join(data_root, f"Big{i}_uni.h5ad"),
        os.path.join(data_root, f"Xenium_V1_FFPE_Human_Breast_IDC_Big_{i}_outs.h5ad"),
    ]
    for tmpl in DEFAULT_EXPR_H5AD:
        names.append(tmpl.format(i=i))
    env_key = f"BIG{i}_EXPR_H5AD"
    if os.environ.get(env_key):
        names.insert(0, os.environ[env_key])
    return names


def _npy_candidates(data_root: str, slice_idx: int) -> list[str]:
    i = int(slice_idx)
    return [
        os.path.join(data_root, f"Big{i}_uni.npy"),
        os.path.join(data_root, f"Big{i}.npy"),
        os.path.join(data_root, f"big_{i}.npy"),
        os.path.join(data_root, f"big{i}.npy"),
    ]


def _prepare_expr_adata(h5ad_path: str, n_he: int) -> AnnData:
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)

    if "spatial" not in adata.obsm:
        if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values.astype(np.float32)
        elif {"x", "y"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values.astype(np.float32)
        else:
            raise ValueError(f"{h5ad_path} missing spatial coordinates")

    x_sample = _as_array(adata.X[: min(1000, adata.n_obs)])
    if x_sample.max() > 20:
        adata = se.pp.Preprocess_adata(adata)
    else:
        sc.pp.filter_cells(adata, min_counts=10)

    adata.obs["x_centroid"] = adata.obsm["spatial"][:, 0]
    adata.obs["y_centroid"] = adata.obsm["spatial"][:, 1]

    if adata.n_obs < n_he:
        raise ValueError(
            f"{h5ad_path} has {adata.n_obs} cells after filtering, "
            f"but HE embeddings have {n_he} rows"
        )
    if adata.n_obs > n_he:
        print(
            f"[load_big] aligning expression to HE rows: "
            f"{adata.n_obs} -> {n_he} (first {n_he} cells; HE export order assumed)"
        )
        adata = adata[:n_he].copy()

    adata.X = _as_array(adata.X).astype(np.float32)
    return adata


def _load_npy_bundle(path: str, data_root: str) -> AnnData:
    obj = np.load(path, allow_pickle=True, mmap_mode="r")
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()

    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"he", "X", "spatial"}.issubset(keys):
            genes = obj.get("gene_names", obj.get("genes", obj.get("var_names")))
            obs = obj.get("obs_names", obj.get("barcodes"))
            return _build_adata(obj["he"], obj["X"], obj["spatial"], genes, obs)
        if {"he", "expr", "spatial"}.issubset(keys):
            genes = obj.get("gene_names", obj.get("genes"))
            obs = obj.get("obs_names")
            return _build_adata(obj["he"], obj["expr"], obj["spatial"], genes, obs)

    arr = np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported npy contents in {path}: type={type(obj)}, shape={getattr(arr, 'shape', None)}"
        )

    stem = os.path.splitext(os.path.basename(path))[0]
    sidecar = _first_existing(
        data_root,
        [
            f"{stem}.h5ad",
            f"{stem}_full.h5ad",
            stem.replace("Big", "Human_Breast_IDC_Big") + ".h5ad",
            stem.replace("big_", "Human_Breast_IDC_Big") + ".h5ad",
        ],
    )
    if sidecar is None:
        idx = 1 if "1" in stem else 2 if "2" in stem else None
        if idx is not None:
            sidecar = _first_existing("", _expr_h5ad_candidates(data_root, idx))
    if sidecar is None:
        raise ValueError(
            f"{path} looks like H&E embeddings only ({arr.shape}). "
            f"Need a matching expression .h5ad; set BIG1_EXPR_H5AD / BIG2_EXPR_H5AD."
        )

    adata = _prepare_expr_adata(sidecar, n_he=arr.shape[0])
    adata.obsm["he"] = np.asarray(arr, dtype=np.float32)
    return adata


def load_big_slice(data_root: str, slice_idx: int) -> AnnData:
    """Load big slice 1 or 2 from h5ad / npy bundles under data_root."""
    i = int(slice_idx)
    h5ad_candidates = [
        f"big_{i}.h5ad",
        f"big{i}.h5ad",
        f"Human_Breast_IDC_Big{i}_uni_resolution64_full.h5ad",
        f"Human_Breast_IDC_Big{i}.h5ad",
        f"Human_breast_cancer_big_{i}.h5ad",
        f"Big{i}_full.h5ad",
        f"Big{i}.h5ad",
    ]

    h5ad_path = _first_existing(data_root, h5ad_candidates)
    if h5ad_path is None:
        h5ad_path = _first_existing("", _expr_h5ad_candidates(data_root, i))

    if h5ad_path and not h5ad_path.endswith(".npy"):
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names = adata.var_names.astype(str)
        adata.obs_names = adata.obs_names.astype(str)
        if "spatial" not in adata.obsm and {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
        if "he" in adata.obsm:
            return adata

    npy_path = _first_existing("", _npy_candidates(data_root, i))
    if npy_path:
        return _load_npy_bundle(npy_path, data_root)

    globs = sorted(
        glob.glob(os.path.join(data_root, f"**/*Big{i}*.h5ad"), recursive=True)
        + glob.glob(os.path.join(data_root, f"**/*big*{i}*.h5ad"), recursive=True)
    )
    if globs:
        return load_big_slice_from_path(globs[0])

    raise FileNotFoundError(
        f"Could not find big slice {i} under {data_root}. "
        f"Tried HE npy + expr h5ad candidates."
    )


def load_big_slice_from_path(path: str) -> AnnData:
    data_root = os.path.dirname(path)
    if path.endswith(".h5ad"):
        adata = sc.read_h5ad(path)
        adata.var_names = adata.var_names.astype(str)
        adata.obs_names = adata.obs_names.astype(str)
        if "spatial" not in adata.obsm and {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
        return adata
    if path.endswith(".npy"):
        return _load_npy_bundle(path, data_root)
    raise ValueError(f"Unsupported big slice file: {path}")


def load_panel_genes(panel_csv: str) -> tuple[list[str], list[str]]:
    panel = pd.read_csv(panel_csv, index_col=0)
    panel.index = panel.index.astype(str)
    panel_a = panel.index[panel["panelA"] == 1].tolist()
    panel_b = panel.index[panel["panelB"] == 1].tolist()
    return panel_a, panel_b


def subset_panel(adata: AnnData, genes: Iterable[str], name: str) -> AnnData:
    genes = [g for g in genes if g in adata.var_names]
    missing = set(genes) - set(adata.var_names)
    if missing:
        raise ValueError(f"{name}: {len(missing)} panel genes missing from adata, e.g. {sorted(missing)[:5]}")
    out = adata[:, genes].copy()
    x = out.X
    out.X = _as_array(x).astype(np.float32)
    return out


def default_gt_h5ad(data_root: str, slice_idx: int) -> Optional[str]:
    return _first_existing("", _expr_h5ad_candidates(data_root, slice_idx))


def discover_big_inputs(data_root: str) -> list[str]:
    found = []
    for i in (1, 2):
        found.extend(_npy_candidates(data_root, i))
        found.extend(_expr_h5ad_candidates(data_root, i))
    return sorted(set(p for p in found if os.path.isfile(p)))
