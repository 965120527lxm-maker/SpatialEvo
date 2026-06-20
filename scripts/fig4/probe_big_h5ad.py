#!/usr/bin/env python3
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import scanpy as sc
import SpatialEx as se

p = os.environ.get("BIG_H5AD", "/root/autodl-tmp/lsy/nicheformer/data/data_unprocessed/10xgenomics_xenium_breast_cancer_big.h5ad")
ad = sc.read_h5ad(p)
ad.var_names = ad.var_names.astype(str)
ad.obs_names = ad.obs_names.astype(str)
print("full", ad.shape)
print("replicates", ad.obs["donor_id"].value_counts().to_dict())
for rep, target in [("replicate 1", 836616), ("replicate 2", 834928)]:
    sub = ad[ad.obs["donor_id"] == rep].copy()
    n0 = sub.n_obs
    sub2 = se.pp.Preprocess_adata(sub.copy())
    print(rep, "raw", n0, "after preprocess", sub2.n_obs, "target he", target)
