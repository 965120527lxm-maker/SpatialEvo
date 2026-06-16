import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import scipy.sparse
import torch
from sklearn.neighbors import NearestNeighbors
import SpatialEx as se
import anndata as ad

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    adata = ad.read_h5ad(args.adata)
    all_genes = adata.var_names.tolist()
    np.random.seed(args.seed)
    perm = np.random.permutation(len(all_genes))
    panelA = [all_genes[i] for i in sorted(perm[:args.panelA_size])]
    panelB = [all_genes[i] for i in sorted(perm[args.panelA_size:])]
    print(f"Panel A: {len(panelA)}, Panel B: {len(panelB)}")
    Y_A = torch.tensor(adata[:, panelA].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, panelA].X, dtype=torch.float32)
    Y_B = torch.tensor(adata[:, panelB].X.toarray() if scipy.sparse.issparse(adata.X) else adata[:, panelB].X, dtype=torch.float32)
    # use Panel A kNN to predict Panel B (oracle, same slice)
    X = Y_A.numpy()
    nbrs = NearestNeighbors(n_neighbors=args.k, metric='cosine').fit(X)
    dist, idx = nbrs.kneighbors(X)
    pred = np.stack([Y_B.numpy()[i].mean(axis=0) for i in idx])
    pred_X = np.array(pred.copy())
    gt_X = Y_B.numpy().copy()
    graph = se.pp.Build_graph(adata.obsm["spatial"], graph_type="knn", weighted="gaussian",
                              apply_normalize="row", return_type="coo")
    pcc, pcc_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="pcc")
    ssim, ssim_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="ssim", graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt_X, pred_X, metric="cmd")
    print(f"Oracle same-slice PanelA->PanelB kNN: PCC={pcc_reduce:.6f}, SSIM={ssim_reduce:.6f}, CMD={cmd_reduce:.6f}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'oracle_metrics.txt'), 'w') as f:
        f.write(f"Oracle same-slice PanelA kNN->PanelB k={args.k}: PCC={pcc_reduce:.6f}, SSIM={ssim_reduce:.6f}, CMD={cmd_reduce:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata', type=str, default=os.path.join(PROJECT_ROOT, 'data', 'Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad'))
    parser.add_argument('--panelA_size', type=int, default=150)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default=os.path.join(PROJECT_ROOT, 'outputs', 'oracles', 'fig3_oracle'))
    args = parser.parse_args()
    main(args)
