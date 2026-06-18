import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import torch
import torch.nn.functional as F
import SpatialEx as se

def load_slice(path, data_root):
    full_path = os.path.join(data_root, path)
    adata = sc.read_h5ad(full_path)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata

def split_panels(adata, panelA_genes, panelB_genes):
    return adata[:, panelA_genes].copy(), adata[:, panelB_genes].copy()

def main():
    device = torch.device('cuda:0')
    data_root = os.path.join(PROJECT_ROOT, 'data')
    rep1_full = load_slice('Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad', data_root)
    rep2_full = load_slice('Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad', data_root)
    genes = rep1_full.var_names.values
    np.random.seed(0); np.random.shuffle(genes)
    panelA_genes = genes[:150].tolist(); panelB_genes = genes[150:].tolist()
    rep1_A, rep1_B = split_panels(rep1_full, panelA_genes, panelB_genes)
    rep2_A, rep2_B = split_panels(rep2_full, panelA_genes, panelB_genes)
    graph1 = se.pp.Build_hypergraph_spatial_and_HE(rep1_A, 7, graph_kind='spatial', return_type='csr')
    graph2 = se.pp.Build_hypergraph_spatial_and_HE(rep2_B, 7, graph_kind='spatial', return_type='csr')
    trainer = se.SpatialExP_Conditional(rep1_A, rep2_B, graph1, graph2,
                                        pseudo_k=5, hidden_dim=512, num_layers=2,
                                        epochs=20, lr=1e-3, device=device, prune=10000, use_dgi=False, seed=0)
    for epoch in range(20):
        loss_sum = 0; n=0
        for data1, data2 in zip(trainer.slice1_dataloader, trainer.slice2_dataloader):
            graph1b = data1[0]['graph'].to(device)
            he1 = data1[0]['he'].to(device)
            target1 = data1[0]['exp'].to(device)
            graph2b = data2[0]['graph'].to(device)
            he2 = data2[0]['he'].to(device)
            target2 = data2[0]['exp'].to(device)
            agg_mtx1 = data1[0]['agg_mtx'].to(device)
            agg_exp1 = data1[0]['agg_exp'].to(device)
            agg_mtx2 = data2[0]['agg_mtx'].to(device)
            agg_exp2 = data2[0]['agg_exp'].to(device)
            selection1 = data1[0]['selection']
            selection2 = data2[0]['selection']
            loss1, _ = trainer.model_AB(he1, graph1b, target1, agg_exp1, agg_mtx1, use_agg=True, selection=selection1)
            loss2, _ = trainer.model_BA(he2, graph2b, target2, agg_exp2, agg_mtx2, use_agg=True, selection=selection2)
            loss = loss1 + loss2
            trainer.optimizer.zero_grad(); loss.backward(); trainer.optimizer.step()
            loss_sum += loss.item(); n += 1
        print(f'epoch {epoch} avg loss {loss_sum/n:.4f}')
    # predict using internal model and graph
    with torch.no_grad():
        he = torch.Tensor(np.concatenate([trainer.HE1, trainer.YA1], axis=1)).to(device)
        pred = trainer.model_AB.predict(he, trainer.graph1, grad=False).cpu().numpy()
    pseudo = trainer.pseudo_YB1
    print('pred mean', pred.mean(), 'std', pred.std(), 'min', pred.min(), 'max', pred.max())
    print('pseudo mean', pseudo.mean(), 'std', pseudo.std())
    # per-gene pcc between pred and pseudo
    from scipy.stats import pearsonr
    pccs = []
    for g in range(pred.shape[1]):
        pccs.append(pearsonr(pred[:,g], pseudo[:,g])[0])
    print('pred-pseudo PCC mean', np.nanmean(pccs))
    # vs ground truth
    Y_B1 = rep1_B.X.toarray() if scipy.sparse.issparse(rep1_B.X) else np.array(rep1_B.X)
    pccs_gt = [pearsonr(pred[:,g], Y_B1[:,g])[0] for g in range(pred.shape[1])]
    print('pred-gt PCC mean', np.nanmean(pccs_gt))

if __name__ == '__main__':
    main()
