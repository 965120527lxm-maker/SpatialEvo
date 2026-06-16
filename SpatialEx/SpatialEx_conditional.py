"""
Measured-panel-conditioned SpatialEx+ trainer.

This trainer reframes panel diagonal integration as a conditional completion
problem: instead of predicting the missing panel from H&E alone (p(Y_B|X)), it
predicts from H&E plus the measured panel on the same slice (p(Y_B|X, Y_A)).

To avoid using the held-out ground truth during training, pseudo-labels for the
missing panel are generated from the adjacent slice via cross-slice H&E nearest
neighbor matching.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import preprocess as pp
from .utils import create_optimizer
from .model import Model_Plus


class SpatialExP_Conditional:
    """Conditional panel completion trainer.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Slice 1 with measured panel A, slice 2 with measured panel B. Both must
        have ``.obsm['he']`` and expression in ``.X``.
    graph1, graph2 : scipy.sparse.spmatrix
        Spatial hypergraphs for the two slices.
    measured_dim1, measured_dim2 : int
        Dimensions of measured panels (used for logging only).
    missing_dim1, missing_dim2 : int
        Dimensions of missing panels (output dims of the two conditional models).
    pseudo_k : int, default 5
        Number of cross-slice H&E nearest neighbors used to build pseudo-labels.
    hidden_dim : int, default 512
        Hidden dimension of the HGNN backbone.
    num_layers : int, default 2
        Number of HGNN layers.
    epochs : int, default 500
        Training epochs.
    lr : float, default 1e-3
    device : torch.device
    prune : int, default 10000
        Pruning parameter for the dataloader.
    use_dgi : bool, default False
        Whether to use DGI auxiliary loss. Disabled by default because the
        conditional task already provides strong supervision.
    """

    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 pseudo_k=5,
                 hidden_dim=512,
                 num_layers=2,
                 epochs=500,
                 lr=1e-3,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 prune=10000,
                 use_dgi=False,
                 save_path=None,
                 seed=0):
        pp.set_random_seed(seed)
        self.adata1 = adata1
        self.adata2 = adata2
        self.graph1 = pp.sparse_mx_to_torch_sparse_tensor(graph1).to(device)
        self.graph2 = pp.sparse_mx_to_torch_sparse_tensor(graph2).to(device)
        self.pseudo_k = pseudo_k
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.prune = prune
        self.use_dgi = use_dgi
        self.save_path = save_path

        self.HE1 = np.asarray(adata1.obsm['he'], dtype=np.float32)
        self.HE2 = np.asarray(adata2.obsm['he'], dtype=np.float32)
        self.YA1 = np.asarray(adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X,
                               dtype=np.float32)
        self.YB2 = np.asarray(adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X,
                               dtype=np.float32)

        self.measured_dim1 = self.YA1.shape[1]
        self.measured_dim2 = self.YB2.shape[1]
        self.missing_dim1 = self.YB2.shape[1]  # slice 1 missing panel B
        self.missing_dim2 = self.YA1.shape[1]  # slice 2 missing panel A
        self.he_dim = self.HE1.shape[1]

        # Build cross-slice pseudo-labels via H&E nearest neighbor matching
        print("Building cross-slice pseudo-labels...")
        self.pseudo_YB1 = self._build_pseudo_labels(self.HE1, self.HE2, self.YB2, k=pseudo_k)
        self.pseudo_YA2 = self._build_pseudo_labels(self.HE2, self.HE1, self.YA1, k=pseudo_k)

        # Construct conditional AnnData: input = [H&E, measured_panel], target = pseudo missing panel
        self.adata1_cond = self._build_conditional_adata(self.adata1, self.HE1, self.YA1, self.pseudo_YB1)
        self.adata2_cond = self._build_conditional_adata(self.adata2, self.HE2, self.YB2, self.pseudo_YA2)

        # Dataloaders
        self.slice1_dataloader = pp.Build_dataloader(
            self.adata1_cond, graph=graph1, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )
        self.slice2_dataloader = pp.Build_dataloader(
            self.adata2_cond, graph=graph2, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )

        # Conditional models: input_dim = he_dim + measured_dim
        self.model_AB = Model_Plus(
            in_dim=self.he_dim + self.measured_dim1,
            hidden_dim=self.hidden_dim,
            out_dim=self.missing_dim1,
            num_layers=self.num_layers,
            platform='Xenium',
            use_dgi=self.use_dgi
        ).to(self.device)

        self.model_BA = Model_Plus(
            in_dim=self.he_dim + self.measured_dim2,
            hidden_dim=self.hidden_dim,
            out_dim=self.missing_dim2,
            num_layers=self.num_layers,
            platform='Xenium',
            use_dgi=self.use_dgi
        ).to(self.device)

        self.models = [self.model_AB, self.model_BA]
        self.optimizer = create_optimizer('adam', self.models, self.lr, weight_decay=0)

    def _build_pseudo_labels(self, he_query, he_ref, y_ref, k=5, batch_size=4096):
        """Generate pseudo-labels for query slice by batched GPU cosine kNN."""
        he_query = torch.tensor(he_query, device=self.device)
        he_ref = torch.tensor(he_ref, device=self.device)
        y_ref = torch.tensor(y_ref, device=self.device)

        he_query_norm = he_query / (he_query.norm(dim=1, keepdim=True) + 1e-8)
        he_ref_norm = he_ref / (he_ref.norm(dim=1, keepdim=True) + 1e-8)

        n_query = he_query.shape[0]
        k = min(k, he_ref.shape[0])
        pseudo = torch.zeros(n_query, y_ref.shape[1], device='cpu')

        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            sim = he_query_norm[start:end] @ he_ref_norm.T
            topk_sim, topk_idx = torch.topk(sim, k, dim=1)
            weights = topk_sim / (topk_sim.sum(dim=1, keepdim=True) + 1e-8)
            pseudo[start:end] = (y_ref[topk_idx] * weights.unsqueeze(-1)).sum(dim=1).cpu()

        return pseudo.numpy().astype(np.float32)

    @staticmethod
    def _build_conditional_adata(original_adata, he, measured, pseudo_missing):
        import scanpy as sc
        adata = sc.AnnData(X=pseudo_missing, obs=original_adata.obs.copy())
        adata.obsm['he'] = np.concatenate([he, measured], axis=1).astype(np.float32)
        if 'spatial' in original_adata.obsm:
            adata.obsm['spatial'] = original_adata.obsm['spatial'].copy()
        return adata

    def train(self):
        self.model_AB.train()
        self.model_BA.train()
        print('\n')
        print('=================================== Start conditional training =========================================')
        for epoch in tqdm(range(self.epochs)):
            batch_iter = zip(self.slice1_dataloader, self.slice2_dataloader)
            for data1, data2 in batch_iter:
                graph1 = data1[0]['graph'].to(self.device)
                he1 = data1[0]['he'].to(self.device)
                target1 = data1[0]['exp'].to(self.device)
                graph2 = data2[0]['graph'].to(self.device)
                he2 = data2[0]['he'].to(self.device)
                target2 = data2[0]['exp'].to(self.device)
                agg_mtx1 = data1[0]['agg_mtx'].to(self.device)
                agg_exp1 = data1[0]['agg_exp'].to(self.device)
                agg_mtx2 = data2[0]['agg_mtx'].to(self.device)
                agg_exp2 = data2[0]['agg_exp'].to(self.device)
                selection1 = data1[0]['selection']
                selection2 = data2[0]['selection']

                loss1, _ = self.model_AB(he1, graph1, target1, agg_exp1, agg_mtx1, use_agg=True, selection=selection1)
                loss2, _ = self.model_BA(he2, graph2, target2, agg_exp2, agg_mtx2, use_agg=True, selection=selection2)

                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_panelB_on_slice1(self, he, measured, graph):
        """Predict missing panel B for slice 1 given H&E and measured panel A."""
        he = torch.Tensor(np.concatenate([he, measured], axis=1)).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        self.model_AB.eval()
        with torch.no_grad():
            pred = self.model_AB.predict(he, graph, grad=False)
        return pred.detach().cpu().numpy()

    def predict_panelA_on_slice2(self, he, measured, graph):
        """Predict missing panel A for slice 2 given H&E and measured panel B."""
        he = torch.Tensor(np.concatenate([he, measured], axis=1)).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        self.model_BA.eval()
        with torch.no_grad():
            pred = self.model_BA.predict(he, graph, grad=False)
        return pred.detach().cpu().numpy()
