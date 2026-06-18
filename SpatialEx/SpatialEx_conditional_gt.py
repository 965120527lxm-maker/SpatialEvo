"""
Graph-Transformer conditional panel completion with strict Fig.3 MNN pseudo-labels.

Strict protocol (Fig.3): Slice1 has H&E + panel A; Slice2 has H&E + panel B.
Pseudo-labels never use held-out Y_B1 or Y_A2.

  Step 1 (Slice1 pseudo-B): H&E cross-slice MNN (HE1 ↔ HE2), transfer YB2.
  Step 2 (Slice2 pseudo-A): B-panel cross-slice MNN (YB2 ↔ pseudo_YB1), transfer YA1.
"""

import torch
import numpy as np
from tqdm import tqdm

from . import preprocess as pp
from .utils import create_optimizer
from .model_improved import Model_Plus_GT


def batched_topk_indices(query, ref, k=20, batch_size=4096, device='cuda'):
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    indices = torch.zeros(n_query, k, dtype=torch.long, device='cpu')
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = query[start:end] @ ref.T
        indices[start:end] = torch.topk(sim, k, dim=1).indices.cpu()
    return indices.numpy()


def build_mnn_pseudo(query, ref, y_ref, k=5, mnn_k=20, batch_size=4096, device='cuda'):
    """MNN-filtered pseudo-labels; fall back to raw top-k when no mutual match."""
    fwd = batched_topk_indices(query, ref, k=mnn_k, batch_size=batch_size, device=device)
    rev = batched_topk_indices(ref, query, k=mnn_k, batch_size=batch_size, device=device)
    rev_sets = [set(rev[j].tolist()) for j in range(ref.shape[0])]
    selected = []
    for i in range(query.shape[0]):
        mnn = [j for j in fwd[i].tolist() if i in rev_sets[j]]
        if len(mnn) == 0:
            mnn = fwd[i, :k].tolist()
        selected.append(mnn)
    y_ref = np.asarray(y_ref, dtype=np.float32)
    return np.stack([y_ref[idx].mean(axis=0) for idx in selected]).astype(np.float32)


def build_strict_mnn_pseudo_labels(he1, he2, ya1, yb2, k=5, mnn_k=20, batch_size=4096, device='cuda'):
    """Fig.3 strict two-step MNN pseudo-labels (no held-out Y_B1 or Y_A2)."""
    he1 = np.asarray(he1, dtype=np.float32)
    he2 = np.asarray(he2, dtype=np.float32)
    ya1 = np.asarray(ya1, dtype=np.float32)
    yb2 = np.asarray(yb2, dtype=np.float32)
    pseudo_yb1 = build_mnn_pseudo(he1, he2, yb2, k=k, mnn_k=mnn_k,
                                  batch_size=batch_size, device=device)
    pseudo_ya2 = build_mnn_pseudo(yb2, pseudo_yb1, ya1, k=k, mnn_k=mnn_k,
                                  batch_size=batch_size, device=device)
    return pseudo_yb1, pseudo_ya2


class SpatialExP_ConditionalGT:
    """GT conditional completion: input [H&E, measured panel] -> missing panel."""

    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 pseudo_k=5,
                 mnn_k=20,
                 hidden_dim=128,
                 num_layers=2,
                 num_heads=8,
                 epochs=500,
                 lr=1e-3,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 prune=10000,
                 use_mfp=False,
                 save_path=None,
                 seed=0):
        pp.set_random_seed(seed)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path

        self.HE1 = np.asarray(adata1.obsm['he'], dtype=np.float32)
        self.HE2 = np.asarray(adata2.obsm['he'], dtype=np.float32)
        self.YA1 = np.asarray(adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X,
                              dtype=np.float32)
        self.YB2 = np.asarray(adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X,
                              dtype=np.float32)
        self.measured_dim1 = self.YA1.shape[1]
        self.measured_dim2 = self.YB2.shape[1]
        self.he_dim = self.HE1.shape[1]

        print('Building strict Fig.3 MNN pseudo-labels (H&E bridge + B-panel bridge)...')
        self.pseudo_YB1, self.pseudo_YA2 = build_strict_mnn_pseudo_labels(
            self.HE1, self.HE2, self.YA1, self.YB2,
            k=pseudo_k, mnn_k=mnn_k, device=device)
        self.missing_dim1 = self.pseudo_YB1.shape[1]
        self.missing_dim2 = self.pseudo_YA2.shape[1]

        self.adata1_cond = self._build_conditional_adata(adata1, self.HE1, self.YA1, self.pseudo_YB1)
        self.adata2_cond = self._build_conditional_adata(adata2, self.HE2, self.YB2, self.pseudo_YA2)

        self.slice1_dataloader = pp.Build_dataloader(
            self.adata1_cond, graph=graph1, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )
        self.slice2_dataloader = pp.Build_dataloader(
            self.adata2_cond, graph=graph2, graph_norm='hpnn', feat_norm=False,
            prune=[prune, prune], drop_last=False
        )

        in_dim1 = self.he_dim + self.measured_dim1
        in_dim2 = self.he_dim + self.measured_dim2
        self.model_AB = Model_Plus_GT(
            in_dim=in_dim1, hidden_dim=hidden_dim, out_dim=self.missing_dim1,
            num_layers=num_layers, num_heads=num_heads, use_mfp=use_mfp,
            platform='Xenium',
        ).to(device)
        self.model_BA = Model_Plus_GT(
            in_dim=in_dim2, hidden_dim=hidden_dim, out_dim=self.missing_dim2,
            num_layers=num_layers, num_heads=num_heads, use_mfp=use_mfp,
            platform='Xenium',
        ).to(device)

        self.models = [self.model_AB, self.model_BA]
        self.optimizer = create_optimizer('adam', self.models, lr, weight_decay=0)

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
        print('=================================== Start GT conditional (strict MNN) training =========================================')
        for _epoch in tqdm(range(self.epochs)):
            for data1, data2 in zip(self.slice1_dataloader, self.slice2_dataloader):
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

                loss1, _ = self.model_AB(
                    he1, graph1, target1, agg_exp1, agg_mtx1, use_agg=True, selection=selection1)
                loss2, _ = self.model_BA(
                    he2, graph2, target2, agg_exp2, agg_mtx2, use_agg=True, selection=selection2)

                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_panelB_on_slice1(self, he, measured, graph):
        x = torch.Tensor(np.concatenate([he, measured], axis=1)).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        self.model_AB.eval()
        with torch.no_grad():
            pred = self.model_AB.predict(x, graph, grad=False)
        return pred.detach().cpu().numpy()

    def predict_panelA_on_slice2(self, he, measured, graph):
        x = torch.Tensor(np.concatenate([he, measured], axis=1)).to(self.device)
        graph = pp.sparse_mx_to_torch_sparse_tensor(graph).to(self.device)
        self.model_BA.eval()
        with torch.no_grad():
            pred = self.model_BA.predict(x, graph, grad=False)
        return pred.detach().cpu().numpy()
