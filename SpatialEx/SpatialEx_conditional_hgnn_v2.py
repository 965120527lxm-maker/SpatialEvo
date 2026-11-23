"""
HGNN conditional panel completion v2 with strict Fig.3 MNN pseudo-labels.

Key differences from the original SpatialEx_conditional_hgnn:
  1. Z-score normalises inputs (H&E and measured panels separately).
  2. Z-score normalises targets (MNN pseudo-labels).
  3. Uses direct cell-level MSE instead of pseudo-spot aggregate loss.
  4. Uses a linear output head (no final LeakyReLU).
  5. Un-normalises predictions at inference to match the original expression scale.

This makes the HGNN+MNN training pipeline directly comparable to the MLP+MNN
pipeline, isolating the effect of the GNN architecture from preprocessing issues.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from . import preprocess as pp
from .model import HGNN
from .SpatialEx_conditional_gt import build_strict_mnn_pseudo_labels


class PanelHGNN(nn.Module):
    """Simple HGNN for panel completion: [features] -> [missing panel]."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
        )
        self.hgnn = HGNN(
            in_dim=hidden_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation='prelu',
        )
        self.predictor = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = self.mlp(x)
        h = F.leaky_relu(self.hgnn(x, adj))
        return self.predictor(h)  # linear output for z-score targets


class SpatialExP_ConditionalHGNNv2:
    """Clean HGNN+MNN panel completion trainer.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Slice 1 (measured panel A) and slice 2 (measured panel B). Both must
        contain ``obsm['he']`` and expression in ``.X``.
    graph1, graph2 : scipy sparse matrix
        Hypergraph / graph adjacency for slice 1 and slice 2.
    use_he : bool
        Whether to concatenate H&E to the measured panel input.
    pseudo_k, mnn_k : int
        MNN pseudo-label construction hyperparameters.
    hidden_dim, num_layers : int
        HGNN architecture hyperparameters.
    epochs, lr, dropout : training hyperparameters.
    standardize : bool
        Whether to z-score inputs and targets (default True; should almost always
        be kept True for this model).
    device : torch.device
    seed : int
    """

    def __init__(self,
                 adata1,
                 adata2,
                 graph1,
                 graph2,
                 use_he=False,
                 pseudo_k=5,
                 mnn_k=20,
                 hidden_dim=512,
                 num_layers=2,
                 epochs=500,
                 lr=1e-3,
                 dropout=0.1,
                 standardize=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 seed=0):
        pp.set_random_seed(seed)
        self.device = device
        self.use_he = use_he
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.standardize = standardize

        # Load raw arrays
        self.HE1_orig = np.asarray(adata1.obsm['he'], dtype=np.float32)
        self.HE2_orig = np.asarray(adata2.obsm['he'], dtype=np.float32)
        self.YA1_orig = np.asarray(adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X,
                                   dtype=np.float32)
        self.YB2_orig = np.asarray(adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X,
                                   dtype=np.float32)

        self.measured_dim1 = self.YA1_orig.shape[1]
        self.measured_dim2 = self.YB2_orig.shape[1]
        self.he_dim = self.HE1_orig.shape[1]

        # Build MNN pseudo-labels
        print('Building strict Fig.3 MNN pseudo-labels (H&E bridge + B-panel bridge)...')
        self.pseudo_YB1_orig, self.pseudo_YA2_orig = build_strict_mnn_pseudo_labels(
            self.HE1_orig, self.HE2_orig, self.YA1_orig, self.YB2_orig,
            k=pseudo_k, mnn_k=mnn_k, device=device)

        # Standardisation
        if self.standardize:
            self.he_mean1 = self.HE1_orig.mean(axis=0)
            self.he_std1 = self.HE1_orig.std(axis=0) + 1e-8
            self.he_mean2 = self.HE2_orig.mean(axis=0)
            self.he_std2 = self.HE2_orig.std(axis=0) + 1e-8
            self.HE1 = (self.HE1_orig - self.he_mean1) / self.he_std1
            self.HE2 = (self.HE2_orig - self.he_mean2) / self.he_std2

            self.meas_mean1 = self.YA1_orig.mean(axis=0)
            self.meas_std1 = self.YA1_orig.std(axis=0) + 1e-8
            self.YA1 = (self.YA1_orig - self.meas_mean1) / self.meas_std1

            self.meas_mean2 = self.YB2_orig.mean(axis=0)
            self.meas_std2 = self.YB2_orig.std(axis=0) + 1e-8
            self.YB2 = (self.YB2_orig - self.meas_mean2) / self.meas_std2

            self.pseudo_mean1 = self.pseudo_YB1_orig.mean(axis=0)
            self.pseudo_std1 = self.pseudo_YB1_orig.std(axis=0) + 1e-8
            self.pseudo_YB1 = (self.pseudo_YB1_orig - self.pseudo_mean1) / self.pseudo_std1

            self.pseudo_mean2 = self.pseudo_YA2_orig.mean(axis=0)
            self.pseudo_std2 = self.pseudo_YA2_orig.std(axis=0) + 1e-8
            self.pseudo_YA2 = (self.pseudo_YA2_orig - self.pseudo_mean2) / self.pseudo_std2
        else:
            self.HE1 = self.HE1_orig.copy()
            self.HE2 = self.HE2_orig.copy()
            self.YA1 = self.YA1_orig.copy()
            self.YB2 = self.YB2_orig.copy()
            self.pseudo_YB1 = self.pseudo_YB1_orig.copy()
            self.pseudo_YA2 = self.pseudo_YA2_orig.copy()
            self.pseudo_mean1 = self.pseudo_std1 = None
            self.pseudo_mean2 = self.pseudo_std2 = None

        # Build model inputs
        if self.use_he:
            self.X1 = np.concatenate([self.HE1, self.YA1], axis=1).astype(np.float32)
            self.X2 = np.concatenate([self.HE2, self.YB2], axis=1).astype(np.float32)
            in_dim1 = self.he_dim + self.measured_dim1
            in_dim2 = self.he_dim + self.measured_dim2
        else:
            self.X1 = self.YA1.astype(np.float32)
            self.X2 = self.YB2.astype(np.float32)
            in_dim1 = self.measured_dim1
            in_dim2 = self.measured_dim2

        # Normalise graphs and convert to torch sparse
        graph1_norm = pp.normalize_graph(graph1, norm_type='hpnn')
        graph2_norm = pp.normalize_graph(graph2, norm_type='hpnn')
        self.adj1 = pp.sparse_mx_to_torch_sparse_tensor(graph1_norm).to(device)
        self.adj2 = pp.sparse_mx_to_torch_sparse_tensor(graph2_norm).to(device)

        # Tensors
        self.X1_t = torch.tensor(self.X1, device=device)
        self.X2_t = torch.tensor(self.X2, device=device)
        self.Y1_t = torch.tensor(self.pseudo_YB1, device=device)
        self.Y2_t = torch.tensor(self.pseudo_YA2, device=device)

        # Models
        self.model_AB = PanelHGNN(in_dim1, hidden_dim, self.pseudo_YB1.shape[1],
                                  num_layers=num_layers, dropout=dropout).to(device)
        self.model_BA = PanelHGNN(in_dim2, hidden_dim, self.pseudo_YA2.shape[1],
                                  num_layers=num_layers, dropout=dropout).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.model_AB.parameters()) + list(self.model_BA.parameters()),
            lr=lr, weight_decay=0)
        self.criterion = nn.MSELoss()

    def train(self, print_every=50):
        print('\n=== Start HGNN v2 conditional (strict MNN) training ==='
              f' use_he={self.use_he}, hidden_dim={self.hidden_dim}, '
              f'epochs={self.epochs}, lr={self.lr}')

        for epoch in tqdm(range(self.epochs), desc='epochs'):
            self.model_AB.train()
            self.model_BA.train()

            pred1 = self.model_AB(self.X1_t, self.adj1)
            loss1 = self.criterion(pred1, self.Y1_t)

            pred2 = self.model_BA(self.X2_t, self.adj2)
            loss2 = self.criterion(pred2, self.Y2_t)

            loss = loss1 + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, '
                      f'loss1 {loss1.item():.4f}, loss2 {loss2.item():.4f}, '
                      f'total {loss.item():.4f}')

    @torch.no_grad()
    def predict_panelB_on_slice1(self, he, measured, graph=None):
        self.model_AB.eval()
        if self.standardize:
            he = (he - self.he_mean1) / self.he_std1 if self.use_he else he
            measured = (measured - self.meas_mean1) / self.meas_std1

        if self.use_he:
            x = np.concatenate([he, measured], axis=1).astype(np.float32)
        else:
            x = measured.astype(np.float32)
        x = torch.tensor(x, device=self.device)

        pred = self.model_AB(x, self.adj1).cpu().numpy()

        if self.standardize:
            pred = pred * self.pseudo_std1 + self.pseudo_mean1
        return pred

    @torch.no_grad()
    def predict_panelA_on_slice2(self, he, measured, graph=None):
        self.model_BA.eval()
        if self.standardize:
            he = (he - self.he_mean2) / self.he_std2 if self.use_he else he
            measured = (measured - self.meas_mean2) / self.meas_std2

        if self.use_he:
            x = np.concatenate([he, measured], axis=1).astype(np.float32)
        else:
            x = measured.astype(np.float32)
        x = torch.tensor(x, device=self.device)

        pred = self.model_BA(x, self.adj2).cpu().numpy()

        if self.standardize:
            pred = pred * self.pseudo_std2 + self.pseudo_mean2
        return pred
