"""
Lightweight measured-panel-conditioned SpatialEx+ baseline.

Concatenates H&E and the measured panel, optionally standardises them, and
predicts the missing panel with a 2-layer MLP.  Cross-slice pseudo-labels can
be generated either from H&E (fully diagonal) or from the measured panels.

Measured-panel pseudo-labels are built as follows:
* Slice 1 missing Panel B: match measured A1 to measured A2 on the other slice
  and transfer measured B2.
* Slice 2 missing Panel A: match measured B2 to the pseudo Panel B computed for
  slice 1, then transfer measured A1.

This uses measured panels from both slices to build training targets, but never
uses the held-out panel of the slice being predicted (Y_B1 or Y_A2).
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import preprocess as pp


class ConditionalMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def batched_cosine_knn_indices(query, ref, k=5, batch_size=4096, device='cuda'):
    """Return indices of k nearest ref vectors for each query vector."""
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    indices = torch.zeros(n_query, k, device='cpu', dtype=torch.long)
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        indices[start:end] = torch.topk(query[start:end] @ ref.T, k, dim=1).indices.cpu()
    return indices.numpy()


def batched_cosine_knn_pseudo(query, ref, y_ref, k=5, batch_size=4096, device='cuda'):
    """Return weighted k-NN average of y_ref for each query vector."""
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
    y_ref = torch.as_tensor(y_ref, dtype=torch.float32, device=device)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    ref = ref / (ref.norm(dim=1, keepdim=True) + 1e-8)
    k = min(k, ref.shape[0])
    n_query = query.shape[0]
    pseudo = torch.zeros(n_query, y_ref.shape[1], device='cpu', dtype=torch.float32)
    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        sim = query[start:end] @ ref.T
        topk_sim, topk_idx = torch.topk(sim, k, dim=1)
        weights = topk_sim / (topk_sim.sum(dim=1, keepdim=True) + 1e-8)
        pseudo[start:end] = (y_ref[topk_idx] * weights.unsqueeze(-1)).sum(dim=1).cpu()
    return pseudo.numpy()


class SpatialExP_ConditionalMLP:
    """Simple conditional panel completion with a 2-layer MLP.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Slice 1 measured panel A, slice 2 measured panel B. Both contain
        ``obsm['he']`` and expression in ``.X``.
    mode : str
        One of ``'he_conditional'`` or ``'measured_pseudo'``.
    measured_A2, measured_B1 : np.ndarray, optional
        Measured panel A on slice 2 and measured panel B on slice 1. Required
        for ``'measured_pseudo'`` mode.
    pseudo_k : int
        k for nearest-neighbour pseudo-label averaging.
    hidden_dim : int
        Hidden dimension of the MLP.
    epochs, lr, dropout : training hyperparameters.
    standardize : bool
        Whether to z-score inputs and pseudo-labels.
    device : torch.device
    """

    def __init__(self,
                 adata1,
                 adata2,
                 mode='he_conditional',
                 measured_A2=None,
                 measured_B1=None,
                 pseudo_k=5,
                 hidden_dim=512,
                 epochs=500,
                 lr=1e-3,
                 dropout=0.1,
                 standardize=True,
                 use_he=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 seed=0):
        pp.set_random_seed(seed)
        self.device = device
        self.mode = mode
        self.pseudo_k = pseudo_k
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.standardize = standardize
        self.use_he = use_he

        # Original arrays
        self.HE1_orig = np.asarray(adata1.obsm['he'], dtype=np.float32)
        self.HE2_orig = np.asarray(adata2.obsm['he'], dtype=np.float32)
        self.YA1_orig = np.asarray(adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X,
                                   dtype=np.float32)
        self.YB2_orig = np.asarray(adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X,
                                   dtype=np.float32)

        self.measured_dim1 = self.YA1_orig.shape[1]
        self.measured_dim2 = self.YB2_orig.shape[1]
        self.missing_dim1 = self.YB2_orig.shape[1]
        self.missing_dim2 = self.YA1_orig.shape[1]
        self.he_dim = self.HE1_orig.shape[1]

        # Standardise inputs for the MLP
        if self.standardize:
            self.he_mean = np.concatenate([self.HE1_orig, self.HE2_orig], axis=0).mean(axis=0)
            self.he_std = np.concatenate([self.HE1_orig, self.HE2_orig], axis=0).std(axis=0) + 1e-8
            self.HE1 = (self.HE1_orig - self.he_mean) / self.he_std
            self.HE2 = (self.HE2_orig - self.he_mean) / self.he_std

            self.meas_mean1 = self.YA1_orig.mean(axis=0)
            self.meas_std1 = self.YA1_orig.std(axis=0) + 1e-8
            self.YA1 = (self.YA1_orig - self.meas_mean1) / self.meas_std1

            self.meas_mean2 = self.YB2_orig.mean(axis=0)
            self.meas_std2 = self.YB2_orig.std(axis=0) + 1e-8
            self.YB2 = (self.YB2_orig - self.meas_mean2) / self.meas_std2
        else:
            self.HE1 = self.HE1_orig.copy()
            self.HE2 = self.HE2_orig.copy()
            self.YA1 = self.YA1_orig.copy()
            self.YB2 = self.YB2_orig.copy()

        # Build pseudo labels
        if self.mode == 'measured_pseudo':
            if measured_A2 is None or measured_B1 is None:
                raise ValueError("measured_A2 and measured_B1 are required for measured_pseudo mode")
            self.YA2_orig = np.asarray(measured_A2.toarray() if hasattr(measured_A2, 'toarray') else measured_A2,
                                       dtype=np.float32)
            self.YB1_orig = np.asarray(measured_B1.toarray() if hasattr(measured_B1, 'toarray') else measured_B1,
                                       dtype=np.float32)
            print('Building measured-panel cross-slice pseudo-labels...')
            self.pseudo_YB1, self.pseudo_YA2 = self._build_measured_pseudo_labels()
        elif self.mode == 'he_conditional':
            print('Building H&E cross-slice pseudo-labels...')
            self.pseudo_YB1 = batched_cosine_knn_pseudo(
                self.HE1, self.HE2, self.YB2, k=self.pseudo_k, device=self.device)
            self.pseudo_YA2 = batched_cosine_knn_pseudo(
                self.HE2, self.HE1, self.YA1, k=self.pseudo_k, device=self.device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Standardise pseudo labels for training
        self.pseudo_mean1 = self.pseudo_YB1.mean(axis=0)
        self.pseudo_std1 = self.pseudo_YB1.std(axis=0) + 1e-8
        self.pseudo_YB1 = (self.pseudo_YB1 - self.pseudo_mean1) / self.pseudo_std1

        self.pseudo_mean2 = self.pseudo_YA2.mean(axis=0)
        self.pseudo_std2 = self.pseudo_YA2.std(axis=0) + 1e-8
        self.pseudo_YA2 = (self.pseudo_YA2 - self.pseudo_mean2) / self.pseudo_std2

        # Models
        if self.mode == 'measured_pseudo' and not self.use_he:
            # Pure panel-to-panel translator.
            self.model_AB = ConditionalMLP(self.measured_dim1, hidden_dim, self.missing_dim1,
                                           dropout=dropout).to(device)
            self.model_BA = ConditionalMLP(self.measured_dim2, hidden_dim, self.missing_dim2,
                                           dropout=dropout).to(device)
        else:
            # H&E + measured panel as input (he_conditional or measured_pseudo+use_he).
            self.model_AB = ConditionalMLP(self.he_dim + self.measured_dim1, hidden_dim, self.missing_dim1,
                                           dropout=dropout).to(device)
            self.model_BA = ConditionalMLP(self.he_dim + self.measured_dim2, hidden_dim, self.missing_dim2,
                                           dropout=dropout).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.model_AB.parameters()) + list(self.model_BA.parameters()),
            lr=lr, weight_decay=0)
        self.criterion = nn.MSELoss()

    def _build_measured_pseudo_labels(self):
        """Measured-panel pseudo labels (see module docstring)."""
        # Slice 1: A1 -> A2, transfer B2
        pseudo_YB1 = batched_cosine_knn_pseudo(
            self.YA1_orig, self.YA2_orig, self.YB2_orig, k=self.pseudo_k, device=self.device)
        # Slice 2: B2 -> pseudo B1, transfer A1
        pseudo_YA2 = batched_cosine_knn_pseudo(
            self.YB2_orig, pseudo_YB1, self.YA1_orig, k=self.pseudo_k, device=self.device)
        return pseudo_YB1, pseudo_YA2

    @staticmethod
    def _make_input(he, measured):
        return np.concatenate([he, measured], axis=1)

    def train(self, batch_size=4096):
        print(f'\n=== Start ConditionalMLP training (mode={self.mode}) ===')
        if self.mode == 'measured_pseudo' and not self.use_he:
            X1 = torch.tensor(self.YA1, device=self.device)
            X2 = torch.tensor(self.YB2, device=self.device)
        else:
            X1 = torch.tensor(self._make_input(self.HE1, self.YA1), device=self.device)
            X2 = torch.tensor(self._make_input(self.HE2, self.YB2), device=self.device)
        Y1 = torch.tensor(self.pseudo_YB1, device=self.device)
        Y2 = torch.tensor(self.pseudo_YA2, device=self.device)

        n1 = X1.shape[0]
        n2 = X2.shape[0]

        for epoch in tqdm(range(self.epochs), desc='epochs'):
            self.model_AB.train()
            self.model_BA.train()
            perm1 = torch.randperm(n1)
            loss_epoch = 0.0
            n_batches = 0
            for start in range(0, n1, batch_size):
                idx = perm1[start:start + batch_size]
                pred = self.model_AB(X1[idx])
                loss = self.criterion(pred, Y1[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            perm2 = torch.randperm(n2)
            for start in range(0, n2, batch_size):
                idx = perm2[start:start + batch_size]
                pred = self.model_BA(X2[idx])
                loss = self.criterion(pred, Y2[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, avg MSE {loss_epoch / n_batches:.4f}')

    @torch.no_grad()
    def predict_panelB_on_slice1(self, he, measured, graph=None):
        self.model_AB.eval()
        if self.standardize:
            measured = (measured - self.meas_mean1) / self.meas_std1
            he = (he - self.he_mean) / self.he_std
        if self.mode == 'measured_pseudo' and not self.use_he:
            x = torch.tensor(measured, dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(np.concatenate([he, measured], axis=1), dtype=torch.float32, device=self.device)
        pred = self.model_AB(x).cpu().numpy()
        pred = pred * self.pseudo_std1 + self.pseudo_mean1
        return pred

    @torch.no_grad()
    def predict_panelA_on_slice2(self, he, measured, graph=None):
        self.model_BA.eval()
        if self.standardize:
            measured = (measured - self.meas_mean2) / self.meas_std2
            he = (he - self.he_mean) / self.he_std
        if self.mode == 'measured_pseudo' and not self.use_he:
            x = torch.tensor(measured, dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(np.concatenate([he, measured], axis=1), dtype=torch.float32, device=self.device)
        pred = self.model_BA(x).cpu().numpy()
        pred = pred * self.pseudo_std2 + self.pseudo_mean2
        return pred
