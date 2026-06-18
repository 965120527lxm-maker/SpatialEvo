"""
MLP with strict Fig.3 MNN pseudo-label supervision + cycle consistency.

Combines:
  - Supervised MSE on strict MNN pseudo-labels (H&E bridge + B-panel bridge)
  - Cycle consistency on measured panels (A->B->A on slice1, B->A->B on slice2)

Never uses held-out Y_B1 or Y_A2.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import preprocess as pp
from .SpatialEx_conditional_gt import build_strict_mnn_pseudo_labels
from .SpatialEx_conditional_mlp import ConditionalMLP


class SpatialExP_ConditionalMNNCycleMLP:
    """Strict MNN pseudo-label + cycle MLP for Fig.3 panel diagonal integration."""

    def __init__(self,
                 adata1,
                 adata2,
                 pseudo_k=5,
                 mnn_k=20,
                 hidden_dim=512,
                 lambda_sup=1.0,
                 lambda_cycle=1.0,
                 use_he=False,
                 epochs=500,
                 lr=1e-3,
                 dropout=0.1,
                 standardize=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 seed=0):
        pp.set_random_seed(seed)
        self.device = device
        self.lambda_sup = lambda_sup
        self.lambda_cycle = lambda_cycle
        self.use_he = use_he
        self.epochs = epochs
        self.standardize = standardize

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

        print('Building strict Fig.3 MNN pseudo-labels...')
        pseudo_yb1, pseudo_ya2 = build_strict_mnn_pseudo_labels(
            self.HE1_orig, self.HE2_orig, self.YA1_orig, self.YB2_orig,
            k=pseudo_k, mnn_k=mnn_k, device=device)
        self.pseudo_YB1_orig = pseudo_yb1
        self.pseudo_YA2_orig = pseudo_ya2

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

        in_dim1 = (self.he_dim + self.measured_dim1) if use_he else self.measured_dim1
        in_dim2 = (self.he_dim + self.measured_dim2) if use_he else self.measured_dim2

        self.model_AB = ConditionalMLP(in_dim1, hidden_dim, self.missing_dim1, dropout=dropout).to(device)
        self.model_BA = ConditionalMLP(in_dim2, hidden_dim, self.missing_dim2, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.model_AB.parameters()) + list(self.model_BA.parameters()), lr=lr, weight_decay=0)
        self.criterion = nn.MSELoss()

    @staticmethod
    def _make_input(he, measured):
        return np.concatenate([he, measured], axis=1)

    def train(self, batch_size=4096):
        print(f'\n=== Start MNN+Cycle MLP (use_he={self.use_he}, '
              f'lambda_sup={self.lambda_sup}, lambda_cycle={self.lambda_cycle}) ===')

        if self.use_he:
            X1 = torch.tensor(self._make_input(self.HE1, self.YA1), device=self.device)
            X2 = torch.tensor(self._make_input(self.HE2, self.YB2), device=self.device)
        else:
            X1 = torch.tensor(self.YA1, device=self.device)
            X2 = torch.tensor(self.YB2, device=self.device)
        Y1 = torch.tensor(self.pseudo_YB1, device=self.device)
        Y2 = torch.tensor(self.pseudo_YA2, device=self.device)
        YA1_t = torch.tensor(self.YA1, device=self.device)
        YB2_t = torch.tensor(self.YB2, device=self.device)

        n1, n2 = X1.shape[0], X2.shape[0]

        for epoch in tqdm(range(self.epochs), desc='epochs'):
            self.model_AB.train()
            self.model_BA.train()
            loss_epoch = 0.0
            n_batches = 0

            perm1 = torch.randperm(n1)
            for start in range(0, n1, batch_size):
                idx = perm1[start:start + batch_size]
                x = X1[idx]
                pred_b = self.model_AB(x)
                sup = self.criterion(pred_b, Y1[idx])
                if self.use_he:
                    he = x[:, :self.he_dim]
                    rec_a = self.model_BA(torch.cat([he, pred_b], dim=1))
                else:
                    rec_a = self.model_BA(pred_b)
                cyc = self.criterion(rec_a, YA1_t[idx])
                loss = self.lambda_sup * sup + self.lambda_cycle * cyc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1

            perm2 = torch.randperm(n2)
            for start in range(0, n2, batch_size):
                idx = perm2[start:start + batch_size]
                x = X2[idx]
                pred_a = self.model_BA(x)
                sup = self.criterion(pred_a, Y2[idx])
                if self.use_he:
                    he = x[:, :self.he_dim]
                    rec_b = self.model_AB(torch.cat([he, pred_a], dim=1))
                else:
                    rec_b = self.model_AB(pred_a)
                cyc = self.criterion(rec_b, YB2_t[idx])
                loss = self.lambda_sup * sup + self.lambda_cycle * cyc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, avg loss {loss_epoch / n_batches:.4f}')

    @torch.no_grad()
    def predict_panelB_on_slice1(self, he, measured, graph=None):
        self.model_AB.eval()
        if self.standardize:
            measured = (measured - self.meas_mean1) / self.meas_std1
            if self.use_he:
                he = (he - self.he_mean) / self.he_std
        if self.use_he:
            x = torch.tensor(self._make_input(he, measured), dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(measured, dtype=torch.float32, device=self.device)
        pred = self.model_AB(x).cpu().numpy()
        if self.standardize:
            pred = pred * self.pseudo_std1 + self.pseudo_mean1
        return pred

    @torch.no_grad()
    def predict_panelA_on_slice2(self, he, measured, graph=None):
        self.model_BA.eval()
        if self.standardize:
            measured = (measured - self.meas_mean2) / self.meas_std2
            if self.use_he:
                he = (he - self.he_mean) / self.he_std
        if self.use_he:
            x = torch.tensor(self._make_input(he, measured), dtype=torch.float32, device=self.device)
        else:
            x = torch.tensor(measured, dtype=torch.float32, device=self.device)
        pred = self.model_BA(x).cpu().numpy()
        if self.standardize:
            pred = pred * self.pseudo_std2 + self.pseudo_mean2
        return pred
