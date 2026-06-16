"""
Conditional cycle-completion baseline for Fig. 3.

Two translators:
    F_{B<-A}(X, Y_A) -> Y_B
    F_{A<-B}(X, Y_B) -> Y_A

Training does **not** use the held-out panels (Y_B^1, Y_A^2).  It uses
cycle consistency on the measured panels plus a marginal distribution-matching
loss to the real panels observed on the opposite slice.

Optionally uses H&E-only anchors P_A(X) and P_B(X) as additional grounding.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import preprocess as pp


class CycleTranslator(nn.Module):
    """2-layer MLP translator."""

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


class HEPredictor(nn.Module):
    """H&E-only panel predictor used as an anchor."""

    def __init__(self, he_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(he_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, he):
        return self.net(he)


class SpatialExP_ConditionalCycleMLP:
    """Conditional cycle MLP for panel diagonal integration.

    Parameters
    ----------
    adata1, adata2 : AnnData
        Slice 1 (measured A) and slice 2 (measured B).  Must contain
        ``obsm['he']`` if ``use_he=True`` and expression in ``.X``.
    use_he : bool
        Whether to concatenate H&E to the measured panel input and use
        H&E-only anchor predictors.
    hidden_dim : int
        Hidden size of all MLPs.
    lambda_dist : float
        Weight for marginal distribution matching of predicted missing panels.
    lambda_cycle : float
        Weight for cycle-consistency loss.
    lambda_he : float
        Weight for H&E-only anchor training loss (only when use_he=True).
    epochs, lr, dropout : training hyperparameters.
    standardize : bool
        Whether to z-score inputs and outputs.
    device : torch.device
    seed : int
    """

    def __init__(self,
                 adata1,
                 adata2,
                 use_he=True,
                 hidden_dim=512,
                 lambda_dist=1.0,
                 lambda_cycle=1.0,
                 lambda_he=1.0,
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
        self.lambda_dist = lambda_dist
        self.lambda_cycle = lambda_cycle
        self.lambda_he = lambda_he
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.standardize = standardize

        # Raw arrays
        self.YA1 = np.asarray(adata1.X.toarray() if hasattr(adata1.X, 'toarray') else adata1.X,
                              dtype=np.float32)
        self.YB2 = np.asarray(adata2.X.toarray() if hasattr(adata2.X, 'toarray') else adata2.X,
                              dtype=np.float32)
        self.dim_A = self.YA1.shape[1]
        self.dim_B = self.YB2.shape[1]

        if self.use_he:
            self.HE1 = np.asarray(adata1.obsm['he'], dtype=np.float32)
            self.HE2 = np.asarray(adata2.obsm['he'], dtype=np.float32)
            self.he_dim = self.HE1.shape[1]

        # Standardization statistics
        if self.standardize:
            if self.use_he:
                self.he_mean = np.concatenate([self.HE1, self.HE2], axis=0).mean(axis=0)
                self.he_std = np.concatenate([self.HE1, self.HE2], axis=0).std(axis=0) + 1e-8
                self.HE1 = (self.HE1 - self.he_mean) / self.he_std
                self.HE2 = (self.HE2 - self.he_mean) / self.he_std

            self.yA_mean1 = self.YA1.mean(axis=0)
            self.yA_std1 = self.YA1.std(axis=0) + 1e-8
            self.YA1 = (self.YA1 - self.yA_mean1) / self.yA_std1

            self.yB_mean2 = self.YB2.mean(axis=0)
            self.yB_std2 = self.YB2.std(axis=0) + 1e-8
            self.YB2 = (self.YB2 - self.yB_mean2) / self.yB_std2

        # Tensors on device
        self.YA1_t = torch.tensor(self.YA1, device=self.device)
        self.YB2_t = torch.tensor(self.YB2, device=self.device)
        if self.use_he:
            self.HE1_t = torch.tensor(self.HE1, device=self.device)
            self.HE2_t = torch.tensor(self.HE2, device=self.device)

        # Models
        if self.use_he:
            self.P_A = HEPredictor(self.he_dim, hidden_dim, self.dim_A, dropout=dropout).to(device)
            self.P_B = HEPredictor(self.he_dim, hidden_dim, self.dim_B, dropout=dropout).to(device)
            in_dim_BA = self.he_dim + self.dim_A
            in_dim_AB = self.he_dim + self.dim_B
        else:
            in_dim_BA = self.dim_A
            in_dim_AB = self.dim_B

        self.F_BA = CycleTranslator(in_dim_BA, hidden_dim, self.dim_B, dropout=dropout).to(device)
        self.F_AB = CycleTranslator(in_dim_AB, hidden_dim, self.dim_A, dropout=dropout).to(device)

        if self.use_he:
            params = (list(self.P_A.parameters()) + list(self.P_B.parameters()) +
                      list(self.F_BA.parameters()) + list(self.F_AB.parameters()))
        else:
            params = list(self.F_BA.parameters()) + list(self.F_AB.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0)
        self.criterion = nn.MSELoss()

    def _input_BA(self, he, measured):
        if self.use_he:
            return torch.cat([he, measured], dim=1)
        return measured

    def _input_AB(self, he, measured):
        if self.use_he:
            return torch.cat([he, measured], dim=1)
        return measured

    def _dist_loss(self, pred, target):
        """Match per-gene mean and std of pred to those of target."""
        mean_pred = pred.mean(dim=0)
        std_pred = pred.std(dim=0)
        mean_tgt = target.mean(dim=0)
        std_tgt = target.std(dim=0)
        return ((mean_pred - mean_tgt) ** 2).mean() + ((std_pred - std_tgt) ** 2).mean()

    def train(self):
        print(f'\n=== Start ConditionalCycleMLP training ==='
              f' use_he={self.use_he}, lambda_dist={self.lambda_dist}, '
              f'lambda_cycle={self.lambda_cycle}, lambda_he={self.lambda_he}, '
              f'hidden_dim={self.hidden_dim}')
        for epoch in tqdm(range(self.epochs), desc='epochs'):
            if self.use_he:
                self.P_A.train()
                self.P_B.train()
            self.F_BA.train()
            self.F_AB.train()

            if self.use_he:
                # H&E-only anchor predictions on both slices
                pA_on_1 = self.P_A(self.HE1_t)
                pA_on_2 = self.P_A(self.HE2_t)
                pB_on_1 = self.P_B(self.HE1_t)
                pB_on_2 = self.P_B(self.HE2_t)

                # H&E-only anchor losses (trained on the measured panels)
                he_loss = self.criterion(pA_on_1, self.YA1_t) + self.criterion(pB_on_2, self.YB2_t)

                # Slice1: A -> B -> A
                yB_hat = pB_on_1 + self.F_BA(self._input_BA(self.HE1_t, self.YA1_t))
                yA_rec = pA_on_1 + self.F_AB(self._input_AB(self.HE1_t, yB_hat))
                cycle_loss1 = self.criterion(yA_rec, self.YA1_t)

                # Slice2: B -> A -> B
                yA_hat = pA_on_2 + self.F_AB(self._input_AB(self.HE2_t, self.YB2_t))
                yB_rec = pB_on_2 + self.F_BA(self._input_BA(self.HE2_t, yA_hat))
                cycle_loss2 = self.criterion(yB_rec, self.YB2_t)

                loss = (self.lambda_cycle * (cycle_loss1 + cycle_loss2) +
                        self.lambda_he * he_loss +
                        self.lambda_dist * (self._dist_loss(yB_hat, self.YB2_t) +
                                            self._dist_loss(yA_hat, self.YA1_t)))
            else:
                # Slice1: A -> B -> A
                yB_hat = self.F_BA(self.YA1_t)
                yA_rec = self.F_AB(yB_hat)
                cycle_loss1 = self.criterion(yA_rec, self.YA1_t)

                # Slice2: B -> A -> B
                yA_hat = self.F_AB(self.YB2_t)
                yB_rec = self.F_BA(yA_hat)
                cycle_loss2 = self.criterion(yB_rec, self.YB2_t)

                loss = (self.lambda_cycle * (cycle_loss1 + cycle_loss2) +
                        self.lambda_dist * (self._dist_loss(yB_hat, self.YB2_t) +
                                            self._dist_loss(yA_hat, self.YA1_t)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                c1 = cycle_loss1.item()
                c2 = cycle_loss2.item()
                if self.use_he:
                    print(f'Epoch {epoch + 1}/{self.epochs}, '
                          f'cycle1 {c1:.4f}, cycle2 {c2:.4f}, '
                          f'he {he_loss.item():.4f}, '
                          f'total {loss.item():.4f}')
                else:
                    print(f'Epoch {epoch + 1}/{self.epochs}, '
                          f'cycle1 {c1:.4f}, cycle2 {c2:.4f}, '
                          f'total {loss.item():.4f}')

    @torch.no_grad()
    def predict_panelB_on_slice1(self, he, measured, graph=None):
        if self.use_he:
            self.P_A.eval()
            self.P_B.eval()
        self.F_BA.eval()
        self.F_AB.eval()
        if self.standardize:
            measured = (measured - self.yA_mean1) / self.yA_std1
            if self.use_he:
                he = (he - self.he_mean) / self.he_std
        measured_t = torch.tensor(measured, dtype=torch.float32, device=self.device)
        if self.use_he:
            he_t = torch.tensor(he, dtype=torch.float32, device=self.device)
            pred = self.P_B(he_t) + self.F_BA(self._input_BA(he_t, measured_t))
        else:
            pred = self.F_BA(measured_t)
        pred = pred.cpu().numpy()
        pred = pred * self.yB_std2 + self.yB_mean2
        return pred

    @torch.no_grad()
    def predict_panelA_on_slice2(self, he, measured, graph=None):
        if self.use_he:
            self.P_A.eval()
            self.P_B.eval()
        self.F_BA.eval()
        self.F_AB.eval()
        if self.standardize:
            measured = (measured - self.yB_mean2) / self.yB_std2
            if self.use_he:
                he = (he - self.he_mean) / self.he_std
        measured_t = torch.tensor(measured, dtype=torch.float32, device=self.device)
        if self.use_he:
            he_t = torch.tensor(he, dtype=torch.float32, device=self.device)
            pred = self.P_A(he_t) + self.F_AB(self._input_AB(he_t, measured_t))
        else:
            pred = self.F_AB(measured_t)
        pred = pred.cpu().numpy()
        pred = pred * self.yA_std1 + self.yA_mean1
        return pred
