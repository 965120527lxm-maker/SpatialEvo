#!/usr/bin/env python3
"""
DeepPT baseline for SpatialEx+ Fig.3 panel diagonal integration.

Protocol (matches paper description of modified DeepPT):
  - Train an independent H&E -> expression model for each panel on the slice
    where that panel was measured.
  - Predict the held-out panel on the other slice using H&E only.

  Slice1 (Rep1): measured panel A  ->  predict panel B on Rep1
  Slice2 (Rep2): measured panel B  ->  predict panel A on Rep2

Training:
  - Panel B model: train on Rep2 (HE2, YB2), predict Rep1 panel B
  - Panel A model: train on Rep1 (HE1, YA1), predict Rep2 panel A

Uses the same UNI H&E embeddings stored in adata.obsm['he'] as other Fig.3
scripts. DeepPT's original slide-level tile averaging is replaced with direct
per-cell regression, consistent with the paper's single-cell DeepPT variant.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import SpatialEx as se


class DeepPTCellMLP(nn.Module):
    """Cell-level DeepPT MLP (512 hidden, dropout 0.2)."""

    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Linear(n_hiddens, n_outputs)
        if bias_init is not None:
            self.layer1.bias = bias_init

    def forward(self, x):
        return self.layer1(self.layer0(x))


def parse_args():
    parser = argparse.ArgumentParser(description="DeepPT Fig.3 panel-diagonal baseline")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--rep1", type=str, default="Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad")
    parser.add_argument("--rep2", type=str, default="Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad")
    parser.add_argument("--panel_csv", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "panel_split_official.csv"))
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--valid_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "baselines", "fig3_deeppt_official"))
    return parser.parse_args()


def load_slice(path, data_root):
    adata = sc.read_h5ad(os.path.join(data_root, path))
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
    return adata


def get_matrix(adata, genes=None):
    if genes is None:
        X = adata.X
    else:
        X = adata[:, genes].X
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X, dtype=np.float32)


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_valid(n, valid_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_valid = max(1, int(n * valid_frac))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return train_idx, valid_idx


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def mean_gene_pcc(y_true, y_pred):
    corrs = []
    for g in range(y_true.shape[1]):
        a = y_true[:, g]
        b = y_pred[:, g]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs))


def train_deeppt(he_train, y_train, he_valid, y_valid, args, device):
    n_inputs = he_train.shape[1]
    n_outputs = y_train.shape[1]

    bias_init = nn.Parameter(torch.tensor(y_train.mean(axis=0), dtype=torch.float32, device=device))
    model = DeepPTCellMLP(n_inputs, args.hidden_dim, n_outputs, args.dropout, bias_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(he_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(he_valid, dtype=torch.float32),
                      torch.tensor(y_valid, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=False,
    )

    best_state = None
    best_valid_pcc = -np.inf
    stale = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        valid_pred = predict(model, valid_loader, device)
        valid_pcc = mean_gene_pcc(y_valid, valid_pred)
        if valid_pcc > best_valid_pcc:
            best_valid_pcc = valid_pcc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 25 == 0 or stale == 0:
            print(f"  epoch {epoch + 1}/{args.epochs}: valid PCC={valid_pcc:.4f}, best={best_valid_pcc:.4f}")

        if stale >= args.patience:
            print(f"  early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_slice(adata_gt, pred, label):
    graph = se.pp.Build_graph(
        adata_gt.obsm["spatial"], graph_type="knn", weighted="gaussian",
        apply_normalize="row", return_type="coo",
    )
    gt = get_matrix(adata_gt)
    pcc, pcc_reduce = se.utils.Compute_metrics(gt, pred, metric="pcc")
    ssim, ssim_reduce = se.utils.Compute_metrics(gt, pred, metric="ssim", graph=graph)
    cmd, cmd_reduce = se.utils.Compute_metrics(gt, pred, metric="cmd")
    print(f"[{label}] PCC={pcc_reduce:.6f}, SSIM={ssim_reduce:.6f}, CMD={cmd_reduce:.6f}")
    return {
        "pcc": float(pcc_reduce),
        "ssim": float(ssim_reduce),
        "cmd": float(cmd_reduce),
        "pcc_per_gene": pcc,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    init_seed(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    rep1 = load_slice(args.rep1, args.data_root)
    rep2 = load_slice(args.rep2, args.data_root)
    panel_df = pd.read_csv(args.panel_csv)
    panelA_genes = panel_df[panel_df["panel"] == "panelA"]["gene"].astype(str).tolist()
    panelB_genes = panel_df[panel_df["panel"] == "panelB"]["gene"].astype(str).tolist()
    print(f"Panel A: {len(panelA_genes)}, Panel B: {len(panelB_genes)}")

    HE1 = np.asarray(rep1.obsm["he"], dtype=np.float32)
    HE2 = np.asarray(rep2.obsm["he"], dtype=np.float32)
    YA1 = get_matrix(rep1, panelA_genes)
    YB2 = get_matrix(rep2, panelB_genes)

    # --- Panel B: train on Rep2, predict Rep1 ---
    print("\n=== DeepPT Panel B: train on Rep2, predict Rep1 panel B ===")
    tr_idx, va_idx = split_train_valid(len(HE2), args.valid_frac, args.seed)
    model_B = train_deeppt(HE2[tr_idx], YB2[tr_idx], HE2[va_idx], YB2[va_idx], args, device)

    pred_loader = DataLoader(
        TensorDataset(torch.tensor(HE1, dtype=torch.float32), torch.zeros(len(HE1), 1)),
        batch_size=args.batch_size, shuffle=False,
    )
    pred_B1 = predict(model_B, pred_loader, device)
    pred_B1_df = pd.DataFrame(pred_B1, index=rep1.obs_names, columns=panelB_genes)
    pred_B1_df.to_csv(os.path.join(args.out_dir, "pred_panelB1_deeppt.csv"))
    torch.save(model_B.state_dict(), os.path.join(args.out_dir, "deeppt_panelB_rep2.pt"))

    gt_B1 = rep1[:, panelB_genes]
    metrics_B1 = evaluate_slice(gt_B1, pred_B1, "Slice1 PanelB (DeepPT)")

    # --- Panel A: train on Rep1, predict Rep2 ---
    print("\n=== DeepPT Panel A: train on Rep1, predict Rep2 panel A ===")
    tr_idx, va_idx = split_train_valid(len(HE1), args.valid_frac, args.seed + 1)
    model_A = train_deeppt(HE1[tr_idx], YA1[tr_idx], HE1[va_idx], YA1[va_idx], args, device)

    pred_loader = DataLoader(
        TensorDataset(torch.tensor(HE2, dtype=torch.float32), torch.zeros(len(HE2), 1)),
        batch_size=args.batch_size, shuffle=False,
    )
    pred_A2 = predict(model_A, pred_loader, device)
    pred_A2_df = pd.DataFrame(pred_A2, index=rep2.obs_names, columns=panelA_genes)
    pred_A2_df.to_csv(os.path.join(args.out_dir, "pred_panelA2_deeppt.csv"))
    torch.save(model_A.state_dict(), os.path.join(args.out_dir, "deeppt_panelA_rep1.pt"))

    gt_A2 = rep2[:, panelA_genes]
    metrics_A2 = evaluate_slice(gt_A2, pred_A2, "Slice2 PanelA (DeepPT)")

    summary = pd.DataFrame([{
        "model": "deeppt",
        "slice1_pcc": metrics_B1["pcc"],
        "slice1_ssim": metrics_B1["ssim"],
        "slice1_cmd": metrics_B1["cmd"],
        "slice2_pcc": metrics_A2["pcc"],
        "slice2_ssim": metrics_A2["ssim"],
        "slice2_cmd": metrics_A2["cmd"],
    }])
    summary.to_csv(os.path.join(args.out_dir, "metrics_deeppt.csv"), index=False)
    print(f"\nSaved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
