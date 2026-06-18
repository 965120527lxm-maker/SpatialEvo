"""Solution for Lesson 09: Graph-Aware Evaluation."""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def graph_ssim(pred: torch.Tensor, true: torch.Tensor, coords: torch.Tensor, img_size: int = 64) -> float:
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    
    # Normalize coords to grid
    min_c = coords_np.min(axis=0)
    max_c = coords_np.max(axis=0)
    grid_coords = ((coords_np - min_c) / (max_c - min_c + 1e-6) * (img_size - 1)).astype(int)
    
    G = pred_np.shape[1]
    ssim_vals = []
    for g in range(G):
        img_pred = np.zeros((img_size, img_size))
        img_true = np.zeros((img_size, img_size))
        for i in range(len(grid_coords)):
            x, y = grid_coords[i]
            img_pred[y, x] += pred_np[i, g]
            img_true[y, x] += true_np[i, g]
        val = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min() + 1e-6)
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))
