"""Starter code for Lesson 09: Graph-Aware Evaluation."""
import torch


def graph_ssim(pred: torch.Tensor, true: torch.Tensor, coords: torch.Tensor, img_size: int = 64) -> float:
    """
    Compute graph-aware SSIM.

    Parameters
    ----------
    pred   : torch.Tensor, shape (N, G)
    true   : torch.Tensor, shape (N, G)
    coords : torch.Tensor, shape (N, 2), spatial coordinates
    img_size : int, grid resolution

    Returns
    -------
    ssim : float, mean SSIM across genes
    """
    # YOUR CODE HERE
    raise NotImplementedError
