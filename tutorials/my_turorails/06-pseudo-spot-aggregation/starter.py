"""Starter code for Lesson 06: Pseudo-Spot Aggregation."""
import torch


def aggregate_pseudo_spots(
    cell_expression: torch.Tensor,
    cell_coords: torch.Tensor,
    spot_coords: torch.Tensor,
    radius: float = 50.0,
) -> torch.Tensor:
    """
    Aggregate single-cell predictions into pseudo-spots.

    Parameters
    ----------
    cell_expression : torch.Tensor, shape (N, G)
    cell_coords     : torch.Tensor, shape (N, 2)
    spot_coords     : torch.Tensor, shape (M, 2)
    radius          : float

    Returns
    -------
    pseudo_spots : torch.Tensor, shape (M, G)
    """
    # YOUR CODE HERE
    raise NotImplementedError
