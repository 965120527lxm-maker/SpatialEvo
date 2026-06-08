"""Solution for Lesson 06: Pseudo-Spot Aggregation."""
import torch


def aggregate_pseudo_spots(
    cell_expression: torch.Tensor,
    cell_coords: torch.Tensor,
    spot_coords: torch.Tensor,
    radius: float = 50.0,
) -> torch.Tensor:
    M, G = spot_coords.size(0), cell_expression.size(1)
    pseudo = torch.zeros(M, G, device=cell_expression.device)
    for i in range(M):
        dists = torch.norm(cell_coords - spot_coords[i], dim=1)
        mask = dists <= radius
        if mask.sum() > 0:
            pseudo[i] = cell_expression[mask].mean(dim=0)
    return pseudo
