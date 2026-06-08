# Lesson 09: Graph-Aware Evaluation

## Learning Goal
Implement graph-based SSIM for spatial expression comparison.

## Background
Standard SSIM compares pixel grids. For single-cell data, we compute SSIM on expression images reconstructed from spatial coordinates, using the graph adjacency to define neighborhoods.

## Exercise
Implement `graph_ssim` in `starter.py`:
- Given two expression matrices `pred, true` (N × G) and coordinates `coords` (N × 2)
- Compute per-gene SSIM on a 2D image grid (simple nearest-neighbor assignment to grid pixels)
- Return mean SSIM across genes

## Validation
Run `python test.py`. Checks output is scalar in [0, 1].
