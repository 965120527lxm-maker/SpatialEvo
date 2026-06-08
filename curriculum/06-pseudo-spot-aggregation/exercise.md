# Lesson 06: Pseudo-Spot Aggregation

## Learning Goal
Aggregate single-cell predictions into pseudo-spot expression for spatial comparison.

## Background
Spatial transcriptomics platforms like 10x Visium measure ~5-10 cells per spot. To compare single-cell predictions against spot-level ground truth, we average the predicted expression of cells within each spot's radius.

## Exercise
Implement `aggregate_pseudo_spots` in `starter.py`:
- Given cell coordinates `coords` (N × 2) and spot coordinates `spot_coords` (M × 2)
- For each spot, find all cells within `radius` pixels
- Return average expression of those cells for each spot (M × genes)

## Validation
Run `python test.py`. It checks output shape and that aggregation is actually averaging.
