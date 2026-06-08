# Lesson 05: Single-Slice Model

## Learning Goal
Combine the MLP encoder and HGNN layer into a single-slice prediction model.

## Background
The basic SpatialEx model for one tissue slice: encode gene expression → propagate on hypergraph → predict expression. This is the core "translates histology to omics" capability.

## Exercise
Implement `SingleSliceModel` in `starter.py`:
- `__init__`: `MLPEncoder` + `HGNNLayer` + final `nn.Linear`
- `forward`: encode → propagate on `adj_norm` → project to output dim

## Validation
Run `python test.py`. It checks end-to-end forward and backward pass.
