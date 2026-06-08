# Lesson 07: Regression Translator

## Learning Goal
Build a cross-panel translator that maps from one gene panel's latent space to another.

## Background
SpatialEx+ enables larger panel analysis by translating between gene panels. The translator is an MLP that takes the hidden representation from slice A and predicts the expression for slice B's gene panel.

## Exercise
Implement `RegressionTranslator` in `starter.py`:
- `__init__`: `nn.Linear(hidden_dim, out_dim)` — a simple linear regressor
- `forward`: project hidden state to output genes

## Validation
Run `python test.py`. Checks shape and gradient flow.
