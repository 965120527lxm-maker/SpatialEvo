# Lesson 04: MLP Encoder

## Learning Goal
Implement a simple multi-layer perceptron (MLP) encoder that maps high-dimensional gene expression to a latent vector.

## Background
Before hypergraph convolutions can operate on gene data, we first compress the raw expression matrix (N × genes) into a hidden dimension suitable for message passing. This is a standard MLP with two linear layers and ReLU.

## Exercise
Implement `MLPEncoder` in `starter.py`:
- `__init__`: two `nn.Linear` layers: `in_dim → hidden_dim → out_dim`
- `forward`: apply ReLU after first layer, then second layer, then dropout

## Validation
Run `python test.py`. It checks the output shape and that the model is trainable (gradients flow).
