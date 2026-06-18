# Lesson 10: Graph Transformer Layer

## Learning Goal
Implement a memory-efficient Graph Transformer layer with chunked sparse attention.

## Background
The improved model replaces HGNN with Graph Transformers. For 164k cells, dense attention is impossible. We use sparse neighbor attention over the hypergraph edges, chunked into blocks to fit GPU memory.

## Exercise
Implement `GraphTransformerLayer` in `starter.py`:
- `__init__`: Q/K/V projections, output projection, FFN, LayerNorm, dropout
- `forward`: extract edge indices from sparse adj, compute multi-head attention in chunks, use `torch.utils.checkpoint`
- Use `scatter_softmax` provided in `solution.py`

## Validation
Run `python test.py`. Checks forward pass and memory stays low.
