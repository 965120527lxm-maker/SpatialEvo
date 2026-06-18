"""Test for Lesson 02: Graph Normalization."""
import numpy as np
import scipy.sparse as sp

try:
    from starter import normalize_graph
except ImportError:
    from solution import normalize_graph


def test_gcn_shape_and_type():
    adj = sp.random(50, 50, density=0.1, format='csr')
    adj = adj + adj.T
    norm = normalize_graph(adj, 'gcn')
    assert isinstance(norm, sp.csr_matrix)
    assert norm.shape == (50, 50)
    print("  [PASS] gcn_shape_and_type")


def test_gcn_row_sum():
    adj = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    norm = normalize_graph(adj, 'gcn')
    row_sums = np.array(norm.sum(axis=1)).flatten()
    # GCN normalization: each row should sum to ~1 for regular graphs
    # GCN normalization D^{-1/2} A D^{-1/2}: deg=[1,2,1] → row_sums=[1/sqrt(2), 2/sqrt(2), 1/sqrt(2)]
    assert np.allclose(row_sums, [0.70710678, 1.41421356, 0.70710678], atol=1e-4), f"Row sums: {row_sums}"
    print("  [PASS] gcn_row_sum")


def test_symmetry_preserved():
    adj = sp.random(30, 30, density=0.1, format='csr')
    adj = adj + adj.T
    norm = normalize_graph(adj, 'gcn')
    diff = (norm - norm.T).toarray()
    assert np.allclose(diff, 0), "Normalized graph must remain symmetric"
    print("  [PASS] symmetry_preserved")


def test_hpnn_basic():
    adj = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    norm = normalize_graph(adj, 'hpnn')
    assert isinstance(norm, sp.csr_matrix)
    assert norm.shape == (3, 3)
    print("  [PASS] hpnn_basic")


if __name__ == "__main__":
    print("Running Lesson 02 tests...")
    test_gcn_shape_and_type()
    test_gcn_row_sum()
    test_symmetry_preserved()
    test_hpnn_basic()
    print("\nAll tests PASSED!")
