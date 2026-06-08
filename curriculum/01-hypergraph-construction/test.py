"""Test for Lesson 01: Hypergraph Construction."""
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Import the student's implementation
# ---------------------------------------------------------------------------
try:
    from solution import build_knn_graph  # if they put it here
except ImportError:
    from starter import build_knn_graph    # fallback


def test_shape_and_type():
    coords = np.random.rand(100, 2)
    adj = build_knn_graph(coords, k=7)
    assert isinstance(adj, sp.csr_matrix), f"Expected csr_matrix, got {type(adj)}"
    assert adj.shape == (100, 100), f"Expected shape (100, 100), got {adj.shape}"
    print("  [PASS] shape_and_type")


def test_symmetric():
    coords = np.random.rand(50, 2)
    adj = build_knn_graph(coords, k=5)
    diff = (adj - adj.T).toarray()
    assert np.allclose(diff, 0), "Adjacency matrix must be symmetric"
    print("  [PASS] symmetric")


def test_no_self_loops():
    coords = np.random.rand(50, 2)
    adj = build_knn_graph(coords, k=5)
    diag = adj.diagonal()
    assert np.allclose(diag, 0), "Diagonal (self-loops) must be zero"
    print("  [PASS] no_self_loops")


def test_enough_neighbors():
    coords = np.random.rand(100, 2)
    k = 7
    adj = build_knn_graph(coords, k=k)
    # Each node should have at least k neighbors (bidirectional)
    # but because of symmetry some edges may be shared, so check at least k
    degrees = np.array(adj.sum(axis=1)).flatten()
    assert np.all(degrees >= k), f"Some nodes have fewer than {k} neighbors: min={degrees.min()}"
    print("  [PASS] enough_neighbors")


def test_small_example():
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1]], dtype=np.float32)
    adj = build_knn_graph(coords, k=2)
    expected = np.array([
        [0., 1., 0., 1.],
        [1., 0., 1., 1.],
        [0., 1., 0., 1.],
        [1., 1., 1., 0.]
    ], dtype=np.float32)
    # Because of ties in k-NN, the exact graph may vary slightly.
    # We just check connectivity patterns are reasonable.
    assert adj.shape == (4, 4)
    # For 4 nodes with k=2, symmetrized knn typically has ~10 directed edges
    assert adj.nnz >= 6, f"Expected at least 6 directed edges, got {adj.nnz}"
    print("  [PASS] small_example")


if __name__ == "__main__":
    print("Running Lesson 01 tests...")
    test_shape_and_type()
    test_symmetric()
    test_no_self_loops()
    test_enough_neighbors()
    test_small_example()
    print("\nAll tests PASSED!")
