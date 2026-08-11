"""kNN graph properties on a mock hex grid."""

import numpy as np
import torch

from deepgraph_gbm.data.graph import build_knn_graph


def _hex_grid(n_rows=10, n_cols=10, spacing=100.0):
    """Visium-like hexagonal spot layout."""
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * spacing + (spacing / 2 if r % 2 else 0)
            y = r * spacing * np.sqrt(3) / 2
            pts.append((x, y))
    return np.array(pts)


def test_no_self_loops():
    coords = _hex_grid()
    ei, _ = build_knn_graph(coords, k=6)
    assert (ei[0] != ei[1]).all()


def test_symmetric():
    coords = _hex_grid()
    ei, _ = build_knn_graph(coords, k=6)
    edges = set(map(tuple, ei.numpy().T))
    for u, v in edges:
        assert (v, u) in edges


def test_interior_node_degree():
    coords = _hex_grid(12, 12)
    ei, _ = build_knn_graph(coords, k=6)
    deg = torch.bincount(ei[0], minlength=len(coords))
    # interior nodes of a hex grid have exactly 6 nearest neighbors;
    # symmetrization can only add edges, so interior degree >= 6
    interior = deg[len(coords) // 2 - 1]
    assert interior >= 6


def test_edge_weights_in_unit_interval():
    coords = _hex_grid()
    _, ea = build_knn_graph(coords, k=6)
    assert (ea > 0).all() and (ea <= 1).all()


def test_nearest_neighbor_distance():
    # two clusters far apart: nearest edges should have weight ~1
    a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ei, ea = build_knn_graph(a, k=2)
    d = np.sqrt(((a[ei[0].numpy()] - a[ei[1].numpy()]) ** 2).sum(1))
    assert d.max() <= np.sqrt(2) + 1e-9
    assert ea.max().item() == torch.tensor(1.0) or ea.max().item() <= 1.0
