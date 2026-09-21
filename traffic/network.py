"""Junction geometry used to add signal / turning delay to everyday travel times."""
from __future__ import annotations

import numpy as np

from store.csr_graph import CSRGraph


def physical_degree(g: CSRGraph) -> np.ndarray:
    """Number of physical roads meeting at each node (two-way pair counts once)."""
    e = np.arange(g.n_edges)
    canon = (g.rev < 0) | (e < g.rev)
    return np.bincount(np.concatenate([g.edge_tails[canon], g.indices[canon]]), minlength=g.n_nodes)
