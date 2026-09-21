"""Strongly connected components: iterative Tarjan.  O(V + E).

Iterative on purpose: Python's default recursion limit (~1000) breaks a recursive DFS on a
city-sized graph.  Use it for VEHICLE reachability (one-way streets matter); keep Union-Find
for physical (undirected) islands.
"""
from __future__ import annotations

import numpy as np

from store.csr_graph import CSRGraph


def strongly_connected_components(g: CSRGraph, usable: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Return (component id per node, number of components).

    ``usable`` is an optional boolean mask over edges; edges with False are ignored (blocked roads).
    """
    n = g.n_nodes
    indptr = g.indptr.tolist()
    indices = g.indices.tolist()
    ok = None if usable is None else np.asarray(usable, dtype=bool).tolist()

    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    comp = [-1] * n
    ptr = indptr[:-1].copy()
    stack: list[int] = []
    counter = 0
    ncomp = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        call = [root]
        while call:
            v = call[-1]
            if ptr[v] < indptr[v + 1]:
                e = ptr[v]
                ptr[v] += 1
                if ok is not None and not ok[e]:
                    continue
                w = indices[e]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    call.append(w)
                elif on_stack[w] and index[w] < low[v]:
                    low[v] = index[w]
            else:
                call.pop()
                if call:
                    u = call[-1]
                    if low[v] < low[u]:
                        low[u] = low[v]
                if low[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp[w] = ncomp
                        if w == v:
                            break
                    ncomp += 1
    return np.array(comp, dtype=np.int32), ncomp


def giant_scc_share(comp: np.ndarray) -> float:
    """Fraction of nodes inside the largest SCC (sanity check: should be >= 0.95 on a real city)."""
    if len(comp) == 0:
        return 0.0
    return float(np.bincount(comp).max() / len(comp))
