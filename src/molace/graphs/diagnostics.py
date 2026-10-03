"""Graph facts that are preconditions, not report lines.

Coverage and component structure travel with every measure, because the gradient potential of
a Hodge decomposition is fixed only up to a constant per component, and on threshold graphs the
measured component count reached 925 with 637 isolated molecules on one target.
"""
from __future__ import annotations

import networkx as nx


def graph_diagnostics(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    comps = list(nx.connected_components(g))
    c = len(comps)
    largest = max((len(x) for x in comps), default=0)
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "mean_degree": (2.0 * m / n) if n else 0.0,
        "n_components": c,
        "largest_component_fraction": (largest / n) if n else 0.0,
        "n_isolated": sum(1 for _, d in g.degree() if d == 0),
        "n_triangles": sum(nx.triangles(g).values()) // 3,
        "dim_gradient": n - c,
        "dim_cycle_space": m - n + c,
    }
