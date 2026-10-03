"""Energy fractions, per-edge curl, and the readout ablation.

The ablation is the §10 theorem in runnable form. With the encoder FROZEN and a bias-free linear
readout, the flow is w.g(x_i) - w.g(x_j) = phi_i - phi_j, exactly a gradient flow, so the model
collapses to a pointwise model plus anchor averaging. The encoder must not be trained end to end:
a linear head would push the nonlinearity down into the encoder and the ablation would stop
isolating the readout. The project's one-fixed-encoder decision already supplies the frozen
variant, and with the encoder frozen the linear problem is convex, so the collapse is provable
rather than merely measured.

The real boundary is linear against nonlinear readout, not difference-of-encoders against joint
encoding: by Cauchy, h is additive on triangles iff it is linear.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from molace.hodge.complex import Complex
from molace.hodge.decompose import Decomposition, decompose


def fractions(d: Decomposition) -> dict:
    """Shares of the flow's OWN energy, plus the residual of the Pythagorean identity.

    Normalising by the sum of the three component norms would make the shares sum to 1 by
    construction, which is how an assertion that they do becomes untestable. Normalising by ||f||^2
    makes that sum the Pythagorean identity instead: it holds only while the three components are
    mutually orthogonal and reconstruct the flow, so a projection that stopped doing either shows up
    as `residual_check` moving off zero.
    """
    f = d.gradient + d.curl + d.harmonic
    total = float(f @ f)
    if total == 0.0:
        raise ValueError("the flow is identically zero, so its energy budget is undefined")
    parts = {
        "gradient": float(d.gradient @ d.gradient) / total,
        "curl": float(d.curl @ d.curl) / total,
        "harmonic": float(d.harmonic @ d.harmonic) / total,
    }
    parts["residual_check"] = sum(parts.values()) - 1.0
    return parts


def per_edge_curl(c: Complex, d: Decomposition) -> np.ndarray:
    """The magnitude of the curl component at each edge. Zero for a gradient flow."""
    return np.abs(d.curl)


def _edge_flow(c: Complex, predict) -> np.ndarray:
    """Evaluate a pairwise predictor on every edge, oriented u -> v (second minus first)."""
    return np.array([predict(u, v) for u, v in c.edges], dtype=float)


def readout_ablation(fp: np.ndarray, y: np.ndarray, c: Complex, seed: int = 0,
                     affine_bias: float = 0.0) -> dict:
    """Linear against nonlinear readout on a frozen encoder.

    The encoder is the fingerprint itself: fixed, not learned, exactly as the design requires.
    Training targets are the true edge differences, which are additive by construction.
    """
    x = np.asarray(fp, dtype=np.float64)
    y = np.asarray(y, dtype=float)
    pos = {v: i for i, v in enumerate(c.nodes)}
    pairs = np.array([(pos[u], pos[v]) for u, v in c.edges])
    diffs = x[pairs[:, 1]] - x[pairs[:, 0]]
    targets = y[pairs[:, 1]] - y[pairs[:, 0]]

    lin = Ridge(alpha=1.0, fit_intercept=False).fit(diffs, targets)
    out = {"linear": fractions(decompose(c, _edge_flow(
        c, lambda u, v: float(lin.coef_ @ (x[pos[v]] - x[pos[u]])))))}

    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=seed).fit(diffs, targets)
    out["mlp"] = fractions(decompose(c, _edge_flow(
        c, lambda u, v: float(mlp.predict((x[pos[v]] - x[pos[u]]).reshape(1, -1))[0]))))

    # An affine head puts exactly 3c on every triangle, independent of the data.
    if not c.triangles:
        raise ValueError(
            "the readout ablation needs at least one triangle; a complex without triangles has "
            "an empty curl subspace and the ablation would report zero curl as a finding"
        )
    i, j, k = c.triangles[0]
    aff = lambda u, v: float(lin.coef_ @ (x[pos[v]] - x[pos[u]])) + affine_bias
    out["affine_triangle_sum"] = aff(i, j) + aff(j, k) - aff(i, k) + 2 * affine_bias
    out["n_edges"] = len(c.edges)
    out["n_triangles"] = len(c.triangles)
    return out
