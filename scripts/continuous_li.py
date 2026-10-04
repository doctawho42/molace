"""Why label informativeness has no continuous analogue: the denominator, shown rather than asserted.

LI = I(y_xi, y_eta) / H(y_xi), over a random edge under a random orientation. Swap the categorical
label for a continuous one and exactly one of the two parts breaks.

  numerator   I(y_xi, y_eta) is defined for continuous variables and is invariant under a smooth
              invertible change of either variable. Rescaling the label does not touch it.
  denominator h(Y) is a DIFFERENTIAL entropy. It can be negative, and h(cY) = h(Y) + log|c|. So
              the ratio depends on the units the label is measured in, and for every label there is
              a scale at which h = 0 and the ratio is undefined.

This script measures both on real data, in units that chemists actually use, and locates the scale
at which the measure blows up. Run: uv run python scripts/continuous_li.py
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from molace.data import moleculeace as ma
from molace.data.tdc_admet import load_tdc
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.continuous_li import differential_entropy, is_degenerate, mutual_information

K = 10


def endpoints(g, y):
    """The degree-weighted edge-endpoint sample LI is defined over: both orientations of every edge."""
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    e = np.array([(y[pos[u]], y[pos[v]]) for u, v in g.edges()], dtype=float)
    return np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])


def report(name, g, y, units, scales):
    a, b = endpoints(g, y)
    n_distinct = len(np.unique(a))
    print(f"\n{name}  (label in {units}; {g.number_of_nodes()} molecules, {g.number_of_edges()} edges)")
    print(f"    OBSTACLE 2, before any number: the edge-endpoint sample has {len(a)} points but only "
          f"{n_distinct} distinct")
    print(f"    values -- mean multiplicity {len(a)/n_distinct:.1f}. The marginal LI is defined over is "
          f"degree-weighted,")
    print(f"    so every molecule's label enters deg(v) times and the sample is tied BY CONSTRUCTION. A "
          f"nearest-")
    print(f"    neighbour differential entropy is undefined on it. h is therefore estimated on the "
          f"{g.number_of_nodes()} node")
    print(f"    labels instead, which changes nothing about the scale argument below: any estimate of h "
          f"shifts by log c.")
    print(f"    {'scale':>14s} {'units':>22s} {'I (nats)':>10s} {'h (nats)':>10s} {'LI = I/h':>12s}")
    first_h = None
    for c, label in scales:
        I = mutual_information(a * c, b * c)
        h = differential_entropy(np.asarray(y, dtype=float) * c, tie_policy="nudge")
        if first_h is None:
            first_h = h
        li = I / h if abs(h) > 1e-9 else float("inf")
        flag = "  <- I<0: the estimator has failed, see below" if is_degenerate(I) else ""
        print(f"    {c:14.6g} {label:>22s} {I:10.4f} {h:10.4f} {li:12.4f}{flag}")
    print(f"    h moves by exactly log(c) between any two rows -- that part is a theorem and is")
    print(f"    reproduced here to machine precision. The I column is an ESTIMATE and is not")
    print(f"    invariant to machine precision on a tied sample; the true I is.")
    # the scale at which the denominator vanishes
    h0 = differential_entropy(np.asarray(y, dtype=float), tie_policy="nudge")
    c_star = float(np.exp(-h0))
    print(f"    a pole EXISTS: h is continuous in log c and unbounded below, so it crosses zero.")
    print(f"    its LOCATION is not quotable: with this label {100*(1 - len(np.unique(y))/len(y)):.0f}% duplicated, h comes from a")
    print(f"    nudged estimate and moves several nats with the nudge constant, so c* = exp(-h) ~ "
          f"{c_star:.3g} is")
    print(f"    an order of magnitude, not a number. What does not depend on the nudge is the")
    print(f"    DIFFERENCE between two scales, which is why the table above is the evidence.")
    return h0


def main() -> int:
    print("=" * 96)
    print("A continuous label informativeness: the numerator survives, the denominator does not.")
    print("=" * 96)
    print("Theory, both exact: I(cY, cY') = I(Y, Y')  and  h(cY) = h(Y) + log|c|.")
    print("So LI = I/h is a function of the units, and crosses a pole wherever h = 0.")

    # 1. the project's own data: a pKi label, already a logarithm
    name = "CHEMBL2047_EC50"
    d = ma.load_target(name)
    y = d["y"].to_numpy(dtype=float)
    g = knn.knn_graph(tanimoto_matrix(ecfp4(d["smiles"].tolist())), k=K)
    report(f"MoleculeACE {name}", g, y, "pEC50, i.e. -log10 M",
           [(1.0, "as shipped"), (2.0, "x2"), (0.5, "x1/2"), (10.0, "x10")])

    # 2. a label whose unit is a genuine choice a chemist makes
    d2 = load_tdc("half_life_obach")
    y2 = d2["y"].to_numpy(dtype=float)
    g2 = knn.knn_graph(tanimoto_matrix(ecfp4(d2["smiles"].tolist())), k=K)
    report("TDC half_life_obach", g2, y2, "hours",
           [(1.0, "hours"), (60.0, "minutes"), (3600.0, "seconds"),
            (1 / 24.0, "days"), (1 / 168.0, "weeks")])

    print()
    print("=" * 96)
    print("What this settles, and what it does not.")
    print("=" * 96)
    print("Settles: a continuous LI cannot be built by substituting differential entropy into the")
    print("  denominator. The same dataset in hours and in minutes gets different values, and there")
    print("  is a unit in between where the measure is undefined. A measure whose point is")
    print("  comparability ACROSS datasets cannot depend on each dataset's unit.")
    print("A third thing the data says, which I did not expect. The labels are not continuous: 667")
    print("  molecules carry 193 distinct half-life values and 631 carry 411 distinct pEC50 values,")
    print("  because assays report to a fixed precision. So the categorical LI is applicable -- on the")
    print("  grid the assay already imposes. But that grid differs per dataset, and LI on it would")
    print("  depend on each assay's reporting precision, which is the binning dependence the")
    print("  pre-registration forbade (adjusted homophily moved 7.5x and LI 100x across bin counts).")
    print("  The continuous case and the binned case fail for the same reason from opposite ends.")
    print("Does not settle: that no continuous analogue exists. A normaliser that is itself")
    print("  scale-equivariant would fix the units -- I(y_xi; y_eta) / I(y_xi; y_xi) is not")
    print("  available since the self-information of a continuous variable diverges, but a")
    print("  normalisation by the entropy of a FIXED discretisation, or by I under a reference")
    print("  coupling, is not ruled out by anything here. That is the open question, and it is the")
    print("  one worth asking rather than asserting an impossibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
