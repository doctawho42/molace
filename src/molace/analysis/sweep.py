"""Per-target measurement, cached to disk.

One JSON per (target, seed). A crash on target 23 keeps the first 22, and every record carries
the blob hash of the frozen plan, so a number produced under a different plan is visible rather
than arguable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data import moleculeace as ma
from molace.graphs import diagnostics as dg
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures import rogi as rogi_mod
from molace.measures.assortativity import target_assortativity
from molace.measures.dirichlet import dirichlet_energy
from molace.measures.homophily import adjusted_homophily
from molace.measures.informativeness import label_informativeness
from molace.measures.unbiased import unbiased_homophily
from molace.models.gap import evaluate_target

log = logging.getLogger(__name__)
CACHE = Path(__file__).resolve().parents[3] / "results" / "tasks"


def _maybe(fn, *a, **kw):
    """Measures that are undefined on a degenerate target return None rather than crashing."""
    try:
        return fn(*a, **kw)
    except (ValueError, TypeError) as exc:
        log.warning("measure %s skipped: %s", getattr(fn, "__name__", fn), exc)
        return None


def measure_target(name: str, seed: int, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}__seed{seed}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text())

    plan = load_prereg()
    k = plan["graphs"]["primary"]["k"]
    df = ma.load_target(name)
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    cliff = df["cliff_mol"].to_numpy(dtype=int)
    g = knn.knn_graph(tanimoto_matrix(fp), k=k)
    diag = dg.graph_diagnostics(g)

    asrt = target_assortativity(g, y)
    ub = _maybe(unbiased_homophily, g, cliff)
    li = _maybe(label_informativeness, g, cliff)
    adj = _maybe(adjusted_homophily, g, cliff)
    rg = _maybe(rogi_mod.roughness, fp, y)
    dr = _maybe(dirichlet_energy, g, y)
    res = evaluate_target(df, seed=seed)

    rec = {
        "dataset": name,
        "receptor_class": ma.receptor_class(name),
        "seed": seed,
        "prereg": prereg_fingerprint(),
        "n_molecules": int(len(df)),
        "assortativity": asrt.value,
        "assortativity_coverage": asrt.coverage,
        "unbiased_homophily": None if ub is None else ub.value,
        "label_informativeness": None if li is None else li.value,
        "adjusted_homophily": None if adj is None else adj.value,
        "rogi": None if rg is None else rg.value,
        "rogi_flavour": rogi_mod.ROGI_FLAVOUR,
        "dirichlet": None if dr is None else dr.value,
        "mean_degree": diag["mean_degree"],
        "n_components": diag["n_components"],
        "largest_component_fraction": diag["largest_component_fraction"],
        "n_triangles": diag["n_triangles"],
        "rmse_pointwise": res.pointwise.rmse_all,
        "rmse_knn_floor": res.knn_floor.rmse_all,
        "rmse_pairwise": res.pairwise.rmse_all,
        "rmse_cliff_pointwise": res.pointwise.rmse_cliff,
        "rmse_cliff_knn_floor": res.knn_floor.rmse_cliff,
        "rmse_cliff_pairwise": res.pairwise.rmse_cliff,
        "gap": res.gap,
        "access": res.access,
        "correction": res.correction,
        "selected_pointwise": res.pointwise.selected,
        "selected_pairwise": res.pairwise.selected,
        "per_learner_gap": res.per_learner_gap,
    }
    path.write_text(json.dumps(rec, indent=2, allow_nan=True))
    return rec


def run_sweep(datasets=None, seeds=(0, 1, 2), force: bool = False) -> pd.DataFrame:
    """One row per target, learned quantities averaged over seeds."""
    names = list(datasets) if datasets is not None else list(ma.DATASETS)
    rows = []
    for name in names:
        per_seed = []
        for seed in seeds:
            try:
                per_seed.append(measure_target(name, seed, force=force))
            except Exception as exc:                      # a bad target must not lose the rest
                log.error("target %s seed %s failed: %s", name, seed, exc)
        if not per_seed:
            continue
        head = per_seed[0]
        row = {k: v for k, v in head.items() if k not in {"seed", "per_learner_gap"}}
        for field in ("rmse_pointwise", "rmse_knn_floor", "rmse_pairwise",
                      "rmse_cliff_pointwise", "rmse_cliff_knn_floor", "rmse_cliff_pairwise",
                      "gap", "access", "correction"):
            row[field] = float(np.nanmean([r[field] for r in per_seed]))
        row["n_seeds"] = len(per_seed)
        rows.append(row)
    return pd.DataFrame(rows)
