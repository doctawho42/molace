"""Four panels, each from one committed CSV.

A missing input raises and names the path rather than drawing an empty panel, because an empty
panel reads as a measured absence. Every caption carries the blob hash of the frozen plan plus the
nominal and effective sample sizes, so a figure separated from this repository still says what plan
produced it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molace.analysis.correlate import cluster_bootstrap_spearman, effective_n
from molace.analysis.prereg import prereg_fingerprint

RESULTS = Path(__file__).resolve().parents[3] / "results"
SPINE = RESULTS / "spine.csv"
CENSUS = RESULTS / "hodge_census.csv"
ABLATION = RESULTS / "readout_ablation.csv"

PALETTE = {"GPCR": "#4C72B0", "Kinase": "#DD8452", "NR": "#55A868",
           "Other": "#C44E52", "Protease": "#8172B3", "Transferase": "#937860"}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the task that writes it before plotting")
    return pd.read_csv(path)


def _fig1(spine: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for cls, grp in spine.groupby("receptor_class"):
        ax.scatter(grp["assortativity"], grp["gap"], s=44, alpha=0.85,
                   color=PALETTE.get(cls, "#666666"), label=cls, edgecolor="white", linewidth=0.6)
    ok = spine[["rogi", "gap"]].notna().all(axis=1)
    if ok.sum() > 2:
        z = np.polyfit(spine.loc[ok, "rogi"].rank(), spine.loc[ok, "gap"].rank(), 1)
        xs = np.linspace(spine["assortativity"].min(), spine["assortativity"].max(), 10)
        ax.plot(xs, np.full_like(xs, np.polyval(z, spine.loc[ok, "rogi"].rank().mean())),
                "--", color="#999999", linewidth=1.2, label="roughness-only level")
    b = cluster_bootstrap_spearman(spine["assortativity"], spine["gap"],
                                   spine["receptor_class"], 10000, 0)
    ax.axhline(0.0, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.set_xlabel("target assortativity")
    ax.set_ylabel("gap  (RMSE pointwise - RMSE pairwise)")
    ax.set_title(f"rho = {b.rho:+.3f}   95% CI [{b.lo:+.3f}, {b.hi:+.3f}]", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    path = out / "fig1_assortativity_vs_gap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig2(spine: pd.DataFrame, out: Path) -> Path:
    d = spine.sort_values("gap").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(d))
    ax.bar(x, d["access"], color="#4C72B0", label="access (label access buys)")
    ax.bar(x, d["correction"], bottom=d["access"], color="#DD8452",
           label="correction (the learned function buys)")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("RMSE difference")
    ax.set_title("heights are error differences, not shares: RMSE is nonlinear", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig2_decomposition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig3(census: pd.DataFrame, out: Path) -> Path:
    d = census.sort_values("dim_harmonic", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for off, col, colour, name in ((-w, "dim_gradient", "#4C72B0", "gradient"),
                                   (0.0, "dim_curl", "#DD8452", "curl"),
                                   (w, "dim_harmonic", "#55A868", "harmonic")):
        hatch = ["//" if m == "estimated" else "" for m in d["rank_method"]]
        bars = ax.bar(x + off, d[col], width=w, color=colour, label=name)
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("subspace dimension")
    ax.set_title("hatched bars: rank estimated, not computed exactly", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig3_hodge_census.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig4(abl: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    floor = 1e-16
    for col, colour, name in (("curl_linear", "#4C72B0", "bias-free linear head"),
                              ("curl_mlp", "#DD8452", "MLP head")):
        ax.scatter(np.arange(len(abl)), np.maximum(abl[col], floor), s=36,
                   color=colour, label=name, edgecolor="white", linewidth=0.5)
    ax.axhline(floor, color="#999999", linestyle="--", linewidth=1.0, label="machine zero")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(abl)))
    ax.set_xticklabels(abl["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("curl fraction of flow energy")
    ax.set_title("a linear readout is a gradient flow; a nonlinear one is not", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig4_readout_ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_figures(out_dir="results/figures") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spine = _read(SPINE)
    census = _read(CENSUS)
    abl = _read(ABLATION)
    paths = [_fig1(spine, out), _fig2(spine, out), _fig3(census, out), _fig4(abl, out)]

    stamp = prereg_fingerprint()
    n = len(spine)
    n_eff = round(effective_n(spine["receptor_class"]), 1)
    captions = [
        f"All panels produced under pre-registration {stamp[:12]} "
        f"(full hash {stamp}). n = {n} tasks, n_eff = {n_eff} "
        "at an assumed intra-receptor-class correlation of 0.3.",
        "",
        "**fig1** Target assortativity against the pointwise-minus-pairwise gap, one point per "
        "target, coloured by receptor class. The interval is a cluster bootstrap over receptor "
        "classes, not over rows. The panel licenses a statement about association across tasks; it "
        "does not license a causal claim, and it does not by itself show the statistic adds "
        "anything to roughness, which is the separate incremental quantity in the report.",
        "",
        "**fig2** The gap split into what test-time label access buys and what the learned pairwise "
        "function buys, per target. Heights are error differences in the label's units. They are "
        "**not** shares of the gap: RMSE is nonlinear, so a percentage would be meaningless.",
        "",
        "**fig3** Hodge subspace dimensions per target on the pre-registered kNN graph. Hatched "
        "bars mark targets whose curl rank was estimated by a randomised range finder rather than "
        "computed exactly, which happens above the stated edge count. An estimated bar does not "
        "license the precision an exact one does.",
        "",
        "**fig4** Curl fraction of the trained edge flow under a bias-free linear readout and under "
        "an MLP readout, both on the frozen fingerprint encoder, log scale with the machine-zero "
        "line drawn. The linear result is a theorem being checked, not a measurement; the MLP "
        "magnitudes are measurements and are specific to this encoder and these graphs.",
    ]
    (out / "captions.md").write_text("\n".join(captions))
    return paths
