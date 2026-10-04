"""The graph constructions whose spread is the measurement.

Increment 1 deliberately fixed ONE fingerprint and ONE k, so that nothing could be tuned. That
discipline is about not choosing a construction after seeing a result. Here the construction is the
independent variable: the question is how much of a homophily measure's spread between datasets is
the data and how much is the graph the analyst built, and answering it requires varying exactly the
thing increment 1 held still.
"""
from __future__ import annotations

import numpy as np
import pytest

from molace.graphs.constructions import FINGERPRINTS, METRICS, fingerprint, similarity, threshold_graph

SMILES = [
    "CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CC(=O)N",
    "CCOCC", "CCCCO", "c1ccncc1", "CC(C)O", "CCCl",
]


@pytest.mark.parametrize("kind", FINGERPRINTS)
def test_every_fingerprint_returns_a_binary_matrix_one_row_per_molecule(kind):
    fp = fingerprint(SMILES, kind)
    assert fp.shape[0] == len(SMILES)
    assert fp.ndim == 2 and fp.shape[1] > 0
    assert set(np.unique(fp)).issubset({0, 1})


def test_the_fingerprints_are_actually_different_from_each_other():
    mats = {k: fingerprint(SMILES, k) for k in FINGERPRINTS}
    sims = {k: similarity(m, "tanimoto") for k, m in mats.items()}
    seen = []
    for k, s in sims.items():
        for prev_k, prev in seen:
            assert not np.allclose(s, prev), f"{k} and {prev_k} give identical similarity matrices"
        seen.append((k, s))


@pytest.mark.parametrize("metric", METRICS)
def test_similarity_is_symmetric_with_a_unit_diagonal(metric):
    s = similarity(fingerprint(SMILES, "ecfp4"), metric)
    assert np.allclose(s, s.T)
    assert np.allclose(np.diag(s), 1.0)
    assert s.min() >= -1e-12 and s.max() <= 1.0 + 1e-12


def test_dice_is_never_below_tanimoto():
    """A closed-form invariant: D = 2c/(a+b) and T = c/(a+b-c) give D >= T for every pair."""
    fp = fingerprint(SMILES, "ecfp4")
    assert (similarity(fp, "dice") + 1e-12 >= similarity(fp, "tanimoto")).all()


def test_threshold_graph_gets_sparser_as_the_threshold_rises():
    s = similarity(fingerprint(SMILES, "ecfp4"), "tanimoto")
    edges = [threshold_graph(s, t).number_of_edges() for t in (0.1, 0.3, 0.5, 0.7)]
    assert edges == sorted(edges, reverse=True)
    assert edges[0] > edges[-1], "the thresholds must actually separate, or the test says nothing"


def test_unknown_names_are_refused_rather_than_silently_defaulted():
    with pytest.raises(KeyError):
        fingerprint(SMILES, "not-a-fingerprint")
    with pytest.raises(KeyError):
        similarity(fingerprint(SMILES, "ecfp4"), "not-a-metric")


def test_an_unparseable_smiles_names_itself():
    with pytest.raises(ValueError, match="index 1"):
        fingerprint(["CCO", "this-is-not-a-molecule"], "ecfp4")
