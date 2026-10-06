import re
import subprocess

import pytest
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data.moleculeace import DATASETS, RECEPTOR_CLASSES


def test_loads_and_has_the_sections_the_sweep_reads():
    p = load_prereg()
    for key in ("representation", "graphs", "task_sets", "independent_variables",
                "arms", "dependent_variables", "analysis", "hodge", "forbidden"):
        assert key in p, key


def test_representation_matches_the_code():
    from molace.graphs.fingerprints import N_BITS, RADIUS
    r = load_prereg()["representation"]
    assert (r["radius"], r["n_bits"], r["binary"]) == (RADIUS, N_BITS, True)


def test_primary_graph_k_and_anchor_m_are_separate_constants():
    p = load_prereg()
    assert p["graphs"]["primary"]["k"] == 10
    assert p["arms"]["knn_floor"]["m"] == 10
    # Same value today, different constants. The test exists so a change to one does not
    # silently move the other.
    assert "m" not in p["graphs"]["primary"]
    assert "k" not in p["arms"]["knn_floor"]


def test_cluster_sizes_match_the_shipped_metadata():
    declared = load_prereg()["analysis"]["uncertainty"]["clusters"]
    actual = {}
    for d in DATASETS:
        c = RECEPTOR_CLASSES[d]
        actual[c] = actual.get(c, 0) + 1
    assert declared == actual


def test_spine_task_count_matches_the_data():
    assert load_prereg()["task_sets"]["spine"]["n_tasks"] == len(DATASETS)


def test_fingerprint_is_a_git_blob_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", prereg_fingerprint())


def test_fingerprint_is_committed_so_results_are_anchored():
    # A dirty pre-registration cannot anchor a result.
    out = subprocess.run(["git", "status", "--porcelain", "prereg/increment1.yaml"],
                         capture_output=True, text=True, encoding="utf-8").stdout
    assert out.strip() == "", f"prereg/increment1.yaml is uncommitted: {out!r}"
