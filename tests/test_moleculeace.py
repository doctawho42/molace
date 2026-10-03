import numpy as np
import pytest
from molace.data import moleculeace as ma


def test_thirty_datasets_and_total_molecule_count():
    assert len(ma.DATASETS) == 30
    assert sum(len(ma.load_target(d)) for d in ma.DATASETS) == 48714


def test_size_extremes_match_the_published_corrected_data():
    sizes = {d: len(ma.load_target(d)) for d in ma.DATASETS}
    assert min(sizes.values()) == 615
    assert max(sizes.values()) == 3657


def test_label_is_the_potency_column_not_the_standardised_one():
    df = ma.load_target("CHEMBL2835_Ki")
    # y [pEC50/pKi] is positive potency; the `y` column is standardised and goes negative.
    assert df["y"].min() > 0.0
    assert 8.0 < df["y"].mean() < 9.5


def test_cliff_flags_come_from_the_csv_not_the_stale_metadata_table():
    # metadata/datasets.csv describes pre-Correction data and reports 120 for this target.
    assert int(ma.load_target("CHEMBL2971_Ki")["cliff_mol"].sum()) == 162


def test_split_column_is_train_test_only():
    df = ma.load_target("CHEMBL4203_Ki")
    assert set(df["split"]) == {"train", "test"}
    assert (df["split"] == "train").sum() == 582


def test_receptor_classes_are_the_six_published_blocks():
    counts = {}
    for d in ma.DATASETS:
        counts[ma.receptor_class(d)] = counts.get(ma.receptor_class(d), 0) + 1
    assert counts == {"GPCR": 12, "NR": 6, "Kinase": 6, "Other": 3,
                      "Protease": 2, "Transferase": 1}


def test_unknown_target_raises():
    with pytest.raises(KeyError):
        ma.load_target("CHEMBL_nope")
