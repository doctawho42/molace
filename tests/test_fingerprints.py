import numpy as np
import pytest
from molace.graphs import fingerprints as fpmod


def test_shape_and_dtype():
    fp = fpmod.ecfp4(["CCO", "CCC", "c1ccccc1"])
    assert fp.shape == (3, 2048)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp)) <= {0, 1}


def test_identical_smiles_give_identical_rows():
    fp = fpmod.ecfp4(["CCO", "CCO"])
    assert np.array_equal(fp[0], fp[1])


def test_unparseable_smiles_raises_and_names_the_offender():
    with pytest.raises(ValueError) as e:
        fpmod.ecfp4(["CCO", "not_a_molecule", "CCC"])
    assert "index 1" in str(e.value)
    assert "not_a_molecule" in str(e.value)


def test_tanimoto_is_one_on_the_diagonal_and_symmetric():
    T = fpmod.tanimoto_matrix(fpmod.ecfp4(["CCO", "CCC", "c1ccccc1O"]))
    assert np.allclose(np.diag(T), 1.0)
    assert np.allclose(T, T.T)
    assert T.min() >= 0.0 and T.max() <= 1.0


def test_tanimoto_matches_the_hand_computed_definition():
    fp = fpmod.ecfp4(["CCO", "CCCCO"])
    a, b = fp[0].astype(bool), fp[1].astype(bool)
    expected = (a & b).sum() / (a | b).sum()
    assert fpmod.tanimoto_matrix(fp)[0, 1] == pytest.approx(expected, abs=1e-6)


def test_empty_fingerprint_row_does_not_divide_by_zero():
    # a molecule with no set bits would make the union zero; force the degenerate case
    fp = np.zeros((2, 2048), dtype=np.uint8)
    T = fpmod.tanimoto_matrix(fp)
    assert T[0, 1] == 0.0
    assert np.allclose(np.diag(T), 1.0)
