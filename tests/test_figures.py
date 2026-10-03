import pandas as pd
import pytest
from molace.analysis import figures


def test_every_figure_is_written(tmp_path):
    paths = figures.make_figures(out_dir=tmp_path)
    assert len(paths) == 4
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


def test_captions_carry_the_prereg_hash(tmp_path):
    from molace.analysis.prereg import prereg_fingerprint
    figures.make_figures(out_dir=tmp_path)
    caption = (tmp_path / "captions.md").read_text()
    assert prereg_fingerprint()[:12] in caption


def test_captions_state_n_and_n_eff(tmp_path):
    figures.make_figures(out_dir=tmp_path)
    caption = (tmp_path / "captions.md").read_text()
    assert "n_eff" in caption and "n = 30" in caption


def test_a_missing_result_file_raises_rather_than_drawing_an_empty_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "SPINE", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        figures.make_figures(out_dir=tmp_path)
