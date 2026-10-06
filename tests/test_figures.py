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
    caption = (tmp_path / "captions.md").read_text(encoding="utf-8")
    assert prereg_fingerprint()[:12] in caption


def test_captions_state_n_and_n_eff(tmp_path):
    figures.make_figures(out_dir=tmp_path)
    caption = (tmp_path / "captions.md").read_text(encoding="utf-8")
    assert "n_eff" in caption and "n = 30" in caption


def test_a_missing_result_file_raises_rather_than_drawing_an_empty_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "SPINE", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        figures.make_figures(out_dir=tmp_path)


def test_fig1_y_axis_covers_the_gap_not_a_rank(tmp_path):
    """The plan plotted a rank-space fit on an RMSE axis, so every point collapsed to one line.

    The reviewer opened the PNG: the axis ran to 16 while the data span about -0.3 to +0.02. The
    headline figure has to be readable, so the bogus reference line is gone and the axis limits are
    asserted to sit inside the data's own range.
    """
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    figures.make_figures(out_dir=tmp_path)
    spine = pd.read_csv(figures.SPINE)
    lo, hi = float(spine["gap"].min()), float(spine["gap"].max())
    span = hi - lo
    ylo, yhi = figures.LAST_FIG1_YLIM
    assert ylo >= lo - 2 * span
    assert yhi <= hi + 2 * span


def test_fig3_marks_the_unavailable_split_rather_than_drawing_it_as_zero(tmp_path):
    """27 of 30 targets have no computable curl/harmonic split.

    Plotting them as absent bars reads as a measured zero, and in the direction of the project's own
    hypothesis. The panel must mark them, and its caption must not describe the randomised estimator
    that was removed.
    """
    import pandas as pd

    figures.make_figures(out_dir=tmp_path)
    census = pd.read_csv(figures.CENSUS)
    n_unavailable = int((census["rank_method"] == "unavailable").sum())
    assert figures.LAST_FIG3_UNAVAILABLE == n_unavailable
    caption = (tmp_path / "captions.md").read_text(encoding="utf-8")
    assert "UNAVAILABLE" in caption
    # The caption may explain why the estimator was removed; it must not describe it as in use.
    assert "estimated by a randomised" not in caption
    assert "whose curl rank was estimated" not in caption
