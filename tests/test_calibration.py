"""Pins for the corpus-cohort machinery in `fishsense_imwut.calibration`.

The August analysis hand-picked its accuracy dives; the corpus analysis derives
them from a rule. These tests pin the rule and the decomposition behind it so
a later edit cannot silently change which dives back the paper's numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


# --- median polish ----------------------------------------------------------


def test_median_polish_recovers_an_exactly_additive_grid():
    dives = [1, 2, 3, 4]
    models = ["A", "B", "C"]
    # Effects are identifiable only up to their median, so build them centred.
    dive_eff = pd.Series([-2.0, -1.0, 1.0, 3.0], index=dives)
    model_eff = pd.Series([-1.0, 0.0, 2.0], index=models)
    grid = pd.DataFrame(
        {m: 0.5 + dive_eff + model_eff[m] for m in models}, index=dives
    )

    fit = cal.median_polish(grid)

    assert fit.overall == pytest.approx(0.5)
    pd.testing.assert_series_equal(fit.dive_effect, dive_eff, check_names=False)
    pd.testing.assert_series_equal(fit.model_effect, model_eff, check_names=False)
    assert fit.residual.abs().max().max() == pytest.approx(0.0)


def test_median_polish_is_robust_to_a_single_outlying_cell():
    dives = [1, 2, 3, 4, 5]
    models = ["A", "B", "C"]
    grid = pd.DataFrame(0.0, index=dives, columns=models)
    grid.loc[3, "B"] = 40.0  # one wild cell must not move any effect

    fit = cal.median_polish(grid)

    assert fit.dive_effect.abs().max() == pytest.approx(0.0)
    assert fit.model_effect.abs().max() == pytest.approx(0.0)
    assert fit.residual.loc[3, "B"] == pytest.approx(40.0)


def test_median_polish_tolerates_missing_cells():
    grid = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3]
    )
    fit = cal.median_polish(grid)
    # Model B sits exactly 1 above model A wherever both are observed.
    assert fit.model_effect["B"] - fit.model_effect["A"] == pytest.approx(1.0)
    assert np.isnan(fit.residual.loc[3, "A"])


# --- cohort rule ------------------------------------------------------------


def _frame(cells):
    """(dive, model, [pct_errors]) -> tidy frame like `cal.to_frame`."""
    rows = []
    for dive, model, errors in cells:
        for e in errors:
            rows.append(
                {
                    "dive_id": dive,
                    "model_name": model,
                    "known_length_m": 0.3,
                    "length_m": 0.3 * (1 + e / 100),
                    "pct_error": e,
                }
            )
    return pd.DataFrame(rows)


def test_accuracy_cohort_keeps_dives_within_the_dive_effect_threshold():
    good = [0.0] * 6
    df = _frame(
        [
            (1, "A", good), (1, "B", good),
            (2, "A", [1.0] * 6), (2, "B", [1.0] * 6),  # +1 pp dive effect
            (3, "A", [-6.0] * 6), (3, "B", [-6.0] * 6),  # -6 pp: out
            (4, "A", [4.0] * 6), (4, "B", [4.0] * 6),  # +4 pp: out
        ]
    )
    assert cal.accuracy_cohort(df, max_dive_effect_pp=2.5, exclude=()) == (1, 2)


def test_accuracy_cohort_holds_out_named_dives_but_still_polishes_them():
    good = [0.0] * 6
    df = _frame(
        [
            (1, "A", good), (1, "B", good),
            (2, "A", good), (2, "B", good),
            (87, "A", good), (87, "B", [3.0] * 6),  # held out; still a row
        ]
    )
    assert cal.accuracy_cohort(df, exclude=(87,)) == (1, 2)
    assert cal.accuracy_cohort(df, exclude=()) == (1, 2, 87)
    # The held-out dive's cells still inform the model effects.
    fit = cal.median_polish(cal.cell_p90_grid(df))
    assert 87 in fit.dive_effect.index


def test_design_exclusions_are_fixed_before_any_corpus_number():
    assert set(cal.ANGLE_TEST_DIVES) <= set(cal.DESIGN_EXCLUDED_DIVES)
    assert {60, 76, 66} <= set(cal.DESIGN_EXCLUDED_DIVES)
    assert 490 not in cal.DESIGN_EXCLUDED_DIVES  # falls out on its own effect


def test_accuracy_cohort_ignores_cells_below_min_frames():
    good = [0.0] * 6
    df = _frame(
        [
            (1, "A", good), (1, "B", good),
            (2, "A", good), (2, "B", good),
            (3, "A", good), (3, "B", [-30.0] * 2),  # 2 frames: not a cell
        ]
    )
    assert cal.accuracy_cohort(df, min_frames=5, exclude=()) == (1, 2, 3)
    assert cal.accuracy_cohort(df, min_frames=2, exclude=()) == (1, 2)


def test_accuracy_cohort_on_the_real_corpus_is_the_published_set():
    """The paper's cohort, pinned. If this changes, Section 4 changes."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES
    assert cal.CORPUS_ACCURACY_DIVES == (
        59, 61, 84, 491, 495, 497, 498, 500, 501, 503, 507, 519, 520, 521, 522
    )


def test_corpus_is_a_superset_of_the_august_export():
    """Every August frame reappears in the corpus with the same length."""
    aug = cal.to_frame(cal.load_rows(DATA / "all.csv"))
    corpus = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    key = ["dive_id", "model_name", "length_m"]
    missing = aug.merge(corpus[key].drop_duplicates(), on=key, how="left", indicator=True)
    assert (missing["_merge"] == "both").all()


# --- angle experiment -------------------------------------------------------


def test_load_angles_reads_the_designed_experiment():
    ang = cal.load_angles(DATA / "angles.csv")
    assert set(ang.dive_id.unique()) == set(cal.ANGLE_TEST_DIVES)
    assert ang.fish_angle_degrees.between(0, 45).all()
    assert "pct_error" in ang.columns
    assert ang.pct_error.notna().all()


def test_binned_angle_error_uses_symmetric_bins_and_min_count():
    ang = pd.DataFrame(
        {
            "fish_angle_degrees": [-2, -1, 0, 1, 2, 18, 19, 20, 21, 22, 45],
            "pct_error": [-1, -2, -3, -4, -5, -8, -9, -10, -11, -12, -30],
        }
    )
    binned = cal.binned_angle_error(ang, bins=(0, 20, 45), half_width=2.5, min_count=5)
    assert binned.loc[0, "median"] == pytest.approx(-3.0)
    assert binned.loc[20, "median"] == pytest.approx(-10.0)
    assert np.isnan(binned.loc[45, "median"])  # 1 frame < min_count
