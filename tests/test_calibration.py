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
                    "depth_m": 1.0,  # one range: the range-trend filter abstains
                    "baseline_m": 0.103,
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
    assert 490 not in cal.DESIGN_EXCLUDED_DIVES  # was never a design call
    assert cal.SPLIT_DIVES == {490: 527}


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
        59, 61, 84, 495, 497, 498, 500, 501, 507, 519, 521, 522, 527
    )


def test_correcting_one_reference_costs_two_cohort_members():
    """The analysis's headline sensitivity, pinned so it cannot be lost.

    Measuring the Weasly Fish (0.310 -> 0.3127 m, +0.86 %) drops dives 59 and
    497. Neither measures that model — 59 is Grouper/Snook/Shark/Purple Angel
    and 497 is all Box — so their calibrations did not change. The grid is
    unbalanced (8 of 32 dives measure only the Weasly Fish), so a per-model
    change moves every dive effect by ~0.5 pp, and those two sat 0.42 and
    0.46 pp inside the 2.5 pp bound."""
    frozen = DATA / "corpus_20260912.csv"
    raw = cal.to_frame(cal.load_rows(frozen), corrected_references=False)
    corrected = cal.to_frame(cal.load_rows(frozen))
    assert cal.accuracy_cohort(raw) == cal.CORPUS_ACCURACY_DIVES_AS_EXPORTED
    assert set(cal.accuracy_cohort(raw)) - set(cal.accuracy_cohort(corrected)) == {59, 497}


def test_the_measured_reference_is_the_tape_value():
    """Two independent tape readings, snout to fork: 12 5/16 in = 312.74 mm and
    312-313 mm metric, so 312.7 +- 0.5 mm. Adopted 313 mm -- three significant
    figures like every other reference, which the +-0.5 mm supports and a
    tenth-millimetre digit does not. The cohort is the same for any value in
    that spread, so the rounding costs nothing."""
    assert cal.MEASURED_REFERENCES_M["Weasly Fish"] == 0.313
    rows = cal.load_rows(DATA / "corpus.csv")
    for metres in (0.3125, 0.3127, 0.3127375, 0.313):
        df = cal.to_frame(rows, corrected_references=False)
        mask = df.model_name == "Weasly Fish"
        df.loc[mask, "known_length_m"] = metres
        df["pct_error"] = 100 * (df.length_m / df.known_length_m - 1)
        assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES


def test_the_range_trend_filter_does_not_depend_on_any_reference():
    """It compares a rigid target against ITSELF across range, so no reference
    length enters. Pinned because it is what makes the pre-filter trustworthy
    when a reference moves."""
    frozen = DATA / "corpus_20260912.csv"
    raw = cal.to_frame(cal.load_rows(frozen), corrected_references=False)
    corrected = cal.to_frame(cal.load_rows(frozen))
    assert cal.range_trend_flagged_dives(raw) == cal.range_trend_flagged_dives(corrected)


# --- range-trend pre-filter ------------------------------------------------


def test_theil_sen_is_exact_on_a_line_and_robust_to_one_outlier():
    x = np.linspace(1, 4, 20)
    y = 2.0 + 0.5 * x
    y[3] += 30.0
    slope, lo, hi = cal.theil_sen(x, y)
    assert slope == pytest.approx(0.5, abs=1e-9)
    assert lo <= slope <= hi


def test_range_trend_recovers_an_injected_angle_without_a_known_length():
    z = np.linspace(0.9, 3.0, 40)
    eps = np.radians(-0.25)
    lengths = 0.31 * (1 + eps * z / 0.103)
    trend = cal.range_trend(z, lengths, baseline_m=0.103)
    assert trend["eps_deg"] == pytest.approx(-0.25, abs=0.01)
    assert trend["flagged"]


def test_range_trend_needs_frames_beyond_0_8_m_over_a_2x_spread():
    z = np.linspace(0.3, 0.7, 30)
    assert cal.range_trend(z, np.full(30, 0.31), 0.103) is None
    z = np.linspace(1.0, 1.5, 30)
    assert cal.range_trend(z, np.full(30, 0.31), 0.103) is None


def test_to_frame_carries_the_dive_baseline():
    """The column the range-trend filter needs, on both exports.

    Dive 503 is the interesting one: it borrows dive 502's calibration, which
    the 2026-09-12 export caught at 8.90 cm — below the 9.7 cm floor — and
    which prod refitted to 10.35 cm on 2026-09-14 once the shipped gates made
    the old fit read as uncalibrated. Both values are pinned so a future edit
    cannot silently swap one export for the other."""
    fresh = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    frozen = cal.to_frame(cal.load_rows(DATA / "corpus_20260912.csv"))
    assert "baseline_m" in fresh.columns
    assert frozen[frozen.dive_id == 503].baseline_m.iloc[0] == pytest.approx(0.0890, abs=5e-4)
    assert fresh[fresh.dive_id == 503].baseline_m.iloc[0] == pytest.approx(0.1035, abs=5e-4)
    assert fresh[fresh.dive_id == 500].baseline_m.iloc[0] == pytest.approx(0.1041, abs=5e-4)


def test_range_trend_flagged_dives_on_the_corpus():
    """Both signs, whole interval beyond +-2 %/m, angle dives excluded. The
    flagged set is every dive already known bad from the known lengths plus
    the short-baseline dives the median had hidden -- and no sound one."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.range_trend_flagged_dives(df) == (76, 491, 492, 494, 503, 504, 509)


def test_the_pre_filter_is_what_removes_491_503_and_504():
    """All three clear the dive-effect bound and are removed by the range trend
    alone. 503 and 504 borrow dive 502's calibration, which prod refitted from
    8.90 cm to a sound 10.35 cm on 2026-09-14 — so a plausible baseline is not
    sufficient, and the scale-free trend still says their lengths vary with
    range. 491 borrows across the mid-session laser movement of §4.2."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    with_filter = cal.accuracy_cohort(df)
    without = cal.accuracy_cohort(df, range_trend_filter=False)
    assert set(without) - set(with_filter) == {491, 503, 504}


def test_the_dive_84_relabels_are_in_the_corpus():
    """The two corrected frames, pinned by the quantity that identified them.

    A purple angelfish reads ~192 mm; as a snook it would read -58 %. Both
    frames now sit in dive 84's purple-angel population (180.9-194.2 mm) and
    nowhere near its snook population (434.9-451.4 mm), and the session reads
    16 purple angel to 13 snook rather than 14 to 15."""
    corpus = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    d84 = corpus[corpus.dive_id == 84]
    counts = d84.model_name.value_counts().to_dict()
    assert counts["Purple Angel"] == 16
    assert counts["Snook"] == 13
    assert d84[d84.model_name == "Snook"].length_m.min() > 0.40
    assert d84[d84.model_name == "Purple Angel"].length_m.max() < 0.20


def test_the_dive_521_relabels_are_in_the_corpus():
    """Four frames labelled as the box are the trout: 292-304 mm, which is
    +95 to +102 % as a 150 mm box and within a few percent of the trout. Their
    implied head/tail spans sit 16-58 px from the nearest trout frame and
    309-659 px from the nearest box."""
    corpus = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    d521 = corpus[corpus.dive_id == 521]
    counts = d521.model_name.value_counts().to_dict()
    assert counts["Box"] == 42
    assert counts["Weasly Fish"] == 37
    assert d521[d521.model_name == "Box"].length_m.max() < 0.20


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


def test_the_rejected_sessions_are_what_the_limitations_paragraph_reports():
    """PAPER.md's "Sessions the rule rejects" states three counts about the same
    nine dives. A draft said six of them borrow their calibration; eight do,
    which makes borrowing the strongest predictor of rejection in the corpus
    rather than a co-equal one. Pin all three so the next edit cannot drift."""
    rows = cal.load_rows(DATA / "corpus.csv")
    df = cal.to_frame(rows, corrected_references=True)
    polished = cal.median_polish(cal.cell_p90_grid(df, cal.POLISH_MIN_FRAMES))
    considered = {int(d) for d in polished.dive_effect.index}
    rejected = sorted(
        considered - set(cal.accuracy_cohort(df)) - set(cal.DESIGN_EXCLUDED_DIVES)
    )
    assert rejected == [58, 491, 492, 494, 503, 504, 506, 509, 520]

    flagged = set(cal.range_trend_flagged_dives(df))
    assert len(set(rejected) & flagged) == 6

    source = df.groupby("dive_id").calibration_dive_id.first()
    borrowed = [d for d in rejected if int(source[d]) != d]
    assert len(borrowed) == 8
    assert [d for d in rejected if d not in borrowed] == [509]

    baselines = df.groupby("dive_id").baseline_m.first() * 100
    assert baselines[rejected].min() == pytest.approx(10.24, abs=0.01)
    assert baselines[rejected].max() == pytest.approx(10.51, abs=0.01)
