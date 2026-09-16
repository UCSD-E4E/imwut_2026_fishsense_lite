"""Pins for the within-group repeatability behind PAPER.md section 4.5.

The field figure is the one result in section 4.5 that needs nothing external:
a calibration error is common to every frame of one fish and cancels in a
relative spread, as does any error in the length convention, and no comparison
population is involved. Two of these tests pin exactly that -- rescaling a
dive's lengths must not move its fish's CVs -- because it is the property the
paper's claim rests on, not an implementation detail.

The two headline numbers are pinned against committed extractions
(`data/field.csv` from `sql/extract_field.sql`, `data/corpus.csv` from
`sql/extract_corpus.sql`) so they are reproducible from the repository rather
than from whatever a notebook last held in memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal
from fishsense_imwut import repeatability as rep

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


# --- the estimator ----------------------------------------------------------


def test_a_group_measured_identically_has_zero_spread():
    assert rep.cv_percent([0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_the_cv_uses_the_sample_standard_deviation():
    # ddof=1, not the population divisor: these are samples of a repeat
    # measurement, and n is 3 often enough that the difference matters.
    values = [1.0, 2.0, 3.0]
    assert rep.cv_percent(values) == pytest.approx(100.0 * 1.0 / 2.0)


def test_the_cv_does_not_depend_on_the_scale():
    """The load-bearing property: a wrong baseline scales every frame of a fish
    by the same factor, so it cancels. This is why section 4.5 can quote the
    field repeatability with no reference object in the water."""
    values = np.array([0.31, 0.33, 0.29, 0.32])
    for factor in (0.5, 1.0, 1.37, 4.0):
        assert rep.cv_percent(values * factor) == pytest.approx(
            rep.cv_percent(values)
        )


def test_groups_below_the_frame_floor_are_dropped_not_counted_as_zero():
    groups = {"a": [1.0, 1.1, 0.9], "b": [1.0, 1.0], "c": [2.0]}
    result = rep.repeatability(groups, min_frames=3)
    assert result.n_groups == 1
    assert list(result.cvs_percent.index) == ["a"]


def test_the_p90_is_nearest_rank_not_interpolated():
    """Section 4.5 quotes p90 by the same rule as the rest of the paper and the
    pipeline -- ceil(0.9n) -- and an interpolated p90 is a different number on
    the 25 field individuals. A draft of the paper quoted the interpolated one."""
    groups = {
        str(i): [1.0, 1.0 + step, 1.0 - step]
        for i, step in enumerate(np.linspace(0.001, 0.25, 25))
    }
    result = rep.repeatability(groups, min_frames=3)
    ordered = np.sort(result.cvs_percent.to_numpy())
    assert result.n_groups == 25
    assert result.p90_percent == pytest.approx(ordered[int(np.ceil(0.9 * 25)) - 1])
    assert result.p90_percent != pytest.approx(np.percentile(ordered, 90))


def test_the_bootstrap_interval_brackets_the_median_and_is_reproducible():
    groups = {
        str(i): [1.0, 1.0 + s, 1.0 - s] for i, s in enumerate(np.linspace(0.01, 0.2, 30))
    }
    first = rep.repeatability(groups, seed=7)
    assert rep.repeatability(groups, seed=7).ci_percent == first.ci_percent
    lo, hi = first.ci_percent
    assert lo <= first.median_percent <= hi


def test_an_empty_cohort_reports_no_groups_rather_than_raising():
    result = rep.repeatability({"a": [1.0, 2.0]}, min_frames=3)
    assert result.n_groups == 0
    assert np.isnan(result.median_percent)


# --- the field cohort -------------------------------------------------------


@pytest.fixture(name="field")
def _field() -> pd.DataFrame:
    return rep.load_field(DATA / "field.csv")


def test_the_field_set_is_162_measurements_of_73_individuals(field):
    assert len(field) == 162
    assert field["fish_id"].nunique() == 73
    assert sorted(field["dive_id"].unique()) == [279, 341, 347, 349, 383, 465, 471]


def test_every_field_measurement_names_the_calibration_it_used(field):
    """A NULL here is an orphaned row -- a frame re-bound between Fish rows
    whose old measurement survived (FINDINGS section 9.4). Two of them reached
    a draft and moved the individual count; they were deleted in prod, and this
    is the tripwire that says so."""
    assert field["laser_extrinsics_id"].notna().all()


def test_the_two_repaired_dives_are_measured_under_their_refits(field):
    # Extrinsics 15 and 53 were the ill-conditioned fits; deleting the rows
    # forced new ids, which is what made stage 14 recompute the lengths.
    repaired = field.loc[field["dive_id"].isin([347, 349]), "laser_extrinsics_id"]
    assert set(repaired) == {54, 55}


def test_the_field_repeatability_is_what_section_4_5_reports(field):
    result = rep.repeatability(rep.by_individual(field))
    assert result.n_groups == 25
    assert result.median_percent == pytest.approx(2.91, abs=0.01)
    assert result.p90_percent == pytest.approx(11.44, abs=0.01)
    lo, hi = result.ci_percent
    assert (lo, hi) == (pytest.approx(1.63, abs=0.01), pytest.approx(4.06, abs=0.01))


def test_the_reported_interval_does_not_depend_on_the_seed(field):
    """The seed is fixed for reproducibility, not to pick a favourable interval.
    Section 4.5 quotes one decimal place, so the bound must be stable well
    inside that -- otherwise the interval is a property of the seed."""
    groups = rep.by_individual(field)
    bounds = [rep.repeatability(groups, seed=s).ci_percent for s in range(8)]
    lows = [lo for lo, _ in bounds]
    highs = [hi for _, hi in bounds]
    assert max(lows) - min(lows) < 0.05
    assert max(highs) - min(highs) < 0.05


def test_rescaling_one_dives_lengths_leaves_the_field_figure_alone(field):
    """The same invariance as `test_the_cv_does_not_depend_on_the_scale`, but on
    the real cohort and per dive -- which is the form section 4.5 uses when it
    says the two repaired calibrations could not have changed this number."""
    baseline = rep.repeatability(rep.by_individual(field))
    shifted = field.copy()
    wrong = shifted["dive_id"] == 347
    shifted.loc[wrong, "length_m"] = shifted.loc[wrong, "length_m"] * 1.18
    after = rep.repeatability(rep.by_individual(shifted))
    assert after.n_groups == baseline.n_groups
    assert after.median_percent == pytest.approx(baseline.median_percent)
    assert after.p90_percent == pytest.approx(baseline.p90_percent)


def test_no_field_individual_spans_the_range_the_trend_check_needs(field):
    """Section 4.5 says the scale-free range check cannot be applied in the
    field. The reason is a property of this sample, so pin it: the check wants
    one rigid object over a >=2x range spread."""
    spreads = field.groupby("fish_id")["range_m"].agg(lambda s: s.max() / s.min())
    assert spreads.max() < 2.0


# --- the pool comparison ----------------------------------------------------


def test_the_pool_repeatability_is_what_section_4_5_reports():
    """The identical statistic on the pool cohort: repeat frames of one target
    in one session. Computed from the CURRENT corpus -- the draft's 1.4 % and
    p90 4.5 % came from the frozen August export, before six mislabelled frames
    were corrected, and both moved when they were.

    Re-derived 2026-09-16 after the 0.04217 m pitch correction added dives 503
    and 504 to the cohort: 29 -> 31 groups and 908 -> 1001 frames. Holding out
    the ruler later the same day took it to 30 groups and 995 frames; its six
    frames were one group, and losing it moved the median 1.364 -> 1.426 %
    while leaving p90 at 3.196 -- the ruler was one of the TIGHTER groups, as a
    board held at a steady wrong angle would be. Note this
    statistic barely moved (median 1.326 -> 1.364 %, p90 3.271 -> 3.196), which
    is the expected behaviour and worth stating -- repeatability is a SPREAD
    within one session at one scale, so a pitch change is very nearly common
    mode inside each group and cancels. The accuracy medians moved; this did
    not. If a scale correction ever moves this number much, suspect the
    correction is not uniform across the session."""
    rows = cal.load_rows(DATA / "corpus.csv")
    df = cal.to_frame(rows, corrected_references=True)
    df = df[df["dive_id"].isin(cal.CORPUS_ACCURACY_DIVES)]
    df = df[~df.model_name.isin(cal.HELD_OUT_MODELS)]
    result = rep.repeatability(rep.by_session_target(df))
    assert result.n_groups == 30
    assert result.n_values == 995
    assert result.median_percent == pytest.approx(1.43, abs=0.01)
    assert result.p90_percent == pytest.approx(3.20, abs=0.01)


def test_the_field_and_pool_intervals_overlap(field):
    """A draft claimed they did not, on figures that included the two orphaned
    rows. They do, so section 4.5 reports a difference in point estimate."""
    rows = cal.load_rows(DATA / "corpus.csv")
    pool_df = cal.to_frame(rows, corrected_references=True)
    pool_df = pool_df[pool_df["dive_id"].isin(cal.CORPUS_ACCURACY_DIVES)]
    pool_df = pool_df[~pool_df.model_name.isin(cal.HELD_OUT_MODELS)]
    pool = rep.repeatability(rep.by_session_target(pool_df))
    field_result = rep.repeatability(rep.by_individual(field))
    assert field_result.ci_percent[0] < pool.ci_percent[1]
    assert field_result.median_percent > 2 * pool.median_percent


# --- the one-estimator convention, and where it degenerates -----------------


def test_every_field_p90_is_that_animals_longest_frame(field):
    """§4.3's caveat, pinned on the data it is about.

    The paper reports p90 by nearest rank wherever frames become a length, the
    field included. `ceil(0.9n)` is n itself for every n <= 10, and no wild
    animal here carries more than eight frames, so a field p90 is always that
    animal's maximum. That is intended -- with a one-sided error the longest
    frame is the least pose-corrupted -- but the claim is stated in §4.3, §4.6
    and three captions, so it is pinned rather than trusted.
    """
    from fishsense_imwut.pubfig import nearest_rank_p90

    per_fish = field.groupby("fish_id").length_m.agg(
        n="size", p90=nearest_rank_p90, longest="max"
    )
    assert per_fish.n.max() <= 8
    assert (per_fish.p90 == per_fish.longest).all()


def test_the_pool_is_where_p90_is_a_real_quantile(field):
    """The contrast that makes the caveat worth stating, rather than alarming.

    A pool (session, target) cell holds 26 frames at the median, so there
    `ceil(0.9n)` is a genuine high quantile and lands strictly below the
    maximum for most cells. The estimator is one convention; only its
    resolution differs between the two settings.
    """
    from fishsense_imwut.pubfig import nearest_rank_p90

    rows = [
        r
        for r in cal.load_rows(DATA / "corpus.csv")
        if int(r["dive_id"]) not in cal.NON_POOL_DIVES
    ]
    pool = cal.to_frame(rows)
    pool = pool[pool.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)]
    cells = pool.groupby(["dive_id", "model_name"]).length_m.agg(
        n="size", p90=nearest_rank_p90, longest="max"
    )
    assert cells.n.median() >= 20
    assert (cells.p90 < cells.longest).mean() > 0.5
