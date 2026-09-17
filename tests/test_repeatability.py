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


# --- how many frames p90 needs --------------------------------------------


@pytest.fixture(name="cohort_cells")
def _cohort_cells():
    rows = [r for r in cal.load_rows(DATA / "corpus.csv")
            if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    a = df[df.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)
           & ~df.model_name.isin(cal.HELD_OUT_MODELS)]
    return {k: v for k, v in rep.by_session_target(a, value="pct_error").items()
            if len(v) >= 30}


def test_the_rarefaction_never_draws_a_group_beyond_its_frames(cohort_cells):
    """n must never be inflated by resampling, or the whole figure is circular.

    Each draw is without replacement from a group that HAS n frames, so a group
    stops contributing once n passes its size rather than padding itself.
    """
    assert len(cohort_cells) == 15
    r = rep.p90_rarefaction(cohort_cells, sizes=(2, 30, 40, 200), draws=20)
    assert 2 in r and 30 in r and 40 in r
    assert 200 not in r, "no group has 200 frames, so n=200 must be absent"
    assert max(len(v) for v in cohort_cells.values()) >= 40


def test_the_two_statistics_are_not_interchangeable(cohort_cells):
    """Which statistic is which, pinned, because the figure mixes two levels.

    Per fish the estimate is p90 of percent length error -- the paper's
    estimator, because one fish's frames are a one-sided pose-corrupted
    distribution. ACROSS draws and cells the summary is a median, as §4.6 also
    summarises across animals: sampling error is not one-sided, so there is no
    tail to reject. `abs_p90` is kept because the minimum-frames threshold is a
    spread question, and at every n it sits well above the median draw -- a
    diver is not protected by being on the good side of a median.
    """
    r = rep.p90_rarefaction(cohort_cells, sizes=(5, 10, 20))
    for n in (5, 10, 20):
        assert r[n].abs_p90 > abs(r[n].median) + 0.3


def test_p90_is_the_maximum_below_ten_frames_and_that_is_the_step(cohort_cells):
    """The figure's whole claim, and it is arithmetic before it is empirical.

    ceil(0.9n) == n for every n <= 10, so below ten frames p90 takes the sample
    maximum -- the noisiest order statistic. Crossing the boundary drops the
    reported error with no extra information.
    """
    for n in range(2, 10):
        assert int(np.ceil(0.9 * n)) == n
    assert int(np.ceil(0.9 * 10)) == 9

    r = rep.p90_rarefaction(cohort_cells, sizes=(8, 9, 10, 11))
    assert r[9].rank == 9 and r[10].rank == 9
    assert r[9].abs_p90 == pytest.approx(1.56, abs=0.15)
    assert r[10].abs_p90 == pytest.approx(1.28, abs=0.15)
    assert r[10].abs_p90 < r[9].abs_p90


def test_the_signed_error_changes_sign_across_the_small_n_range(cohort_cells):
    """Why the signed band is kept alongside the unsigned curve: the maximum of
    two draws sits BELOW the true 90th percentile and the maximum of nine sits
    above it, so a small sample is not conservative in either direction."""
    r = rep.p90_rarefaction(cohort_cells, sizes=(2, 9))
    assert r[2].median < -0.3
    assert r[9].median > 0.1


def test_thirteen_frames_is_the_reported_minimum(cohort_cells):
    """Section 4.3's guidance: 13 frames puts p90 within 1 pp of its own limit
    90 % of the time, and every larger n tested stays there."""
    r = rep.p90_rarefaction(cohort_cells)
    assert rep.p90_min_frames(r) == 13
    assert r[13].abs_p90 <= 1.0
    assert r[9].abs_p90 > 1.0
    assert all(r[n].abs_p90 <= 1.0 for n in r if n >= 13)


def test_the_tolerance_actually_reaches_the_minimum(cohort_cells):
    """Regression: an earlier draft took a tolerance argument on the wrong
    function and ignored it. `p90_min_frames` now applies the threshold itself,
    reading `abs_p90`, so asking for a tighter one moves the answer."""
    r = rep.p90_rarefaction(cohort_cells)
    assert rep.p90_min_frames(r, tolerance=2.0) < rep.p90_min_frames(r)
    assert rep.p90_min_frames(r, tolerance=0.5) > rep.p90_min_frames(r)


# --- the budget view, and why it is a different figure -----------------------


def test_level_traces_are_kept_per_cell_because_pooling_flattens_them(cohort_cells):
    """The reason `p90_level_traces` exists next to `p90_rarefaction`.

    Pooled, the reported error's spread is dominated by the cells' own p90s --
    which differ by nearly 9 pp and do not depend on n -- so a pooled band is
    flat across the axis and describes between-cell variation while appearing
    to describe sample size. Per cell, each trace still settles.
    """
    sizes = (2, 5, 10, 30)
    rare = rep.p90_rarefaction(cohort_cells, sizes=sizes, draws=100)
    traces, worst = rep.p90_level_traces(cohort_cells, sizes=sizes, draws=100)

    pooled_width = {n: rare[n].level_hi - rare[n].level_lo for n in sizes}
    between = max(traces[k][30] for k in traces) - min(traces[k][30] for k in traces)
    # the pooled band never narrows to anything like the sampling term
    assert min(pooled_width.values()) > 0.5 * between
    # and it is the between-cell spread that sets its size
    assert pooled_width[30] == pytest.approx(between, rel=0.5)


def test_level_traces_settle_where_the_pooled_band_does_not(cohort_cells):
    sizes = (2, 3, 5, 10, 20, 30)
    traces, _ = rep.p90_level_traces(cohort_cells, sizes=sizes, draws=200)
    # every trace moves less between n=20 and n=30 than between n=2 and n=5
    early = [abs(t[5] - t[2]) for t in traces.values()]
    late = [abs(t[30] - t[20]) for t in traces.values()]
    assert np.median(late) < np.median(early)


def test_no_draw_at_any_n_leaves_the_error_budget(cohort_cells):
    """The budget figure's headline, and it is a negative result about sample
    size: frame count is not what puts a measurement outside 15 %.

    Pinned because it is the claim the paper makes from Figure D2, and because
    the worst case is the whole point -- a budget is a bound, not an average.
    """
    _, worst = rep.p90_level_traces(cohort_cells, sizes=(2, 3, 5, 10, 30), draws=500)
    assert max(abs(v) for v in worst.values()) < 15.0
    # the worst case is at the smallest n, where p90 is the max of two frames
    assert abs(worst[2]) == max(abs(v) for v in worst.values())


def test_level_traces_never_draw_a_cell_beyond_its_frames(cohort_cells):
    traces, worst = rep.p90_level_traces(cohort_cells, sizes=(2, 40, 200), draws=10)
    assert 200 not in worst
    assert all(200 not in t for t in traces.values())
    assert all(len(cohort_cells[k]) >= 40 for k, t in traces.items() if 40 in t)
