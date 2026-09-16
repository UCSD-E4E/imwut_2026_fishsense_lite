"""Pins for the 2023-08-03 paired-instrument comparison (FINDINGS section 12).

The numbers are pinned against the two committed extractions
(`data/stereo_pairs.csv` from `sql/extract_stereo_pairs.sql`, and
`data/stereo_reference.csv` lifted from the collaborators' archive) so the
comparison is reproducible from this repository rather than from a prod query
that may not return the same rows next week.

The paper reports ONE estimator, p90, everywhere a set of frames becomes a
length, so `test_p90_is_the_reported_estimator_between_the_other_two` is the
test that carries the analysis and the rest are context. An out-of-plane fish
can only read SHORT, so a per-fish median inherits the pose loss (-3.7 %, reads
as "we measure slightly short") while p90 rejects it (+3.5 %, reads as "four of
seven fish are longer than the stereo says"). Both are true of the same data;
the median is the wrong summary for a one-sided error, which is the same reason
section 4.3 reports p90 rather than a mean.

`test_the_estimator_choice_flips_the_sign` and the maximum-based tests are kept
because the spread between estimators is what justifies the choice -- not
because any of them is reported.
"""

from pathlib import Path

import numpy as np
import pytest

from fishsense_imwut import stereo_pairs as sp

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


@pytest.fixture(name="pairs")
def _pairs():
    ours = sp.load_ours(DATA / "stereo_pairs.csv")
    stereo = sp.load_stereo(DATA / "stereo_reference.csv")
    return sp.build_pairs(ours, stereo)


@pytest.fixture(name="ours")
def _ours():
    return sp.load_ours(DATA / "stereo_pairs.csv")


# --- the cohort --------------------------------------------------------------


def test_seven_individuals_are_paired(pairs):
    """Eight morning dives, seven with a stereo counterpart. The dog snapper
    (dive 8) has none -- it is measured but unpaired, and must not silently
    become an eighth pair."""
    assert len(pairs) == 7
    assert 8 not in [p.dive_id for p in pairs]
    assert sorted(p.dive_id for p in pairs) == [5, 16, 20, 25, 28, 35, 39]


def test_every_measurement_shares_one_calibration(ours):
    """The load-bearing property of the whole comparison: all eight dives
    borrow dive 32's fit, so a calibration scale error is COMMON-MODE and
    cannot produce disagreement *between* individuals. Without this the
    per-fish differences would be unattributable."""
    assert set(ours.laser_extrinsics_id.unique()) == {sp.SHARED_EXTRINSICS_ID}
    assert len(ours) == 41


# --- the estimator, which is the finding ------------------------------------


def test_the_estimator_choice_flips_the_sign(pairs):
    med = sp.summary(pairs, "median")
    mx = sp.summary(pairs, "max")
    assert med["median_pct"] == pytest.approx(-3.7, abs=0.2)
    assert mx["median_pct"] == pytest.approx(+8.6, abs=0.2)
    # the scatter is essentially unchanged -- it is the CENTRE that moves,
    # which is what makes this an estimator question and not a noise question
    assert med["sd_pct"] == pytest.approx(11.0, abs=0.3)
    assert mx["sd_pct"] == pytest.approx(10.9, abs=0.3)


def test_four_individuals_exceed_the_stereo_even_at_their_maximum(pairs):
    """Our maximum is a lower bound on the animal, so these four cannot be
    explained by our own pose loss. This is the residual that the comparison
    cannot attribute to us rather than to the stereo."""
    over = sp.over_measured(pairs)
    assert len(over) == 4
    assert sorted(p.dive_id for p in over) == [5, 16, 25, 28]
    assert min(p.max_diff_pct for p in over) > 8.0


def test_the_mean_difference_interval_spans_zero(pairs):
    """Seven pairs cannot establish a bias in either direction, under either
    estimator. Section 4.5's arithmetic, confirmed on real pairs."""
    for estimator in ("median", "max"):
        low, high = sp.summary(pairs, estimator)["ci95"]
        assert low < 0 < high


# --- what the comparison rules out ------------------------------------------


def test_the_shared_calibration_shows_no_range_trend(ours):
    """The scale-free check: a rigid object must read the same length at every
    distance. Dive 5 gives 11 frames over a 1.55x range spread, and the slope
    is flat -- so the shared 10.551 cm fit is not producing a range-dependent
    scale error, and the disagreement is not a bad baseline."""
    slope, r, n = sp.range_trend_pct_per_m(ours, 5)
    assert n == 11
    assert abs(slope) < 2.0
    assert abs(r) < 0.2

    spread = ours[(ours.dive_id == 5) & ours.range_m.notna()].range_m
    assert spread.max() / spread.min() > 1.5


def test_the_projection_is_self_consistent_and_one_sided(ours):
    """`length == head_tail_px * range / fx` to a fraction of a percent, and
    never above 1.0. The equality says the error is in the clicks or the depth
    rather than the maths; the bound is the direct evidence that pose only
    subtracts, which is what licenses using the maximum."""
    g = sp.geometry_consistency(ours)
    assert len(g) >= 16
    assert g.ratio.min() > 0.99
    assert g.ratio.max() <= 1.0


def test_landmark_over_reach_does_not_explain_the_signs(pairs):
    """Recorded because it was the first hypothesis and the frames refuted it.

    The two Blue Parrotfish are the same species on the same day with opposite
    signs. Inspecting the tail clicks: dive 25 (+8.0 %) lands in the fork
    notch, dive 39 (-10.3 %) lands out at the caudal fin tip. Clicking the tip
    LENGTHENS a measurement, so over-reach predicts the opposite of what is
    observed -- it cannot be the mechanism.
    """
    by_dive = {p.dive_id: p for p in pairs}
    assert by_dive[25].median_diff_pct > 0
    assert by_dive[39].median_diff_pct < 0
    assert by_dive[25].species == by_dive[39].species == "Blue Parrotfish"


# --- the p90 estimator, which is what Figure 15 draws -----------------------


def test_p90_is_the_reported_estimator_between_the_other_two(pairs):
    """Section 4.3's estimator applied to the paired day: +3.5 %.

    It sits between the median's -3.7 % and the maximum's +8.6 % because it is
    a high-order statistic of the same one-sided distribution, and it is the
    one Figure 15 draws -- the same estimator Figure 1 draws against the models,
    so the two figures can be read without learning a second convention.
    """
    s = sp.summary(pairs, "p90")
    assert s["median_pct"] == pytest.approx(+3.5, abs=0.2)
    assert s["mean_pct"] == pytest.approx(+0.8, abs=0.2)
    assert s["sd_pct"] == pytest.approx(10.1, abs=0.3)
    assert s["within_10_pct"] == 4

    med = sp.summary(pairs, "median")["median_pct"]
    mx = sp.summary(pairs, "max")["median_pct"]
    assert med < s["median_pct"] < mx


def test_nearest_rank_selects_the_top_sample_for_five_of_seven(pairs):
    """Why p90 lands close to the maximum here, stated so it cannot surprise.

    `ceil(0.9n)` is n itself for every n <= 10, and five of the seven fish have
    3 to 6 frames. p90 on this day is therefore a high-order statistic and not
    a tail estimate, and it is only the two best-sampled fish -- 11 and 10
    frames -- that separate it from the maximum at all.
    """
    at_max = [p for p in pairs if p.ours_p90_mm == p.ours_max_mm]
    assert len(at_max) == 5
    assert all(p.n_frames <= 10 for p in at_max)

    separated = [p for p in pairs if p.ours_p90_mm < p.ours_max_mm]
    assert sorted(p.dive_id for p in separated) == [5, 25]
    assert all(p.n_frames >= 10 for p in separated)


def test_four_individuals_still_exceed_the_stereo_at_p90(pairs):
    """The finding survives the softer estimator, which is the point of using it.

    Switching from the maximum to p90 costs the two best-sampled fish some
    length (+9.4 -> +3.5 % and +12.3 -> +10.7 %) and changes nothing about the
    membership: the same four individuals read longer than the stereo does, by
    3.5 to 11.2 %, and pose loss cannot produce a positive.
    """
    over = [p for p in pairs if p.p90_diff_pct > 0]
    assert sorted(p.dive_id for p in over) == [5, 16, 25, 28]
    assert min(p.p90_diff_pct for p in over) == pytest.approx(3.5, abs=0.2)
    assert max(p.p90_diff_pct for p in over) == pytest.approx(11.2, abs=0.2)


def test_the_p90_mean_interval_also_spans_zero(pairs):
    """Seven pairs establish no bias under this estimator either."""
    low, high = sp.summary(pairs, "p90")["ci95"]
    assert low < 0 < high


def test_figure_15_draws_the_numbers_the_analysis_reports(pairs):
    """The diamond in Figure 15 is `Pair.ours_p90_mm`, not a recomputation.

    Pins the figure to the analysis module rather than to a second p90 written
    beside it, so the number quoted from the figure equals the one the text
    quotes. `nearest_rank_p90` is the single implementation both go through.
    """
    from fishsense_imwut.pubfig import nearest_rank_p90

    ours = sp.load_ours(DATA / "stereo_pairs.csv")
    by_dive = {int(d): g for d, g in ours.groupby("dive_id")}
    for p in pairs:
        frames_mm = by_dive[p.dive_id]["length_m"].to_numpy(float) * 1000.0
        assert p.ours_p90_mm == pytest.approx(nearest_rank_p90(frames_mm))
