"""Pins for the 2023-08-03 paired-instrument comparison (FINDINGS section 12).

The numbers are pinned against the two committed extractions
(`data/stereo_pairs.csv` from `sql/extract_stereo_pairs.sql`, and
`data/stereo_reference.csv` lifted from the collaborators' archive) so the
comparison is reproducible from this repository rather than from a prod query
that may not return the same rows next week.

The test that carries the analysis is `test_the_estimator_choice_flips_the_sign`.
An out-of-plane fish can only read SHORT, so a per-fish median is biased
downward and a per-fish maximum is a lower bound on the animal. Reporting the
median alone gives -3.7 % and reads as "we measure slightly short"; the
maximum gives +8.6 % and reads as "four of seven fish are longer than the
stereo says". Both are true statements about the same data, and the first one
is the wrong summary for a one-sided error -- the same reason section 4.3
reports p90 rather than a mean.
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
