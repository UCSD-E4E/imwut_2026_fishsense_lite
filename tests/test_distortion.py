"""Bounds on what residual lens distortion can do to a length (section 4.5).

These pin a NEGATIVE result and the ceiling behind it. The near-field trend in
Figure 3 is not distortion, and the numbers here are what rules it out: the
whole modelled effect is far too small, and it has the wrong sign.
"""

from pathlib import Path

import numpy as np
import pytest

from fishsense_imwut import calibration as cal
from fishsense_imwut import distortion as dz

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


@pytest.fixture(name="table")
def _table():
    rows = [r for r in dz.load_head_tail(DATA / "head_tail.csv")
            if r["model"] not in cal.HELD_OUT_MODELS]
    return dz.leverage_table(rows)


def test_the_export_covers_the_reported_cohort(table):
    """995 frames, the same set section 4.3 reports -- so the bound is on the
    measurements the paper actually makes, not a convenience subset."""
    assert len(table["pct_error"]) == 995


def test_distortion_leverage_grows_toward_the_near_field(table):
    """The mechanism is real: a near target subtends more pixels and reaches
    further out, and both push the same way."""
    z, lev = table["depth_m"], table["leverage_pct"]
    near, far = z < 0.8, z > 2.0
    assert np.median(lev[near]) == pytest.approx(0.76, abs=0.05)
    assert np.median(lev[far]) == pytest.approx(0.02, abs=0.02)
    assert np.median(lev[near]) > 20 * np.median(lev[far])


def test_radial_reach_compounds_the_apparent_size_effect(table):
    """Section 4.5's "about half again on top of the size": the same span placed
    symmetrically about the principal point carries noticeably less."""
    z = table["depth_m"]
    near = z < 0.8
    ratio = np.median(table["leverage_pct"][near]) / np.median(table["centred_pct"][near])
    assert ratio == pytest.approx(1.5, abs=0.15)
    assert ratio > 1.0


def test_distortion_cannot_be_the_near_field_trend(table):
    """The verdict, with the arithmetic that reaches it.

    The whole modelled effect between near and far is 0.74 pp against 3.68 pp
    observed, so the model would have to be wrong by five times itself. And
    leverage is POSITIVE -- undistortion lengthens a span -- so an under-corrected
    model reads long where the near field reads short. Wrong size and wrong sign.
    """
    z, lev, e = table["depth_m"], table["leverage_pct"], table["pct_error"]
    near, far = z < 0.8, z > 2.0
    d_lev = np.median(lev[near]) - np.median(lev[far])
    d_err = np.median(e[near]) - np.median(e[far])

    assert d_lev == pytest.approx(0.74, abs=0.05)
    assert d_err == pytest.approx(-3.68, abs=0.10)
    assert abs(d_err / d_lev) > 4.0
    assert d_lev > 0 > d_err, "leverage lengthens; the near field shortens"

    # a generous 20 % residual covers only a few percent of what is observed
    assert abs(0.20 * d_lev / d_err) < 0.05


def test_the_worst_frame_still_bounds_it(table):
    """Even where distortion has the most to work with, the ceiling holds."""
    worst = int(np.argmax(table["leverage_pct"]))
    assert table["leverage_pct"][worst] == pytest.approx(4.15, abs=0.10)
    assert table["r_max_px"][worst] > 1600
    assert table["span_px"][worst] > 2800


def test_the_laser_dot_is_the_wrong_proxy(table):
    """Why the endpoints had to be exported at all.

    Dot radius differs from endpoint radius by half the apparent span, which is
    itself proportional to 1/range -- so the dot reports a radial effect the
    endpoints do not show. Pinned as a warning, not as a result.
    """
    assert np.median(table["r_max_px"]) > 0
    # endpoint radius and span are strongly coupled; that coupling IS the trap
    r = np.corrcoef(table["r_max_px"], table["span_px"])[0, 1]
    assert r > 0.7
