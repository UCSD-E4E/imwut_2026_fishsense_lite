"""Section 5.5's counterfactual: calibrate underwater as the fleet was, but through
the bare flat port, fit the laser under that calibration, and measure.

Noise-free simulation (`fishsense_imwut.svp_counterfactual`). Assumptions the data
do not fix: 55 views (the manual's 50-60), the E4E board, tilts to 35 deg, a
calibration-distance spread of 0.5-1.5 m (no record gives the real one), ten slate
frames at 0.8-3.0 m for the laser, `refraction.py`'s housing (the standoff that
makes the port most nearly single-viewpoint, a 6 mm placeholder pane, salt water).
The housing geometry is the assumption that matters, so two looser ones are run.
"""

import numpy as np
import pytest

from fishsense_imwut import refraction as rf
from fishsense_imwut import svp_counterfactual as sc


@pytest.fixture(scope="module")
def runs():
    return {
        "assumed": sc.run((0.5, 1.5)),
        "narrow": sc.run((0.8, 1.2)),
        "wide": sc.run((0.5, 4.0)),
        "optic": sc.run((0.5, 1.5), n_water=1.0),
        "d0_5mm": sc.run((0.5, 1.5), d0=0.005, d1=0.006),
        "loose": sc.run((0.5, 1.5), d0=0.025, d1=0.010),
    }


def _worst(r):
    return round(float(np.abs(np.r_[r["length_mid"], r["length_end"]]).max()), 3)


def test_the_in_water_fit_absorbs_the_port(runs):
    """The fit lands on the paraxial focal length (f * n_water = 3824.7 px) with
    large radial terms, and reprojects to 0.003 px: the port looks like a lens."""
    r = runs["assumed"]
    assert round(r["intrinsics"][0], 1) == 3824.6 and round(rf.FOCAL_LENGTH_PX * rf.SALTY_WATER, 1) == 3824.7
    assert round(r["intrinsics"][4], 2) == 0.40
    assert round(r["rms_px"], 3) == 0.003
    # a whole board fits the bare port's narrower underwater field only from 0.91 m
    assert round(r["view_distances"].min(), 2) == 0.91


def test_no_optic_length_error_is_a_hundredth_of_a_percent(runs):
    """0.5-5 m, dot mid-body and dot at the target's end (the furthest off axis a
    target can sit and still carry the dot): worst 0.009 %. Range: -0.07 % at 0.5 m."""
    r = runs["assumed"]
    assert _worst(r) == 0.009
    assert round(float(r["range_mid"][0]), 2) == -0.07
    assert _worst(r) < sc.BUDGET_PCT / 1000


def test_the_calibration_spread_does_not_matter(runs):
    assert (_worst(runs["narrow"]), _worst(runs["wide"])) == (0.008, 0.007)
    assert [round(v, 2) for v in (runs["narrow"]["view_distances"].min(), runs["narrow"]["view_distances"].max(),
                                  runs["wide"]["view_distances"].min(), runs["wide"]["view_distances"].max())] == [0.83, 1.2, 0.9, 3.94]


def test_the_optic_baseline(runs):
    assert _worst(runs["optic"]) < 0.03


def test_looser_housings(runs):
    """The single-viewpoint fit degrades with standoff and pane, as WUWNet's §4.3
    shows; at a 25 mm standoff and a 10 mm pane the worst is 0.36 %, at 0.5 m with
    the dot at the target's end. Range reads +1.75 % there and cancels out of length."""
    assert _worst(runs["d0_5mm"]) == 0.064
    r = runs["loose"]
    assert _worst(r) == 0.363
    assert round(float(r["length_end"][0]), 3) == -0.363 and round(float(r["range_mid"][0]), 2) == 1.75
