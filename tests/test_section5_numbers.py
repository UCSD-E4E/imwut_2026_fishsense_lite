"""Section 5's numbers that were computed but not pinned until 2026-10-14.

Each was already reported in PAPER.md and recomputed here from the committed
cache when the manuscript asked for one table of every number in Section 5.
Numbers pinned elsewhere are not repeated; this file fills the gaps, grouped by
the manuscript's subsections. A number that changes fails here rather than
drifting silently between the analysis and the text.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fishsense_imwut import calibration as cal
from fishsense_imwut import pubfig, refraction
from fishsense_imwut import repeatability as rep
from fishsense_imwut import stereo_pairs as sp

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


@pytest.fixture(scope="module")
def pairs():
    return sp.build_pairs(sp.load_ours(DATA / "stereo_pairs.csv"),
                          sp.load_stereo(DATA / "stereo_reference.csv"))


@pytest.fixture(scope="module")
def cohort():
    rows = [r for r in cal.load_rows(DATA / "corpus.csv")
            if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    keep = cal.accuracy_cohort(df)
    return df[df.dive_id.isin(keep) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]


# --- comparison with a stereo-video system ---------------------------------------

def _diffs(pairs):
    return np.array([100 * (p.ours_p90_mm - p.stereo_mm) / p.stereo_mm for p in pairs])


def test_stereo_agreement_statistics(pairs):
    ours = np.array([p.ours_p90_mm for p in pairs]) / 10
    theirs = np.array([p.stereo_mm for p in pairs]) / 10
    assert np.corrcoef(ours, theirs)[0, 1] ** 2 == pytest.approx(0.838, abs=0.001)
    assert np.sqrt(np.mean((ours - theirs) ** 2)) == pytest.approx(3.0, abs=0.05)

    frames = pd.read_csv(DATA / "stereo_pairs.csv", sep="|")
    ref = {p.dive_id: p.stereo_mm / 1000 for p in pairs}
    frames = frames[frames.dive_id.isin(ref)]
    assert pubfig.r2_about_identity(frames.dive_id.map(ref), frames.length_m) == \
        pytest.approx(0.786, abs=0.001)


def test_stereo_per_individual_differences(pairs):
    d = _diffs(pairs)
    assert (round(d.min(), 1), round(d.max(), 1)) == (-13.8, 11.2)
    assert round(np.median(d), 1) == 3.5 and round(d.mean(), 1) == 0.8
    long = d[d > 0]
    assert len(long) == 4 and (round(long.min(), 1), round(long.max(), 1)) == (3.5, 11.2)
    assert round(np.median(np.abs(d)), 1) == 8.6           # the "+8.6 %" scale question


def test_every_paired_fish_agrees_within_fifteen_percent(pairs):
    """Section 5's sentence: each of the seven per-fish differences, FishSense
    Lite p90 against the stereo reference, lies inside the 15 % budget."""
    d = _diffs(pairs)
    assert len(d) == 7 and (np.abs(d) < 15).all()


def test_stereo_intervals_and_tests(pairs):
    d = _diffs(pairs)
    t = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    assert (round(d.mean() - t, 1), round(d.mean() + t, 1)) == (-8.6, 10.2)
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(d, len(d))) for _ in range(20_000)]
    assert (round(np.percentile(boot, 2.5), 1), round(np.percentile(boot, 97.5), 1)) == (-8.1, 10.7)
    assert round(stats.ttest_1samp(d, 0).pvalue, 2) == 0.84
    assert round(stats.wilcoxon(d).pvalue, 2) == 0.81
    assert round(stats.binomtest(int((d > 0).sum()), len(d)).pvalue, 2) == 1.00


def test_stereo_day_calibration_against_the_fleet():
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv")
    assert round(fits.set_index("dive_id").loc[32, "baseline_cm"], 2) == 10.55
    sound = fits[fits.dive_id != 107].baseline_cm
    assert len(sound) == 30 and len(fits) == 31
    assert (round(sound.min(), 2), round(sound.max(), 2)) == (9.87, 10.55)


# --- against known lengths ---------------------------------------------------------

def test_known_length_accuracy_numbers(cohort):
    e = cohort.pct_error
    assert [round(100 * (e.abs() <= k).mean()) for k in (5, 10, 15)] == [80, 98, 99]
    p90 = {m: round(pubfig.nearest_rank_p90(g.pct_error), 1) for m, g in cohort.groupby("model_name")}
    assert p90 == {"Box": 0.6, "Grouper": 0.9, "Purple Angelfish": 0.8,
                   "Rainbow Trout": -1.0, "Snook": -1.2}
    beyond = e[e < -15]
    assert len(beyond) == 9 and (e > 15).sum() == 0
    implied = np.degrees(np.arccos(1 + beyond / 100))
    assert (round(implied.min()), round(implied.max())) == (32, 46)
    assert round(e.min(), 1) == -30.6
    # The "-2.0 % is a typical pose of 11.5 deg" argument is cut from the
    # manuscript (2026-10-16) but still flagged in PAPER.md section 4.2; pinned
    # until that passage goes.
    assert round(np.degrees(np.arccos(1 + e.median() / 100)), 1) == 11.5


def test_measured_against_known_r2(cohort):
    per = cohort.groupby("model_name").agg(known=("known_length_m", "first"),
                                           p90=("length_m", pubfig.nearest_rank_p90))
    assert pubfig.r2_about_identity(per.known, per.p90) == pytest.approx(0.999, abs=0.0005)
    # 0.97545: the notebook prints 0.9755 and the caption rounded that again to
    # 0.976. Stated at three places it is 0.975.
    assert pubfig.r2_about_identity(cohort.known_length_m, cohort.length_m) == \
        pytest.approx(0.97545, abs=0.00001)


# --- calibration stability -----------------------------------------------------------

def test_months_to_the_2024_deployments():
    """Three of the four units with a well-conditioned fit in both years moved
    about 3 deg; the fourth 0.9 deg."""
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv")
    d = cal.laser_drift(fits)
    by = {r.dive_id: r for r in fits.itertuples()}
    moved = {}
    for cam, g in d[d.well_conditioned].sort_values("when").groupby("camera_id"):
        y23, y24 = g[g.when.dt.year == 2023], g[g.when.dt.year == 2024]
        if y23.empty or y24.empty:
            continue
        a, b = cal._beam(by[int(y23.iloc[-1].dive_id)]), cal._beam(by[int(y24.iloc[0].dive_id)])
        moved[cam] = float(np.hypot(*cal.laser_beam_change_deg(b, a)))
    assert sorted(moved) == [1, 2, 4, 6]
    assert {c: round(v, 1) for c, v in moved.items()} == {1: 2.9, 2: 3.0, 4: 0.9, 6: 3.1}


# --- fish pose -----------------------------------------------------------------------

def test_pose_curve_and_windows():
    a = cal.load_angles(DATA / "angles.csv")
    med = a.groupby("fish_angle_degrees").pct_error.median()
    assert [round(med[k], 1) for k in (0, 15, 30, 45)] == [-3.8, -5.8, -15.0, -31.4]
    offset = np.median([med[k] - 100 * (np.cos(np.radians(k)) - 1) for k in med.index])
    assert round(offset, 1) == -2.4
    assert round(np.degrees(np.arccos(0.85)), 1) == 31.8      # predicted crossing, analytic

    def window(lim):
        w = a[a.fish_angle_degrees <= lim].pct_error
        return len(w), round(w.median(), 1), round(pubfig.nearest_rank_p90(w), 1), \
            round(w.min(), 1), int((w < -15).sum())
    assert window(15) == (542, -4.7, -0.8, -10.7, 0)
    assert window(20)[0] == 684 and window(20)[3] == -14.2 and window(20)[4] == 0
    assert window(25)[0] == 833 and window(25)[4] == 32


# --- what the port correction buys --------------------------------------------------

def test_worst_point_in_frame_is_the_corner():
    f = refraction.flat_port_error_field()
    assert round(float(np.nanmax(f["error_pct"])), 1) == 29.5


# --- the field cohort --------------------------------------------------------------

def test_field_species_lengths():
    per = rep.field_per_animal(rep.load_field(DATA / "field.csv"))
    s = per.groupby("species").p90_m
    assert round(100 * s.median()["Hogfish"], 1) == 20.9
    assert round(100 * s.median()["Stoplight Parrotfish"], 1) == 21.8
    # the upper tail: the two longest animals in the field cohort
    top = per.sort_values("p90_m", ascending=False).head(2)
    assert list(top.species) == ["Goliath Grouper", "Nassau Grouper"]
    assert [round(100 * v, 1) for v in top.p90_m] == [84.7, 75.6]
    assert sorted(k for k, v in per.groupby("species").size().items() if v == 1) == \
        ["Blue Parrotfish", "Goliath Grouper", "Midnight Parrotfish", "Mutton Snapper"]


def test_the_paired_fish_size_range(pairs):
    """The seven paired fish span 197.9-426.5 mm by the stereo reference (the
    "20 to 43 cm" Section 5 quotes) and 220.0-472.1 mm by FishSense Lite p90.
    Pinned in millimetres: 426.5 mm is exactly 42.65 cm, which binary floating
    point rounds to 42.6 at one decimal."""
    stereo = [p.stereo_mm for p in pairs]
    ours = [p.ours_p90_mm for p in pairs]
    assert (min(stereo), max(stereo)) == (pytest.approx(197.9, abs=0.05), pytest.approx(426.5, abs=0.05))
    assert (min(ours), max(ours)) == (pytest.approx(220.0, abs=0.05), pytest.approx(472.1, abs=0.05))
