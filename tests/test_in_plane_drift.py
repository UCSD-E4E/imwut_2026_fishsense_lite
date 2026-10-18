"""Bounding in-plane drift within a dive: the numbers behind
`fish_model_analysis/IN_PLANE_DRIFT.md`, sections A-E.

In-plane drift rescales every length and is invisible in a single dive's
frames. These tests pin what the companion repo can say about it: whether the
visible out-of-plane component predicts it (A), whether field or stereo-day
lengths trend with time since calibration (B), how far the field cohort sits
from its tolerance under assumed rates (C), whether any dive carries a second,
separately fittable calibration (D), and one step restated on the
in-plane/out-of-plane basis (E).

"Time since calibration" is measured to the frames the dive's fit rests on: a
completed slate label with a laser dot, the rule `calibration_fits.n_frames`
counts by. Angles convert to length at range z through `pointing_budget_deg`'s
relation, % ~ 100 * delta * z / B for small delta.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fishsense_imwut import calibration as cal
from fishsense_imwut import repeatability as rep
from fishsense_imwut import stereo_pairs as spm

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
FIELD = (279, 341, 347, 349, 383, 465, 471)
STEREO_CALIBRATION = 32


@pytest.fixture(scope="module")
def fits():
    return cal.load_calibration_fits(DATA / "calibration_fits.csv")


@pytest.fixture(scope="module")
def timeline():
    return cal.load_dive_timeline(DATA / "dive_timeline.csv")


def _fit_frame_times(timeline, dive):
    t = timeline[(timeline.dive_id == dive) & timeline.slate & timeline.has_dot]
    return t.taken.sort_values()


def _tolerance_deg(baseline_m, z):
    lo, hi = cal.pointing_budget_deg(baseline_m, z)
    return min(abs(lo), abs(hi))            # the sign of a drift is unknown


def _ols(y, X):
    X = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / dof)
    q = stats.t.ppf(0.975, dof)
    return b[1], b[1] - q * se[1], b[1] + q * se[1], (q + stats.norm.ppf(0.8)) * se[1]


@pytest.fixture(scope="module")
def field(fits, timeline):
    f = rep.load_field(DATA / "field.csv").merge(
        timeline[["image_id", "taken"]].drop_duplicates("image_id"), on="image_id", how="left")
    gaps = [(tk - _fit_frame_times(timeline, d)).abs().dt.total_seconds() / 60
            for d, tk in zip(f.dive_id, f.taken)]
    f["nearest_min"] = [float(g.min()) for g in gaps]
    f["farthest_min"] = [float(g.max()) for g in gaps]
    ref = rep.field_per_animal(f).groupby("species").p90_m.median()
    f["norm_pct"] = 100 * (f.length_m / f.species.map(ref) - 1)
    f["tol_deg"] = [_tolerance_deg(fits.set_index("dive_id").loc[d, "baseline_m"], z)
                    for d, z in zip(f.dive_id, f.depth_m)]
    return f


@pytest.fixture(scope="module")
def stereo(fits, timeline):
    pairs = spm.build_pairs(spm.load_ours(DATA / "stereo_pairs.csv"),
                            spm.load_stereo(DATA / "stereo_reference.csv"))
    ref = {p.dive_id: p.stereo_mm / 1000 for p in pairs}
    s = cal.load_laser_state(DATA / "laser_state.csv")
    K = {int(r["camera_id"]): np.array(json.loads(r["km"])) for r in cal.load_rows(DATA / "corpus.csv")}
    fit = fits.set_index("dive_id").loc[STEREO_CALIBRATION]
    sp = pd.read_csv(DATA / "stereo_pairs.csv", sep="|")
    sp = sp[sp.dive_id.isin(ref)].merge(s[["image_id", "dot_x", "dot_y", "taken"]], on="image_id")
    # 25 of the 39 frames carry no stored depth; triangulate every frame under
    # the day's beam, which reproduces the 14 stored ones to 0.2 mm
    sp["z"] = cal.beam_depth_m(sp[["dot_x", "dot_y"]].values, K[int(fit.camera_id)], *cal._beam(fit))
    stored = sp.depth_m.notna()
    assert stored.sum() == 14 and np.abs(sp.z[stored] - sp.depth_m[stored]).max() < 0.0005
    cal_t = _fit_frame_times(timeline, STEREO_CALIBRATION)
    sp["mins"] = (sp.taken - cal_t.max()).dt.total_seconds() / 60
    sp["diff_pct"] = 100 * (sp.length_m / sp.dive_id.map(ref) - 1)
    sp["tol_deg"] = [_tolerance_deg(fit.baseline_m, z) for z in sp.z]
    return sp, pairs, cal_t, fit


# --- A. does visible drift predict invisible drift? ---------------------------------

def _steps(fits):
    d = cal.laser_drift(fits).sort_values(["camera_id", "when"])
    ok = d.groupby("camera_id").well_conditioned.shift(1)
    return d[d.well_conditioned & (ok == True)].dropna(subset=["step_in_plane_deg"])  # noqa: E712


def test_A_out_of_plane_steps_do_not_predict_in_plane_steps(fits):
    b = _steps(fits)
    i, o = b.step_in_plane_deg.to_numpy(), b.step_out_of_plane_deg.to_numpy()
    assert len(b) == 18
    rho, p = stats.spearmanr(np.abs(i), np.abs(o))
    assert (round(rho, 2), round(p, 2)) == (0.23, 0.36)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 18, (20_000, 18))
    boot = [stats.spearmanr(np.abs(i[k]), np.abs(o[k]))[0] for k in idx]
    assert [round(v, 2) for v in np.nanpercentile(boot, [2.5, 97.5])] == [-0.31, 0.65]
    assert round(np.corrcoef(i, o)[0, 1], 2) == -0.09                       # signed
    lr = stats.linregress(np.abs(o), np.abs(i))
    assert (round(lr.intercept, 2), round(lr.slope, 2)) == (0.84, 0.10)
    # the two within-session moves: large in plane, nearly nothing out of it
    d = cal.laser_drift(fits).set_index("dive_id")
    assert (round(d.loc[490, "step_in_plane_deg"], 2), round(d.loc[490, "step_out_of_plane_deg"], 2)) == (0.85, 0.06)
    # dive 509's pause (0.29 in, 0.014 out) is pinned in test_laser_switching


# --- B. do lengths trend with time since calibration? -------------------------------

def test_B_field_times_since_calibration(field):
    assert len(field) == 162 and field.taken.notna().all()
    assert [round(v, 1) for v in field.nearest_min.quantile([0, 0.5, 0.9, 1])] == [0.9, 16.5, 35.5, 51.3]


def test_B_field_length_trend_is_undetectable_at_useful_rates(field):
    """Species-normalized length on hours to the nearest calibration frame,
    with dive intercepts: +18.8 %/h (-0.6 to +38.2), smallest detectable 27.6
    %/h -- about 1.1 deg/h in plane at the cohort's median 1.48 m. Animals
    filmed earlier or later in a dive differ in size, and that, not the
    instrument, is what this regression mostly sees."""
    y, h = field.norm_pct.to_numpy(), field.nearest_min.to_numpy() / 60
    dives = pd.get_dummies(field.dive_id, drop_first=True).to_numpy(float)
    b, lo, hi, mde = _ols(y, np.c_[h, dives])
    assert [round(v, 1) for v in (b, lo, hi, mde)] == [18.8, -0.6, 38.2, 27.6]
    b, lo, hi, mde = _ols(y, h[:, None])                                    # no dive intercepts
    assert [round(v, 1) for v in (b, lo, hi, mde)] == [12.3, -2.9, 27.5, 21.7]
    keep = ~field.species.str.startswith("Other").to_numpy()               # nontarget left out
    b, lo, hi, _ = _ols(y[keep], np.c_[h[keep], dives[keep]])
    assert [round(v, 1) for v in (b, lo, hi)] == [14.5, -4.9, 33.8]
    z, B = field.depth_m.median(), 0.1026
    assert round(z, 2) == 1.48
    assert round(27.6 / (100 * z / B * np.pi / 180), 1) == 1.1


def test_B_within_an_animal_there_is_no_time_to_regress_on(field):
    """An animal's frames are seconds apart, so 'the same fish later in the
    dive' does not exist in this cohort."""
    multi = field.groupby("fish_id").filter(lambda g: len(g) >= 2)
    span = multi.groupby("fish_id").nearest_min.agg(lambda x: x.max() - x.min())
    assert len(span) == 42
    assert round(span.median() * 60) == 1 and round(span.max(), 1) == 2.4


def test_B_stereo_difference_against_time(stereo):
    """Seven fish, one per dive, 26 to 103 minutes after calibration: the
    per-fish difference from stereo falls -6.8 %/h (-26.7 to +13.0), smallest
    detectable 26.3 %/h -- about 0.9 deg/h at the frames' median 1.78 m. Frame by frame the slope
    is -8.5 %/h (-15.4 to -1.6), but 39 frames of 7 fish are not 39 samples."""
    sp, pairs, _, fit = stereo
    fish = sp.groupby("dive_id").mins.median() / 60
    d = pd.Series({p.dive_id: 100 * (p.ours_p90_mm / p.stereo_mm - 1) for p in pairs})
    b, lo, hi, mde = _ols(d[fish.index].to_numpy(), fish.to_numpy()[:, None])
    assert [round(v, 1) for v in (b, lo, hi, mde)] == [-6.8, -26.7, 13.0, 26.3]
    lr = stats.linregress(sp.mins / 60, sp.diff_pct)
    q = stats.t.ppf(0.975, len(sp) - 2)
    assert [round(v, 1) for v in (lr.slope, lr.slope - q * lr.stderr, lr.slope + q * lr.stderr)] == [-8.5, -15.4, -1.6]
    per_deg = 100 * sp.z.median() / fit.baseline_m * np.pi / 180
    assert round(sp.z.median(), 2) == 1.78 and round(26.3 / per_deg, 1) == 0.9


# --- C. the field cohort's exposure --------------------------------------------------

def test_C_field_exposure_under_assumed_rates(field):
    """Tolerance: the in-plane angle that moves a length 15 % at the frame's
    own range, on the tighter side. Under 0.24 deg/h every field length stays
    inside it; under 0.48 deg/h 157 of 162 do (147 timing from the farthest
    calibration frame instead of the nearest)."""
    assert [round(v, 2) for v in field.tol_deg.quantile([0, 0.5, 1])] == [0.19, 0.52, 1.73]
    out = {}
    for col in ("nearest_min", "farthest_min"):
        for rate in (0.24, 0.48):
            ratio = rate * field[col] / 60 / field.tol_deg
            out[(col, rate)] = (int((ratio <= 1).sum()), round(ratio.max(), 2))
    assert out == {("nearest_min", 0.24): (162, 0.70), ("nearest_min", 0.48): (157, 1.40),
                   ("farthest_min", 0.24): (162, 0.77), ("farthest_min", 0.48): (147, 1.54)}
    over = field[0.48 * field.nearest_min / 60 > field.tol_deg].groupby("dive_id").size().to_dict()
    assert over == {347: 4, 383: 1}


def test_C_stereo_day_exposure_and_timing(stereo):
    """The day's calibration frames are 08:58:20 and 08:58:27 UTC; its measured
    fish run 25.9 to 102.6 minutes after the later one -- 26 min to 1 h 42 min
    33 s, which rounds to 1 h 43."""
    sp, _, cal_t, _ = stereo
    assert [f"{x:%H:%M:%S}" for x in cal_t] == ["08:58:20", "08:58:27"]
    assert (round(sp.mins.min(), 1), round(sp.mins.max(), 1)) == (25.9, 102.6)
    assert round(sp.mins.max() * 60) == 6153                                 # 1 h 42 min 33 s
    res = {}
    for rate in (0.24, 0.48):
        ratio = rate * sp.mins / 60 / sp.tol_deg
        res[rate] = (int((ratio <= 1).sum()), round(ratio.max(), 2))
    assert res == {0.24: (39, 0.90), 0.48: (29, 1.80)}


# --- D. second calibrations within a dive --------------------------------------------

def test_D_no_dive_has_two_separately_fittable_calibrations(timeline):
    """Frames a fit can rest on (completed slate label + dot), by dive. Three
    field dives already straddle their measurements with them, but each end
    holds at most one frame except dive 279's, whose fit is all closing burst.
    So none splits into two fits."""
    fit_frames = {d: [f"{x:%H:%M:%S}" for x in _fit_frame_times(timeline, d)] for d in FIELD}
    assert fit_frames == {
        279: ["12:33:55", "12:33:55", "12:34:01"],
        341: fit_frames[341], 347: fit_frames[347], 349: fit_frames[349],
        383: ["07:52:25", "07:52:37"],
        465: ["09:18:08", "10:09:32", "10:09:36"],
        471: ["09:16:33", "10:08:37"]}
    # 341, 347, 349: one opening burst each
    for d in (341, 347, 349):
        t = _fit_frame_times(timeline, d)
        assert (t.max() - t.min()) < pd.Timedelta(minutes=2)
    # 279's opening burst shows the slate in 13 frames, none completed for a fit
    g = timeline[(timeline.dive_id == 279) & timeline.content.str.startswith("Slate")]
    opening = g[g.taken < pd.Timestamp("2023-11-13 12:05:00Z")]
    assert len(opening) == 13 and int((opening.slate & opening.has_dot).sum()) == 0


# --- E. one restatement --------------------------------------------------------------

def test_E_the_496_to_509_step_on_both_bases(fits):
    d = cal.laser_drift(fits).set_index("dive_id")
    assert (round(d.loc[509, "step_in_plane_deg"], 2), round(d.loc[509, "step_out_of_plane_deg"], 2)) == (0.46, 1.54)
    s = cal.laser_step_directions(fits).set_index(["from_dive", "to_dive"]).loc[(496, 509)]
    assert (round(s.vertical_deg, 2), round(s.horizontal_deg, 2)) == (0.00, 1.61)
