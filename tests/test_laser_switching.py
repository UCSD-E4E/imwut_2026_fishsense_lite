"""Does switching the laser on and off move the beam within the plane?

Laser state is not logged. What the images record is whether a dot is visible
(a human label, cross-checked by the automated detector) and when each frame
was shot, so a switch can only hide in a run of dark frames or in a pause in
recording. This file pins how far those signals go, what the beam does across
each candidate event, the two within-session moves the manuscript cites, and,
for contrast, how the dots sit against their calibration line through the
seven field dives.

In-plane motion is read from the known lengths (a scale change, converted to an
angle by inverting `pointing_budget_deg`); out-of-plane motion from the dots'
offset from the calibration's image line (offset / f, independent of range).
The angle-experiment dives step the fish's pose through time by design, so
their length error is not a laser measure and they are left out of in-plane
comparisons.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
ANGLE_DIVES = (87, 94, 103, 107, 114)
FIELD = (279, 341, 347, 349, 383, 465, 471)


@pytest.fixture(scope="module")
def ctx():
    raw = cal.load_rows(DATA / "corpus.csv")
    rows = [r for r in raw if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    s = cal.load_laser_state(DATA / "laser_state.csv")
    s["kx"], s["ky"] = s.dot_x.round(4), s.dot_y.round(4)
    err = pd.DataFrame({
        "dive_id": [int(r["dive_id"]) for r in rows],
        "kx": [round(float(r["dot"].split(";")[0]), 4) for r in rows],
        "ky": [round(float(r["dot"].split(";")[1]), 4) for r in rows],
        "model": df.model_name.values, "pct": df.pct_error.values, "z": df.depth_m.values,
        "held": df.model_name.isin(cal.HELD_OUT_MODELS).values})
    # frames join to images on (dive, dot pixel); the few dot positions shared
    # by two images are dropped rather than guessed
    jj = err.merge(s[["dive_id", "kx", "ky", "image_id", "taken"]], on=["dive_id", "kx", "ky"])
    jj = jj[~jj.duplicated(["dive_id", "kx", "ky"], keep=False)]
    return dict(
        s=s, jj=jj, pool=sorted(set(err.dive_id)),
        K={int(r["camera_id"]): np.array(json.loads(r["km"])) for r in raw},
        cam={int(r["dive_id"]): int(r["camera_id"]) for r in rows},
        calof={int(r["dive_id"]): int(r["calibration_dive_id"]) for r in rows},
        fits=cal.load_calibration_fits(DATA / "calibration_fits.csv").set_index("dive_id"))


def _compare(ctx, d, g, lo, cut, hi):
    """In-plane and out-of-plane change between frames [lo, cut) and [cut, hi)."""
    f = ctx["fits"].loc[ctx["calof"][d]]
    beam, K = cal._beam(f), ctx["K"][ctx["cam"][d]]
    A, B = g.iloc[lo:cut], g.iloc[cut:hi]

    def res(x):
        x = x[x.label_state == "dot"]
        return cal.dot_line_residual_px(x[["dot_x", "dot_y"]].values, K, *beam)
    ra, rb = res(A), res(B)
    out = np.degrees((np.median(rb) - np.median(ra)) / K[0, 0]) \
        if len(ra) >= 3 and len(rb) >= 3 else np.nan

    jj = ctx["jj"]
    ea = jj[jj.image_id.isin(A.image_id) & ~jj.held]
    eb = jj[jj.image_id.isin(B.image_id) & ~jj.held]
    parts = []
    for m in sorted(set(ea.model) & set(eb.model)):
        x, y = ea[ea.model == m], eb[eb.model == m]
        if len(x) >= 3 and len(y) >= 3:
            ratio = (1 + np.median(y.pct) / 100) / (1 + np.median(x.pct) / 100)
            z = np.median(np.r_[x.z, y.z])
            parts.append((min(len(x), len(y)), np.degrees(f.baseline_m / z * (1 - 1 / ratio))))
    inp = np.nan
    if parts and d not in ANGLE_DIVES:
        w = np.array([p[0] for p in parts], float)
        inp = float(np.dot(w / w.sum(), [p[1] for p in parts]))
    return inp, out


def _events(ctx):
    out = []
    for d in ctx["pool"]:
        g = ctx["s"][ctx["s"].dive_id == d].reset_index(drop=True)
        ev = cal.switch_candidates(g)
        b = [0] + [e[0] for e in ev] + [len(g)]
        for k, e in enumerate(ev):
            out.append((d, e[1], *_compare(ctx, d, g, b[k], e[0], b[k + 2])))
    return pd.DataFrame(out, columns=["dive", "kind", "in_deg", "out_deg"])


# --- 1. what records a switch ------------------------------------------------------

def test_no_signal_records_the_laser_state(ctx):
    """Over the 31 pool sessions: 3,389 frames with a labeled dot, 317 labeled
    empty, 292 never labeled. The detector backs the empty labels (median
    confidence 0.17 against 1.00 on dot frames; 199 of the 284 it ran on found
    nothing), so 'empty' is a fair 'no visible dot'. 'None' is not 'off': the
    81 unlabeled frames before dive 509's burst carry a detected dot."""
    s = ctx["s"]
    p = s[s.dive_id.isin(ctx["pool"])]
    assert len(ctx["pool"]) == 31 and len(p) == 3998
    assert p.label_state.value_counts().to_dict() == {"dot": 3389, "empty": 317, "none": 292}
    e = p[p.label_state == "empty"]
    assert int(e.pred_conf.notna().sum()) == 284 and int((e.pred_gate == "no_prediction").sum()) == 199
    assert round(e.pred_conf.median(), 2) == 0.17
    assert p[p.label_state == "dot"].pred_conf.median() == 1.0
    pre = s[(s.dive_id == 509) & (s.taken < pd.Timestamp("2023-08-18 09:05:40Z"))]
    assert (pre.label_state == "none").sum() == 81 and pre.pred_conf.median() == 1.0


def test_candidate_events_in_the_pool(ctx):
    """19 dark runs (>= 2 dark frames between dots) and 12 recording pauses
    (> 2 min) over 31 sessions; most sessions last 1-6 minutes."""
    kinds = {"dark": 0, "gap": 0}
    for d in ctx["pool"]:
        g = ctx["s"][ctx["s"].dive_id == d].reset_index(drop=True)
        for e in cal.switch_candidates(g):
            kinds[e[1]] += 1
    assert kinds == {"dark": 19, "gap": 12}


# --- 2. the beam across each event --------------------------------------------------

def test_events_with_known_lengths_both_sides_barely_move_the_beam(ctx):
    """Eight events have the same target on both sides (angle experiment
    excluded): in-plane at most 0.06 deg, out of plane at most 0.04 deg --
    inside what splitting a session where nothing happened gives."""
    ev = _events(ctx)
    m = ev.dropna(subset=["in_deg"])
    assert len(m) == 8
    assert round(m.in_deg.abs().max(), 2) == 0.06
    assert round(m.out_deg.abs().max(), 2) == 0.04
    # every event with dots both sides, out of plane: at most 0.13 deg (dive 84)
    o = ev.dropna(subset=["out_deg"])
    assert round(o.out_deg.abs().max(), 2) == 0.13
    assert o.loc[o.out_deg.abs().idxmax(), "dive"] == 84


def test_the_noise_floor_of_a_split_with_no_event(ctx):
    """Each segment between events split at its middle: the same comparison
    where no candidate event lies."""
    ins, outs = [], []
    for d in ctx["pool"]:
        g = ctx["s"][ctx["s"].dive_id == d].reset_index(drop=True)
        b = [0] + [e[0] for e in cal.switch_candidates(g)] + [len(g)]
        for lo, hi in zip(b[:-1], b[1:]):
            if hi - lo >= 20:
                i, o = _compare(ctx, d, g, lo, (lo + hi) // 2, hi)
                ins.append(i)
                outs.append(o)
    ins, outs = pd.Series(ins).dropna().abs(), pd.Series(outs).dropna().abs()
    # 31 splits with the same target both sides: in-plane median 0.02 deg, up
    # to 0.36 where a split also separates ranges in a session with a range
    # trend (509); out of plane at most 0.04 deg
    assert (len(ins), round(ins.median(), 2), round(ins.max(), 2)) == (31, 0.02, 0.36)
    assert round(outs.max(), 2) == 0.04


# --- 3. the two events the manuscript cites ----------------------------------------

def test_489_to_490_spans_a_pause_in_recording_not_a_dark_run(ctx):
    """Nothing was recorded for 4.9 minutes between dive 527's last frame and
    dive 490's first; the frames either side show a dot. The step itself, from
    the fitted beams: 0.85 deg in plane, 0.06 deg out of it."""
    s = ctx["s"]
    t = s[s.dive_id.isin([489, 527, 490])].sort_values(["taken", "image_id"])
    dt = t.taken.diff().dt.total_seconds()
    k = dt.idxmax()
    assert round(dt[k] / 60, 1) == 4.9
    assert f"{t.loc[k, 'taken']:%H:%M:%S}" == "19:07:25"
    assert int(t.dark.sum()) == 2                                 # both inside 527, seconds apart
    assert t.pred_conf.loc[t.index[t.index.get_loc(k) - 1]] > 0.99 and t.pred_conf[k] > 0.99
    d = cal.laser_drift(cal.load_calibration_fits(DATA / "calibration_fits.csv")).set_index("dive_id")
    assert (round(d.loc[490, "step_in_plane_deg"], 2), round(d.loc[490, "step_out_of_plane_deg"], 2)) == (0.85, 0.06)


def test_509s_trout_steps_at_a_ten_minute_pause_and_the_step_is_in_plane(ctx):
    """The trout reads -4.37 % before a 10.6-minute pause in recording and
    -9.57 % after it (-3.90 vs -9.26 % at matched range, 1.0-1.5 m). Before,
    error barely changes with range (+0.75 %/m); after, it falls at -3.25 %/m, the signature of an
    in-plane swing, and one angle of -0.29 deg reproduces every later frame.
    The dots' offset from the calibration line moves 0.014 deg."""
    s, jj = ctx["s"], ctx["jj"]
    g = s[s.dive_id == 509]
    dt = g.taken.diff().dt.total_seconds() / 60
    k = dt.idxmax()
    assert round(dt[k], 1) == 10.6
    gap0, gap1 = g.taken[g.index[g.index.get_loc(k) - 1]], g.taken[k]

    tr = jj[(jj.dive_id == 509) & (jj.model == "Rainbow Trout")]
    pre, post = tr[tr.taken <= gap0], tr[tr.taken >= gap1]
    assert (len(pre), round(pre.pct.median(), 2), len(post), round(post.pct.median(), 2)) == (58, -4.37, 38, -9.57)
    a, b = pre[pre.z.between(1.0, 1.5)], post[post.z.between(1.0, 1.5)]
    assert (len(a), round(a.pct.median(), 2), len(b), round(b.pct.median(), 2)) == (8, -3.90, 13, -9.26)
    assert round(stats.linregress(pre.z, pre.pct).slope, 2) == 0.75
    assert round(stats.linregress(post.z, post.pct).slope, 2) == -3.25

    B = ctx["fits"].loc[509, "baseline_m"]
    f = (1 + post.pct / 100) / (1 + pre.pct.median() / 100)
    delta = np.degrees(B / post.z * (1 - 1 / f))
    assert [round(v, 2) for v in delta.quantile([0.25, 0.5, 0.75])] == [-0.32, -0.29, -0.26]

    dots = g[g.label_state == "dot"]
    r = cal.dot_line_residual_px(dots[["dot_x", "dot_y"]].values, ctx["K"][4],
                                 *cal._beam(ctx["fits"].loc[509]))
    # from the first trout frame after the burst; the burst frames are the fit
    before = r[((dots.taken >= pd.Timestamp("2023-08-18 09:10:46Z")) & (dots.taken <= gap0)).to_numpy()]
    after = r[(dots.taken >= gap1).to_numpy()]
    shift = np.median(after) - np.median(before)
    assert (len(before), len(after)) == (112, 110)
    assert round(np.degrees(shift / ctx["K"][4][0, 0]), 3) == 0.014


def test_the_fleets_steps_on_the_same_basis():
    """Between successive well-conditioned calibrations: median 0.65 deg in
    plane, 0.35 deg out of it, in-plane the larger in 11 of 18 -- about 2:1,
    where the two within-session moves run 13:1 and 20:1."""
    d = cal.laser_drift(cal.load_calibration_fits(DATA / "calibration_fits.csv")).sort_values(["camera_id", "when"])
    ok = d.groupby("camera_id").well_conditioned.shift(1)
    b = d[d.well_conditioned & (ok == True)].dropna(subset=["step_in_plane_deg"])  # noqa: E712
    assert len(b) == 18
    assert round(b.step_in_plane_deg.abs().median(), 2) == 0.65
    assert round(b.step_out_of_plane_deg.abs().median(), 2) == 0.35
    assert int((b.step_in_plane_deg.abs() > b.step_out_of_plane_deg.abs()).sum()) == 11


# --- 4. the field, out of plane only -------------------------------------------------

def test_field_dots_drift_slowly_off_the_line_in_four_of_seven_dives(ctx):
    """Through each field dive, the dot's offset from its own calibration's
    line, regressed on time and 1/range together: four dives drift +0.10 to
    +0.24 deg per hour (p < 1e-8), three are flat. Out of plane only."""
    fld = pd.read_csv(DATA / "field.csv", sep="|")
    rate, p = {}, {}
    for d in FIELD:
        g = ctx["s"][(ctx["s"].dive_id == d) & (ctx["s"].label_state == "dot")]
        K = ctx["K"][int(fld[fld.dive_id == d].camera_id.iloc[0])]
        beam = cal._beam(ctx["fits"].loc[d])
        xy = g[["dot_x", "dot_y"]].values
        r, z = cal.dot_line_residual_px(xy, K, *beam), cal.beam_depth_m(xy, K, *beam)
        h = (g.taken - g.taken.min()).dt.total_seconds().to_numpy() / 3600
        keep = (z > 0.2) & (z < 10)
        X = np.c_[np.ones(keep.sum()), h[keep], 1 / z[keep]]
        coef, *_ = np.linalg.lstsq(X, r[keep], rcond=None)
        e = r[keep] - X @ coef
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(e) - 3))
        rate[d] = round(float(np.degrees(coef[1] / K[0, 0])), 2)
        p[d] = 2 * stats.t.sf(abs(coef[1] / se[1]), len(e) - 3)
    drifting = sorted(d for d in FIELD if p[d] < 1e-8)
    assert drifting == [341, 347, 349, 465]
    assert [rate[d] for d in drifting] == [0.15, 0.16, 0.24, 0.10]
    assert all(p[d] > 0.1 for d in FIELD if d not in drifting)
