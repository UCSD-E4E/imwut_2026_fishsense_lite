"""In-plane drift measured across a field dive: what the slate frames at both ends allow.

`IN_PLANE_DRIFT.md` section F. Each slate frame's dot, placed on the slate plane by PnP on
its corner labels, is a 3D point whose range does not depend on the laser
(`data/slate_points.csv`, from `derive_slate_points.py` through production's stage-13
geometry). Under a laser line, the same dot has a laser-derived range. If the beam swung
in plane between two ends of a dive, a frame at one end disagrees with a line fitted at
the other. A two-point line is exact, so these tests need no compiled kernel.

No new labels are used. A label-free fit of either end is not possible on these dives,
for want of frames with the dot on the slate (test below).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


@pytest.fixture(scope="module")
def ctx():
    pts = pd.read_csv(DATA / "slate_points.csv")
    pts["taken"] = pd.to_datetime(pts.taken_utc)
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv").set_index("dive_id")
    K = {int(r["camera_id"]): np.array(json.loads(r["km"])) for r in cal.load_rows(DATA / "corpus.csv")}
    return pts, fits, K


def _line(P):
    """The line through points (exact for two): origin where it meets z = 0, unit axis."""
    P = np.asarray(P, float)
    c = P.mean(0)
    a = np.linalg.svd(P - c)[2][0]
    a = a if a[2] > 0 else -a
    return c - a * c[2] / a[2], a


def _disagree_pct(row, beam, K):
    """Laser-derived over pose-derived range, percent."""
    z = cal.beam_depth_m(np.array([[row.dot_x, row.dot_y]]), K, *beam)[0]
    return 100 * (z / row.Z - 1)


def test_the_slate_points(ctx):
    pts, fits, K = ctx
    assert pts.groupby("dive_id").size().to_dict() == {279: 3, 341: 30, 347: 14, 349: 8, 465: 3, 471: 2}
    # the stored fits pass through their own points: 279's three agree to 0.01 %,
    # and 471's two exactly, since a line through two points has no residual
    for d in (279, 471):
        g = pts[pts.dive_id == d]
        beam, Kc = cal._beam(fits.loc[d]), K[int(fits.loc[d, "camera_id"])]
        assert max(abs(_disagree_pct(r, beam, Kc)) for r in g.itertuples()) < 0.011


def test_no_end_of_these_dives_supports_its_own_fit():
    """Frames with the dot ON the slate, per end. 279's opening burst: all 13
    slate frames answered 'Laser not on slate', 12 with no dot at all, so no
    calibration of any kind -- labeled or label-free -- can come from it. 465
    and 471: one or two per end. Size constancy (the SenSys label-free method)
    needs at least eight."""
    t = cal.load_dive_timeline(DATA / "dive_timeline.csv")
    s = t[t.content.str.startswith("Slate")].copy()
    s["on"] = s.content.str.contains("Laser on slate")
    opening279 = s[(s.dive_id == 279) & (s.taken < pd.Timestamp("2023-11-13 12:05:00Z"))]
    assert len(opening279) == 13 and int(opening279.on.sum()) == 0 and int(opening279.has_dot.sum()) == 1
    ends = {}
    for d in (279, 465, 471):
        g = s[(s.dive_id == d) & s.on].sort_values("taken")
        burst = (g.taken.diff() > pd.Timedelta(minutes=5)).cumsum()
        ends[d] = g.groupby(burst.values).size().tolist()
    assert ends == {279: [3], 465: [1, 2], 471: [1, 1]}


def test_465_opening_frame_against_its_closing_line(ctx):
    """The two closing frames (1.37 and 2.51 m, 1.15 m apart) fix a line; the
    opening frame, 51.4 minutes earlier at 1.16 m, reads 1.91 % short against
    it -- an in-plane change of -0.095 deg, against a tolerance of 0.66 deg at
    that range. Within the noise of the design (next test)."""
    pts, fits, K = ctx
    g = pts[pts.dive_id == 465].sort_values("taken").reset_index(drop=True)
    Kc = K[int(fits.loc[465, "camera_id"])]
    assert [round(z, 2) for z in g.Z] == [1.16, 1.37, 2.51]
    assert round((g.taken[1] - g.taken[0]).total_seconds() / 60, 1) == 51.4
    beam = _line(g.loc[1:, ["X", "Y", "Z"]].to_numpy())
    pct = _disagree_pct(g.iloc[0], beam, Kc)
    assert round(pct, 2) == -1.91
    B = float(np.hypot(*beam[0][:2]))
    angle = np.degrees(B / g.Z[0] * (1 - 1 / (1 + pct / 100)))
    assert round(angle, 3) == -0.095
    lo, hi = cal.pointing_budget_deg(fits.loc[465, "baseline_m"], g.Z[0])
    assert round(min(abs(lo), abs(hi)), 2) == 0.66
    # the closing frames under the dive's single stored line: +1.08 and -0.53 %
    stored = cal._beam(fits.loc[465])
    assert [round(_disagree_pct(r, stored, Kc), 2) for r in g.iloc[1:].itertuples()] == [1.08, -0.53]


def test_the_noise_floor_of_that_design(ctx):
    """The same test where nothing can have drifted: inside one burst (341, 347;
    349 has no pair 1 m apart), fit a line through two frames at least 1 m
    apart in range and test a third. |disagreement| median 0.57 %, 90th
    percentile 1.97 %, 95th 4.19 %; 13 % of draws reach 465's 1.91 %."""
    pts, fits, K = ctx
    rng = np.random.default_rng(0)
    null = []
    for d in (341, 347, 349):
        g = pts[pts.dive_id == d].reset_index(drop=True)
        P, Kc = g[["X", "Y", "Z"]].to_numpy(), K[int(fits.loc[d, "camera_id"])]
        for _ in range(4000):
            i, j = rng.choice(len(g), 2, replace=False)
            if abs(P[i, 2] - P[j, 2]) < 1.0:
                continue
            k = rng.choice([x for x in range(len(g)) if x not in (i, j)])
            null.append((d, abs(_disagree_pct(g.iloc[k], _line(P[[i, j]]), Kc))))
    N = pd.DataFrame(null, columns=["dive", "abs_pct"])
    assert N.groupby("dive").size().to_dict() == {341: 971, 347: 1779}
    assert [round(v, 2) for v in N.abs_pct.quantile([0.5, 0.9, 0.95])] == [0.57, 1.97, 4.19]
    assert round(float((N.abs_pct >= 1.91).mean()), 2) == 0.13
    # in degrees at the 465 opening frame's range (1.16 m) and closing-line
    # baseline (99.1 mm): the 95th percentile is 0.20 deg, the median 0.03
    g = pts[pts.dive_id == 465].sort_values("taken").reset_index(drop=True)
    beam = _line(g.loc[1:, ["X", "Y", "Z"]].to_numpy())
    B, z = float(np.hypot(*beam[0][:2])), g.Z[0]
    assert round(B * 1000, 1) == 99.1
    to_deg = lambda pct: np.degrees(B / z * (1 - 1 / (1 + pct / 100)))
    assert (round(to_deg(4.19), 2), round(to_deg(0.57), 2)) == (0.20, 0.03)


# --- with every frame's automated predictions (v2) and SAM slate size ------------------

def _sizes():
    s = pd.read_csv(DATA / "slate_sizes.csv")
    s["taken"] = pd.to_datetime(s.taken, utc=True, format="mixed")
    s["end"] = s.groupby("dive_id").taken.transform(lambda t: (t.diff() > pd.Timedelta(minutes=5)).cumsum())
    return s


def _ray_point(xy, K, z):
    r = np.linalg.solve(K, [xy[0], xy[1], 1.0])
    return r * z / r[2]


def test_automated_predictions_reach_frames_nobody_labeled():
    """Slate-presence >= 0.5 with a human dot or a detector dot >= 0.5, and of
    those, frames whose SAM slate mask contains the dot (sized), per end."""
    s = _sizes()
    s = s[s.dive_id.isin([279, 465, 471])]
    got = {d: [(len(g), int(g["size"].notna().sum())) for _, g in s[s.dive_id == d].groupby("end")]
           for d in (279, 465, 471)}
    assert got == {279: [(20, 3), (7, 3)], 465: [(4, 4), (3, 3)], 471: [(3, 3), (1, 1)]}
    assert int((s[(s.dive_id == 279) & (s.end == 0)].dot_source == "detector").sum()) == 19


def test_size_gives_range_to_a_percent_or_two(ctx):
    """Z * size is constant within a single burst: coefficient of variation
    0.79 % (341), 1.03 % (347), 2.30 % (349) over frames with a pose range."""
    pts, _, _ = ctx
    r = pts.merge(_sizes()[["image_id", "size"]], on="image_id").dropna(subset=["size"])
    k = r.assign(k=r.Z * r["size"]).groupby("dive_id").k
    cv = (100 * k.std() / k.mean()).round(2).to_dict()
    assert {d: cv[d] for d in (341, 347, 349)} == {341: 0.79, 347: 1.03, 349: 2.30}


def test_279s_opening_detections_are_not_its_laser(ctx):
    """279's opening frames 78686 and 78688 put a detected dot on the slate at
    y = 903 and 810 px; under the closing fit that is 0.49 and 0.42 m, while the
    slate's size puts it at 0.91 and 0.98 m. Within one second (78690) the
    detector puts the dot 440 px further along the line on the diver's chest.
    The dive's laser is red and blooms at these ranges (closing frame 78797);
    these are small red specks. Not the measuring laser on the slate, so the
    opening end of 279 stays unmeasurable."""
    pts, fits, K = ctx
    s = _sizes()
    Kc = K[int(fits.loc[279, "camera_id"])]
    ref = s.merge(pts[["image_id", "Z"]], on="image_id")
    ref = ref[ref.dive_id == 279]
    k = float(np.median(ref.Z * ref["size"]))
    g = s[s.image_id.isin([78686, 78688])].set_index("image_id")
    z_laser = cal.beam_depth_m(g[["dot_x", "dot_y"]].values, Kc, *cal._beam(fits.loc[279]))
    assert [round(v, 2) for v in z_laser] == [0.49, 0.42]
    assert [round(k / v, 2) for v in g["size"]] == [0.91, 0.98]
    other = s[s.image_id == 78690]
    assert round(float(other.dot_y.iloc[0] - g.loc[78688, "dot_y"])) == 445


def _end_line(d, end, ctx, s):
    """A line from one end's frames alone: pose range where a slate pose exists,
    otherwise size-implied range, the size scale from that end's pose frames."""
    pts, fits, K = ctx
    Kc = K[int(fits.loc[d, "camera_id"])]
    A = s[(s.dive_id == d) & (s.end == end)].dropna(subset=["size"])
    ref = A.merge(pts[["image_id", "Z"]], on="image_id")
    k = float(np.median(ref.Z * ref["size"]))
    zs = {r.image_id: r.Z for r in pts.itertuples()}
    P = np.array([_ray_point((r.dot_x, r.dot_y), Kc, zs.get(r.image_id, k / r.size)) for r in A.itertuples()])
    return _line(P), P, Kc


def _test_end(d, fit_end, test_end, ctx):
    pts, _, _ = ctx
    s = _sizes()
    (o, a), P, Kc = _end_line(d, fit_end, ctx, s)
    T = s[(s.dive_id == d) & (s.end == test_end)].merge(pts[["image_id", "Z"]], on="image_id")
    z = cal.beam_depth_m(T[["dot_x", "dot_y"]].values, Kc, o, a)
    pct = 100 * (z / T.Z.to_numpy() - 1)
    B = float(np.hypot(*o[:2]))
    deg = np.degrees(B / T.Z.to_numpy() * (1 - 1 / (1 + pct / 100)))
    return P, [round(v, 2) for v in pct], [round(v, 3) for v in deg]


def test_465_both_directions(ctx):
    """Opening line (4 frames, 1.16-2.72 m) reads the closing pose frames +3.09
    and +3.72 % (+0.127, +0.083 deg); closing line (3 frames, 1.25-2.51 m)
    reads the opening pose frame -1.42 % (-0.071 deg). Same sign both ways:
    about 0.1 deg over 51 minutes."""
    P, pct, deg = _test_end(465, 0, 1, ctx)
    assert (len(P), round(P[:, 2].min(), 2), round(P[:, 2].max(), 2)) == (4, 1.16, 2.72)
    assert (pct, deg) == ([3.09, 3.72], [0.127, 0.083])
    P, pct, deg = _test_end(465, 1, 0, ctx)
    assert (len(P), round(P[:, 2].min(), 2), round(P[:, 2].max(), 2)) == (3, 1.25, 2.51)
    assert (pct, deg) == ([-1.42], [-0.071])
    _, fits, _ = ctx
    lo, hi = cal.pointing_budget_deg(fits.loc[465, "baseline_m"], 2.5)
    assert round(min(abs(lo), abs(hi)), 2) == 0.31                        # tolerance at 2.5 m


def test_471_opening_line_against_the_closing_frame(ctx):
    """Opening line (3 frames, 1.12-3.51 m: a human pose frame and two detector
    frames ranged by size) reads the closing pose frame, 51.6 minutes later,
    -0.76 % (-0.017 deg). The closing end has one frame and cannot fit a line."""
    P, pct, deg = _test_end(471, 0, 1, ctx)
    assert (len(P), round(P[:, 2].min(), 2), round(P[:, 2].max(), 2)) == (3, 1.12, 3.51)
    assert (pct, deg) == ([-0.76], [-0.017])
    s = _sizes()
    t = s[s.dive_id == 471].groupby("end").taken
    assert round((t.min()[1] - t.max()[0]).total_seconds() / 60, 1) == 51.6


def test_noise_floor_of_the_size_ranged_design(ctx):
    """Inside single reef bursts (341, 347; 349 has no 1 m spread): scale from
    one pose frame, line through it and two size-ranged frames >= 1 m apart,
    test a fourth frame's pose range. |disagreement| median 1.06 %, 90th
    percentile 3.19 %, 95th 4.25 %; in angle at the test frame's range, median
    0.031 deg, 90th 0.103, 95th 0.127. 465's +3.09 / +3.72 % sit at the 89th
    and 93rd percentiles; 471's -0.76 % and 465's -1.42 % are ordinary."""
    pts, fits, K = ctx
    R = pts[pts.dive_id.isin([341, 347, 349])].merge(_sizes()[["image_id", "size"]], on="image_id").dropna(subset=["size"])
    rng = np.random.default_rng(0)
    rows = []
    for d, g in R.groupby("dive_id"):
        g = g.reset_index(drop=True)
        Kc = K[int(fits.loc[d, "camera_id"])]
        for _ in range(3000):
            i, j, m, t = rng.choice(len(g), 4, replace=False)
            k = g.Z[i] * g["size"][i]
            P = np.array([_ray_point((g.dot_x[i], g.dot_y[i]), Kc, g.Z[i])] +
                         [_ray_point((g.dot_x[q], g.dot_y[q]), Kc, k / g["size"][q]) for q in (j, m)])
            if np.ptp(P[:, 2]) < 1.0:
                continue
            o, a = _line(P)
            z = cal.beam_depth_m(np.array([[g.dot_x[t], g.dot_y[t]]]), Kc, o, a)[0]
            pct = 100 * (z / g.Z[t] - 1)
            rows.append((d, abs(pct), abs(np.degrees(np.hypot(*o[:2]) / g.Z[t] * (1 - 1 / (1 + pct / 100))))))
    N = pd.DataFrame(rows, columns=["dive", "pct", "deg"])
    assert N.groupby("dive").size().to_dict() == {341: 1418, 347: 2005}
    assert [round(v, 2) for v in N.pct.quantile([0.5, 0.9, 0.95])] == [1.06, 3.19, 4.25]
    assert [round(v, 3) for v in N.deg.quantile([0.5, 0.9, 0.95])] == [0.031, 0.103, 0.127]
    assert [round(float((N.pct >= v).mean()), 2) for v in (3.09, 3.72)] == [0.11, 0.07]
