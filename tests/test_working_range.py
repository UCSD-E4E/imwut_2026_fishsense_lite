"""Section 5.7's working range: the near-range drop, and whether body depth
accounts for it; plus how much of the field cohort was taken close in.

The near bin is the range figure's nearest equal-count bin (`cal.range_bins`,
below 0.81 m). Half the trout's 58.7 mm body depth predicts a shortfall of
(depth / 2) / z, the convention of `test_body_depth.py`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal
from fishsense_imwut import repeatability as rep
from fishsense_imwut import stereo_pairs as spm

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
HALF_DEPTH_M = 0.0587 / 2


@pytest.fixture(scope="module")
def cohort():
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    return df[df.dive_id.isin(cal.accuracy_cohort(df)) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]


@pytest.fixture(scope="module")
def near_edge(cohort):
    return float(cal.range_bins(cohort).hi_m.iloc[0])


def test_the_near_bin(cohort, near_edge):
    """0.28-0.81 m, 125 frames, median -5.27 %, median range 0.57 m; only two
    targets: the trout (68 frames, -6.99 %) and the box (57, -1.71 %)."""
    near = cohort[cohort.depth_m < near_edge]
    assert round(near_edge, 2) == 0.81 and len(near) == 125
    assert round(near.pct_error.median(), 2) == -5.27 and round(near.depth_m.median(), 2) == 0.57
    by = near.groupby("model_name").pct_error.agg(["size", "median"]).round(2)
    assert by.to_dict("index") == {"Box": {"size": 57, "median": -1.71},
                                   "Rainbow Trout": {"size": 68, "median": -6.99}}


def test_the_arithmetic():
    """29.35 mm / 810 mm = 3.62 %; same convention as 1.47 % at 2 m and 1.71 %
    at 1.72 m."""
    assert [round(100 * HALF_DEPTH_M / z, 2) for z in (0.81, 2.0, 1.72)] == [3.62, 1.47, 1.71]


def test_the_gap_widens_close_in_but_by_less_than_depth_predicts(cohort, near_edge):
    """Pooled: trout minus box is -5.29 points in the near bin and -2.90 beyond
    1.2 m, widening 2.39; half the trout's depth predicts 5.14 % and 1.28 % at
    its median ranges there (0.57 and 2.29 m), a widening of 3.86. The box
    itself drops 1.08 points (-0.63 to -1.71 %)."""
    tb = cohort[cohort.model_name.isin(["Rainbow Trout", "Box"])]
    med = {}
    for zone, sel in (("near", tb.depth_m < near_edge), ("far", tb.depth_m >= 1.2)):
        g = tb[sel]
        t, b = g[g.model_name == "Rainbow Trout"], g[g.model_name == "Box"]
        med[zone] = (t.pct_error.median(), b.pct_error.median(), 100 * HALF_DEPTH_M / t.depth_m.median(), t.depth_m.median())
    gap = {z: round(v[0] - v[1], 2) for z, v in med.items()}
    assert gap == {"near": -5.29, "far": -2.90}
    assert round(med["near"][0] - med["near"][1] - (med["far"][0] - med["far"][1]), 2) == -2.39
    assert [round(med[z][2], 2) for z in ("near", "far")] == [5.14, 1.28]
    assert [round(med[z][3], 2) for z in ("near", "far")] == [0.57, 2.29]
    assert round(med["near"][2] - med["far"][2], 2) == 3.86
    assert (round(med["far"][1], 2), round(med["near"][1], 2)) == (-0.63, -1.71)
    # the trout's own median in the bin against its prediction there
    near_t = tb[(tb.depth_m < near_edge) & (tb.model_name == "Rainbow Trout")]
    assert round((100 * HALF_DEPTH_M / near_t.depth_m).median(), 2) == 5.14


def test_the_same_within_shared_calibrations(cohort, near_edge):
    """Under the seven calibrations that served both targets, with >= 3 frames
    of each per zone: the gap widens close in under 5 of 7 (518 flat, 522
    reversed), by 1.99 points on average against 4.22 predicted -- close in the
    gap is about the predicted size (mean -5.44 against -5.50), beyond 1.2 m it
    is larger (-3.45 against -1.29), a range-independent offset on top."""
    tb = cohort[cohort.model_name.isin(["Rainbow Trout", "Box"])].copy()
    tb["zone"] = np.where(tb.depth_m < near_edge, "near", np.where(tb.depth_m >= 1.2, "far", "mid"))
    rows = []
    for c, g in tb.groupby("calibration_dive_id"):
        rec = {"calib": c}
        for zone in ("near", "far"):
            t = g[(g.model_name == "Rainbow Trout") & (g.zone == zone)]
            b = g[(g.model_name == "Box") & (g.zone == zone)]
            ok = len(t) >= 3 and len(b) >= 3
            rec[zone] = t.pct_error.median() - b.pct_error.median() if ok else np.nan
            rec[f"pred_{zone}"] = -(100 * HALF_DEPTH_M / t.depth_m).median() if ok else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows).dropna()
    assert list(R.calib) == [496, 499, 502, 505, 518, 521, 522]
    widen = R.near - R.far
    assert int((widen < -0.05).sum()) == 5
    assert round(widen.mean(), 2) == -1.99 and round((R.pred_near - R.pred_far).mean(), 2) == -4.22
    assert [round(v, 2) for v in (R.near.mean(), R.pred_near.mean(), R.far.mean(), R.pred_far.mean())] == [-5.44, -5.50, -3.45, -1.29]


def test_how_much_of_the_field_was_taken_close_in():
    """35 of 162 field measurements (13 animals) closer than 1 m, 20 closer than
    0.81 m; none of the 39 stereo-day frames is closer than 1.24 m."""
    f = rep.load_field(DATA / "field.csv")
    assert len(f) == 162 and f.depth_m.notna().all()
    assert (int((f.depth_m < 1).sum()), int((f.depth_m < 0.81).sum())) == (35, 20)
    assert f[f.depth_m < 1].fish_id.nunique() == 13

    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv").set_index("dive_id")
    K = {int(r["camera_id"]): np.array(json.loads(r["km"])) for r in cal.load_rows(DATA / "corpus.csv")}
    pairs = spm.build_pairs(spm.load_ours(DATA / "stereo_pairs.csv"), spm.load_stereo(DATA / "stereo_reference.csv"))
    s = cal.load_laser_state(DATA / "laser_state.csv")
    sp = pd.read_csv(DATA / "stereo_pairs.csv", sep="|")
    sp = sp[sp.dive_id.isin({p.dive_id for p in pairs})].merge(s[["image_id", "dot_x", "dot_y"]], on="image_id")
    # 25 of 39 carry no stored depth; every frame triangulated under the day's beam
    z = cal.beam_depth_m(sp[["dot_x", "dot_y"]].values, K[int(fits.loc[32, "camera_id"])], *cal._beam(fits.loc[32]))
    assert len(z) == 39 and (int((z < 1).sum()), int((z < 0.81).sum())) == (0, 0)
    assert round(float(z.min()), 2) == 1.24


def test_section_5_2s_frame_thresholds():
    """Figure 16's rarefaction: 90 % of draws within 2 % of the cell's own p90
    from 4 frames, within 1.5 % from 10, within 1 % from 13 (each staying there
    for every larger n tested)."""
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    a = df[df.dive_id.isin(cal.CORPUS_ACCURACY_DIVES) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]
    cells = {k: v for k, v in rep.by_session_target(a, value="pct_error").items() if len(v) >= 30}
    r = rep.p90_rarefaction(cells)
    assert [rep.p90_min_frames(r, t) for t in (2.0, 1.5, 1.0)] == [4, 10, 13]


def test_figure_21_draws_these_values(cohort):
    """Every point Figure 21 draws: per target, fixed range bins whose first edge
    is the range figure's own near edge (0.81 m), so the first points are the
    near-bin medians the text quotes (trout -6.99 %, box -1.71 %). Each point
    sits at the target's median range in its bin."""
    m = cal.target_range_medians(cohort)
    got = {t: [(int(r.n), round(r.median, 2), round(r.z_m, 2)) for r in g.itertuples()]
           for t, g in m.groupby("target")}
    assert got == {
        "Rainbow Trout": [(68, -6.99, 0.57), (26, -4.61, 0.91), (31, -4.26, 1.14), (33, -4.27, 1.37),
                          (66, -3.58, 1.75), (65, -3.86, 2.27), (42, -3.19, 2.72), (35, -3.14, 3.26),
                          (41, -2.09, 3.95)],
        "Box": [(57, -1.71, 0.56), (21, -1.26, 0.9), (31, -0.92, 1.13), (30, -0.82, 1.37),
                (52, -0.53, 1.8), (54, -0.69, 2.27), (51, -0.66, 2.69), (25, -0.47, 3.22),
                (40, -0.48, 3.84)]}
    edges = sorted(set(round(v, 2) for v in np.r_[m.lo_m, m.hi_m]))
    assert edges == [0.28, 0.81, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 5.06]
