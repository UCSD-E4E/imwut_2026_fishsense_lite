"""Section 5.2's body-depth paragraph: do the thick targets read short against
flat ones measured under the same calibration, and by how much?

The laser lands on a solid model's near flank, about half its body depth in
front of the snout and fork, so the range -- and the length -- reads short by
(depth / 2) / z. The trout's depth is calipered (58.69 mm); the snook's has
never been measured, so it gets a comparison but no prediction.

"Under the same calibration" is the unit, because calibration is the confound.
The trout and the box are photographed in the same dive only in 521 and 522;
otherwise each checkerboard calibration serves one box dive and one trout dive
minutes apart. The snook shares its slate sessions with the angelfish and the
grouper. A target needs at least 8 frames in a group to carry a p90, the
cohort rule's own minimum.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fishsense_imwut import calibration as cal
from fishsense_imwut import pubfig

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
TROUT_DEPTH_M = 0.0587          # FINDINGS 9.5: 58.69 mm, calipered 2026-08-20
p90 = pubfig.nearest_rank_p90


@pytest.fixture(scope="module")
def cohort():
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    keep = cal.accuracy_cohort(df)
    return df[df.dive_id.isin(keep) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]


def _t_interval(x):
    x = np.asarray(x, float)
    q = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return round(x.mean(), 2), round(x.mean() - q, 2), round(x.mean() + q, 2)


def test_which_targets_share_a_calibration(cohort):
    by = cohort.groupby("dive_id").model_name.agg(lambda s: tuple(sorted(set(s))))
    both = sorted(d for d, m in by.items() if {"Rainbow Trout", "Box"} <= set(m))
    assert both == [521, 522]                                           # same dive
    tb = cohort[cohort.model_name.isin(["Rainbow Trout", "Box"])]
    n = tb.groupby("calibration_dive_id").model_name.nunique()
    assert sorted(n[n == 2].index) == [496, 499, 502, 505, 518, 521, 522]    # same calibration
    alone = tb[(tb.model_name == "Rainbow Trout") & ~tb.calibration_dive_id.isin(n[n == 2].index)]
    assert (sorted(alone.dive_id.unique()), len(alone), int((tb.model_name == "Rainbow Trout").sum())) == ([495, 527], 90, 407)
    snook = sorted(d for d, m in by.items() if "Snook" in m)
    assert snook == [59, 61, 66, 84]
    assert all({"Purple Angelfish", "Grouper"} <= set(by[d]) for d in snook)
    assert set(by[58]) == {"Grouper", "Purple Angelfish"}                    # the slate session without one


def test_trout_against_box_under_each_shared_calibration(cohort):
    """The trout's p90 is below the box's under all seven: gaps -1.23 to
    -5.82 points, mean -2.95 (95 % CI -4.47 to -1.42) -- against the 1.65-point
    gap of the whole-cohort figures (-1.00 against +0.65)."""
    tb = cohort[cohort.model_name.isin(["Rainbow Trout", "Box"])]
    gaps = {}
    for c, g in tb.groupby("calibration_dive_id"):
        t, b = g[g.model_name == "Rainbow Trout"], g[g.model_name == "Box"]
        if len(t) >= 8 and len(b) >= 8:
            gaps[c] = round(p90(t.pct_error) - p90(b.pct_error), 2)
    assert gaps == {496: -1.34, 499: -2.87, 502: -1.23, 505: -3.09, 518: -4.24, 521: -2.03, 522: -5.82}
    assert _t_interval(list(gaps.values())) == (-2.95, -4.47, -1.42)
    whole = {m: round(p90(g.pct_error), 2) for m, g in cohort.groupby("model_name")}
    assert (whole["Rainbow Trout"], whole["Box"]) == (-1.00, 0.65)
    assert round(whole["Rainbow Trout"] - whole["Box"], 2) == -1.65


def test_snook_against_the_flats_in_its_sessions(cohort):
    """Snook p90 minus each flat target's, per session; a cell needs 8 frames,
    which drops dive 66's grouper (4 frames, p90 -12.1 %). Against the
    angelfish the snook is lower in 4 of 4 sessions, against the grouper in 3
    of 3. Pooled over sessions (gap to the mean of the flats present): -1.65
    points, 95 % CI -2.96 to -0.35."""
    s = cohort[cohort.model_name.isin(["Snook", "Purple Angelfish", "Grouper"]) & cohort.dive_id.isin([59, 61, 66, 84])]
    per = {}
    for d, g in s.groupby("dive_id"):
        sn = p90(g[g.model_name == "Snook"].pct_error)
        flats = {m: p90(x.pct_error) for m, x in g.groupby("model_name") if m != "Snook" and len(x) >= 8}
        per[d] = {m: round(sn - v, 2) for m, v in flats.items()}
    assert per == {59: {"Grouper": -1.02, "Purple Angelfish": -3.32},
                   61: {"Grouper": -0.57, "Purple Angelfish": -1.63},
                   66: {"Purple Angelfish": -0.82},
                   84: {"Grouper": -2.52, "Purple Angelfish": -2.52}}
    g66 = s[(s.dive_id == 66) & (s.model_name == "Grouper")]
    assert (len(g66), round(p90(g66.pct_error), 1)) == (4, -12.1)
    pooled = [np.mean(list(v.values())) for v in per.values()]
    assert _t_interval(pooled) == (-1.65, -2.96, -0.35)
    whole = {m: round(p90(g.pct_error), 2) for m, g in cohort.groupby("model_name")}
    assert (whole["Snook"], whole["Purple Angelfish"], whole["Grouper"]) == (-1.16, 0.85, 0.90)


def test_the_predicted_shortfall(cohort):
    """Half the trout's 58.7 mm depth: 1.47 % at 2 m; 1.71 % at the median of
    its own frames' ranges (1.72 m, in the shared-calibration groups)."""
    assert round(100 * TROUT_DEPTH_M / 2 / 2.0, 2) == 1.47
    tb = cohort[cohort.model_name == "Rainbow Trout"]
    tb = tb[tb.calibration_dive_id.isin([496, 499, 502, 505, 518, 521, 522])]
    pred = 100 * TROUT_DEPTH_M / 2 / tb.depth_m
    assert (round(tb.depth_m.median(), 2), round(pred.median(), 2)) == (1.72, 1.71)
    assert round(-100 * TROUT_DEPTH_M / 2, 2) == -2.94                    # the 1/z coefficient it implies, %·m


def _ols(y, X):
    X = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    dof = len(y) - np.linalg.matrix_rank(X)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (e @ e) / dof)
    q = stats.t.ppf(0.975, dof)
    return b, b - q * se, b + q * se


def _gap_slope(g, thick, group):
    th = (g.model_name == thick).to_numpy(float)
    iz = 1 / g.depth_m.to_numpy()
    G = pd.get_dummies(g[group].astype(str), drop_first=True).to_numpy(float)
    T = pd.get_dummies(g.model_name, drop_first=True).to_numpy(float)
    b, lo, hi = _ols(g.pct_error.to_numpy(), np.c_[iz, th * iz, G, T])
    return round(b[2], 2), round(lo[2], 2), round(hi[2], 2)


def test_the_gap_against_one_over_range(cohort):
    """pct = group + target + b/z + delta_b * [thick]/z, within shared groups:
    delta_b is the 1/z term of the thick-minus-flat gap. Trout: -0.81 %·m
    (-1.19 to -0.42), right sign but a quarter of the -2.94 half its depth
    predicts. Snook: -0.44 (-6.24 to +5.36), unresolved."""
    tb = cohort[cohort.model_name.isin(["Rainbow Trout", "Box"])
                & cohort.calibration_dive_id.isin([496, 499, 502, 505, 518, 521, 522])]
    assert _gap_slope(tb, "Rainbow Trout", "calibration_dive_id") == (-0.81, -1.19, -0.42)
    s = cohort[cohort.model_name.isin(["Snook", "Purple Angelfish", "Grouper"]) & cohort.dive_id.isin([59, 61, 66, 84])]
    s = s[~((s.dive_id == 66) & (s.model_name == "Grouper"))]
    assert _gap_slope(s, "Snook", "dive_id") == (-0.44, -6.24, 5.36)
