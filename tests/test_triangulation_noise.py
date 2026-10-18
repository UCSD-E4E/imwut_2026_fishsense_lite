"""Section 5.7: where triangulation noise starts to matter.

Labeling noise from repeat labels of the same frame (`data/label_repeats.csv`):
- dot, along the dive's image line (the component that moves range): 267 human
  pairs that are not identical and differ by under 20 px. 461 of the 733 repeat
  pairs are exact duplicates and the detector matches the live human dot exactly
  on 2,242 of 2,364 frames: labels were seeded from predictions, so those pairs
  measure nothing and are dropped. Even the kept pairs may be a second labeler
  nudging the first, which would make these values low.
- snout/fork, along the fish's own axis: 11 frames with two labels (8 of them one
  labeler relabeling), 20 endpoint differences after dropping two fork
  disagreements of 57 and 108 px.

Terms, for a target of length L at range z, fleet medians B = 0.104 m and
f = 2832.7 px: dot 100 z sd / (B f); endpoints 100 sqrt(2) z se / (L f) -- two
endpoints, hence sqrt(2). Combined in quadrature; a Gaussian's IQR is 1.349 sd.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
L_M = 0.30


@pytest.fixture(scope="module")
def reps():
    return pd.read_csv(DATA / "label_repeats.csv", sep="|")


def _per_label(v):
    v = np.asarray(v, float)
    return float(v.std() / np.sqrt(2)), float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(2))


@pytest.fixture(scope="module")
def sigma_dot(reps):
    d = reps[(reps.kind == "dot") & (reps.source == "human")].sort_values(["image_id", "label_id"])
    g = d.groupby("image_id")
    a, b = g.first(), g.last()
    n = np.hypot(a.line_a, a.line_b)
    ux, uy = a.line_b / n, -a.line_a / n
    dx, dy = b.x - a.x, b.y - a.y
    dist = np.hypot(dx, dy)
    keep = (dist >= 0.01) & (dist < 20)
    along, across = (dx * ux + dy * uy)[keep], (-dx * uy + dy * ux)[keep]
    return dict(n_pairs=len(a), duplicates=int((dist < 0.01).sum()), kept=int(keep.sum()),
                gross=int((dist >= 20).sum()), along=_per_label(along), across=_per_label(across))


@pytest.fixture(scope="module")
def sigma_end(reps):
    h = reps[(reps.kind == "headtail") & (reps.source == "human")].sort_values(["image_id", "label_id"])
    g = h.groupby("image_id")
    a, b = g.first(), g.last()
    ax = np.c_[a.x - a.x2, a.y - a.y2]
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    head = (np.c_[b.x - a.x, b.y - a.y] * ax).sum(1)
    tail = (np.c_[b.x2 - a.x2, b.y2 - a.y2] * ax).sum(1)
    v = np.r_[head, tail[np.abs(tail) < 20]]
    return dict(frames=len(a), same_labeler=int((a.user_id == b.user_id).sum()),
                dropped=sorted(round(float(t), 1) for t in tail[np.abs(tail) >= 20]), n=len(v), sigma=_per_label(v))


def test_dot_label_noise_along_the_line(sigma_dot):
    s = sigma_dot
    assert (s["n_pairs"], s["duplicates"], s["kept"], s["gross"]) == (733, 461, 267, 5)
    assert [round(v, 2) for v in s["along"]] == [0.90, 0.28]          # SD-based, robust
    assert [round(v, 2) for v in s["across"]] == [0.65, 0.28]
    assert round(s["along"][0] / s["across"][0], 2) == 1.38
    # the 0.64 px perpendicular field scatter matches the across value; scaled by
    # the measured anisotropy it gives about the along value
    assert round(0.64 * s["along"][0] / s["across"][0], 2) == 0.88


def test_the_detector_is_not_an_independent_check(reps):
    d = reps[reps.kind == "dot"]
    det = d[d.source == "detector"].set_index("image_id")[["x", "y"]]
    hum = d[d.source == "human_live"].groupby("image_id")[["x", "y"]].first()
    j = det.join(hum, rsuffix="_h", how="inner")
    dist = np.hypot(j.x - j.x_h, j.y - j.y_h)
    assert (len(j), int((dist < 0.01).sum()), int(((dist >= 0.01) & (dist < 20)).sum())) == (2364, 2242, 29)


def test_endpoint_label_noise(sigma_end):
    s = sigma_end
    assert (s["frames"], s["same_labeler"], s["n"]) == (11, 8, 20)
    assert s["dropped"] == [-108.3, -56.6]
    assert [round(v, 2) for v in s["sigma"]] == [2.29, 2.08]


def _terms(sd, se, B=0.104, f=2832.7):
    k_dot = 100 * sd / (B * f)
    k_end = 100 * np.sqrt(2) * se / (f * L_M)
    return k_dot, k_end, float(np.hypot(k_dot, k_end))


def test_the_fleet_medians():
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv")
    assert round(float(fits.baseline_m.median()), 3) == 0.104
    fx = [json.loads(r["km"])[0][0] for r in cal.load_rows(DATA / "corpus.csv")]
    assert round(float(np.median(fx)), 1) == 2832.7


def test_the_noise_terms(sigma_dot, sigma_end):
    """Percent length error for a 300 mm target, at 1-5 m."""
    k_dot, k_end, k = _terms(sigma_dot["along"][0], sigma_end["sigma"][0])
    z = np.arange(1, 6)
    assert [round(v, 2) for v in k_dot * z] == [0.30, 0.61, 0.91, 1.22, 1.52]
    assert [round(v, 2) for v in k_end * z] == [0.38, 0.76, 1.14, 1.52, 1.90]
    assert [round(v, 2) for v in k * z] == [0.49, 0.98, 1.46, 1.95, 2.44]
    assert [round(v, 2) for v in 1.349 * k * z] == [0.66, 1.32, 1.97, 2.63, 3.29]


def test_against_the_observed_spread(sigma_dot, sigma_end):
    """Predicted IQR sits below the observed IQR in every range bin; held at the
    observed spread beyond 0.81 m (median IQR 3.40) they meet at 5.16 m, and at
    the last bin's 3.27 at 4.97 m. Noise alone reaches 15 % at 30.8 m (one sd)
    or 18.7 m (90 % of frames inside)."""
    _, _, k = _terms(sigma_dot["along"][0], sigma_end["sigma"][0])
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    a = df[df.dive_id.isin(cal.accuracy_cohort(df)) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]
    b = cal.range_bins(a)
    z = [a[(a.depth_m >= lo) & (a.depth_m <= hi)].depth_m.median() for lo, hi in zip(b.lo_m, b.hi_m)]
    pred = 1.349 * k * np.array(z)
    assert [round(v, 2) for v in pred] == [0.37, 0.67, 0.88, 1.09, 1.28, 1.50, 1.80, 2.40]
    assert (pred < b.iqr.to_numpy()).all()
    far = float(b.iqr.iloc[1:].median())
    assert round(far, 2) == 3.40 and round(far / (1.349 * k), 2) == 5.16
    assert round(float(b.iqr.iloc[-1]) / (1.349 * k), 2) == 4.97
    assert (round(15 / k, 1), round(15 / (1.645 * k), 1)) == (30.8, 18.7)


def test_the_farthest_data():
    """The pooled cohort reaches 5.06 m; its last bin, 3.14-5.06 m, holds 125
    frames at median -1.51 %, IQR 3.27, median range 3.65 m. 30 frames lie beyond
    4 m and 8 beyond 4.5 m."""
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    a = df[df.dive_id.isin(cal.accuracy_cohort(df)) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]
    last = cal.range_bins(a).iloc[-1]
    assert round(float(a.depth_m.max()), 2) == 5.06
    assert (round(last.lo_m, 2), round(last.hi_m, 2), int(last.n), round(last["median"], 2), round(last.iqr, 2)) == \
        (3.14, 5.06, 125, -1.51, 3.27)
    assert round(float(a[a.depth_m >= last.lo_m].depth_m.median()), 2) == 3.65
    assert (int((a.depth_m > 4).sum()), int((a.depth_m > 4.5).sum())) == (30, 8)


def test_the_trout_reference_cannot_make_it_read_short():
    """Scored against 313 mm; the SfM scan and LiDAR (FINDINGS §7.6) say about
    315 mm. Were 315 true, the trout's shortfall would be 0.63 points larger."""
    assert cal.MEASURED_REFERENCES_M["Rainbow Trout"] == 0.313
    assert round(100 * (0.313 / 0.315 - 1), 2) == -0.63
