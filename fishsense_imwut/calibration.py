"""Laser-calibration geometry for the fish-model validation set.

Ported from the 2026-08-26 calibration-repair working session (see
`fish_model_analysis/HANDOFF.md`) so the publication figures are computed by the
same code that produced the handoff's tables, not by a re-derivation that might
drift from them. `test_reproduces_handoff` in the notebook pins the key numbers.

The physical model, in one line: a fish-model dive does not self-calibrate. It
**borrows** a sibling slate dive's extrinsics shot ~30 minutes earlier, and in
between the laser rotates about an effectively fixed pivot. So the borrow error
is a **rotation**, not an arbitrary 4-DOF line change -- and of that rotation
only the *in-plane* component phi matters, because:

  * the out-of-plane component is observable from the dive's own dots and is
    worth ~0.1 % of depth;
  * phi is **invisible** to the dots. Rotating the axis in the camera-laser
    plane moves the projected dot by ~1e-13 px. The image of a 3-D line fixes
    only the plane through the camera centre containing it; where the line sits
    *within* that plane projects identically. That is monocular scale
    ambiguity, and it is why no amount of dot-reprojection refinement can
    recover scale.

phi therefore has to come from a known length (an anchor), or from the same
object seen at two well-separated ranges (scale-free).
"""

import json
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

# Physical floor for the implied-yaw diagnostic: you cannot present a rigid
# object better than side-on, so a sound calibration must reach ~0 degrees.
_YAW_SEARCH_MAX_DEG = 75.0


def rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of `v` about `axis` by `angle` radians."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def triangulate_depth(
    pixel: Sequence[float],
    laser_origin: np.ndarray,
    laser_axis: np.ndarray,
    inv_intrinsics: np.ndarray,
) -> float:
    """Z of the closest-approach midpoint between the camera ray through
    `pixel` and the laser ray. This is the quantity stage 14 back-projects head
    and tail against."""
    d = inv_intrinsics @ np.array([pixel[0], pixel[1], 1.0])
    d /= np.linalg.norm(d)
    a = laser_axis / np.linalg.norm(laser_axis)
    b = d @ a
    d1 = -(d @ laser_origin)
    e1 = -(a @ laser_origin)
    den = 1 - b * b
    return (
        0.5 * (((b * e1 - d1) / den) * d + laser_origin + ((e1 - b * d1) / den) * a)
    )[2]


def dive_geometry(rows: Sequence[dict]):
    """Unpack the per-dive geometry shared by every row: laser origin `o`,
    laser axis `a`, inverse intrinsics, and `n` -- the normal of the
    camera-laser plane, which is the axis phi rotates about."""
    o = np.array(json.loads(rows[0]["pos"]))
    a = np.array(json.loads(rows[0]["ax"]))
    a /= np.linalg.norm(a)
    inv_k = np.linalg.inv(np.array(json.loads(rows[0]["km"])))
    n = np.cross(o, a)
    n /= np.linalg.norm(n)
    return o, a, inv_k, n


def length_ratios(
    rows: Sequence[dict], phi: float, o, a, inv_k, n
) -> np.ndarray:
    """Measured/known length ratio for each row under an in-plane rotation
    `phi` (radians).

    Length is linear in depth -- head and tail are back-projected at the single
    laser-derived depth -- so re-scaling by the ratio of new to recorded depth
    is exact, and no re-labelling is needed to test a candidate phi.
    """
    a2 = rotate(a, n, phi)
    out = []
    for r in rows:
        u = [float(x) for x in r["dot"].split(";")]
        z0 = float(r["depth_m"])
        length = float(r["length_m"])
        known = float(r["known_length_m"])
        out.append((length * triangulate_depth(u, o, a2, inv_k) / z0) / known)
    return np.array(out)


def fit_phi(anchor_rows, o, a, inv_k, n, q: float = 0.90) -> float:
    """Solve for the phi that makes the anchor's q-quantile length ratio 1.

    The quantile rather than the mean is deliberate and load-bearing:
    foreshortening is one-sided negative, so a mean anchors to the pose
    distribution instead of the object.
    """
    f = lambda p: np.quantile(length_ratios(anchor_rows, p, o, a, inv_k, n), q) - 1.0
    return brentq(f, np.deg2rad(-1.5), np.deg2rad(1.5))


def fit_phi_joint(rows, o, a, inv_k, n, min_frames: int = 5) -> float:
    """phi from every model on the dive at once, with no known length trusted
    individually -- each model's p90 ratio is driven to 1 simultaneously.

    This is the estimator behind the mount-state distribution: it needs no
    anchor, so it is available on all seven dives.
    """
    models = np.array([r["model"] for r in rows])
    usable = [m for m in sorted(set(models)) if (models == m).sum() >= min_frames]
    residual = lambda p: [
        np.quantile(
            length_ratios([rows[k] for k in np.where(models == m)[0]], p[0], o, a, inv_k, n),
            0.9,
        )
        - 1
        for m in usable
    ]
    return float(least_squares(residual, [0.0]).x[0])


def implied_yaw_deg(known_length: float, measured: float, z: float) -> float:
    """The out-of-plane yaw a rigid plate of `known_length` would need for a
    single-depth back-projection at range `z` to read `measured`.

    The second-order term matters at ~1.1 m, so it is kept:

        measured = L cos(t) / (1 - ((L/2) sin(t) / z)^2)

    Returns 0.0 when the measurement is at or above the known length -- the
    diagnostic **clips at zero and is blind to over-correction**, so it must
    always be read alongside the signed error. That blindness is what flagged
    dive 66 in the handoff.
    """
    f = (
        lambda t: known_length
        * np.cos(t)
        / (1 - ((known_length / 2) * np.sin(t) / z) ** 2)
        - measured
    )
    try:
        return float(np.degrees(brentq(f, 0, np.radians(_YAW_SEARCH_MAX_DEG))))
    except ValueError:
        return 0.0


def yaw_floor(
    rows, phi: float, o, a, inv_k, n, quantile: float = 0.10, min_frames: int = 5
) -> dict:
    """Per-model implied-yaw floor: the 10th percentile over frames, i.e. the
    best-presented ones.

    A **calibration** error lifts every object's floor uniformly; **pose** only
    adds a one-sided tail above it. That asymmetry is what makes the floor able
    to separate the two.

    Models with fewer than `min_frames` frames are skipped: a 10th percentile
    over 3 frames is just the minimum, and it reads as a confident floor.
    """
    models = np.array([r["model"] for r in rows])
    depths = np.array([float(r["depth_m"]) for r in rows])
    out = {}
    for m in sorted(set(models)):
        idx = np.where(models == m)[0]
        if idx.size < min_frames:
            continue
        known = float(rows[idx[0]]["known_length_m"])
        lengths = length_ratios([rows[k] for k in idx], phi, o, a, inv_k, n) * known
        out[m] = float(
            np.quantile(
                [implied_yaw_deg(known, v, z) for v, z in zip(lengths, depths[idx])],
                quantile,
            )
        )
    return out


# --- data ----------------------------------------------------------------

# Per-dive borrow map, from `Dive.calibration_dive_id`. Each fish-model dive is
# fish-only and borrows a sibling slate dive shot ~30 min earlier.
BORROW_MAP = {58: 71, 59: 77, 60: 65, 61: 80, 66: 83, 76: 63, 84: 62}

# The two repairs the handoff proposes. NOT applied to prod.
#   60: ruler anchor, fitted scale-free -- the ruler's 342.9 mm is never spent,
#       so it stays held out and grades the result at +0.1 %.
#   76: all four models jointly; costs the models on that dive, which is why 76
#       demonstrates the method but is not accuracy evidence.
REPAIR_PHI_DEG = {60: -0.1458, 76: -0.1505}

# Dive 66's ruler is 4 frames at one depth, so its fit must spend the 342.9 mm.
# Applying it over-corrects two held-out models (-1.0 -> +5.0, -0.4 -> +4.2).
# Recorded for the figure that shows the discrepancy; deliberately not applied.
DISPUTED_PHI_DEG = {66: -0.1350}

# --- August 2026 seven-dive cohorts (historical) --------------------------
#
# These back the HANDOFF reproduction and the repair figures (4-7). They were
# hand-picked. The paper's accuracy numbers now come from the rule-derived
# CORPUS_ACCURACY_DIVES further down; these stay so the August tables remain
# reproducible, not because they are the accuracy claim.
#
# Which dives back an accuracy number is a claim about the paper, not a
# plotting detail, so every cohort lives here rather than inline in a notebook.
ACCURACY_DIVES = (58, 59, 60, 61, 84)  # 60 included: its anchor is scale-free
UNTOUCHED_DIVES = (58, 59, 61, 84)  # no repair applied at all

# The per-model "ladder" cohort. NOT the same as UNTOUCHED_DIVES, and the
# difference is deliberate: HANDOFF section 6 quotes the ladder over 59/61/84,
# excluding 58, while section 3 puts 58 in the accuracy cohort. Dive 58 carries
# the largest positive calibration offset of any dive (+2.98 pp; +2.14 % median
# on its own) over just 24 frames, and 12 of those are Shark -- so folding it in
# inflates exactly the model whose bias is under investigation, while Shark
# still gets nothing from dive 61, the near-zero-offset dive. Keeping the two
# cohorts separate is what stops that confound leaking into the ladder.
LADDER_DIVES = (59, 61, 84)
METHOD_DEMO_DIVES = (76,)  # the repair works here; not accuracy evidence
DISPUTED_DIVES = (66,)  # ruler and models disagree by ~4 %, unresolved


def load_rows(path) -> list[dict]:
    """Read a '|'-delimited handoff CSV into row dicts.

    Pipe-delimited because the geometry columns are JSON and contain commas.
    Rows without a `dot` are dropped, which also discards psql's trailing
    `(437 rows)` footer.
    """
    import csv

    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh, delimiter="|") if r.get("dot")]


def group_by_dive(rows: Sequence[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(int(r["dive_id"]), []).append(r)
    return out


def to_frame(rows: Sequence[dict]):
    """Tidy per-frame DataFrame for the distribution figures."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "dive_id": [int(r["dive_id"]) for r in rows],
            "calibration_dive_id": [int(r["calibration_dive_id"]) for r in rows],
            "model_name": [r["model"] for r in rows],
            "known_length_m": [float(r["known_length_m"]) for r in rows],
            "length_m": [float(r["length_m"]) for r in rows],
            "depth_m": [float(r["depth_m"]) for r in rows],
        }
    )
    df["error_m"] = df.length_m - df.known_length_m
    df["pct_error"] = 100 * df.error_m / df.known_length_m
    return df


# --- full corpus (2026-09) --------------------------------------------------
#
# `data/corpus.csv` is every measurement of a rigid target in prod as of
# 2026-09-12: 2,927 frames over 32 dives, extracted by `sql/extract_corpus.sql`
# in the same schema as `all.csv` (of which it is a strict superset; pinned by
# a test). The new pool dives (480-522) self-calibrate from a checkerboard
# rather than borrowing a slate, so `calibration_dive_id` equals `dive_id` for
# them and BORROW_MAP above does not cover them.

# The designed foreshortening experiment: one Snook (455 mm) presented at
# 0-45 deg in 5 deg steps, five sessions at two ranges. Single-object dives,
# so the polish cannot separate their dive effect from Snook's model effect,
# and their purpose is to grade the angle model, not the system.
ANGLE_TEST_DIVES = (87, 94, 103, 107, 114)
ANGLE_TEST_KNOWN_M = 0.455

# Dive 490 read every model ~14 % short with verified labels and a clean board
# fit. Resolved 2026-09-12: its fish frames were shot nine seconds after the
# PRECEDING board burst (dive 489) and five minutes before its own, and the
# laser had rotated 0.82 deg in-plane in between. The fish frames were split
# into dive 527 (borrows 489) and re-measured at -1.7 % median; 490 now holds
# only its board burst and no measurements. Nothing is unresolved any more, so
# this is empty; it stays so the notebook's status column keeps its vocabulary.
UNRESOLVED_DIVES: tuple[int, ...] = ()
SPLIT_DIVES = {490: 527}  # original -> the dive its fish frames became

# Dives held out of the accuracy cohort BY DESIGN, decided before any corpus
# number was looked at, so the threshold below cannot be tuned around them:
#   * the angle experiment (above);
#   * 60 and 76, whose borrowed calibration the August analysis repaired --
#     keeping their raw rows would assert the borrow was fine, contradicting
#     the repair the paper reports;
#   * 66, whose ruler and models disagree by ~4 % (unresolved, reported).
# They still take part in the polish: every extra (dive, model) cell sharpens
# the model effects, and the polish is robust to the rows it is about to drop.
DESIGN_EXCLUDED_DIVES = ANGLE_TEST_DIVES + tuple(REPAIR_PHI_DEG) + DISPUTED_DIVES

# The accuracy-cohort rule. A dive is accuracy evidence when its calibration
# offset -- the dive effect of a Tukey median polish over the p90 of each
# (dive, model) cell, fitted on the whole corpus -- is within this many
# percentage points of the corpus median. One number applied uniformly to all
# 32 dives, in place of the August hand-pick, and model-agnostic: a per-model
# reference error lands in the model effect and cannot get a dive in or out.
MAX_DIVE_EFFECT_PP = 2.5
POLISH_MIN_FRAMES = 5

# What the rule selects on the corpus, pinned as data so a change is a diff.
# 2026-09-12, after the 490 -> 527 split: 491 (-2.73) and 527 (+2.95) both sit
# just outside the band; with 490's -13.8 gone the corpus median moved ~0.3 pp
# and took 491 with it. The rule is not tuned to keep them.
CORPUS_ACCURACY_DIVES = (
    59, 61, 84, 495, 497, 498, 500, 501, 503, 507, 519, 520, 521, 522,
)


class MedianPolish:
    """Result of `median_polish`: grid = overall + dive_effect + model_effect
    + residual, with every effect centred on a median of zero."""

    def __init__(self, overall, dive_effect, model_effect, residual):
        self.overall = overall
        self.dive_effect = dive_effect
        self.model_effect = model_effect
        self.residual = residual


def median_polish(grid, iterations: int = 50) -> MedianPolish:
    """Tukey median polish of a dives x models grid (NaN = unobserved cell).

    A **calibration** error is per-dive and moves every model on that dive
    together; a **reference or landmark** error is per-model and follows the
    model across dives. The polish separates the two additive terms robustly,
    so one wild cell cannot pull a dive in or out of the cohort.
    """
    import pandas as pd

    m = grid.astype(float).copy()
    overall = 0.0
    dive_effect = pd.Series(0.0, index=m.index)
    model_effect = pd.Series(0.0, index=m.columns)
    for _ in range(iterations):
        rm = m.median(axis=1, skipna=True)
        m = m.sub(rm, axis=0)
        dive_effect += rm
        d = dive_effect.median()
        dive_effect -= d
        overall += d
        cm = m.median(axis=0, skipna=True)
        m = m.sub(cm, axis=1)
        model_effect += cm
        d = model_effect.median()
        model_effect -= d
        overall += d
    return MedianPolish(float(overall), dive_effect, model_effect, m)


def cell_p90_grid(df, min_frames: int = POLISH_MIN_FRAMES):
    """p90 percent error per (dive, model) cell, as a dives x models grid.
    Cells with fewer than `min_frames` frames are left unobserved: a p90 over
    two frames is the maximum, and it reads as a confident cell."""
    from .pubfig import nearest_rank_p90

    cells = (
        df.groupby(["dive_id", "model_name"])["pct_error"]
        .agg(n="size", p90=nearest_rank_p90)
        .reset_index()
    )
    return cells[cells.n >= min_frames].pivot(
        index="dive_id", columns="model_name", values="p90"
    )


def accuracy_cohort(
    df,
    max_dive_effect_pp: float = MAX_DIVE_EFFECT_PP,
    exclude: Sequence[int] = DESIGN_EXCLUDED_DIVES,
    min_frames: int = POLISH_MIN_FRAMES,
) -> tuple[int, ...]:
    """The dives whose calibration offset is within `max_dive_effect_pp` of
    the corpus median, less the dives held out by design.

    The polish is fitted on every dive, `exclude` included, and the hold-out is
    applied to the result: the held-out dives contribute cells to the model
    effects but cannot themselves qualify. Returned sorted, as a tuple, so it
    compares directly to CORPUS_ACCURACY_DIVES.
    """
    fit = median_polish(cell_p90_grid(df, min_frames))
    return tuple(
        int(d) for d in sorted(fit.dive_effect.index)
        if abs(fit.dive_effect[d]) <= max_dive_effect_pp and int(d) not in set(exclude)
    )


def load_angles(path, known_m: float = ANGLE_TEST_KNOWN_M):
    """The foreshortening experiment export: one row per frame with the
    labelled `fish_angle_degrees` (read off the card in frame), the measured
    length and the laser depth. Frames without an angle are dropped."""
    import pandas as pd

    df = pd.read_csv(path, na_values=["\\N", ""])
    df = df.dropna(subset=["fish_angle_degrees", "length_m"]).copy()
    df["dive_id"] = df.dive_id.astype(int)
    df["pct_error"] = 100 * (df.length_m - known_m) / known_m
    return df


def binned_angle_error(
    ang,
    bins: Sequence[float] = tuple(range(0, 50, 5)),
    half_width: float = 2.5,
    min_count: int = 5,
):
    """Median / IQR of percent error in a symmetric window about each
    designed angle. The experiment stepped in 5 deg increments, so the window
    is +-2.5 deg; a bin with fewer than `min_count` frames reads NaN rather
    than a confident median over a handful."""
    import pandas as pd

    out = []
    for b in bins:
        sel = ang.pct_error[(ang.fish_angle_degrees - b).abs() < half_width]
        if len(sel) < min_count:
            out.append((b, len(sel), np.nan, np.nan, np.nan))
        else:
            out.append((b, len(sel), sel.median(), sel.quantile(0.25), sel.quantile(0.75)))
    return pd.DataFrame(out, columns=["angle", "n", "median", "q1", "q3"]).set_index("angle")
