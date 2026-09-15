"""Does any camera+laser unit carry a bias of its own?

The corpus was spread over seven units so that no one of them could drive the
accuracy figure. This module supplies the two groupings and the one-way fit that
check it, so the notebook can draw Figures 10 and 10b without carrying the
reasoning inline.

WHAT IS COMPARED, AND WHY NOT THE OBVIOUS THING
-----------------------------------------------
Not the raw per-unit percent error. Units photographed different target sets --
the 2023 sessions carry the fish models, the 2025 sessions the box and the trout
-- so a raw per-unit mean is confounded with which targets that unit happened to
see, and on this corpus that confound is most of the apparent spread: 3.2 points
raw against 1.6 once each target is centred on its own median.

`per_camera_offsets` therefore returns the **session effect of the median
polish**, the calibration offset with the per-target term already removed, which
is the quantity a rig bias would live in. `per_camera_frames` returns
target-centred per-frame errors for the distribution view, which is the same
correction applied one level down.

The angle-experiment sessions are excluded from both. One target at deliberately
oblique poses gives the polish nothing to separate the session from the target,
so what it returns for them is pose rather than a calibration offset.

CAVEAT WORTH CARRYING
---------------------
`camera_id` is the database's camera row, and `post_labeling_analysis/HANDOFF.md`
records that it does **not** track the physical rig number for the 2025 sessions.
If two ids are one rig, or one id is two rigs, the grouping is wrong. Note the
direction: mis-grouping mixes rigs and so tends to *hide* a real bias rather than
manufacture the null this finds.
"""

from typing import Mapping, Sequence

import numpy as np

from . import calibration as cal


def per_camera_offsets(rows: Sequence[dict], df=None) -> dict[int, list]:
    """camera id -> [(session id, calibration offset pp, in cohort), ...]."""
    if df is None:
        df = cal.to_frame(rows)
    dive_to_camera = {int(r["dive_id"]): int(r["camera_id"]) for r in rows}
    effects = cal.median_polish(cal.cell_p90_grid(df)).dive_effect
    cohort = set(cal.accuracy_cohort(df))
    angle = set(cal.ANGLE_TEST_DIVES)

    out: dict[int, list] = {}
    for dive in effects.index:
        d = int(dive)
        if d in angle:
            continue
        out.setdefault(dive_to_camera[d], []).append(
            (d, float(effects[dive]), d in cohort)
        )
    for cam in out:
        out[cam].sort(key=lambda t: t[0])
    return out


def per_camera_frames(rows: Sequence[dict], df=None) -> dict[int, np.ndarray]:
    """camera id -> per-frame percent error, cohort only, target offset removed."""
    if df is None:
        df = cal.to_frame(rows)
    df = df.copy()
    df["camera_id"] = [int(r["camera_id"]) for r in rows]
    s = df[df.dive_id.isin(cal.accuracy_cohort(df))].copy()
    s["e"] = s.pct_error - s.groupby("model_name").pct_error.transform("median")
    return {int(c): g.e.to_numpy() for c, g in s.groupby("camera_id")}


def one_way(groups: Mapping[int, Sequence]) -> dict:
    """Between- versus within-group spread, as a one-way variance-components fit.

    Accepts either the `(dive, value, keep)` triples of `per_camera_offsets` or
    plain sequences of values. The between-group component is clipped at zero:
    it is a variance, and an F below 1 -- groups differing by less than chance
    predicts -- would otherwise return a negative one.
    """
    from scipy import stats

    def values(v):
        return [x[1] if isinstance(x, tuple) else float(x) for x in v]

    data = [values(groups[c]) for c in sorted(groups)]
    k = len(data)
    n = sum(len(g) for g in data)
    if k < 2 or n <= k:
        raise ValueError("need at least two groups and more values than groups")

    grand = np.mean([x for g in data for x in g])
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in data)
    ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in data)
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)

    f = ms_between / ms_within
    p = float(1 - stats.f.cdf(f, k - 1, n - k))
    # Unbalanced groups, so the variance component uses the effective group size.
    n0 = (n - sum(len(g) ** 2 for g in data) / n) / (k - 1)
    return dict(
        k=k,
        n=n,
        f=float(f),
        p=p,
        df=(k - 1, n - k),
        within_sd=float(np.sqrt(ms_within)),
        between_sd=float(np.sqrt(max(0.0, (ms_between - ms_within) / n0))),
    )
