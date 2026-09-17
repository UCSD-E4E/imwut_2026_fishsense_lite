"""Within-group repeatability -- the one field result that needs nothing
external (PAPER.md section 4.5).

A calibration error is common to every frame of one fish and cancels in a
relative spread, as does any error in the length convention, and no comparison
population is involved. So a within-fish coefficient of variation measures the
system on wild animals with no reference object in the water, which is the only
thing this corpus permits: no known-length target was carried on the reef
dives, so accuracy there is not measurable at all.

What it cannot see, and section 4.5 says so:

* **Anything common to a fish's frames.** A calibration that scales every
  length by 1.18 leaves every CV untouched -- that invariance is the point, and
  it is also the blindness. Repeatability is not accuracy.
* **Whether the animal is the species the labeler named.** The spread is
  computed within one Fish row, whatever it is; a mislabel changes which
  reference a length would be compared against, and this statistic compares it
  against nothing.
* **A fish that moved between frames** in a way that changes its projected
  length consistently rather than noisily -- a slow turn reads as a small CV.

The estimator is deliberately the same shape as the rest of the paper: the
median across groups, a percentile bootstrap on that median, and p90 by nearest
rank (`ceil(0.9n)`), matching `pubfig.nearest_rank_p90` and the pipeline's own
`fish_length_estimate` view. An interpolated p90 is a materially different
number on 25 groups, and a draft of section 4.5 quoted one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: Frames of one individual needed before its spread means anything. Two
#: frames give a CV, but one determined entirely by a single difference.
MIN_FRAMES = 3

#: Bootstrap draws for the interval on the median. Enough that the reported
#: two-decimal bounds are stable across seeds.
RESAMPLES = 20_000

#: The seven Florida-reef deployments of section 4.5.
FIELD_DIVES = (279, 341, 347, 349, 383, 465, 471)


@dataclass(frozen=True)
class Repeatability:
    """The reported figure and everything needed to audit it."""

    n_groups: int
    n_values: int
    median_percent: float
    ci_percent: tuple[float, float]
    p90_percent: float
    cvs_percent: "object"  # pandas Series, keyed by group

    def summary(self) -> str:
        if not self.n_groups:
            return "no group reaches the frame floor"
        lo, hi = self.ci_percent
        return (
            f"{self.median_percent:.2f} % (median; bootstrap 95 % CI "
            f"{lo:.2f}-{hi:.2f} %), p90 {self.p90_percent:.2f} %, "
            f"{self.n_groups} groups / {self.n_values} measurements"
        )


def cv_percent(values: Sequence[float]) -> float:
    """Coefficient of variation in percent, on the sample standard deviation.

    `ddof=1` because these are samples of a repeat measurement and n is 3 often
    enough for the divisor to matter -- the population form would understate
    every three-frame fish by 18 %.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(100.0 * arr.std(ddof=1) / arr.mean())


def repeatability(
    groups: Mapping[object, Sequence[float]],
    *,
    min_frames: int = MIN_FRAMES,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> Repeatability:
    """Median within-group CV, its bootstrap interval, and its nearest-rank p90.

    Groups with fewer than `min_frames` values are dropped, not counted as
    zero. `seed` is fixed so a quoted interval is reproducible.
    """
    import pandas as pd

    kept = {k: np.asarray(v, dtype=float) for k, v in groups.items()}
    kept = {k: v for k, v in kept.items() if v.size >= min_frames}
    if not kept:
        empty = pd.Series(dtype=float)
        return Repeatability(0, 0, float("nan"), (float("nan"), float("nan")), float("nan"), empty)

    cvs = pd.Series({k: cv_percent(v) for k, v in kept.items()}, dtype=float)
    arr = cvs.to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(arr, size=(resamples, arr.size), replace=True), axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ordered = np.sort(arr)
    p90 = float(ordered[int(np.ceil(0.9 * ordered.size)) - 1])
    return Repeatability(
        n_groups=int(arr.size),
        n_values=int(sum(v.size for v in kept.values())),
        median_percent=float(np.median(arr)),
        ci_percent=(float(lo), float(hi)),
        p90_percent=p90,
        cvs_percent=cvs,
    )


def by_individual(df, *, key: str = "fish_id", value: str = "length_m"):
    """Lengths grouped by animal -- the field cohort's grouping.

    One `Fish` row per animal per dive, bound by stage 14 from the frame's
    LABEL_STUDIO cluster, so repeat frames of one animal share it.
    """
    return {k: g[value].to_numpy() for k, g in df.groupby(key)}


def by_session_target(
    df, *, keys: Sequence[str] = ("dive_id", "model_name"), value: str = "length_m"
):
    """Lengths grouped by (session, target) -- the pool cohort's grouping, and
    the closest analogue to one wild individual: one rigid object, one session,
    one calibration."""
    return {k: g[value].to_numpy() for k, g in df.groupby(list(keys))}


def load_field(path):
    """Read `data/field.csv` as extracted by `sql/extract_field.sql`."""
    import pandas as pd

    return pd.read_csv(path, sep="|")


# --- how many frames does p90 need? --------------------------------------
#
# The ruler forced this question (HELD_OUT_MODELS): six frames, and its p90 was
# simply the largest of the six. Nearest rank is ceil(0.9n), which equals n for
# every n <= 10, so below ten frames "p90" is not a quantile at all -- it is the
# maximum, the noisiest order statistic there is. This measures what that costs.


@dataclass(frozen=True)
class RarefactionPoint:
    """One sample size's worth of subsample behaviour.

    `abs_p90` is the REPORTED statistic, for the same reason p90 is reported
    everywhere else: it answers the question actually being asked, which is how
    far wrong a p90 from n frames can be, not how far wrong it typically is.
    `median`, `lo` and `hi` describe the SIGNED error and are kept because the
    sign flips across the small-n range -- the maximum of two draws sits below
    the true 90th percentile, the maximum of nine sits above it -- so a small
    sample is not conservative in either direction.
    """

    n: int
    rank: int
    abs_p90: float      # 90th percentile of |error|, in pp
    within: float       # fraction of draws inside +-tolerance
    median: float       # median SIGNED error
    lo: float           # 10th percentile of the signed error
    hi: float           # 90th percentile of the signed error


def p90_rarefaction(groups, sizes=range(2, 41), draws: int = 1000, seed: int = 0,
                    tolerance: float = 1.0) -> dict:
    """How far a p90 from `n` frames lands from the same group's full-sample p90.

    `groups` maps a key to that group's per-frame percent errors -- use
    `by_session_target` on the cohort. Only groups with enough frames to draw
    `n` without replacement contribute at each `n`, so a group never competes
    with itself and `n` is never inflated by resampling.

    Returns ``{n: RarefactionPoint}``. The reference is each group's own
    full-sample p90, so this isolates the estimator's sampling behaviour from
    how accurate that group happened to be. Read it as the error a diver with
    `n` frames of one fish actually faces.
    """
    from .pubfig import nearest_rank_p90

    rng = np.random.default_rng(seed)
    full = {k: nearest_rank_p90(v) for k, v in groups.items()}
    out = {}
    for n in sizes:
        errs = [
            np.array([nearest_rank_p90(rng.choice(v, n, replace=False))
                      for _ in range(draws)]) - full[k]
            for k, v in groups.items() if len(v) >= n
        ]
        if not errs:
            continue
        e = np.concatenate(errs)
        lo, hi = np.percentile(e, [10, 90])
        out[int(n)] = RarefactionPoint(
            n=int(n),
            rank=int(np.ceil(0.9 * n)),
            abs_p90=float(np.percentile(np.abs(e), 90)),
            within=float((np.abs(e) < tolerance).mean()),
            median=float(np.median(e)),
            lo=float(lo),
            hi=float(hi),
        )
    return out


def p90_min_frames(rarefaction, tolerance: float = 1.0) -> int | None:
    """Smallest n whose p90 is within `tolerance` pp of its own limit 90 % of
    the time, and stays there for every larger n tested.

    Reads `abs_p90` directly rather than the `within` column, so the threshold
    asked for here is the one applied -- `p90_rarefaction`'s `tolerance` only
    shapes `within`, which is the same statement seen from the other side.
    """
    ns = sorted(rarefaction)
    for i, n in enumerate(ns):
        if all(rarefaction[m].abs_p90 <= tolerance for m in ns[i:]):
            return n
    return None
