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
