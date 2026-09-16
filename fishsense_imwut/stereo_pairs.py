"""The 2023-08-03 paired-instrument comparison (FINDINGS section 12).

Seven wild fish measured twice on one day: once by FishSense Lite, once by a
calibrated stereo-video rig, as the *same named individual* rather than as two
population samples. Each dive folder is one animal
(`Hogfish01_MolHITW_0926_080323`) and the SMILE archive numbers the same fish
the same way, which is what makes the pairing possible at all.

**The estimator matters more than the statistics here, and getting it wrong
inverts the result.** Stage 14 back-projects head and tail at a single
laser-derived depth, so an out-of-plane fish can only read SHORT — verified on
this data: every stored length is <= the flat in-plane span its clicked pixels
and range imply (`geometry_consistency`). With a one-sided error, the median
over frames of one fish is biased downward by however much the pose varied, and
the MAXIMUM is a lower bound on the animal's true length.

    per-fish median -> median difference  -3.5 %
    per-fish max    -> median difference  +8.7 %

Both are computed here because the flip is the finding. The same reasoning is
why section 4.3 reports p90 rather than a mean.

What the comparison can and cannot say is in FINDINGS section 12; the short
version is that four of the seven fish have a *maximum* exceeding the stereo by
8.7-12.4 %, which our own pose loss cannot explain, and that we cannot separate
"we read long" from "the stereo reads short" from "the pairing is wrong on
those individuals" with seven pairs and no third instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

#: The eight 2023-08-03 morning dives. Seven have a stereo counterpart; the dog
#: snapper (dive 8) has none, so it carries a length but no pair.
MORNING_DIVES = (5, 8, 16, 20, 25, 28, 35, 39)

#: Every measurement resolves through this one fit — dive 32's `H Slate Dive 1`
#: at 09:11, 10.551 cm — borrowed by all eight dives. A calibration scale error
#: is therefore COMMON-MODE: it can move the median difference and cannot
#: produce disagreement *between* individuals.
SHARED_EXTRINSICS_ID = 56
SHARED_BASELINE_CM = 10.551

#: Camera 1 (FSL-01), `cameraintrinsics` id 3. Used only by
#: `geometry_consistency`.
FX_PX = 2832.6628438838216


@dataclass(frozen=True)
class Pair:
    """One individual measured by both instruments."""

    dive_id: int
    species: str
    n_frames: int
    ours_median_mm: float
    ours_max_mm: float
    stereo_mm: float

    @property
    def median_diff_pct(self) -> float:
        return 100.0 * (self.ours_median_mm - self.stereo_mm) / self.stereo_mm

    @property
    def max_diff_pct(self) -> float:
        """Difference using our per-fish maximum.

        The honest comparison when the error is one-sided: our maximum is a
        lower bound on the animal, so a positive value here is a disagreement
        our own pose loss cannot account for.
        """
        return 100.0 * (self.ours_max_mm - self.stereo_mm) / self.stereo_mm


def load_ours(path):
    """`data/stereo_pairs.csv`, as extracted by `sql/extract_stereo_pairs.sql`."""
    import pandas as pd

    return pd.read_csv(path, sep="|")


def load_stereo(path):
    """`data/stereo_reference.csv` — the stereo side, lifted from the
    collaborators' `SMILE_Archive_LengthData.csv`, which is not in this repo."""
    import pandas as pd

    return pd.read_csv(path, sep="|")


def build_pairs(ours, stereo) -> list[Pair]:
    """One `Pair` per individual that both instruments measured."""
    by_dive = {int(r.dive_id): r for r in stereo.itertuples()}
    pairs: list[Pair] = []
    for dive_id, group in ours.groupby("dive_id"):
        ref = by_dive.get(int(dive_id))
        if ref is None or not np.isfinite(getattr(ref, "stereo_length_mm", np.nan)):
            continue
        mm = 1000.0 * group["length_m"].to_numpy(float)
        pairs.append(
            Pair(
                dive_id=int(dive_id),
                species=str(group["species"].iloc[0]),
                n_frames=len(mm),
                ours_median_mm=float(np.median(mm)),
                ours_max_mm=float(mm.max()),
                stereo_mm=float(ref.stereo_length_mm),
            )
        )
    return sorted(pairs, key=lambda p: p.dive_id)


def summary(pairs: list[Pair], estimator: Literal["median", "max"] = "median") -> dict:
    """Median / mean / sd of the paired differences under one estimator."""
    a = np.array(
        [p.median_diff_pct if estimator == "median" else p.max_diff_pct for p in pairs]
    )
    se = a.std(ddof=1) / np.sqrt(a.size)
    return {
        "n": int(a.size),
        "median_pct": float(np.median(a)),
        "mean_pct": float(a.mean()),
        "sd_pct": float(a.std(ddof=1)),
        "mean_abs_pct": float(np.abs(a).mean()),
        "ci95": (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se)),
        "within_10_pct": int((np.abs(a) < 10).sum()),
    }


def over_measured(pairs: list[Pair]) -> list[Pair]:
    """Individuals whose MAXIMUM exceeds the stereo length.

    These are the ones that cannot be explained away by our own foreshortening,
    because pose only ever subtracts length.
    """
    return [p for p in pairs if p.ours_max_mm > p.stereo_mm]


def range_trend_pct_per_m(ours, dive_id: int) -> tuple[float, float, int]:
    """Slope of length against range for one individual, in % per metre.

    The scale-free calibration check: a rigid object must read the same length
    at every distance, so a non-zero slope indicts the calibration rather than
    the animal. Returns `(slope, pearson_r, n)`; needs frames carrying a
    `range_m`, which the hourly laser-depth stage supplies.
    """
    g = ours[(ours.dive_id == dive_id) & ours.range_m.notna()]
    if len(g) < 3:
        return (float("nan"), float("nan"), len(g))
    length = 1000.0 * g["length_m"].to_numpy(float)
    rng = g["range_m"].to_numpy(float)
    slope = np.polyfit(rng, length, 1)[0]
    return (
        float(100.0 * slope / np.median(length)),
        float(np.corrcoef(rng, length)[0, 1]),
        len(g),
    )


def geometry_consistency(ours) -> "object":
    """`length / (head_tail_px * range / fx)` per frame.

    1.0 means the stored length is exactly the flat in-plane span the clicked
    pixels subtend at that range; below 1.0 is the out-of-plane correction
    removing length, which is the only direction it can move. Anything above
    1.0 would mean the projection is not doing what we think.
    """
    g = ours[ours.range_m.notna() & ours.head_tail_px.notna()].copy()
    implied_mm = 1000.0 * g["head_tail_px"] * g["range_m"] / FX_PX
    g["implied_mm"] = implied_mm
    g["ratio"] = (1000.0 * g["length_m"]) / implied_mm
    return g
