"""Figure 10: no camera unit carries a bias of its own.

Run from the repository root:

    uv run python fish_model_analysis/rig_bias.py

Writes `fish_model_analysis/figures/fig10_no_rig_bias.{pdf,png}` and prints the
one-way test the caption quotes.

WHAT IS PLOTTED, AND WHY IT IS NOT THE OBVIOUS THING
----------------------------------------------------
Not the per-camera percent error. Cameras photographed different target sets --
the 2023 sessions carry the fish models, the 2025 sessions the box and the trout
-- so a raw per-camera mean is confounded with which targets that camera
happened to see, and the spread it shows is mostly the target ladder.

What is plotted is the **session effect of the median polish**: the calibration
offset with the per-target term already removed. That is the quantity a rig bias
would live in, and it is comparable across cameras that saw different targets.

THE ARGUMENT
------------
A camera with a bias of its own would put its sessions together and away from
the rest. Two things say none does:

  * the scatter *within* one camera is as large as the scatter *between*
    cameras, so a one-way fit puts the between-camera variance component at
    zero and F below 1 -- the units differ less than chance would predict;
  * the sessions that fail the checks of the accuracy rule are spread across
    six of the seven cameras rather than clustered on one, which is what a bad
    unit rather than a bad session would look like.

Six of the seven cameras also span both the 2023 and 2025 campaigns, so camera
is crossed with era rather than nested inside it, and this comparison is not
secretly a comparison of the two eras.

CAVEAT WORTH CARRYING
---------------------
`camera_id` is the database's camera row, and the post-labeling handoff records
that it does **not** track the physical rig number for the 2025 sessions. If two
ids are one rig, or one id is two rigs, the grouping is wrong and this test is
weaker than it looks -- though note that mis-grouping would tend to *hide* a
real bias by mixing rigs, not manufacture the null seen here.
"""

import numpy as np
from pathlib import Path
from scipy import stats

from fishsense_imwut import calibration as cal
from fishsense_imwut import pubfig

CORPUS = Path(__file__).resolve().parent / "data" / "corpus.csv"
OUTDIR = Path(__file__).resolve().parent / "figures"


def per_camera_offsets():
    """camera id -> [(session id, calibration offset pp, in cohort), ...]."""
    rows = cal.load_rows(CORPUS)
    df = cal.to_frame(rows)
    dive_to_camera = {int(r["dive_id"]): int(r["camera_id"]) for r in rows}
    effects = cal.median_polish(cal.cell_p90_grid(df)).dive_effect
    cohort = set(cal.accuracy_cohort(df))

    out: dict[int, list] = {}
    for dive in effects.index:
        d = int(dive)
        out.setdefault(dive_to_camera[d], []).append(
            (d, float(effects[dive]), d in cohort)
        )
    for cam in out:
        out[cam].sort(key=lambda t: t[0])
    return out


def one_way(per_camera):
    """Between- vs within-camera spread of the session offsets."""
    groups = [[v for _, v, _ in per_camera[c]] for c in sorted(per_camera)]
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = np.mean([x for g in groups for x in g])

    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)

    f = ms_between / ms_within
    p = float(1 - stats.f.cdf(f, k - 1, n - k))
    # Unbalanced groups, so the variance component uses the effective n.
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    between_var = max(0.0, (ms_between - ms_within) / n0)
    return dict(k=k, n=n, f=f, p=p, df=(k - 1, n - k),
                within_sd=float(np.sqrt(ms_within)),
                between_sd=float(np.sqrt(between_var)))


def per_camera_frames():
    """camera id -> per-frame percent error, cohort only, target offset removed.

    Removing each target's own median is what makes the units comparable: unit 1
    saw only the 2023 fish models and units 2/3/4/10 only the box and the trout,
    so the raw per-unit medians span 3.2 points of target ladder before any
    property of a unit is involved. Centring each target drops that to 1.7.
    """
    rows = cal.load_rows(CORPUS)
    df = cal.to_frame(rows)
    df["camera_id"] = [int(r["camera_id"]) for r in rows]
    s = df[df.dive_id.isin(cal.accuracy_cohort(df))].copy()
    s["e"] = s.pct_error - s.groupby("model_name").pct_error.transform("median")
    return {int(c): g.e.values for c, g in s.groupby("camera_id")}


def main() -> None:
    pubfig.use_publication_style()
    per_camera = per_camera_offsets()
    stat = one_way(per_camera)

    print(f"{stat['k']} cameras, {stat['n']} sessions")
    for cam in sorted(per_camera):
        vals = [v for _, v, _ in per_camera[cam]]
        eras = {"2023" if d < 200 else "2025" for d, _, _ in per_camera[cam]}
        print(f"  cam {cam:2d}: n={len(vals)}  median {np.median(vals):+6.2f} pp"
              f"   eras {'+'.join(sorted(eras))}")
    print(f"  between-camera sd {stat['between_sd']:.2f} pp "
          f"(variance component, clipped at zero)")
    print(f"  within-camera  sd {stat['within_sd']:.2f} pp")
    print(f"  F{stat['df']} = {stat['f']:.2f}, p = {stat['p']:.3f}")

    fig = pubfig.fig_no_rig_bias(
        per_camera,
        within_sd=stat["within_sd"],
        f_stat=(stat["f"], stat["df"][0], stat["df"][1]),
        p_value=stat["p"],
    )
    for path in pubfig.save_figure(fig, "fig10_no_rig_bias", outdir=OUTDIR):
        print("wrote", path)

    # Companion view: the frame-level distributions behind those session points.
    frames = per_camera_frames()
    meds = {c: float(np.median(v)) for c, v in frames.items()}
    span = max(meds.values()) - min(meds.values())
    print(f"\nper-frame, target-centred: unit medians span {span:.2f} points")
    fig2 = pubfig.fig_error_by_camera(
        frames, span_note=f"unit medians span {span:.1f} points", ylim=(-15, 10)
    )
    for path in pubfig.save_figure(fig2, "fig10b_error_by_camera", outdir=OUTDIR):
        print("wrote", path)


if __name__ == "__main__":
    main()
