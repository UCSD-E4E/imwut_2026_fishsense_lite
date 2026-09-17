"""Publication figures for the fish-model measurement validation set.

The figures here are built for a static two-column ACM paper, so two things the
interactive-chart guidance asks for are deliberately absent: there is no dark
mode and no hover layer. Everything else -- the validated palette, thin marks,
hairline chrome, selective direct labels -- carries over unchanged.

Palette: slots 1 (blue) and 2 (orange) of the reference categorical palette.
Validated all-pairs against a white paper surface: CVD dE 24.7, normal-vision
dE 33.6, both slots >= 3:1 contrast. Never add a third hue here without
re-running the validator -- these figures are scatter/all-pairs forms, which cap
at three slots.

Figures carry no title by default: the caption is the title in LaTeX.
"""

import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette -------------------------------------------------------------
# Categorical slots (identity). Do not reorder -- the order is the CVD-safety
# mechanism, not cosmetics.
SERIES_1 = "#2a78d6"  # blue   -- per-frame measurements
SERIES_2 = "#eb6834"  # orange -- the summary/estimator drawn over them

# Chart chrome & ink. Print surface is paper white, not the screen surface.
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Column widths for the ACM `acmart` two-column layout, in inches.
COL_WIDTH = 3.33
FULL_WIDTH = 7.00

REQUIRED_COLUMNS = (
    "dive_id",
    "model_name",
    "known_length_m",
    "length_m",
    "pct_error",
)


def use_publication_style() -> None:
    """Set rcParams for print figures. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Vector text stays text, so the PDF is searchable and reflows
            # cleanly at any zoom.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            # Hairline, recessive chrome: only the left/bottom rules survive.
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
        }
    )


def _grid(ax: plt.Axes, axis: str = "y") -> None:
    """Hairline grid, drawn under the marks."""
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _zero_line(ax: plt.Axes, orientation: str = "h") -> None:
    """The 0 % reference. Solid, one shade darker than the grid -- it is a real
    datum, so it must not read as another gridline."""
    if orientation == "h":
        ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    else:
        ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)


def _r2_annotation(ax: plt.Axes, r2: float | None, basis: str = "per frame") -> None:
    """Stamp R^2-about-1:1 in the corner the 1:1 line leaves empty.

    Lower right on both agreement figures: the legend holds the upper left and
    the diagonal runs between them, so this is the one corner no mark reaches.
    Text ink, not a series colour -- it is a label, not a third series.

    `basis` names which marks the number was computed over, because both
    figures draw two: a per-frame cloud and a per-target estimator. Without it
    the reader cannot tell a statement about single frames from a statement
    about the reported estimate, and the two differ by a lot.
    """
    if r2 is None:
        return
    ax.text(
        0.97,
        0.04,
        f"$R^2 = {r2:.3f}$ about 1:1\n({basis})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        linespacing=1.3,
        color=INK_SECONDARY,
        zorder=5,
    )


def nearest_rank_p90(values: Sequence[float]) -> float:
    """p90 by nearest rank -- ceil(0.9n) -- matching the `fish_length_estimate`
    view exactly, so a number quoted from a figure equals the number the
    pipeline reports.

    p90 rather than the mean because per-frame error is one-sided negative:
    stage 14 back-projects head and tail at a single laser-derived depth, so it
    measures the projection, and an out-of-plane fish can only read short. A
    high quantile rejects that tail; a mean inherits it.
    """
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n == 0:
        return float("nan")
    return float(arr[int(np.ceil(0.9 * n)) - 1])


def r2_about_identity(reference: Sequence[float], measured: Sequence[float]) -> float:
    """Fraction of the variance in `reference` that `measured` reproduces *on
    the 1:1 line*: 1 - sum((y - x)^2) / sum((x - mean(x))^2).

    The denominator is the **reference's** variance, not the measurement's,
    which makes the null model "a flat guess at the mean reference length" --
    the only baseline that means anything. Dividing by the measurement's
    variance instead is a different number (0.844 rather than 0.807 on the
    stereo pairs) whose null model is "the reference equals the mean of our own
    readings", which is not a baseline at all. A draft of section 4.3 quoted
    that number before this function existed.

    This is the statistic an agreement figure needs, and it is not the one
    `scipy.stats.linregress` reports. Pearson r^2 is scale- and offset-free, so
    a set of measurements that were all 20 % short, or all 3 cm long, would
    still score near 1.0 -- against eight targets that differ in size by a
    factor of four, r^2 mostly certifies that the targets have different
    lengths. R^2 about identity has no free parameters to absorb a bias, so it
    falls whenever the cloud sits off the 1:1 datum the figure draws.

    It can go negative, which is the honest behaviour: a negative value says
    the measurements are further from the 1:1 line than a flat guess at the
    mean reference length would have been.
    """
    x = np.asarray(reference, dtype=float)
    y = np.asarray(measured, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"reference and measured differ in shape: {x.shape} vs {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return float("nan")
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator == 0.0:
        # Every reference identical: there is no variance on x to explain, so
        # the ratio is undefined rather than perfect.
        return float("nan")
    return float(1.0 - np.sum((y - x) ** 2) / denominator)


# --- data ----------------------------------------------------------------


def load_measurements(path: Path | str) -> pd.DataFrame:
    """Load the `fish_model_measurement_accuracy` export and fail loudly on a
    schema that is missing anything the figures need."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Expected at least {', '.join(REQUIRED_COLUMNS)} -- see "
            f"export_fish_model_measurements.sql."
        )
    return df


def _model_order(df: pd.DataFrame) -> list[str]:
    """Models ordered by known length. The categories are ordinal here (size),
    so the reader should see that order on the axis."""
    return (
        df.groupby("model_name")["known_length_m"]
        .first()
        .sort_values()
        .index.tolist()
    )


def _annotate_counts(
    ax: plt.Axes, groups: Iterable[Sequence[float]], positions: Sequence[float]
) -> None:
    """n= in the right margin, in muted ink -- the honest caveat on every box.

    Placed OUTSIDE the axes in blended coordinates (x in axes fraction, y in
    data): inside the axes it collides with whisker caps and outlier fliers on
    exactly the widest, most interesting groups.
    """
    transform = ax.get_yaxis_transform()  # x: axes fraction, y: data
    for pos, values in zip(positions, groups):
        ax.text(
            1.02,
            pos,
            f"n={len(values)}",
            transform=transform,
            va="center",
            ha="left",
            fontsize=6,
            color=INK_MUTED,
            clip_on=False,
        )


def _style_box(bp: dict, color: str) -> None:
    """Thin marks, hollow boxes -- saturated fill is for small marks, never
    large blocks."""
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor=color, linewidth=0.9)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set(color=color, linewidth=0.8)
    for median in bp["medians"]:
        median.set(color=color, linewidth=1.6)
    for flier in bp.get("fliers", []):
        flier.set(
            marker="o",
            markersize=1.8,
            markerfacecolor=color,
            markeredgecolor="none",
            alpha=0.35,
        )


def _clip_x(ax: plt.Axes, groups, xlim, positions=None) -> None:
    """Narrow the x-axis, and make what that hides VISIBLE, not just counted.

    An earlier version set the limit and wrote "N frame(s) beyond axis" below
    the panel. That is not enough and it was rightly called disingenuous: the
    count says how MANY but not how FAR, and "12 beyond axis" reads the same
    whether the worst frame is -12.1 % or -30.6 %. The tail is one-sided and it
    is the paper's own argument for reporting p90, so hiding its size understates
    exactly the thing under discussion.

    Now: every clipped frame is drawn as a caret pinned at the boundary, so the
    reader sees that they exist and roughly where, and the note carries the
    extreme value. Clipping stays because a -30 % tail would squeeze every box
    into a third of the width, but it no longer costs the reader the number.
    """
    if xlim is None:
        return
    lo, hi = xlim
    ax.set_xlim(lo, hi)

    hidden, worst, excursion = 0, None, -1.0
    for i, values in enumerate(groups):
        v = np.asarray(values, dtype=float)
        out = v[(v < lo) | (v > hi)]
        if not out.size:
            continue
        hidden += out.size
        # the worst is the frame that overshoots ITS OWN boundary furthest;
        # comparing |value| or mixing the two sides gets it wrong, and an
        # earlier version reported -19.9 % on a tail that reaches -30.6 %
        for side, candidates in ((lo, out[out < lo]), (hi, out[out > hi])):
            if candidates.size:
                pick = candidates.min() if side == lo else candidates.max()
                if abs(pick - side) > excursion:
                    excursion, worst = abs(pick - side), float(pick)
        if positions is not None:
            below, above = out[out < lo], out[out > hi]
            for side, count, marker in ((lo, below.size, "<"), (hi, above.size, ">")):
                if count:
                    ax.plot([side], [positions[i]], marker=marker, markersize=4.5,
                            color=SERIES_1, markeredgecolor=SURFACE,
                            markeredgewidth=0.8, clip_on=False, zorder=6)
    if hidden:
        note = f"{hidden} frame(s) beyond axis, worst {worst:+.1f} %"
        ax.annotate(
            note,
            xy=(1.0, 0.0),
            xycoords="axes fraction",
            xytext=(0, -40),
            textcoords="offset points",
            fontsize=6,
            color=INK_SECONDARY,
            ha="right",
            va="top",
        )


def fig_measured_vs_known(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (COL_WIDTH, 2.7),
    jitter: float = 0.0025,
    seed: int = 0,
    r2: float | None = None,
    r2_basis: str = "per-model $p_{90}$",
) -> plt.Figure:
    """Per-frame measured length against the model's known length, with the
    p90 estimator overlaid and a 1:1 reference.

    The eight models are already separated along x by their known lengths, so
    identity needs no colour -- which is what keeps this inside the three-slot
    all-pairs cap. Blue is the per-frame cloud, orange the estimator drawn over
    it; that is the whole legend.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")

    rng = np.random.default_rng(seed)
    x = df["known_length_m"].to_numpy(dtype=float)
    y = df["length_m"].to_numpy(dtype=float)
    # Known lengths are eight discrete values; without jitter the cloud
    # collapses into eight opaque vertical rules and the density is unreadable.
    x_jittered = x + rng.uniform(-jitter, jitter, size=x.size)

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.05 * (hi - lo)
    span = np.array([lo - pad, hi + pad])

    # Dashed because it is a reference datum, not chrome -- the one place
    # dashing carries meaning.
    ax.plot(
        span,
        span,
        color=INK_MUTED,
        linewidth=0.8,
        linestyle=(0, (4, 3)),
        zorder=2,
        label="1:1 (exact)",
    )
    ax.scatter(
        x_jittered,
        y,
        s=5,
        color=SERIES_1,
        alpha=0.28,
        linewidths=0,
        zorder=3,
        label="Per-frame measurement",
    )

    p90 = (
        df.groupby("model_name")
        .agg(
            known_length_m=("known_length_m", "first"),
            length_p90_m=("length_m", nearest_rank_p90),
        )
        .sort_values("known_length_m")
    )
    ax.scatter(
        p90["known_length_m"],
        p90["length_p90_m"],
        s=26,
        marker="D",
        color=SERIES_2,
        # A 2px surface ring, not a border -- this mark sits on top of the cloud.
        edgecolors=SURFACE,
        linewidths=1.0,
        zorder=4,
        label="Per-model $p_{90}$ estimate",
    )

    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Known fork length (m)")
    ax.set_ylabel("Measured length (m)")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    # Passed in, never computed here: the notebook owns every number the paper
    # quotes, so the annotation and the text cannot drift apart.
    _r2_annotation(ax, r2, r2_basis)
    return fig


# --- figure 2: error distribution by model -------------------------------


def fig_error_by_model(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (COL_WIDTH, 2.9),
    show_points: bool = True,
    seed: int = 0,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Percent error per model, ordered by known length.

    Horizontal because the model names are long. One hue for every box: the
    categories are identity, and colouring them by their own value would spend
    the identity channel re-encoding what box position already shows.

    The raw strip behind each box is the point of the figure -- it is where the
    one-sided negative tail is visible, which a box alone flattens.
    """
    order = _model_order(df)
    groups = [df.loc[df["model_name"] == m, "pct_error"].to_numpy() for m in order]
    positions = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")
    _zero_line(ax, orientation="v")

    if show_points:
        rng = np.random.default_rng(seed)
        for pos, values in zip(positions, groups):
            ax.scatter(
                values,
                pos + rng.uniform(-0.16, 0.16, size=values.size),
                s=3,
                color=SERIES_1,
                alpha=0.20,
                linewidths=0,
                zorder=2,
            )

    bp = ax.boxplot(
        groups,
        positions=positions,
        orientation="horizontal",
        widths=0.55,
        patch_artist=True,
        showfliers=not show_points,
        zorder=3,
    )
    _style_box(bp, SERIES_1)

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{m}\n{df.loc[df['model_name'] == m, 'known_length_m'].iloc[0]*100:.1f} cm"
         for m in order]
    )
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Length error (%)")
    ax.invert_yaxis()  # smallest model at the top, reading order
    _clip_x(ax, groups, xlim, positions)
    _annotate_counts(ax, groups, positions)
    return fig


# --- figure 3: error vs. range -------------------------------------------


def fig_error_vs_depth(
    df: pd.DataFrame,
    depth_column: str = "depth_m",
    figsize: tuple[float, float] = (COL_WIDTH, 2.5),
    n_bins: int = 8,
    ylim: tuple[float, float] | None = (-15, 10),
) -> plt.Figure:
    """Percent error against the laser-derived distance to the target.

    The binned median and IQR band are what carry the trend; the raw cloud is
    context behind them. Triangulation conditioning goes as Z^2, so any
    range-dependence in the real measurements should show here as a widening
    band rather than a drifting median.
    """
    if depth_column not in df.columns:
        raise ValueError(
            f"No '{depth_column}' column. Re-export with the laserdepth join "
            f"(see export_fish_model_measurements.sql) or pass "
            f"depth_column='range_m'."
        )

    sub = df[[depth_column, "pct_error"]].dropna()
    if sub.empty:
        raise ValueError(f"Every '{depth_column}' value is null -- nothing to plot.")

    x = sub[depth_column].to_numpy(dtype=float)
    y = sub["pct_error"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")
    _zero_line(ax, orientation="h")

    ax.scatter(
        x, y, s=5, color=SERIES_1, alpha=0.25, linewidths=0, zorder=2,
        label="Per-frame measurement",
    )

    # Equal-count bins, not equal-width: the range distribution is heavily
    # skewed toward short distances, so equal-width bins would put almost every
    # frame in the first bin and leave the far bins on one or two points.
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    centres, medians, q1s, q3s = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if mask.sum() < 3:
            continue
        centres.append(float(np.median(x[mask])))
        medians.append(float(np.median(y[mask])))
        q1s.append(float(np.quantile(y[mask], 0.25)))
        q3s.append(float(np.quantile(y[mask], 0.75)))

    if centres:
        ax.fill_between(
            centres, q1s, q3s, color=SERIES_2, alpha=0.16, linewidth=0, zorder=3,
            label="Binned IQR",
        )
        ax.plot(
            centres, medians, color=SERIES_2, linewidth=1.6, zorder=4,
            marker="o", markersize=3, markeredgecolor=SURFACE, markeredgewidth=0.8,
            label="Binned median",
        )

    ax.set_xlabel(
        "Distance to laser dot (m)"
        if depth_column == "range_m"
        else "Laser depth, optical axis (m)"
    )
    ax.set_ylabel("Length error (%)")
    # Above the axes, not in a corner: at this aspect every corner of this plot
    # holds data, so an inside legend covers the cloud it is describing.
    # A handful of gross outliers (species mislabels, see the notebook) otherwise
    # compress the entire distribution into the top tenth of the panel. Clipped
    # by default, and the count of what that hides is printed.
    if ylim is not None:
        hidden = int(((y < ylim[0]) | (y > ylim[1])).sum())
        ax.set_ylim(*ylim)
        if hidden:
            ax.annotate(
                f"{hidden} frame(s) beyond axis",
                xy=(1.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -40),
                textcoords="offset points",
                fontsize=6,
                color=INK_MUTED,
                va="top",
                ha="right",
            )

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncols=3,
        handletextpad=0.4,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    return fig


# --- figure 4: per-dive calibration scale --------------------------------


def fig_error_by_dive(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (COL_WIDTH, 3.2),
    min_frames: int = 8,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Percent error grouped by dive, ordered by median.

    Per-dive calibration scale is bidirectional and invisible to reprojection
    (which pins 2 of 4 DOF and is blind to scale), so a dive-ordered spread
    straddling zero is the signature to look for -- as distinct from the
    one-sided negative foreshortening tail within each dive.

    `min_frames` drops dives too small for a box to mean anything; the count is
    reported on the figure rather than silently trimmed.
    """
    counts = df.groupby("dive_id")["pct_error"].size()
    keep = counts[counts >= min_frames].index
    dropped_dives = int(len(counts) - len(keep))
    dropped_frames = int(counts[counts < min_frames].sum())
    sub = df[df["dive_id"].isin(keep)]

    order = (
        sub.groupby("dive_id")["pct_error"].median().sort_values().index.tolist()
    )
    groups = [sub.loc[sub["dive_id"] == d, "pct_error"].to_numpy() for d in order]
    positions = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")
    _zero_line(ax, orientation="v")

    bp = ax.boxplot(
        groups,
        positions=positions,
        orientation="horizontal",
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        zorder=3,
    )
    _style_box(bp, SERIES_1)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"Dive {d}" for d in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Length error (%)")
    ax.invert_yaxis()
    _clip_x(ax, groups, xlim, positions)
    _annotate_counts(ax, groups, positions)

    if dropped_dives:
        # Never a silent cap -- say what was left out, on the figure.
        ax.set_title(
            f"{dropped_dives} dive(s) with <{min_frames} frames omitted "
            f"({dropped_frames} frames)",
            fontsize=6,
            color=INK_MUTED,
            loc="left",
            pad=4,
        )
    return fig


# --- output --------------------------------------------------------------


#: Title per figure, used ONLY on the previewable PNG -- see `save_figure`.
#: Keyed by the name passed to `save_figure`, so a notebook cell needs no change.
#:
#: **No figure numbers here, deliberately.** The numbering is the paper's to
#: choose and will not survive a reordering of the draft; a title that says
#: "Fig 15" on a file that becomes Figure 11 is worse than no title at all.
#: These name what the figure SHOWS, which does not change when it moves.
FIGURE_TITLES = {
    "fig1_measured_vs_known": "Measured length against known length, pool cohort",
    "fig2_error_by_model": "Percent length error by target",
    "fig3_error_vs_range": "Percent length error against laser range",
    "fig4_phi_mount_state": "Fitted in-plane laser angle, seven sessions of one unit",
    "fig5_yaw_floor_repair": "Implied yaw floor before and after the August repair",
    "fig6_shark_anomaly": "The shark model's disputed reference length",
    "fig7_fork_probe": "Tail-landmark probe: implied shift against apparent length",
    "fig8_error_vs_angle": "Percent length error against fish angle to the image plane",
    "fig9_flat_port_cost": "Uncorrected flat-port cost across the frame",
    "fig9b_flat_port_error_field":
        "Flat-port cost on the frame, with and without the corrective optic",
    "fig10_no_rig_bias": "Session calibration offset by camera unit",
    "fig10b_error_by_camera": "Percent length error by camera unit (repository only)",
    "fig11_field_repeatability":
        "Within-individual repeatability, wild fish against posed models",
    "fig12_field_species": "Measured fork length by species, seven deployments",
    "fig13_field_by_camera": "Hogfish fork length by camera unit",
    "fig14_field_vs_stereo":
        "FishSense Lite against underwater stereo video, per-species medians",
    "fig15_paired_vs_stereo":
        "FishSense Lite against underwater stereo video, the same individuals",
    "fig16_p90_vs_sample_size":
        "Convergence of the $p_{90}$ estimate with frames per fish",
    "figA_all_dives_percent": "Percent length error for every session",
    "figD1_depth_correction": "The range correction, and why it is not applied",
    "figD2_p90_budget":
        "Reported length error against frames per fish, with the error budget marked",
}


def save_figure(
    fig: plt.Figure,
    name: str,
    outdir: Path | str = "figures",
    synthetic: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    title: str | None = None,
    title_in_pdf: bool = True,
) -> list[Path]:
    """Write a figure as vector PDF (for LaTeX) and PNG (for previewing).

    **Both formats carry the title.** A figure opened on its own -- in a file
    browser, a review thread, an IDE tab, a slide -- has no caption, and twenty
    of these are not distinguishable by their axes.

    This was PNG-only at first, on the argument that `acmart` makes the caption
    the title and a title inside the figure duplicates it. That argument holds
    only for the one place a figure is typeset, and it cost the author a title
    in every other place these files are read. So it is reversed, and the
    duplication is dealt with where it occurs: pass `title_in_pdf=False` for a
    figure going straight into the paper with a LaTeX caption, or leave the
    caption to carry the number and the detail while the stamped title carries
    the name. `title` overrides the `FIGURE_TITLES` entry for `name`.

    `synthetic=True` stamps the figure so a placeholder can never be mistaken
    for a result. Refuses to write an unstamped file from synthetic data.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC DATA\nNOT FOR PUBLICATION",
            transform=fig.transFigure,
            ha="center",
            va="center",
            fontsize=13,
            color="#d03b3b",
            alpha=0.30,
            weight="bold",
            rotation=24,
            zorder=100,
        )
        name = f"SYNTHETIC_{name}"

    label = title if title is not None else FIGURE_TITLES.get(
        name[len("SYNTHETIC_"):] if name.startswith("SYNTHETIC_") else name)

    written = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        stamp = label is not None and (fmt != "pdf" or title_in_pdf)
        art = None
        if stamp:
            # wrap to the panel's own width: a 68-character title is 3.5 in at
            # 7.5 pt and a column figure is 3.33, so an unwrapped one would run
            # off the crop on exactly the figures that need it most
            size = 7.5
            per_line = max(24, int(fig.get_size_inches()[0] * 72 / (0.52 * size)))
            text = "\n".join(textwrap.wrap(label, per_line)) if len(label) > per_line else label
            # ABOVE the figure rectangle (y=1, anchored by its bottom), not
            # inside it: `savefig.bbox = "tight"` grows the saved image to
            # include the artist, so the title gains its own band instead of
            # landing on an axes that reaches the top of the panel
            art = fig.text(0.5, 1.0, text, ha="center", va="bottom",
                           fontsize=size, color=INK_SECONDARY)
        # no bbox_inches/pad_inches here: use_publication_style already sets
        # them, and overriding would silently re-crop every typeset PDF
        fig.savefig(path, dpi=300 if fmt == "png" else None)
        if art is not None:
            art.remove()
        written.append(path)
    return written


# --- calibration figures --------------------------------------------------
#
# Added after the 2026-08-26 calibration handoff. Two conventions from it are
# enforced here rather than left to the caller:
#
#   * per-dive error is reported as an ANGLE, never a percentage -- a -8 % dive
#     is not eight times worse than a -1 % dive, it is a mount 0.29 deg off
#     instead of 0.03 deg, and percentages are confounded by shooting distance;
#   * a dive's frames aggregate at p90, never a mean.

# De-emphasis ink for the emphasis form (one series accented, the rest recessive).
DEEMPHASIS = "#b9c6d6"


def _p90_marker(ax, x, y, label=None, **kw):
    kw.setdefault("marker", "D")
    kw.setdefault("s", 22)
    kw.setdefault("color", SERIES_2)
    kw.setdefault("edgecolors", SURFACE)
    kw.setdefault("linewidths", 0.9)
    kw.setdefault("zorder", 5)
    return ax.scatter(x, y, label=label, **kw)


def fig_error_by_model_p90(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (COL_WIDTH, 2.4),
    xlim: tuple[float, float] | None = None,
    emphasise: str | None = None,
) -> plt.Figure:
    """Per-model error distribution with the p90 estimator marked.

    The box shows the frame-level spread; the orange diamond is the number the
    pipeline actually reports. The gap between the median and the p90 is the
    foreshortening tail, which is the reason the estimator is a high quantile
    and not a mean.

    **NOT CLIPPED, deliberately.** This defaulted to +-12 % with a footnote
    counting what fell outside, and that reads as choosing which frames to show
    -- fairly, since the footnote sized nothing and `showfliers` was off besides.
    Rendering it unclipped settled it: the boxes stay legible across the full
    -31 to +4 range, and the long negative tail on the two largest models is the
    figure's actual argument, so hiding it cost more than the width it bought.
    `xlim` remains for a caller that needs it; nothing in the paper uses it.

    `emphasise` recolours one model as the accent and greys the rest -- the
    emphasis form, for when a single model is the point (Shark).
    """
    order = _model_order(df)
    groups = [df.loc[df["model_name"] == m, "pct_error"].to_numpy() for m in order]
    positions = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")
    _zero_line(ax, orientation="v")

    for pos, model, values in zip(positions, order, groups):
        accent = emphasise is None or model == emphasise
        colour = SERIES_1 if accent else DEEMPHASIS
        bp = ax.boxplot(
            [values],
            positions=[pos],
            orientation="horizontal",
            widths=0.5,
            patch_artist=True,
            # ON. With them off the reader saw whiskers stopping at -3.8 % on the
            # Box whose worst frame is -11.2 %, and had no way to know: 41 of 995
            # frames sat past a whisker, four times the number the clip note
            # admitted to. The tail is the paper's own reason for reporting p90.
            showfliers=True,
            flierprops=dict(marker="o", markersize=2.2, markerfacecolor=colour,
                            markeredgecolor="none", alpha=0.55),
            zorder=3,
        )
        _style_box(bp, colour)
        _p90_marker(
            ax,
            [nearest_rank_p90(values)],
            [pos],
            color=SERIES_2 if accent else DEEMPHASIS,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            f"{m}\n{df.loc[df['model_name'] == m, 'known_length_m'].iloc[0]*100:.1f} cm"
            for m in order
        ]
    )
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Length error (%)")
    ax.invert_yaxis()
    _clip_x(ax, groups, xlim, positions)
    _annotate_counts(ax, groups, positions)

    handles = [
        plt.Line2D([], [], color=SERIES_1, linewidth=1.4, label="Frame spread (box)"),
        plt.Line2D(
            [], [], color=SERIES_2, marker="D", markersize=4.5, linestyle="none",
            markeredgecolor=SURFACE, label="$p_{90}$ (reported)",
        ),
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
        handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0,
    )
    return fig


def fig_yaw_floor_repair(
    panels: Sequence[tuple[str, dict, dict]],
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Implied-yaw floor per model, before and after a phi repair.

    `panels` is a sequence of `(title, before, after)`, each dict mapping model
    name to its floor in degrees.

    The physical floor is 0: you cannot present a rigid object better than
    side-on, so a sound calibration with at least one well-presented frame must
    reach ~0. A calibration error lifts **every** object's floor uniformly --
    which is exactly what a dumbbell across models makes visible, and what a
    pose problem could not produce.
    """
    if figsize is None:
        figsize = (FULL_WIDTH, 0.34 * sum(len(b) for _, b, _ in panels) + 1.1)

    fig, axes = plt.subplots(
        1, len(panels), figsize=figsize, squeeze=False, sharex=True
    )
    axes = axes[0]

    for ax, (title, before, after) in zip(axes, panels):
        models = sorted(before, key=lambda m: -before[m])
        positions = np.arange(len(models), dtype=float)
        b = np.array([before[m] for m in models])
        a = np.array([after[m] for m in models])

        _grid(ax, axis="x")
        ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.hlines(
            positions, b, a, color=INK_MUTED, linewidth=0.9, zorder=2,
        )
        # "Before" is drawn larger so that when a model was already at the
        # floor (before == after, e.g. dive 60's Shark) it still shows as a
        # ring behind the diamond instead of vanishing under it and reading as
        # missing data.
        ax.scatter(b, positions, s=38, color=SERIES_1, edgecolors=SURFACE,
                   linewidths=0.9, zorder=4, label="Borrowed calibration")
        ax.scatter(a, positions, s=20, marker="D", color=SERIES_2,
                   edgecolors=SURFACE, linewidths=0.9, zorder=5,
                   label="After $\\varphi$ repair")

        ax.set_yticks(positions)
        ax.set_yticklabels(models)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8, color=INK_PRIMARY, loc="left", pad=6)
        ax.set_xlabel("Implied out-of-plane yaw floor (deg)")

    # One explicit limit across all panels. `sharex` alone is not enough: each
    # panel's autoscale runs against its own data, so the widest panel's points
    # were being clipped off the right edge.
    x_max = max(max(max(b.values()), max(a.values())) for _, b, a in panels)
    for ax in axes:
        ax.set_xlim(-1.0, x_max * 1.08)

    axes[0].legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.10), ncols=2,
        handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig


def fig_phi_mount_state(
    phi_by_dive: dict[int, float],
    cohort: dict[int, str],
    borrow_map: dict[int, int] | None = None,
    figsize: tuple[float, float] = (COL_WIDTH, 2.4),
) -> plt.Figure:
    """Per-dive in-plane mount angle phi -- the honest form of "per-dive error".

    phi is the one degree of freedom that is both unknown and consequential: it
    is invisible to the laser dots (monocular scale ambiguity) yet sets metric
    scale. Plotting it instead of percent error removes the shooting-distance
    confound, and puts every dive on a scale where the physical zero means
    "mount unmoved".
    """
    order = sorted(phi_by_dive, key=lambda d: phi_by_dive[d])
    positions = np.arange(len(order), dtype=float)

    style = {
        "sound": (SERIES_1, "o", "Sound (left as-is)"),
        "repaired": (SERIES_2, "D", "Repaired"),
        "disputed": (INK_MUTED, "s", "Disputed (unresolved)"),
    }

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")
    ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1,)

    seen = set()
    for pos, dive in zip(positions, order):
        kind = cohort.get(dive, "sound")
        colour, marker, label = style[kind]
        ax.hlines(pos, 0.0, phi_by_dive[dive], color=BASELINE, linewidth=0.8, zorder=2)
        ax.scatter(
            [phi_by_dive[dive]], [pos], s=26, marker=marker, color=colour,
            edgecolors=SURFACE, linewidths=0.9, zorder=4,
            label=None if label in seen else label,
        )
        seen.add(label)

    ax.set_yticks(positions)
    if borrow_map:
        ax.set_yticklabels([f"{d} ← {borrow_map.get(d, '?')}" for d in order])
        ax.set_ylabel("Dive ← borrowed calibration")
    else:
        ax.set_yticklabels([f"Dive {d}" for d in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("In-plane mount rotation $\\varphi$ (deg)")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
        handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0,
    )
    return fig


def fig_fork_probe(
    sep_px: np.ndarray,
    shift_px: np.ndarray,
    figsize: tuple[float, float] = (COL_WIDTH, 2.4),
) -> plt.Figure:
    """Does Shark's over-read behave like a label bias or a scale error?

    Each point is a well-presented Shark frame: x is its apparent body length in
    pixels, y is how far the tail label would have to move for the frame to read
    the 605 mm reference, after that dive's calibration has been fitted from
    Grouper.

    The two hypotheses separate cleanly on this axis:

      * a **labeler's click bias** lives in the image plane, so it is a constant
        number of PIXELS regardless of how large the fish appears -> flat line;
      * a **short reference** is a pure scale error, so the shift is a constant
        FRACTION of the body -> line through the origin.

    Frames span a 4x range of apparent size, which is what makes the test work.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")

    x = np.linspace(0, sep_px.max() * 1.05, 50)
    frac = np.median(shift_px / sep_px)

    ax.plot(x, np.full_like(x, np.median(shift_px)), color=INK_MUTED, linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=2, label="Label bias (constant px)")
    ax.plot(x, frac * x, color=SERIES_2, linewidth=1.6, zorder=3,
            label=f"Short reference ({100*frac:.1f}% of body)")
    ax.scatter(sep_px, shift_px, s=16, color=SERIES_1, alpha=0.85, linewidths=0.8,
               edgecolors=SURFACE, zorder=4, label="Shark frame")

    ax.set_xlim(0, x.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Apparent body length (px)")
    ax.set_ylabel("Implied tail shift (px)")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3)
    return fig


# --- foreshortening experiment --------------------------------------------


def fig_error_vs_angle(
    ang: pd.DataFrame,
    bins: Sequence[float] = tuple(range(0, 50, 5)),
    budget_pct: float = 15.0,
    figsize: tuple[float, float] = (COL_WIDTH, 2.4),
) -> plt.Figure:
    """Percent error against the fish's angle to the image plane, from the
    designed experiment: one object, stepped through known angles.

    Each session (dive) is a thin line so the reader can see five independent
    repeats agree; the pooled median and IQR carry the result. cos(theta) - 1
    is the pure-projection prediction -- stage 14 measures the projection of
    the fish onto the image plane at the laser's depth -- so the gap between
    it and the pooled median is the broadside bias, not a pose effect.
    """
    from .calibration import binned_angle_error

    fig, ax = plt.subplots(figsize=figsize)
    for i, (dive, g) in enumerate(sorted(ang.groupby("dive_id"))):
        b = binned_angle_error(g, bins)
        ax.plot(
            b.index, b["median"], marker="o", ms=2.2, lw=0.8, alpha=0.7,
            color=f"C{i}", label=f"session {i + 1}", zorder=3,
        )
    pooled = binned_angle_error(ang, bins)
    ax.fill_between(
        pooled.index, pooled.q1, pooled.q3, color=INK_MUTED, alpha=0.18, lw=0,
        label="pooled IQR", zorder=2,
    )
    ax.plot(pooled.index, pooled["median"], color=INK_PRIMARY, lw=1.6,
            label="pooled median", zorder=4)
    theta = np.arange(0, max(bins) + 1)
    ax.plot(theta, 100 * (np.cos(np.radians(theta)) - 1), ls="--", lw=0.9,
            color=INK_SECONDARY, label=r"$\cos\theta - 1$", zorder=1)
    ax.axhline(-budget_pct, color="#d03b3b", lw=0.8, ls=":")
    ax.text(0.5, -budget_pct + 0.8, f"{budget_pct:.0f}% budget", fontsize=6,
            color="#d03b3b", va="bottom")
    ax.set_xlabel("fish angle to image plane (deg)")
    ax.set_ylabel("length error (%)")
    ax.set_xlim(-1, max(bins) + 1)
    ax.set_ylim(-40, 5)
    _grid(ax)
    ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


# --- flat-port refraction -------------------------------------------------
#
# The one refraction figure this paper carries: what it costs to ignore the
# water's refractive index entirely. The corrections -- Pinax, and the in-water
# single-viewpoint calibration it is compared against -- belong to the WUWNet
# paper and are deliberately absent from both `fishsense_imwut.refraction` and
# this figure. See that module's docstring for the scope cut.


def fig_flat_port_cost(
    field_angle_deg: Sequence[float],
    length_pct_error: Sequence[float],
    range_pct_error: float | None = None,
    budget_pct: float = 15.0,
    half_frame_m: float | None = None,
    depth_m: float | None = None,
    figsize: tuple[float, float] = (COL_WIDTH, 2.7),
) -> plt.Figure:
    """Length error against field position with no refraction correction.

    The shape is the point. An uncorrected flat port expands the scene
    transversely by the water index and shortens the laser range by its
    reciprocal; on the optical axis those cancel almost exactly, so a centred
    target measures correctly *by accident*. Off axis the angular compression is
    not a pure scale, the cancellation fails, and the error depends on nothing
    but where in the frame the target happened to fall -- which is why it cannot
    be averaged away and why it is easy to miss on axis.

    `range_pct_error` annotates the laser-range error behind the cancellation.

    **The x axis is WHERE IN THE PICTURE the target sits, not how it is posed.**
    Section 4.4's figure is also in degrees and means the opposite thing -- the
    fish's angle to the image plane -- and the two are independent: a fish held
    perfectly broadside in the corner of the frame has 0 deg of pose and 20 deg
    of frame position, and an uncorrected port would read it +17.7 % long. Pass
    `half_frame_m` and `depth_m` (both returned by `refraction.flat_port_cost`)
    to get the second axis that says this in picture terms, centre to edge.
    """
    x = np.asarray(field_angle_deg, dtype=float)
    y = np.asarray(length_pct_error, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")
    _zero_line(ax, orientation="h")

    ax.plot(x, y, color=SERIES_2, linewidth=1.8, zorder=4,
            label="No refraction correction")

    if budget_pct is not None:
        ax.axhline(budget_pct, color=INK_MUTED, linestyle=":", linewidth=1.0, zorder=3)
        crossing = np.interp(budget_pct, y, x) if y[-1] >= budget_pct else None
        ax.text(
            x[0] + 0.02 * (x[-1] - x[0]), budget_pct + 0.8,
            f"{budget_pct:g} % error budget", fontsize=6, color=INK_SECONDARY,
            va="bottom", ha="left",
        )
        if crossing is not None:
            ax.plot([crossing], [budget_pct], marker="o", markersize=3.5,
                    color=INK_PRIMARY, zorder=5)
            ax.annotate(
                f"crossed at {crossing:.0f}°",
                xy=(crossing, budget_pct), xytext=(-6, -13),
                textcoords="offset points", fontsize=6,
                color=INK_PRIMARY, ha="right", va="top",
            )

    if range_pct_error is not None:
        # in the clear block BELOW the curve on the right; the curve sweeps the
        # diagonal, so the two free corners are upper-left and lower-right and
        # the budget callouts already own the first
        ax.annotate(
            f"on axis the laser range reads {range_pct_error:+.0f} %,\n"
            "cancelling the transverse error exactly",
            xy=(0.40, 0.06), xycoords="axes fraction",
            fontsize=6, color=INK_SECONDARY, ha="left", va="bottom",
        )

    ax.set_xlabel("Target's distance from the image centre (degrees off axis)")
    ax.set_ylabel("Length error (%)")

    if half_frame_m is not None and depth_m is not None:
        edge_deg = float(np.degrees(np.arctan(half_frame_m / depth_m)))
        # show the whole half-frame, so the reader can see the plotted range
        # stops short of the edge -- and why
        ax.set_xlim(x.min(), edge_deg)
        ax.axvspan(x.max(), edge_deg, color=INK_MUTED, alpha=0.07, zorder=0)
        ax.annotate("a 30 cm target\nno longer fits",
                    xy=(0.5 * (x.max() + edge_deg), ax.get_ylim()[1]),
                    xytext=(0, -4), textcoords="offset points", fontsize=5.8,
                    color=INK_SECONDARY, ha="center", va="top", zorder=5)

        def _to_frac(deg):
            return depth_m * np.tan(np.radians(deg)) / half_frame_m

        def _to_deg(frac):
            return np.degrees(np.arctan(np.asarray(frac) * half_frame_m / depth_m))

        top = ax.secondary_xaxis("top", functions=(_to_frac, _to_deg))
        top.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        top.set_xticklabels(["centre", "\u00bc", "\u00bd", "\u00be", "edge"])
        top.set_xlabel("Position across the frame", labelpad=2)
        top.tick_params(length=2.5, width=0.6)
    else:
        ax.set_xlim(x.min(), x.max())

    ax.set_ylim(bottom=min(0.0, float(np.min(y))) - 0.5)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


# --- rig bias -------------------------------------------------------------


def fig_no_rig_bias(
    per_camera,
    within_sd: float,
    f_stat: float | None = None,
    p_value: float | None = None,
    figsize: tuple[float, float] = (COL_WIDTH, 2.6),
) -> plt.Figure:
    """Per-session calibration offset, grouped by camera.

    `per_camera` maps camera id -> list of (session id, offset_pp, in_cohort).

    The quantity plotted is the session effect of the median polish, not the raw
    per-camera error, and the distinction is load-bearing: cameras photographed
    different target sets, so a raw per-camera mean is confounded with which
    targets that camera happened to see. The polish removes the target term, so
    what is left is the calibration offset alone.

    The argument the figure makes is a comparison of two spreads. If a camera
    carried a bias, its sessions would sit together and away from the rest. They
    do not: the scatter within one camera is as large as the scatter between
    cameras, and the sessions that fail the checks of the accuracy rule (open
    markers) are spread across cameras rather than clustered on one.
    """
    cams = sorted(per_camera)
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="y")

    ax.axhspan(-within_sd, within_sd, color=GRIDLINE, alpha=0.55, zorder=0,
               label="±1 sd within a camera")
    _zero_line(ax, orientation="h")

    for i, cam in enumerate(cams):
        pts = per_camera[cam]
        n = len(pts)
        offsets = np.linspace(-0.17, 0.17, n) if n > 1 else np.array([0.0])
        for (dive, val, keep), dx in zip(pts, offsets):
            if keep:
                ax.plot(i + dx, val, marker="o", markersize=4.2, color=SERIES_1,
                        linestyle="none", zorder=4)
            else:
                ax.plot(i + dx, val, marker="o", markersize=4.2, markerfacecolor="none",
                        markeredgecolor=INK_MUTED, markeredgewidth=0.9,
                        linestyle="none", zorder=3)
        med = float(np.median([v for _, v, _ in pts]))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=SERIES_2, linewidth=1.9, zorder=5)

    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([str(c) for c in cams])
    ax.set_xlim(-0.5, len(cams) - 0.5)
    ax.set_xlabel("Camera unit")
    ax.set_ylabel("Session calibration offset (pp)")

    # Headroom for the legend and the footnote, so neither lands on a session.
    flat = [v for pts in per_camera.values() for _, v, _ in pts]
    lo, hi = min(flat), max(flat)
    ax.set_ylim(lo - 0.34 * (hi - lo), hi + 0.20 * (hi - lo))

    if f_stat is not None and p_value is not None:
        ax.annotate(
            f"$F({int(f_stat[1])},{int(f_stat[2])}) = {f_stat[0]:.2f}$,  "
            f"$p = {p_value:.2f}$",
            xy=(0.985, 0.975), xycoords="axes fraction", ha="right", va="top",
            fontsize=6.5, color=INK_SECONDARY,
        )

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=4.2, color=SERIES_1,
               label="in accuracy cohort"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor="none", markeredgecolor=INK_MUTED, label="rejected"),
        Line2D([], [], color=SERIES_2, linewidth=1.9, label="camera median"),
    ], fontsize=6, loc="lower left", ncol=1, handletextpad=0.5,
        borderpad=0.2, labelspacing=0.35, framealpha=0.0)
    fig.tight_layout()
    return fig


def fig_error_by_camera(
    per_camera,
    figsize: tuple[float, float] = (COL_WIDTH, 2.7),
    span_note: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Percent-error distribution per camera unit, as violins.

    `per_camera` maps camera id -> array of per-frame percent errors.

    **One hue, not seven.** The reader's question is whether these differ, and
    the answer is no; giving each unit its own colour would assert that unit
    identity carries meaning. Identity is on the axis, where it belongs.

    The KDE is clipped to each unit's observed range, so no violin shows density
    where no frame was measured -- the usual way a violin overstates a small
    sample. Bodies are hollow for the same reason the boxes elsewhere are:
    saturated fill is for small marks.

    The quantity should be percent error with each target's own offset removed
    (see `fish_model_analysis/rig_bias.py`). Units photographed different target
    sets, so raw per-unit error is confounded with the target ladder rather than
    with the unit.
    """
    from scipy.stats import gaussian_kde

    cams = sorted(per_camera)
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="y")
    _zero_line(ax, orientation="h")

    half = 0.38
    for i, cam in enumerate(cams):
        v = np.asarray(per_camera[cam], dtype=float)
        lo, hi = v.min(), v.max()
        grid = np.linspace(lo, hi, 200)
        # Silverman's rule on the ROBUST scale, min(sd, IQR/1.34), not on the sd
        # alone. scipy's default bandwidth is proportional to the sample sd, so a
        # couple of gross frames widen the kernel and smear the whole violin --
        # the bulk then looks different between units whose bulks are the same.
        q1v, q3v = np.percentile(v, [25, 75])
        scale = min(v.std(ddof=1), (q3v - q1v) / 1.349)
        bw = 0.9 * scale * v.size ** (-0.2)
        dens = gaussian_kde(v, bw_method=bw / v.std(ddof=1))(grid)
        dens = dens / dens.max() * half

        ax.fill_betweenx(grid, i - dens, i + dens, facecolor=SERIES_1,
                         alpha=0.13, linewidth=0, zorder=2)
        ax.plot(i - dens, grid, color=SERIES_1, linewidth=0.8, zorder=3)
        ax.plot(i + dens, grid, color=SERIES_1, linewidth=0.8, zorder=3)

        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK_SECONDARY, linewidth=1.4, zorder=4)
        ax.plot([i - 0.17, i + 0.17], [med, med], color=SERIES_2,
                linewidth=2.0, zorder=5, solid_capstyle="butt")

    if ylim is not None:
        ax.set_ylim(*ylim)
        hidden = sum(int(((np.asarray(v) < ylim[0]) | (np.asarray(v) > ylim[1])).sum())
                     for v in per_camera.values())
        total = sum(len(v) for v in per_camera.values())
        if hidden:
            # A clipped axis that does not admit it is a lie about the spread.
            ax.annotate(f"{hidden} of {total} frames lie beyond the axis",
                        xy=(0.015, 0.012), xycoords="axes fraction", ha="left",
                        va="bottom", fontsize=5.5, color=INK_MUTED)

    ax.set_xticks(range(len(cams)))
    # n goes in the tick label: inside the axes it collides with the tails.
    ax.set_xticklabels([f"{c}\nn={len(per_camera[c])}" for c in cams])
    ax.set_xlim(-0.6, len(cams) - 0.4)
    ax.set_xlabel("Camera unit")
    ax.set_ylabel("Target-centred error (%)")

    if span_note:
        ax.annotate(span_note, xy=(0.015, 0.985), xycoords="axes fraction",
                    ha="left", va="top", fontsize=6.5, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


# --- field deployments (§4.5) --------------------------------------------


def fig_field_repeatability(
    field_cvs,
    pool_cvs,
    figsize: tuple[float, float] = (COL_WIDTH, 2.5),
) -> plt.Figure:
    """Within-individual repeatability, wild fish against posed models.

    This is the one field result that needs nothing external: a calibration
    error is common to every frame of one animal and cancels in a relative
    spread, as does any error in the length convention, and no comparison
    population is involved. So it is the only §4.5 number that is a measurement
    of the system rather than of the sample.

    Each point is one group -- one wild individual, or one (session, target)
    cell in the pool. Points, not a density: 25 and 29 groups is too few for a
    KDE to be anything but an assertion, and the reader should be able to count
    them. The bar is the median and the band its bootstrap interval, which is
    what §4.5 quotes; the intervals overlap, and the figure should show that
    rather than hide it behind non-overlapping summary marks.
    """
    rng = np.random.default_rng(0)
    groups = [("Wild fish", np.asarray(field_cvs, float)),
              ("Pool models", np.asarray(pool_cvs, float))]
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")

    for i, (label, v) in enumerate(groups):
        y = i + rng.uniform(-0.13, 0.13, v.size)          # jitter, so ties are countable
        ax.scatter(v, y, s=11, facecolor=SERIES_1, edgecolor="none",
                   alpha=0.55, zorder=3)
        med = float(np.median(v))
        draws = np.median(rng.choice(v, size=(20000, v.size), replace=True), axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        ax.plot([lo, hi], [i - 0.30, i - 0.30], color=INK_SECONDARY,
                linewidth=1.3, solid_capstyle="butt", zorder=4)
        ax.plot([med, med], [i - 0.38, i - 0.22], color=SERIES_2,
                linewidth=2.2, solid_capstyle="butt", zorder=5)
        ax.annotate(f"{med:.1f} %", xy=(med, i - 0.46), ha="center", va="top",
                    fontsize=7, color=INK_SECONDARY)

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{l}\nn={len(v)}" for l, v in groups])
    ax.set_ylim(len(groups) - 0.5, -0.75)
    ax.set_xlabel("Within-individual CV (%)")
    ax.set_xlim(left=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_field_species(
    field,
    figsize: tuple[float, float] = (COL_WIDTH, 3.0),
    min_fish: int = 2,
) -> plt.Figure:
    """Measured length by species, one point per measurement.

    Descriptive only, and the caption must say so: the species is a labeler's
    judgement that nothing in the field data can check (§4.5), so a per-species
    offset and a systematic misidentification are the same picture here. Species
    with fewer than `min_fish` individuals are pooled into "other", because a
    row that is one animal invites a comparison the sample cannot support.
    """
    # NB the corpus already has a species literally called "Other" (identifiable
    # but nontarget), so the pooled bucket must not be called that too.
    keep = [s for s, g in field.groupby("species") if g.fish_id.nunique() >= min_fish]
    f = field.assign(sp=np.where(field.species.isin(keep), field.species,
                                 "Species with one individual"))
    order = (f.groupby("sp").length_m.median().sort_values().index.tolist())
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="x")
    for i, sp in enumerate(order):
        v = f.loc[f.sp == sp, "length_m"].to_numpy() * 100
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, v.size), s=10,
                   facecolor=SERIES_1, edgecolor="none", alpha=0.55, zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.26, i + 0.26], color=SERIES_2,
                linewidth=2.0, solid_capstyle="butt", zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{sp}\n{f.loc[f.sp == sp].fish_id.nunique()} fish, "
                        f"{int((f.sp == sp).sum())} meas." for sp in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Measured fork length (cm)")
    ax.set_xlim(left=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def fig_field_by_camera(
    field,
    species: str = "Hogfish",
    figsize: tuple[float, float] = (COL_WIDTH, 2.5),
) -> plt.Figure:
    """One species per camera unit -- the field analogue of Figure 10.

    **One point per ANIMAL, not per measurement**, and the distinction decides
    what the figure says. Repeat frames of one fish are not independent, and
    plotting all 74 hogfish measurements returns F(5,68) = 4.05, p = 0.004 --
    an apparently significant unit effect that is pseudo-replication. Collapsed
    to the 33 individuals §4.6 actually tests, it is F(5,27) = 1.10, p = 0.38.

    An animal's frames reduce through `nearest_rank_p90`, the one estimator the
    paper reports anywhere a set of frames becomes a length -- Figures 1, 14 and
    15 included. Read what nearest rank does at this sample size: no field
    animal has more than 8 frames and `ceil(0.9n)` is n for every n <= 10, so a
    field p90 IS that animal's longest frame. That is the intended behaviour and
    not an accident of the sample. The single-depth back-projection can only
    read short, so the longest frame is the one least corrupted by pose; it is
    also why the choice barely moves this figure (a per-animal mean gives
    F = 1.17, p = 0.35) while keeping one convention across the paper.

    The figure is drawn to show that a unit effect is *not resolvable* here, not
    that there is none: with 4 to 18 fish per unit and an 18 % between-fish size
    spread, the standard error on a unit's median is larger than any bias worth
    detecting. The band behind the points is the between-fish interquartile
    range, so the reader can see what the units are being compared against.
    """
    from scipy import stats

    f = field[field.species == species]
    per_fish = (
        f.groupby(["camera_id", "fish_id"])
        .length_m.agg(nearest_rank_p90)
        .reset_index()
    )
    cams = sorted(per_fish.camera_id.unique())
    rng = np.random.default_rng(2)

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="y")
    allv = per_fish.length_m.to_numpy() * 100
    q1, q3 = np.percentile(allv, [25, 75])
    ax.axhspan(q1, q3, facecolor=INK_MUTED, alpha=0.10, zorder=0)
    ax.axhline(np.median(allv), color=INK_MUTED, linewidth=0.8,
               linestyle=(0, (4, 3)), zorder=1)

    groups = []
    for i, c in enumerate(cams):
        v = per_fish.loc[per_fish.camera_id == c, "length_m"].to_numpy() * 100
        groups.append(v)
        ax.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=14,
                   facecolor=SERIES_1, edgecolor="none", alpha=0.7, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=SERIES_2,
                linewidth=2.0, solid_capstyle="butt", zorder=5)

    F, pv = stats.f_oneway(*groups)
    ax.annotate(f"$F({len(cams)-1},{len(allv)-len(cams)}) = {F:.2f}$, $p = {pv:.2f}$",
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([f"{c}\nn={len(g)}" for c, g in zip(cams, groups)])
    ax.set_xlabel("Camera unit")
    ax.set_ylabel(f"{species} fork length (cm)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def fig_field_vs_stereo(
    ours,
    stereo,
    figsize: tuple[float, float] = (COL_WIDTH, 3.0),
    min_fish: int = 5,
    seed: int = 0,
) -> plt.Figure:
    """Our per-species median against an independent stereo-video archive.

    The field analogue of Figure 1, with one crucial difference the caption must
    carry: Figure 1 measures ONE object against its own known length, so a
    departure from the 1:1 line is error. Here the two axes are *different
    animals* -- our fish and theirs, drawn from the same reef and season but
    never the same individual -- so a departure is error OR a difference in
    which fish each happened to encounter, and nothing in the data separates
    them. It is a consistency check, not a bias measurement.

    Both axes therefore carry bootstrap intervals on the median, and the
    vertical one is computed per ANIMAL rather than per frame: repeat frames of
    one fish are not independent and would shrink the interval spuriously.

    `ours` maps species -> per-animal lengths (cm); `stereo` maps species ->
    archive lengths (cm).
    """
    rng = np.random.default_rng(seed)

    def ci(v):
        d = np.median(rng.choice(v, size=(20000, len(v)), replace=True), axis=1)
        return np.percentile(d, [2.5, 97.5])

    sp = sorted(set(ours) & set(stereo), key=lambda k: np.median(stereo[k]))
    sp = [k for k in sp if len(ours[k]) >= min_fish]
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")

    lo = min(min(np.min(ours[k]), np.min(stereo[k])) for k in sp) * 0.85
    hi = max(max(np.median(ours[k]), np.median(stereo[k])) for k in sp) * 1.25
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=0.9,
            linestyle=(0, (5, 4)), zorder=1, label="1:1 (agreement)")

    pts = []
    for k in sp:
        o, t = np.asarray(ours[k], float), np.asarray(stereo[k], float)
        om, tm = float(np.median(o)), float(np.median(t))
        ol, oh = ci(o)
        tl, th = ci(t)
        # Capped, so a 2 cm arm still reads as an interval next to a 23 cm one.
        # The arms differ by 12x across these five species because the samples
        # do (5 to 391 fish); that asymmetry IS the result, so it must be legible.
        ax.errorbar([tm], [om], yerr=[[om - ol], [oh - om]], xerr=[[tm - tl], [th - tm]],
                    fmt="none", ecolor=SERIES_1, elinewidth=1.0, capsize=2.2,
                    capthick=1.0, zorder=3)
        ax.scatter([tm], [om], s=26, facecolor=SERIES_2, edgecolor=SURFACE,
                   linewidth=0.7, zorder=4)
        pts.append((tm, om, k))          # counts go in the caption, not the plot

    # Greedy label placement. Hogfish and Stoplight Parrotfish sit almost on top
    # of one another, so a fixed offset overlaps whatever is drawn next; try
    # candidate positions around each point and take the first that is clear.
    span = hi - lo
    placed: list[tuple[float, float, float, float]] = []
    cands = [(8, -3, "left", "top"), (8, 5, "left", "bottom"),
             (-8, -3, "right", "top"), (-8, 5, "right", "bottom"),
             (8, -16, "left", "top"), (-8, -16, "right", "top"),
             (8, 16, "left", "bottom"), (-8, 16, "right", "bottom")]
    for x, y, text in pts:
        w = 0.014 * span * len(text)                 # rough text box, in data units
        h = 0.055 * span
        for dx, dy, ha, va in cands:
            cx = x + dx / 72 / fig.get_size_inches()[0] * span
            cy = y + dy / 72 / fig.get_size_inches()[1] * span
            x0 = cx if ha == "left" else cx - w
            y0 = cy if va == "bottom" else cy - h
            if all(x0 + w < q[0] or q[0] + q[2] < x0 or y0 + h < q[1] or q[1] + q[3] < y0
                   for q in placed):
                break
        placed.append((x0, y0, w, h))
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, color=INK_SECONDARY, ha=ha, va=va, zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stereo-video archive, median (cm)")
    ax.set_ylabel("FishSense Lite, median (cm)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --- figure 15: the paired-instrument day --------------------------------
#
# Figure 14's population comparison and this one are deliberately different
# forms for a reason the captions have to carry: there, the two axes are
# different animals, so a departure from 1:1 is error OR a difference in which
# fish each instrument met. Here the two axes are the SAME animal, so a
# departure is the instruments disagreeing and nothing else.


def fig_paired_vs_stereo(
    ours: pd.DataFrame,
    pairs: Sequence[object],
    figsize: tuple[float, float] = (COL_WIDTH, 2.7),
    r2: float | None = None,
    r2_basis: str = "per-fish $p_{90}$",
) -> plt.Figure:
    """Our per-frame lengths against the stereo length, per paired individual.

    Figure 1's form applied to the field, and deliberately its estimator too:
    the reference length on x, every frame on y, the per-individual $p_{90}$
    drawn over the cloud, and a 1:1 datum. Reading the two figures side by side
    should require learning nothing new.

    `nearest_rank_p90` for the same reason it is used against the models -- the
    single-depth back-projection means an out-of-plane fish can only read short,
    so a mean or median is biased down by however much the pose varied. Note
    what nearest rank does at these sample sizes: with 3 to 11 frames per fish
    it selects the top sample for five of the seven, so $p_{90}$ here is close
    to a per-fish maximum and should be read as a high-order statistic rather
    than as a tail estimate.

    Two things differ from Figure 1, each because the data differs. There is no
    jitter: Figure 1 jitters because thousands of frames collapse onto eight
    known lengths, where 39 frames on seven are already legible and jitter
    would move points off the very reference they are compared against. And the
    1:1 line is agreement, not truth -- both axes are instruments, so a point
    above it means we read longer than the stereo did, which is a disagreement
    and not yet an error.

    `ours` is `data/stereo_pairs.csv`; `pairs` is
    `stereo_pairs.build_pairs(...)`, one entry per paired individual.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")

    paired = sorted(pairs, key=lambda p: p.stereo_mm)
    by_dive = {int(d): g for d, g in ours.groupby("dive_id")}

    stereo_cm, p90_cm = [], []
    frame_x, frame_y = [], []
    for p in paired:
        lengths_cm = by_dive[int(p.dive_id)]["length_m"].to_numpy(float) * 100.0
        stereo_cm.append(p.stereo_mm / 10.0)
        # From the Pair, not recomputed here, so the diamond is the same number
        # `stereo_pairs.summary("p90")` reports and the text quotes.
        p90_cm.append(p.ours_p90_mm / 10.0)
        frame_x.extend([p.stereo_mm / 10.0] * lengths_cm.size)
        frame_y.extend(lengths_cm)

    stereo_cm, p90_cm = np.array(stereo_cm), np.array(p90_cm)
    frame_x, frame_y = np.array(frame_x), np.array(frame_y)

    lo = float(min(frame_y.min(), stereo_cm.min()))
    hi = float(max(frame_y.max(), stereo_cm.max()))
    pad = 0.08 * (hi - lo)
    span = np.array([lo - pad, hi + pad])

    ax.plot(span, span, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
            zorder=2, label="1:1 (agreement)")

    # Frames are drawn at full size and low transparency; Figure 1's heavier
    # alpha exists to survive a cloud two orders of magnitude denser than this.
    ax.scatter(frame_x, frame_y, s=11, color=SERIES_1, alpha=0.5, linewidths=0,
               zorder=3, label="Per-frame measurement")
    ax.scatter(stereo_cm, p90_cm, s=26, marker="D", color=SERIES_2,
               edgecolors=SURFACE, linewidths=1.0, zorder=4,
               label="Per-fish $p_{90}$ estimate")

    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Stereo-video length, same individual (cm)")
    ax.set_ylabel("FishSense Lite length (cm)")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    _r2_annotation(ax, r2, r2_basis)
    fig.tight_layout()
    return fig


# --- figure 16: how many frames p90 needs --------------------------------
#
# The estimator's own sampling behaviour, which the ruler exposed. Nearest rank
# is ceil(0.9n) and that equals n for every n <= 10, so below ten frames p90 is
# the sample maximum rather than a quantile. The step at n = 10 is that fact,
# not noise, and it is the figure's whole point.


def fig_p90_budget(
    traces,
    worst,
    marks: tuple[float, ...] = (5.0, 10.0, 15.0),
    budget: float = 15.0,
    figsize: tuple[float, float] = (COL_WIDTH, 3.1),
) -> plt.Figure:
    """The reported measurement's error against the number of frames behind it,
    in percent length error -- the unit §4.2 and Figures 1-3 use.

    `traces`, `worst` are `repeatability.p90_level_traces(...)`.

    This is a **budget** figure, and the distinction from the convergence view
    (Figure D2) is the whole reason it exists. A draw's $p_{90}$ is the
    measurement a diver with `n` frames would report, so plotting its level
    against the 5, 10 and 15 % marks answers "does the measurement meet the
    budget?" directly. Figure D2 plots the deviation from each cell's own
    full-sample $p_{90}$ instead, which answers "has the estimator converged?"
    -- a standard fifteen times tighter, and a different question.

    **One trace per cell, never pooled.** The fifteen cells' own $p_{90}$ values
    span nearly 9 pp; pooling the draws folds that constant spread into the band
    and the band then goes flat in `n`, which is a figure about between-cell
    variation wearing the label of one about sample size. Drawn apart, each
    trace settles -- that is the sample-size effect -- and the fan's width at
    large `n` is the between-cell term, which is §4.3's subject and not this
    figure's.

    The worst single draw over every cell carries the budget claim, because a
    budget is a bound rather than an average: what matters is whether *any* draw
    crosses it.

    The positive side is marked only at the innermost band. Nothing reaches
    +5 % at any `n`, so drawing +10 and +15 would spend a third of the panel
    certifying that an empty region is empty. Nothing is clipped.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")
    _zero_line(ax, orientation="h")

    all_ns = sorted({n for t in traces.values() for n in t})
    wx = np.array(sorted(worst), dtype=float)
    wy = np.array([worst[int(n)] for n in wx])
    floor = min(wy.min(), -max(marks))
    ceiling = max([v for t in traces.values() for v in t.values()] + [min(marks)])

    # Single-hue ramp: the budget is the darkest mark because it is the one the
    # paper commits to; the tighter marks are context.
    shades = np.linspace(0.30, 1.0, len(marks))
    for mark, shade in zip(marks, shades):
        for sign in (-1.0, 1.0):
            if sign > 0 and mark != min(marks):
                continue
            ax.axhline(sign * mark, color=INK_MUTED, linestyle=":",
                       linewidth=0.7 + 0.5 * shade, alpha=0.35 + 0.65 * shade,
                       zorder=1)
        label = f"{mark:g} %" + (" budget" if mark == budget else "")
        # OUTSIDE the axes, in the right margin: a cell trace runs at -5.84 %,
        # so any label placed inside near the 5 % mark lands on it. `clip_on`
        # off and `bbox = "tight"` together give the labels their own gutter.
        ax.annotate(label, xy=(1.01, -mark),
                    xycoords=ax.get_yaxis_transform(), xytext=(0, 0),
                    textcoords="offset points", fontsize=6,
                    color=INK_SECONDARY, ha="left", va="center",
                    annotation_clip=False, zorder=5)

    ax.axvspan(min(all_ns) - 1.0, 9.5, color=INK_MUTED, alpha=0.07, zorder=0)
    for i, trace in enumerate(traces.values()):
        ns = sorted(trace)
        ax.plot(ns, [trace[n] for n in ns], color=SERIES_1, linewidth=0.8,
                alpha=0.55, zorder=2,
                label="One session-target cell" if i == 0 else None)
    ax.plot(wx, wy, color=SERIES_2, linewidth=1.8, zorder=4,
            label="Worst single draw")

    # Under the worst-draw trace inside the shaded small-n region, the only
    # clear space: the budget mark and its label own the bottom row.
    ax.annotate(r"$p_{90}$ here is just the" "\n" r"maximum ($\lceil 0.9n \rceil = n$)",
                xy=(5.6, wy.min()), xytext=(0, -5),
                textcoords="offset points", fontsize=6.2, color=INK_SECONDARY,
                ha="center", va="top", zorder=5)

    ax.set_xlim(min(all_ns) - 1.0, max(all_ns) + 1.0)
    ax.set_ylim(floor - 2.4, ceiling + 1.0)
    ax.set_xlabel("Frames of one fish")
    ax.set_ylabel("Reported $p_{90}$ length error (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    fig.tight_layout()
    return fig


def fig_p90_vs_sample_size(
    rarefaction,
    tolerance: float = 1.0,
    min_frames: int | None = None,
    figsize: tuple[float, float] = (COL_WIDTH, 3.1),
) -> plt.Figure:
    """Error of a p90 length estimate against the number of frames behind it.

    `rarefaction` is `repeatability.p90_rarefaction(...)`.

    **Two statistics, and which is which matters.** The per-fish statistic is
    $p_{90}$ of percent length error -- the paper's estimator everywhere, because
    a fish's own frames are a one-sided pose-corrupted distribution and a high
    quantile rejects that tail. What this figure plots is how that estimate
    moves when it is computed from `n` frames instead of all of them, and the
    summary ACROSS draws and cells is a **median**, for the same reason §4.6
    takes a median across animals: sampling error is not one-sided, and nothing
    about a median of it discards a tail that needs discarding.

    Blue is the central 80 % of the signed error. It is kept signed because the
    sign flips across the small-n range -- the maximum of two draws sits below
    the true 90th percentile, the maximum of nine sits above it -- so a small
    sample is not conservative in either direction.

    The step at n = 10 is annotated because a reader will otherwise take it for
    a glitch: it is where nearest rank stops returning the maximum.
    """
    pts = [rarefaction[n] for n in sorted(rarefaction)]
    ns = np.array([p.n for p in pts], dtype=float)
    med = np.array([p.median for p in pts])
    lo = np.array([p.lo for p in pts])
    hi = np.array([p.hi for p in pts])

    fig, ax = plt.subplots(figsize=figsize)
    _grid(ax, axis="both")
    _zero_line(ax, orientation="h")

    # ignore n = 2's long lower tail, which would otherwise spend a third of
    # the panel on one sample size. The generous padding is deliberate: the
    # legend sits above the axes so the interior is the data's, and both
    # annotations need a margin they are not fighting the marks for.
    top = max(hi.max(), tolerance) + 0.55
    bottom = min(lo[ns >= 4].min(), -tolerance) - 0.55

    ax.axvspan(ns.min() - 0.5, 9.5, color=INK_MUTED, alpha=0.07, zorder=0)
    ax.fill_between(ns, lo, hi, color=SERIES_1, alpha=0.20, linewidth=0, zorder=2,
                    label="Central 80 % of draws")
    ax.plot(ns, med, color=SERIES_2, linewidth=1.8, zorder=4,
            # "change", not "error": the reference is the cell's own full-sample
            # p90, so zero is its own answer and not the known length.
            label="Median change")

    for y in (-tolerance, tolerance):
        ax.axhline(y, color=INK_MUTED, linestyle=":", linewidth=1.0, zorder=3)
    ax.annotate(f"$\\pm${tolerance:g} %", xy=(ns.max(), tolerance),
                xytext=(-2, 2), textcoords="offset points", fontsize=6,
                color=INK_SECONDARY, ha="right", va="bottom")

    # The comparison a reader needs and the panel cannot hold: the sampling term
    # is a few pp, the accuracy budget is 15 %, and drawing the latter would
    # crush the former into a flat line. So it is stated at the scale it is.
    ax.annotate("for scale: the \u00b115 % error budget\n"
                f"spans {30.0 / (top - bottom):.0f}\u00d7 this panel",
                xy=(ns.max(), bottom), xytext=(-2, 3), textcoords="offset points",
                fontsize=6, color=INK_SECONDARY, ha="right", va="bottom", zorder=5)

    if min_frames is not None:
        ax.axvline(min_frames, color=INK_MUTED, linewidth=1.0,
                   linestyle=(0, (4, 2)), zorder=3)
        ax.annotate(f"{min_frames} frames:\n90 % of draws\ninside $\\pm${tolerance:g} %",
                    xy=(min_frames, top), xytext=(5, -5),
                    textcoords="offset points", fontsize=6.2,
                    color=INK_SECONDARY, ha="left", va="top", zorder=5)

    ax.annotate(r"$p_{90}$ here is just the" "\n" r"maximum ($\lceil 0.9n \rceil = n$)",
                xy=(5.75, bottom), xytext=(0, 5), textcoords="offset points",
                fontsize=6.2, color=INK_SECONDARY, ha="center", va="bottom", zorder=5)

    ax.set_xlim(ns.min() - 1.0, ns.max() + 1.0)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Frames of one fish")
    # Name the quantity concretely. `delta` can be read two ways -- as a
    # difference of two percent errors, whose unit is percentage points, or via
    # the cancellation below as one length difference over a length, whose unit
    # is plainly % of the known length. Same number; the second needs no
    # reconciling, so it is what the axis says.
    #
    #   delta = p90(draw) - p90(all frames)
    #         = 100 (L_draw - K)/K - 100 (L_all - K)/K
    #         = 100 (L_draw - L_all)/K
    #
    # What no axis label can carry is the ORIGIN: zero here is the cell's own
    # full-sample p90, not the known length, so delta = 0 means "the same answer
    # every frame would have given", never "the right answer". The caption has
    # to say so, or a reader takes this for a measurement converging on truth.
    ax.set_ylabel("Change from the all-frames $p_{90}$\n(% of known length)")
    ax.margins(y=0.05)
    # above the axes, as Figure 4 does it, so the panel interior is all data
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    fig.tight_layout()
    return fig


# --- figure 9b: the same cost drawn on the frame itself ------------------
#
# Sequential, so ONE hue light -> dark, stepped from this paper's own orange.
# Never a rainbow: the quantity is magnitude and one-signed, and a rainbow would
# invent category boundaries where the field is smooth. The categorical slots
# are untouched -- this is a different form, not a third series.

_PORT_ERROR_RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "port_error",
    ["#fdf3ee", "#fbd9c6", "#f6ad86", "#ef7f4b", "#d9541f", "#a43a12", "#6b2409"],
).with_extremes(bad=GRIDLINE)


def fig_flat_port_error_field(
    field,
    budget_pct: float = 15.0,
    figsize: tuple[float, float] = (COL_WIDTH, 2.9),
) -> plt.Figure:
    """The flat-port length error drawn over the image frame it happens in.

    `field` is `refraction.flat_port_error_field(...)`. The panel IS the frame:
    axes are image pixels, so "where in the picture" needs no translation into
    degrees and cannot be mistaken for the fish's pose.

    **Read the asymmetry, it is the whole argument.** The port is rotationally
    symmetric but the target is not a point -- it is held horizontal, so at the
    left and right edges it lies along a radius and at the top and bottom across
    one. Radial and tangential magnification differ, which is exactly why this
    is not a scale error a calibration could absorb: a horizontal fish reads
    +23 % at the side of the frame and +6 % at the top, at the same distance
    from the centre.

    Grey is where a target of this length would not fit in frame.
    """
    err = np.asarray(field["error_pct"], dtype=float)
    W, H = field["image_size_px"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(err, extent=field["extent"], origin="upper", aspect="equal",
                   cmap=_PORT_ERROR_RAMP, vmin=0.0, vmax=float(np.nanmax(err)),
                   # NEAREST, not bilinear: matplotlib will not interpolate
                   # across a NaN, so a smoothing filter bleeds the no-fit mask
                   # outward and notches its corners. At an 8 px cell the raw
                   # samples are already finer than the printed figure.
                   interpolation="nearest", zorder=1)

    if budget_pct is not None and np.nanmax(err) >= budget_pct:
        ys = np.linspace(field["extent"][3], field["extent"][2], err.shape[0])
        xs = np.linspace(field["extent"][0], field["extent"][1], err.shape[1])
        cs = ax.contour(xs, ys, err, levels=[budget_pct], colors=[SURFACE],
                        linewidths=1.4, zorder=3)
        ax.clabel(cs, fmt=lambda v: f"{v:g} %", fontsize=6, inline=True,
                  inline_spacing=6)

    # the frame's own border, and its centre
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor=INK_MUTED,
                               linewidth=0.8, zorder=4))
    ax.plot([W / 2], [H / 2], marker="+", markersize=6, markeredgewidth=1.0,
            color=INK_PRIMARY, zorder=5)

    # the asymmetry, stated where it happens: same distance from the centre,
    # very different error, because the target lies along a radius at the side
    # and across one at the top
    mid_row = err[err.shape[0] // 2, :]
    mid_col = err[:, err.shape[1] // 2]
    side = float(np.nanmax(mid_row))
    top = float(np.nanmax(mid_col))
    for xy, text, va, ha in (
        ((0.985 * W, H / 2), f"side\n{side:+.0f} %", "center", "right"),
        ((W / 2, 0.02 * H), f"top {top:+.0f} %", "top", "center"),
        ((W / 2, H / 2), "centre +0.1 %", "bottom", "center"),
    ):
        ax.annotate(text, xy=xy, xytext=(0, 5 if va == "bottom" else 0),
                    textcoords="offset points", fontsize=6,
                    color=INK_PRIMARY if va == "bottom" else SURFACE,
                    ha=ha, va=va, zorder=6)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(
        f"The {W} \u00d7 {H} frame, target held horizontal.\n"
        "Grey: a target this long no longer fits.", fontsize=6.5, labelpad=3)

    # horizontal bar beneath a 4:3 panel; a vertical one is taller than the
    # picture and takes the eye off it
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055,
                      pad=0.12, aspect=34)
    cb.set_label("Length error (%)", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(BASELINE)
    fig.tight_layout()
    return fig


def fig_flat_port_before_after(
    uncorrected,
    corrected,
    budget_pct: float = 15.0,
    figsize: tuple[float, float] = (FULL_WIDTH, 2.85),
) -> plt.Figure:
    """The port's cost beside what the corrective optic leaves, one colour scale.

    Both panels are the same frame, the same target, the same scale, so the
    comparison is the picture rather than a pair of numbers in the text.

    **What the right panel is, and is not.** It is the same model with the index
    step removed -- the air path the M52 lens restores at the port -- and NOT a
    refraction *correction*. Pinax and the in-water single-viewpoint calibration
    are the companion paper's contribution and neither they nor their code are
    in this repository. So the right panel is close to tautological: take away
    the water interface and there is no refraction error to have. Its job is
    only to put the magnitude of what the optic removes on a scale the eye can
    compare, which a sentence cannot do.
    """
    left = np.asarray(uncorrected["error_pct"], dtype=float)
    right = np.asarray(corrected["error_pct"], dtype=float)
    W, H = uncorrected["image_size_px"]
    vmax = float(np.nanmax(left))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, err, title in (
        (axes[0], left, "Flat port, no corrective optic"),
        (axes[1], right, "Air path restored by the optic"),
    ):
        im = ax.imshow(err, extent=uncorrected["extent"], origin="upper",
                       aspect="equal", cmap=_PORT_ERROR_RAMP, vmin=0.0, vmax=vmax,
                       interpolation="nearest", zorder=1)
        ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor=INK_MUTED,
                                   linewidth=0.8, zorder=4))
        ax.plot([W / 2], [H / 2], marker="+", markersize=5, markeredgewidth=0.9,
                color=INK_PRIMARY, zorder=5)
        peak = float(np.nanmax(err))
        ax.annotate(f"worst in frame {peak:+.1f} %", xy=(W / 2, 0.985 * H),
                    xytext=(0, 0), textcoords="offset points", fontsize=6.5,
                    color=INK_PRIMARY if peak < 1 else SURFACE,
                    ha="center", va="bottom", zorder=6)
        ax.set_title(title, fontsize=7, pad=4)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    if budget_pct is not None and np.nanmax(left) >= budget_pct:
        xs = np.linspace(uncorrected["extent"][0], uncorrected["extent"][1],
                         left.shape[1])
        ys = np.linspace(uncorrected["extent"][3], uncorrected["extent"][2],
                         left.shape[0])
        cs = axes[0].contour(xs, ys, left, levels=[budget_pct], colors=[SURFACE],
                             linewidths=1.3, zorder=3)
        axes[0].clabel(cs, fmt=lambda v: f"{v:g} %", fontsize=6, inline=True,
                       inline_spacing=5)

    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.06, aspect=48)
    cb.set_label("Length error for a 30 cm target held horizontal (%)",
                 fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(BASELINE)
    return fig


# --- figure D1 (repository only): the candidate range correction ---------
#
# Two panels because the answer has two halves and one of them is negative. A
# panel showing only the pool would sell a fix the external check does not
# support.


def fig_depth_correction(
    depths_m,
    before_pct,
    after_pct,
    curve,
    paired,
    bins: int = 12,
    figsize: tuple[float, float] = (FULL_WIDTH, 2.8),
) -> plt.Figure:
    """What the `a + b/z` range correction does, and what it does not.

    Left: the pool cohort's binned median before and after, with the fitted
    curve. `after_pct` should be LEAVE-ONE-SESSION-OUT residuals -- correcting
    a session with a curve fitted on that same session flatters the fix and
    measures nothing.

    Right: the seven fish of the paired stereo day, each a line from its
    as-measured difference to its corrected one. This is the only external
    reference in the corpus, and it is where the fix stops working: the
    correction only ever adds length, so it helps the fish that read short and
    hurts the ones that read long.

    `paired` is a sequence of `(label, before, after)` in percent.
    """
    z = np.asarray(depths_m, float)
    b = np.asarray(before_pct, float)
    a = np.asarray(after_pct, float)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- left: does it flatten the pool profile? ---
    _grid(ax, axis="both")
    _zero_line(ax, orientation="h")
    edges = np.quantile(z, np.linspace(0, 1, bins + 1))
    mid = [np.median(z[(z >= lo) & (z <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
    for values, colour, label in ((b, SERIES_1, "As measured"),
                                  (a, SERIES_2, "Corrected (held out)")):
        med = [np.median(values[(z >= lo) & (z <= hi)])
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mid, med, marker="o", markersize=3.5, linewidth=1.5, color=colour,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4, label=label)
    grid_z = np.linspace(z.min(), z.max(), 200)
    ax.plot(grid_z, curve[0] + curve[1] / grid_z, color=INK_MUTED, linewidth=1.0,
            linestyle=(0, (4, 2)), zorder=3,
            label=f"${curve[0]:+.2f} {curve[1]:+.2f}/z$")
    # the fit overshoots the shortest bin, which is worth seeing but not worth
    # half the panel; clip to the data and let the curve run off
    lows = [np.median(b[(z >= lo) & (z <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])]
    ax.set_ylim(min(lows) - 1.2, 1.6)
    ax.set_xlabel("Laser depth (m)")
    ax.set_ylabel("Binned median length error (%)")
    ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.3, fontsize=6)
    ax.set_title("Pool cohort: the fix works here", fontsize=7, pad=4)

    # --- right: the only external reference says otherwise ---
    _grid(bx, axis="y")
    _zero_line(bx, orientation="h")
    closer = 0
    for label, was, now in paired:
        closer += abs(now) < abs(was)
        bx.plot([0, 1], [was, now], color=INK_MUTED, linewidth=0.9, zorder=2)
        bx.plot([0], [was], marker="o", markersize=4, color=SERIES_1,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
        bx.plot([1], [now], marker="o", markersize=4, color=SERIES_2,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["as measured", "corrected"], fontsize=6.5)
    bx.set_xlim(-0.35, 1.35)
    bx.set_ylabel("Difference from the stereo rig (%)")
    bx.set_title(f"Paired day: {closer} of {len(paired)} moved closer",
                 fontsize=7, pad=4)
    fig.tight_layout()
    return fig
