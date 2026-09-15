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


def _clip_x(ax: plt.Axes, groups: Iterable[Sequence[float]], xlim) -> None:
    """Optionally narrow the x-axis, and SAY how many points that hides.

    A long one-sided tail compresses every box into a third of the width, so
    clipping is often the readable choice -- but a clipped axis that does not
    admit it is a lie about the spread. The count goes on the figure.
    """
    if xlim is None:
        return
    lo, hi = xlim
    hidden = sum(
        int(((np.asarray(v) < lo) | (np.asarray(v) > hi)).sum()) for v in groups
    )
    ax.set_xlim(lo, hi)
    if hidden:
        # Below the axes, right-aligned, at a FIXED point offset rather than an
        # axes fraction: a fraction scales with figure height, so the same
        # offset that cleared the x-label on a tall panel landed on top of it
        # on a short one. Above the plot is no good either -- the legend is
        # there. The x-label is centred, so the right edge stays free.
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


# --- figure 1: measured vs. known ----------------------------------------


def fig_measured_vs_known(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (COL_WIDTH, 2.7),
    jitter: float = 0.0025,
    seed: int = 0,
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
    _clip_x(ax, groups, xlim)
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
    _clip_x(ax, groups, xlim)
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


def save_figure(
    fig: plt.Figure,
    name: str,
    outdir: Path | str = "figures",
    synthetic: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
) -> list[Path]:
    """Write a figure to `outdir` as vector PDF (for LaTeX) and PNG (for
    previewing in a notebook or slide).

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

    written = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None)
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
    figsize: tuple[float, float] = (COL_WIDTH, 2.2),
    xlim: tuple[float, float] | None = (-12, 12),
    emphasise: str | None = None,
) -> plt.Figure:
    """Per-model error distribution with the p90 estimator marked.

    The box shows the frame-level spread; the orange diamond is the number the
    pipeline actually reports. The gap between the median and the p90 is the
    foreshortening tail, which is the reason the estimator is a high quantile
    and not a mean.

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
            showfliers=False,
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
    _clip_x(ax, groups, xlim)
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
    figsize: tuple[float, float] = (COL_WIDTH, 2.5),
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
        ax.annotate(
            f"laser range reads {range_pct_error:+.0f} %,\n"
            "cancelling the transverse error on axis",
            xy=(x[0], y[0]), xytext=(8, 10), textcoords="offset points",
            fontsize=6, color=INK_SECONDARY, ha="left", va="bottom",
        )

    ax.set_xlabel("Target position in frame (degrees off axis)")
    ax.set_ylabel("Length error (%)")
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
    to the 33 individuals §4.5 actually tests, it is F(5,27) = 1.17, p = 0.35.

    The figure is drawn to show that a unit effect is *not resolvable* here, not
    that there is none: with 4 to 18 fish per unit and an 18 % between-fish size
    spread, the standard error on a unit's median is larger than any bias worth
    detecting. The band behind the points is the between-fish interquartile
    range, so the reader can see what the units are being compared against.
    """
    from scipy import stats

    f = field[field.species == species]
    per_fish = f.groupby(["camera_id", "fish_id"]).length_m.mean().reset_index()
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
        ax.plot([tm, tm], [ol, oh], color=SERIES_1, linewidth=1.0, zorder=3)
        ax.plot([tl, th], [om, om], color=SERIES_1, linewidth=1.0, zorder=3)
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
