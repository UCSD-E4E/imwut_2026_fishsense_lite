"""Helpers for saving figures into the shared figures/ directory.

Notebooks that produce figures referenced by the paper should use
``save_paper_figure`` rather than ``fig.savefig`` directly. This keeps
all paper-bound artifacts in a single location with consistent styling
and makes it obvious which notebook produces which figure.

The filename passed to ``save_paper_figure`` should match the basename
used in the paper's ``\\includegraphics{Images/graphs/...}`` reference.
"""
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

from fishsense_imwut.constants import FIGURES_DIR


def save_paper_figure(
    fig: plt.Figure,
    name: str,
    directory: Union[str, Path, None] = None,
    **savefig_kwargs,
) -> Path:
    """Save ``fig`` into the repo's ``figures/`` directory.

    The filename ``name`` (e.g. ``"laser_spot_path.pdf"``) should match
    the paper's ``\\includegraphics`` reference so figures can be
    copied straight into ``Images/graphs/`` in the paper source.
    """
    out_dir = Path(directory) if directory is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name

    kwargs = {"bbox_inches": "tight"}
    kwargs.update(savefig_kwargs)
    fig.savefig(path, **kwargs)
    return path
