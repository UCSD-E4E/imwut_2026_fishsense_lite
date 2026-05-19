"""Simulation defaults matching the hardware characterized in the paper.

These values correspond to the Olympus Tough TG6 in its underwater housing
with the Backscatter M52 corrective optic — the reference FishCamera
implementation described in Section 3.3 of the paper.
"""
from pathlib import Path

IMAGE_WIDTH = 4014
IMAGE_HEIGHT = 3016
FOCAL_LENGTH_PX = 2850

LASER_DIAMETER_M = 5.6e-3

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
