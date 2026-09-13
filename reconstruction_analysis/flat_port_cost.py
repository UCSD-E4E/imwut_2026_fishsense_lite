"""Figure 9: what it costs to ignore the water's refractive index.

Run from the repository root:

    uv run python reconstruction_analysis/flat_port_cost.py

Writes `fish_model_analysis/figures/fig9_flat_port_cost.{pdf,png}` and prints
the three numbers the paper quotes.

WHAT THIS IS
------------
A simulation, not a measurement. It traces rays through a flat pane with
`fishsense_imwut.refraction` (validated against Table 1 of the Pinax paper by
`tests/test_refraction.py`) and asks what a length reads if the in-air
calibration is used underwater with no correction at all -- which is what the
corrective optic at the housing port exists to prevent.

The result is not the simple scale error one might expect. Ignoring refraction
expands the scene transversely by n_water and shortens the laser range by very
nearly 1/n_water. On the optical axis those cancel almost exactly, so a centred
target measures correctly *by accident*; off axis the angular compression is not
a pure scale, the cancellation fails, and the error depends on nothing but where
in the frame the target fell. That is why it cannot be averaged away, and why it
is easy to miss.

!! THE HOUSING IS NOT MEASURED !!
---------------------------------
GLASS_THICKNESS_M and N_GLASS below are placeholders carried over from the WUWNet
simulation, and the pane is assumed to sit at its optimal camera-to-glass
spacing. The headline number is driven by the index ratio and the field of view
rather than by the pane, so it should be robust to these -- but it has not been
checked against the real housing. Put calipers on the TG6 housing before this
figure goes in a submission.
"""

import numpy as np

from fishsense_imwut import pubfig
from fishsense_imwut.refraction import (
    SALTY_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    back_project_uncorrected,
    measure_length,
    optimal_d0,
    project_water_points,
    reconstruct_points,
    water_radius,
)

# --- the imaged system ----------------------------------------------------
IMAGE_WIDTH, IMAGE_HEIGHT = 4014, 3016
FOCAL_LENGTH_PX = 2850.0
CAMERA_INTRINSICS = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2],
        [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2],
        [0.0, 0.0, 1.0],
    ]
)
HALF_FOV = np.arctan(np.hypot(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2) / FOCAL_LENGTH_PX)

# --- the housing (see the warning above) ----------------------------------
GLASS_THICKNESS_M = 0.006  # 6 mm acrylic pane -- PLACEHOLDER, not measured
N_GLASS = 1.49  # acrylic
N_WATER = SALTY_WATER  # ocean deployment

# --- the rig and the target -----------------------------------------------
LASER_POSITION = np.array([-0.04, -0.11, 0.0])
LASER_DIRECTION = np.array([0.0, 0.0, 1.0])
FISH_LENGTH_M = 0.30
MEASUREMENT_DEPTH_M = 2.0
MAX_FRAME_FRACTION = 0.75  # of the half-frame; beyond this a fish is clipped


def main() -> None:
    pubfig.use_publication_style()

    d0_star, _, _ = optimal_d0(GLASS_THICKNESS_M, N_GLASS, N_WATER, HALF_FOV)
    port = FlatPort(d0_star, GLASS_THICKNESS_M, N_GLASS, N_WATER)

    back_project = lambda q: back_project_uncorrected(q, CAMERA_INTRINSICS)

    def to_pixels(points):
        alpha, azimuth = project_water_points(points, port)
        return alpha_azimuth_to_pixel(alpha, azimuth, CAMERA_INTRINSICS)

    # The range the pipeline actually has: triangulated from the laser dot with
    # the same uncorrected back-projection, not the true depth.
    laser_point = LASER_POSITION + MEASUREMENT_DEPTH_M * LASER_DIRECTION
    _, laser_dirs = back_project(to_pixels(laser_point))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), LASER_POSITION, LASER_DIRECTION
    )[0][2]
    range_pct = 100 * (measured_depth - MEASUREMENT_DEPTH_M) / MEASUREMENT_DEPTH_M

    half_frame = water_radius(
        np.arctan((IMAGE_WIDTH / 2) / FOCAL_LENGTH_PX), MEASUREMENT_DEPTH_M, port
    )
    offsets = np.linspace(0.0, MAX_FRAME_FRACTION * half_frame, 60)
    zeros = np.zeros_like(offsets)
    depths = np.full_like(offsets, MEASUREMENT_DEPTH_M)

    head = to_pixels(np.stack([offsets + FISH_LENGTH_M / 2, zeros, depths], -1))
    tail = to_pixels(np.stack([offsets - FISH_LENGTH_M / 2, zeros, depths], -1))
    measured = measure_length(head, tail, measured_depth, back_project)

    length_pct = 100 * (measured - FISH_LENGTH_M) / FISH_LENGTH_M
    field_deg = np.degrees(np.arctan(offsets / MEASUREMENT_DEPTH_M))

    print(f"housing: d0* = {d0_star * 1000:.3f} mm, pane {GLASS_THICKNESS_M * 1000:.1f} mm "
          f"(PLACEHOLDER), n_glass {N_GLASS}, n_water {N_WATER}")
    print(f"laser range error:      {range_pct:+.1f} %")
    print(f"length error on axis:   {length_pct[0]:+.2f} %")
    print(f"length error at {field_deg[-1]:.0f} deg: {length_pct[-1]:+.1f} %")
    print(f"crosses 15 %% budget at: {np.interp(15.0, length_pct, field_deg):.1f} deg off axis")

    fig = pubfig.fig_flat_port_cost(field_deg, length_pct, range_pct_error=range_pct)
    for path in pubfig.save_figure(
        fig, "fig9_flat_port_cost", outdir="fish_model_analysis/figures"
    ):
        print("wrote", path)


if __name__ == "__main__":
    main()
