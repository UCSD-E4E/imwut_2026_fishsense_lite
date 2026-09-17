"""Flat-port refraction: what it costs to ignore it.

**Scope.** This module answers exactly one question — how wrong is a length, and
a laser range, if an in-air calibration is used underwater with no refraction
correction at all? That is the number this paper needs to justify the corrective
optic at the housing port, and nothing more.

It is a deliberate subset of `fishsense_wuwnet/refraction.py` in the sibling
WUWNet repository, copied rather than imported so this paper stands alone. The
forward model below (`FlatPort` through `project_water_points`) and the pixel and
reconstruction helpers are **verbatim**, so the two cannot silently disagree; the
tests ported alongside pin them against the Pinax paper's own Table 1.

What is deliberately **not** here, and belongs to the WUWNet paper:

  * the Pinax correction itself (virtual centre of projection, `back_project_pinax`);
  * the in-water single-viewpoint calibration it is compared against
    (`fit_svp_model`, `svp_scene_radius`, `back_project_svp`);
  * `LayeredPort`, the multi-pane stack an already-waterproof camera inside a
    second enclosure really forms;
  * the housing-geometry, salinity and calibration-noise residual budgets.

So this module has one back-projection, `back_project_uncorrected`, and no way to
correct anything. That is the point.

Reference for the geometry:

> Łuczyński, Pfingsthorn & Birk, "The Pinax-model for accurate and efficient
> refraction correction of underwater cameras in flat-pane housings," Ocean
> Engineering 133 (2017) 9-22.

Conventions
-----------
- Camera centre of projection at the origin, optical axis +z into the scene.
- The port is rotationally symmetric about the optical axis, so every ray is
  described by a field angle `alpha` and an azimuth; no vector Snell is needed.
- Lens distortion is assumed already removed by the in-air calibration.
- Any single length unit, used consistently. The notebooks use metres.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

# Refraction indices used throughout Łuczyński et al. (2017) for field work.
SWEET_WATER = 1.333
SALTY_WATER = 1.342

# Bisection halves [0, pi/2] sixty times -> ~1.4e-18 rad, past float64 on that
# interval. Bisecting the ANGLE rather than a distance keeps this unit-free.
_BISECT_ITERS = 60


@dataclass(frozen=True)
class FlatPort:
    """Geometry of a single-pane housing, measured along the optical axis."""

    d0: float  # centre of projection -> inner glass surface
    d1: float  # glass thickness
    n_glass: float
    n_water: float
    n_air: float = 1.0

    @property
    def layers(self) -> Tuple[Tuple[float, float], ...]:
        """``((thickness, index), ...)`` from the camera outward."""
        return ((self.d0, self.n_air), (self.d1, self.n_glass))

    @property
    def d2(self) -> float:
        """Depth of the outer glass surface."""
        return self.d0 + self.d1


def snell(alpha, n_from, n_to):
    """Refraction angle for an incident field angle `alpha`. NaN past the critical angle."""
    s = (n_from / n_to) * np.sin(alpha)
    return np.where(np.abs(s) <= 1.0, np.arcsin(np.clip(s, -1.0, 1.0)), np.nan)


def exit_ray(alpha, port) -> Tuple[np.ndarray, np.ndarray]:
    """Where a camera ray at field angle `alpha` leaves the port, and its angle in water.

    Returns ``(r_exit, gamma)``: the radial offset from the optical axis at the
    outermost surface, and the ray's angle to the axis once in water. `gamma`
    depends only on the water index -- the port shifts a ray sideways but cannot
    change its final direction.
    """
    gamma = snell(alpha, port.n_air, port.n_water)
    r_exit = sum(
        thickness * np.tan(snell(alpha, port.n_air, index))
        for thickness, index in port.layers
    )
    return r_exit, gamma


def water_radius(alpha, z, port: FlatPort):
    """Radius from the optical axis at which the ray at `alpha` reaches depth `z`."""
    r_exit, gamma = exit_ray(alpha, port)
    return r_exit + (z - port.d2) * np.tan(gamma)


def field_angle(radius, z, port: FlatPort) -> np.ndarray:
    """Field angle of the camera ray that images a water point at ``(radius, z)``.

    The axial-camera forward projection. Because `water_radius` is strictly
    increasing in `alpha`, bisection on ``[0, pi/2)`` is unconditionally robust
    and needs no spurious-root filtering.
    """
    radius = np.asarray(radius, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.any(z <= port.d2):
        raise ValueError(
            f"all points must lie beyond the outer glass surface (z > {port.d2}); "
            f"got a minimum z of {np.min(z)}"
        )

    shape = np.broadcast_shapes(radius.shape, z.shape)
    lo = np.zeros(shape)
    hi = np.full(shape, 0.5 * np.pi - 1e-12)

    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        too_small = water_radius(mid, z, port) < radius
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)

    return 0.5 * (lo + hi)


def project_water_points(points, port: FlatPort) -> Tuple[np.ndarray, np.ndarray]:
    """Field angle and azimuth of the camera rays imaging 3D water points.

    `points` has shape ``(..., 3)`` in the camera frame.
    """
    points = np.asarray(points, dtype=float)
    radius = np.hypot(points[..., 0], points[..., 1])
    azimuth = np.arctan2(points[..., 1], points[..., 0])
    return field_angle(radius, points[..., 2], port), azimuth


def axis_crossing(alpha, port: FlatPort):
    """Depth at which a water ray, traced back into the housing, crosses the optical axis.

    For a true pinhole this would be the same depth for every `alpha`; the spread
    over the field of view is the "focus section".
    """
    r_exit, gamma = exit_ray(alpha, port)
    return port.d2 - r_exit / np.tan(gamma)


def focus_section(port: FlatPort, half_fov, n_rays=256) -> Tuple[float, float]:
    """``(midpoint, length)`` of the focus section over a field of view."""
    alpha = np.linspace(half_fov / n_rays, half_fov, n_rays)
    z = axis_crossing(alpha, port)
    return float(0.5 * (z.min() + z.max())), float(z.max() - z.min())


def optimal_d0(d1, n_glass, n_water, half_fov, n_air=1.0, n_rays=256):
    """Camera-to-glass distance that best collapses the axial camera to a pinhole.

    Returns ``(d0_star, virtual_cop, focus_length)``, all in the unit of `d1`.
    Kept here only to place the simulated housing at a defensible spacing; this
    paper does not use the virtual centre of projection to correct anything.

    The optimum depends on `half_fov`. Solved over the dimensionless ratio
    ``d0 / d1`` so the result does not depend on the length unit.
    """

    def cost(ratio):
        port = FlatPort(ratio * d1, d1, n_glass, n_water, n_air)
        return focus_section(port, half_fov, n_rays)[1] / d1

    res = minimize_scalar(
        cost, bounds=(1e-6, 1.0), method="bounded", options={"xatol": 1e-10}
    )

    d0_star = float(res.x * d1)
    virtual_cop, focus_length = focus_section(
        FlatPort(d0_star, d1, n_glass, n_water, n_air), half_fov, n_rays
    )
    return d0_star, virtual_cop, focus_length


def ray_directions(alpha, azimuth) -> np.ndarray:
    """Unit ray directions from a field angle and azimuth, shape ``(..., 3)``."""
    sin_a = np.sin(alpha)
    return np.stack(
        [
            sin_a * np.cos(azimuth),
            sin_a * np.sin(azimuth),
            np.cos(alpha) * np.ones_like(azimuth),
        ],
        axis=-1,
    )


def alpha_azimuth_to_pixel(alpha, azimuth, camera_intrinsics) -> np.ndarray:
    """Project a field angle and azimuth to pixel coordinates, shape ``(..., 2)``."""
    r = np.tan(alpha)
    return np.stack(
        [
            camera_intrinsics[0, 0] * r * np.cos(azimuth) + camera_intrinsics[0, 2],
            camera_intrinsics[1, 1] * r * np.sin(azimuth) + camera_intrinsics[1, 2],
        ],
        axis=-1,
    )


def pixel_to_alpha_azimuth(pixels, camera_intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Recover field angle and azimuth from pixel coordinates."""
    pixels = np.asarray(pixels, dtype=float)
    x = (pixels[..., 0] - camera_intrinsics[0, 2]) / camera_intrinsics[0, 0]
    y = (pixels[..., 1] - camera_intrinsics[1, 2]) / camera_intrinsics[1, 1]
    return np.arctan(np.hypot(x, y)), np.arctan2(y, x)


def reconstruct_points(directions, ray_origin, laser_origin, laser_axis):
    """Closest point on each camera ray to the laser line.

    `ray_origin` is carried so this stays character-identical to the WUWNet
    implementation; for the uncorrected pipeline it is always the origin, and
    with ``ray_origin = 0`` this reduces to `fishsense_imwut.camera.reconstruct_points`.

    Returns ``(world_points, denom)``, where `denom` is ``sin^2`` of the angle
    between the two lines -- the conditioning metric that bounds usable range.
    """
    u = np.asarray(directions, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    ray_origin = np.asarray(ray_origin, dtype=float)
    laser_origin = np.asarray(laser_origin, dtype=float)
    v = np.asarray(laser_axis, dtype=float)
    v = v / np.linalg.norm(v)

    w0 = ray_origin - laser_origin
    d = u @ v
    denom = 1.0 - d**2

    s = (d * (v @ w0) - (u @ w0)) / denom
    return ray_origin + s[..., None] * u, denom


def back_project_uncorrected(pixels, camera_intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """In-air intrinsics applied underwater with no refraction correction.

    The only pipeline in this module. Returns ``(ray_origin, directions)`` ready
    for `reconstruct_points`.
    """
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, camera_intrinsics)
    return np.zeros(3), ray_directions(alpha, azimuth)


def measure_length(pixels_head, pixels_tail, depth, back_project) -> np.ndarray:
    """Length of a fish from its head and tail pixels, given a range estimate.

    The measurement itself: the laser gives the range, both endpoints are
    back-projected to that depth, and the length is the distance between them.
    Error enters twice -- through the range and through the back-projection
    geometry -- and off-axis it is the geometry that dominates.
    """
    origin, head = back_project(pixels_head)
    _, tail = back_project(pixels_tail)

    depth = np.asarray(depth, dtype=float)
    head_point = origin + head * ((depth - origin[2]) / head[..., 2])[..., None]
    tail_point = origin + tail * ((depth - origin[2]) / tail[..., 2])[..., None]

    return np.linalg.norm(head_point - tail_point, axis=-1)


# --- the paper's scenario ---------------------------------------------------
#
# Figure 9 asks one question of the model above: what does a length read if the
# in-air calibration is used underwater with no correction at all? That is the
# number the corrective optic at the housing port exists to prevent.
#
# !! THE HOUSING IS NOT MEASURED !!
# GLASS_THICKNESS_M and N_GLASS are placeholders carried over from the WUWNet
# simulation, and the pane is assumed to sit at its optimal camera-to-glass
# spacing. The headline is driven by the index ratio and the field of view
# rather than by the pane, so it should be robust to these -- but it has not
# been checked against the real housing. Put calipers on the TG6 before this
# figure goes in a submission.

IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX = 4014, 3016
FOCAL_LENGTH_PX = 2850.0
GLASS_THICKNESS_M = 0.006  # 6 mm acrylic pane -- PLACEHOLDER, not measured
N_GLASS = 1.49  # acrylic
LASER_POSITION_M = np.array([-0.04, -0.11, 0.0])
LASER_DIRECTION = np.array([0.0, 0.0, 1.0])
FISH_LENGTH_M = 0.30
MEASUREMENT_DEPTH_M = 2.0
MAX_FRAME_FRACTION = 0.75  # of the half-frame; beyond this a fish is clipped


def camera_intrinsics(
    focal_px: float = FOCAL_LENGTH_PX,
    width: int = IMAGE_WIDTH_PX,
    height: int = IMAGE_HEIGHT_PX,
) -> np.ndarray:
    return np.array(
        [[focal_px, 0.0, width / 2], [0.0, focal_px, height / 2], [0.0, 0.0, 1.0]]
    )


def flat_port_cost(
    n_water: float = SALTY_WATER,
    glass_thickness_m: float = GLASS_THICKNESS_M,
    n_glass: float = N_GLASS,
    fish_length_m: float = FISH_LENGTH_M,
    depth_m: float = MEASUREMENT_DEPTH_M,
    n_points: int = 60,
) -> dict:
    """Length error against position in the frame, with no refraction correction.

    Returns `field_angle_deg`, `length_pct_error`, `range_pct_error` and the
    fitted `d0_star`.

    The range error is taken the way the pipeline would take it -- triangulated
    from the laser dot with the same uncorrected back-projection, not the true
    depth -- because that is what makes the two errors cancel on the optical
    axis. Ignoring refraction expands the scene transversely by `n_water` and
    shortens the range by very nearly its reciprocal; on axis those cancel, so a
    centred target measures correctly by accident and the error appears only off
    axis, where the angular compression stops being a pure scale.
    """
    K = camera_intrinsics()
    half_fov = np.arctan(
        np.hypot(IMAGE_WIDTH_PX / 2, IMAGE_HEIGHT_PX / 2) / FOCAL_LENGTH_PX
    )
    d0_star, _, _ = optimal_d0(glass_thickness_m, n_glass, n_water, half_fov)
    port = FlatPort(d0_star, glass_thickness_m, n_glass, n_water)

    back_project = lambda q: back_project_uncorrected(q, K)

    def to_pixels(points):
        alpha, azimuth = project_water_points(points, port)
        return alpha_azimuth_to_pixel(alpha, azimuth, K)

    laser_point = LASER_POSITION_M + depth_m * LASER_DIRECTION
    _, laser_dirs = back_project(to_pixels(laser_point))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), LASER_POSITION_M, LASER_DIRECTION
    )[0][2]

    half_frame = water_radius(
        np.arctan((IMAGE_WIDTH_PX / 2) / FOCAL_LENGTH_PX), depth_m, port
    )
    offsets = np.linspace(0.0, MAX_FRAME_FRACTION * half_frame, n_points)
    zeros = np.zeros_like(offsets)
    depths = np.full_like(offsets, depth_m)

    head = to_pixels(np.stack([offsets + fish_length_m / 2, zeros, depths], -1))
    tail = to_pixels(np.stack([offsets - fish_length_m / 2, zeros, depths], -1))
    measured = measure_length(head, tail, measured_depth, back_project)

    return dict(
        field_angle_deg=np.degrees(np.arctan(offsets / depth_m)),
        length_pct_error=100 * (measured - fish_length_m) / fish_length_m,
        range_pct_error=float(100 * (measured_depth - depth_m) / depth_m),
        d0_star=float(d0_star),
        # the frame's own half-width at this range, so a caller can say where in
        # the PICTURE a given field angle falls. Without it "19.6 degrees off
        # axis" is unreadable: the figure's x axis is the target's position in
        # frame, not its pose, and the two are easy to confuse (section 4.4's
        # figure is in degrees too, and means the opposite thing).
        half_frame_m=float(half_frame),
        depth_m=float(depth_m),
    )


def flat_port_error_field(
    n_water: float = SALTY_WATER,
    glass_thickness_m: float = GLASS_THICKNESS_M,
    n_glass: float = N_GLASS,
    fish_length_m: float = FISH_LENGTH_M,
    depth_m: float = MEASUREMENT_DEPTH_M,
    cell_px: float = 8.0,
) -> dict:
    """Length error as a field over the whole image, not just along one axis.

    `flat_port_cost` walks the target out along +x; this evaluates the same
    measurement wherever in the frame the target's centre falls, which is what a
    picture of the frame needs.

    **The field is not radially symmetric, and that is the physics.** The port
    is rotationally symmetric, so a *radius* is, but the target is not a point:
    it is held horizontal in the image, so near the left and right edges it lies
    along the radius and near the top and bottom it lies across one. Radial and
    tangential magnification differ under this distortion -- that difference is
    precisely why the error is not a scale error -- so a horizontal target reads
    differently at the side of the frame than at the top.

    `cell_px` is the sampling pitch in IMAGE PIXELS, and it is square on
    purpose. The fit/no-fit boundary is a smooth curve that can only land on a
    cell edge, so a coarse grid staircases it, and a grid with the same count on
    both axes staircases it unevenly -- the frame is 4:3, so equal counts give
    cells half again as wide as they are tall. Square cells at 8 px put the
    steps below the resolution of a column-width figure.

    Returns `error_pct` (NaN where a target of this length would not fit),
    `extent` in pixels for `imshow`, and the scalars `range_pct_error` and
    `budget_crossing_px`.
    """
    K = camera_intrinsics()
    W, H = IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX
    half_fov = np.arctan(np.hypot(W / 2, H / 2) / FOCAL_LENGTH_PX)
    d0_star, _, _ = optimal_d0(glass_thickness_m, n_glass, n_water, half_fov)
    port = FlatPort(d0_star, glass_thickness_m, n_glass, n_water)

    back_project = lambda q: back_project_uncorrected(q, K)

    def to_pixels(points):
        alpha, azimuth = project_water_points(points, port)
        return alpha_azimuth_to_pixel(alpha, azimuth, K)

    # the range the pipeline actually has, from the laser dot, uncorrected
    laser_point = LASER_POSITION_M + depth_m * LASER_DIRECTION
    _, laser_dirs = back_project(to_pixels(laser_point))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), LASER_POSITION_M, LASER_DIRECTION
    )[0][2]

    # every pixel is a camera ray; follow it to `depth_m` to place the target
    u = np.linspace(0.0, float(W), int(round(W / cell_px)) + 1)
    v = np.linspace(0.0, float(H), int(round(H / cell_px)) + 1)
    uu, vv = np.meshgrid(u, v)
    alpha, azimuth = pixel_to_alpha_azimuth(np.stack([uu, vv], -1), K)
    radius = water_radius(alpha, depth_m, port)
    cx = radius * np.cos(azimuth)
    cy = radius * np.sin(azimuth)
    z = np.full_like(cx, depth_m)

    half = fish_length_m / 2
    head_px = to_pixels(np.stack([cx + half, cy, z], -1))
    tail_px = to_pixels(np.stack([cx - half, cy, z], -1))
    measured = measure_length(head_px, tail_px, measured_depth, back_project)
    error = 100 * (measured - fish_length_m) / fish_length_m

    inside = np.ones_like(error, dtype=bool)
    for px in (head_px, tail_px):
        inside &= (px[..., 0] >= 0) & (px[..., 0] <= W)
        inside &= (px[..., 1] >= 0) & (px[..., 1] <= H)
    error = np.where(inside, error, np.nan)

    # where the budget is first crossed along the horizontal centre line
    mid = error[np.argmin(np.abs(v - H / 2)), :]
    right = mid[u >= W / 2]
    ur = u[u >= W / 2]
    ok = np.isfinite(right)
    crossing = (float(np.interp(15.0, right[ok], ur[ok]))
                if ok.any() and np.nanmax(right) >= 15.0 else float("nan"))

    return dict(
        error_pct=error,
        extent=(0.0, float(W), float(H), 0.0),
        range_pct_error=float(100 * (measured_depth - depth_m) / depth_m),
        budget_crossing_px=crossing,
        image_size_px=(W, H),
    )
