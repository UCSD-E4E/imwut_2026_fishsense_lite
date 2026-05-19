"""Flat-port refraction simulation (Section 3.3.2).

Models the camera housing as two planar refractive interfaces perpendicular
to the optical axis: air -> acrylic -> water. Given a 3D point in water,
solves for the air-side ray direction the pinhole camera observes, then
quantifies the depth error that results from applying the pinhole laser
triangulation of Section 3.2 to the refracted ray without accounting for
refraction. This is the analysis that motivates the Backscatter M52
corrective optic.

Ported from the MATLAB `fishsense-optics-simulator` repo
(https://github.com/UCSD-E4E/fishsense-optics-simulator).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class FlatPort:
    """Parameters of a flat-port underwater housing.

    Distances are in millimeters and refractive indices are unitless.
    Defaults match the MATLAB ``testingparameters.m`` configuration used
    to generate the box plot in the paper.
    """

    d_air: float = 2.0
    d_glass: float = 11.56
    mu_air: float = 1.0
    mu_glass: float = 1.495
    mu_water: float = 1.34995


def snell_refract(
    v_in: np.ndarray, normal: np.ndarray, mu_in: float, mu_out: float
) -> np.ndarray:
    """Refract a unit direction ``v_in`` at an interface with unit
    ``normal`` going from medium ``mu_in`` to ``mu_out``.

    Returns the refracted unit direction. Raises if total internal
    reflection occurs.
    """
    n = normal / np.linalg.norm(normal)
    cos_i = v_in @ n
    kk = mu_in**2 * cos_i**2 - (mu_in**2 - mu_out**2) * (v_in @ v_in)
    if kk < 0:
        raise ValueError("total internal reflection")
    a = mu_in / mu_out
    b = (-mu_in * cos_i - np.sqrt(kk)) / mu_out
    return a * v_in + b * n


def solve_forward_projection(
    fish_point: np.ndarray, port: FlatPort = FlatPort()
) -> np.ndarray:
    """Given a 3D fish point in water, return the unit air-side ray
    direction the camera observes.

    The camera is at the origin; interfaces are at ``z = d_air`` and
    ``z = d_air + d_glass`` with normal along the optical axis. The
    problem is planar in the plane-of-refraction (POR) containing the
    optical axis and the fish point, reducing to a 1D root-find on the
    radial entry point at the first interface.
    """
    x_fish, y_fish, z_fish = fish_point
    u_fish = np.hypot(x_fish, y_fish)

    if u_fish < 1e-12:
        return np.array([0.0, 0.0, 1.0])

    d_air = port.d_air
    d_glass = port.d_glass
    z_water_start = d_air + d_glass
    mu_a, mu_g, mu_w = port.mu_air, port.mu_glass, port.mu_water

    def residual(x1: float) -> float:
        sin1 = x1 / np.hypot(x1, d_air)
        sin2 = mu_a * sin1 / mu_g
        if abs(sin2) >= 1.0:
            return np.inf
        tan2 = sin2 / np.sqrt(1.0 - sin2**2)
        x2 = x1 + d_glass * tan2
        sin3 = mu_g * sin2 / mu_w
        if abs(sin3) >= 1.0:
            return np.inf
        tan3 = sin3 / np.sqrt(1.0 - sin3**2)
        u_predicted = x2 + (z_fish - z_water_start) * tan3
        return u_predicted - u_fish

    # Upper bracket: angle-of-incidence near 90° in air.
    x1_hi = d_air * np.tan(np.radians(89.0))
    x1_sol = brentq(residual, 1e-10, x1_hi, xtol=1e-12)

    u_hat = np.array([x_fish, y_fish, 0.0]) / u_fish
    direction = x1_sol * u_hat + np.array([0.0, 0.0, d_air])
    return direction / np.linalg.norm(direction)


def trace_refracted_ray(
    v_air: np.ndarray, z_target: float, port: FlatPort = FlatPort()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-trace a unit ``v_air`` direction from the camera origin
    through both interfaces into water, returning the intersection points
    at plane 1, plane 2, and at depth ``z_target``.
    """
    v_air = v_air / np.linalg.norm(v_air)
    normal_into_glass = np.array([0.0, 0.0, -1.0])
    normal_into_water = np.array([0.0, 0.0, -1.0])

    t1 = port.d_air / v_air[2]
    p1 = t1 * v_air

    v_glass = snell_refract(v_air, normal_into_glass, port.mu_air, port.mu_glass)
    t2 = port.d_glass / v_glass[2]
    p2 = p1 + t2 * v_glass

    v_water = snell_refract(v_glass, normal_into_water, port.mu_glass, port.mu_water)
    t3 = (z_target - p2[2]) / v_water[2]
    p3 = p2 + t3 * v_water

    return p1, p2, p3


def estimate_depth_pinhole(
    v_air: np.ndarray, laser_origin: np.ndarray, laser_direction: np.ndarray
) -> float:
    """Apply the pinhole laser-triangulation estimator (Equation 5) to
    the refracted air-side direction, ignoring refraction. Returns the
    estimated depth ``P_z``.

    ``v_air`` is the unit camera ray direction recovered from the image
    pixel via ``K^{-1} q`` and normalization. Under the pinhole model
    assumed by Section 3.2, this *is* the geometric ray from the camera
    into the scene — but in the presence of refraction, it is only the
    air-side direction, so the reconstructed depth is biased.
    """
    v = v_air / np.linalg.norm(v_air)
    alpha = laser_direction / np.linalg.norm(laser_direction)
    ell = np.asarray(laser_origin, dtype=float)

    r = v / v[2]
    alpha_dot_r = alpha @ r
    denom = (r @ r) - alpha_dot_r**2
    numer = (ell @ r) - (alpha @ ell) * alpha_dot_r
    return numer / denom


def depth_error_percent(
    fish_depth_mm: float,
    laser_origin: np.ndarray,
    laser_direction: np.ndarray,
    port: FlatPort = FlatPort(),
) -> float:
    """Percent depth error from applying the pinhole geometry to a
    single refracted laser-fish configuration at depth ``fish_depth_mm``.

    The "fish" is placed at the point where the laser ray (in the
    camera/world frame) crosses ``z = fish_depth_mm``. The air-side ray
    the camera would actually observe is computed via
    ``solve_forward_projection``; that ray is then fed into
    ``estimate_depth_pinhole`` to get the depth the pinhole model
    recovers. Percent error is reported against the true depth.
    """
    alpha = laser_direction
    scale = (fish_depth_mm - laser_origin[2]) / alpha[2]
    fish_point = laser_origin + scale * alpha

    v_air = solve_forward_projection(fish_point, port)
    z_est = estimate_depth_pinhole(v_air, laser_origin, laser_direction)

    return abs(z_est - fish_point[2]) / fish_point[2] * 100.0


# Eight simulated laser poses used for the box plot in the paper.
# Origins and directions are copied verbatim from
# testingparameters.m in fishsense-optics-simulator. Origins are
# converted to millimeters (the MATLAB code multiplies by 1000).
LASER_CONFIGURATIONS_MM = [
    ("cam2", np.array([-0.03090316, -0.0995172, 0.0]) * 1000.0,
        np.array([0.03036567, 0.03888114, 0.99878215])),
    ("cam3", np.array([-0.02938965, -0.09794052, 0.0]) * 1000.0,
        np.array([-0.00760419, 0.01835205, 0.9998026])),
    ("cam4", np.array([-0.03088575, -0.09819288, 0.0]) * 1000.0,
        np.array([0.01611468, 0.02188467, 0.99963054])),
    ("cam5", np.array([-0.02968504, -0.09813952, 0.0]) * 1000.0,
        np.array([0.01520644, 0.02237695, 0.99963369])),
    ("cam6", np.array([-0.02991985, -0.09841192, 0.0]) * 1000.0,
        np.array([0.02536296, 0.00915088, 0.99963614])),
    ("cam7", np.array([-0.03142921, -0.09906525, 0.0]) * 1000.0,
        np.array([0.02814415, 0.01302515, 0.99951866])),
    ("pool", np.array([-0.02991158, -0.09781963, 0.0]) * 1000.0,
        np.array([0.00877752, 0.04130509, 0.99910793])),
    ("dive", np.array([-0.03163025, -0.10136059, 0.0]) * 1000.0,
        np.array([0.01214438, 0.02188269, 0.99968678])),
]
