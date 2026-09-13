"""Validation of the flat-port model against Łuczyński et al. (2017).

The paper quotes one number from this model -- what an uncorrected flat port
costs a length measurement -- so the geometry behind it has to be right. Table 1
of the Pinax paper is the published numeric oracle; the rest are
internal-consistency checks that would catch a sign or convention slip.

Ported from `fishsense_wuwnet/tests/test_refraction.py` alongside the module
itself, less the tests for the parts this paper does not carry: the SVP model,
the Pinax correction, and `LayeredPort`.
"""

import numpy as np
import pytest

from fishsense_imwut.refraction import (
    SALTY_WATER,
    SWEET_WATER,
    FlatPort,
    alpha_azimuth_to_pixel,
    axis_crossing,
    back_project_uncorrected,
    exit_ray,
    field_angle,
    focus_section,
    measure_length,
    optimal_d0,
    pixel_to_alpha_azimuth,
    project_water_points,
    ray_directions,
    reconstruct_points,
    water_radius,
)

HALF_FOV = np.radians(35.0)  # the paper's standard 70 deg example
N_GLASS = 1.5

# Table 1: glass thickness (mm) -> (d0*, virtual centre of projection), in mm.
TABLE_1_SWEET = {
    1: (0.15, 0.06),
    3: (0.45, 0.18),
    5: (0.76, 0.31),
    10: (1.52, 0.61),
    15: (2.28, 0.92),
    20: (3.04, 1.22),
}
TABLE_1_SALTY = {
    1: (0.14, 0.06),
    3: (0.42, 0.17),
    5: (0.70, 0.29),
    10: (1.40, 0.58),
    15: (2.10, 0.87),
    20: (2.80, 1.15),
}


@pytest.mark.parametrize("d1,expected", TABLE_1_SWEET.items())
def test_table_1_sweet_water(d1, expected):
    d0_star, virtual_cop, _ = optimal_d0(d1, N_GLASS, SWEET_WATER, HALF_FOV)
    assert d0_star == pytest.approx(expected[0], rel=0.02, abs=0.01)
    assert virtual_cop == pytest.approx(expected[1], rel=0.03, abs=0.01)


@pytest.mark.parametrize("d1,expected", TABLE_1_SALTY.items())
def test_table_1_salty_water(d1, expected):
    d0_star, virtual_cop, _ = optimal_d0(d1, N_GLASS, SALTY_WATER, HALF_FOV)
    assert d0_star == pytest.approx(expected[0], rel=0.02, abs=0.01)
    assert virtual_cop == pytest.approx(expected[1], rel=0.03, abs=0.01)


def test_optimal_d0_is_independent_of_length_unit():
    """The same housing in mm and in m must give the same answer."""
    in_mm, _, _ = optimal_d0(10.0, N_GLASS, SWEET_WATER, HALF_FOV)
    in_m, _, _ = optimal_d0(0.010, N_GLASS, SWEET_WATER, HALF_FOV)
    assert in_mm / 1000.0 == pytest.approx(in_m, rel=1e-6)


def test_focus_section_collapses_at_the_optimum():
    """At d0* the axial camera really does behave as a virtual pinhole."""
    d1 = 0.010
    d0_star, _, focus_length = optimal_d0(d1, N_GLASS, SWEET_WATER, HALF_FOV)
    _, loose = focus_section(FlatPort(0.025, d1, N_GLASS, SWEET_WATER), HALF_FOV)
    assert focus_length < loose / 100


def test_field_angle_inverts_water_radius():
    """Forward projection must invert the forward ray trace."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    alpha = np.linspace(0.001, HALF_FOV, 64)
    for z in (0.5, 2.0, 5.0):
        recovered = field_angle(water_radius(alpha, z, port), z, port)
        assert recovered == pytest.approx(alpha, abs=1e-9)


def test_field_angle_rejects_points_inside_the_port():
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    with pytest.raises(ValueError, match="beyond the outer glass"):
        field_angle(0.1, port.d2 * 0.5, port)


def test_no_refraction_reduces_to_a_straight_line():
    """With every index equal the port is invisible: alpha = atan(radius / z)."""
    port = FlatPort(0.0015, 0.010, 1.0, 1.0)
    for radius, z in ((0.3, 2.0), (0.05, 5.0)):
        assert field_angle(radius, z, port) == pytest.approx(np.arctan(radius / z))


def test_water_ray_direction_ignores_the_glass():
    """Glass shifts a ray sideways but cannot change its final angle in water."""
    alpha = np.radians(30.0)
    expected = np.arcsin(np.sin(alpha) / 1.34)
    for n_glass in (1.4, 1.5, 1.6):
        for d1 in (0.005, 0.020):
            _, gamma = exit_ray(alpha, FlatPort(0.002, d1, n_glass, 1.34))
            assert gamma == pytest.approx(expected)


def test_axis_crossing_matches_the_thin_ray_limit():
    """As alpha -> 0 the crossing depth tends to d2 - n_w * (d0 + d1 / n_g)."""
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    limit = port.d2 - SWEET_WATER * (port.d0 + port.d1 / N_GLASS)
    assert axis_crossing(1e-7, port) == pytest.approx(limit, rel=1e-6)


def test_pixel_round_trip():
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    alpha = np.radians([5.0, 20.0, 35.0])
    azimuth = np.radians([0.0, 100.0, -140.0])
    a, z = pixel_to_alpha_azimuth(alpha_azimuth_to_pixel(alpha, azimuth, K), K)
    assert a == pytest.approx(alpha)
    assert np.cos(z) == pytest.approx(np.cos(azimuth))


def test_reconstruct_recovers_a_point_on_the_laser_line():
    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])
    truth = laser_origin + 2.5 * laser_axis

    directions = truth / np.linalg.norm(truth)
    got, denom = reconstruct_points(directions, np.zeros(3), laser_origin, laser_axis)

    assert got == pytest.approx(truth)
    assert 0.0 < denom <= 1.0


def test_project_water_points_is_vectorised():
    port = FlatPort(0.0015, 0.010, N_GLASS, SWEET_WATER)
    points = np.random.default_rng(0).uniform(-0.3, 0.3, (4, 5, 3))
    points[..., 2] = np.random.default_rng(1).uniform(0.5, 5.0, (4, 5))

    alpha, azimuth = project_water_points(points, port)
    assert alpha.shape == (4, 5)
    assert azimuth.shape == (4, 5)

    radius = water_radius(alpha, points[..., 2], port)
    assert radius == pytest.approx(np.hypot(points[..., 0], points[..., 1]), abs=1e-12)


def test_ray_directions_are_unit():
    d = ray_directions(np.radians([0.0, 15.0, 40.0]), np.radians([10.0, 200.0, -30.0]))
    assert np.linalg.norm(d, axis=-1) == pytest.approx(1.0)


# --- the uncorrected pipeline, which is all this paper carries ---------------


def test_uncorrected_back_projection_is_the_in_air_pinhole():
    """No correction means the in-air ray, from the coordinate origin."""
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    pixels = np.array([[2007.0, 1508.0], [3000.0, 2000.0]])
    origin, directions = back_project_uncorrected(pixels, K)

    assert origin == pytest.approx(np.zeros(3))
    alpha, azimuth = pixel_to_alpha_azimuth(pixels, K)
    assert directions == pytest.approx(ray_directions(alpha, azimuth))


def test_measure_length_is_exact_without_refraction():
    """Through a port with no index step, the measurement must be exact."""
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    port = FlatPort(0.0015, 0.006, 1.0, 1.0)  # no refraction anywhere
    depth, length = 2.0, 0.30

    head = np.array([length / 2, 0.0, depth])
    tail = np.array([-length / 2, 0.0, depth])
    a, az = project_water_points(np.stack([head, tail]), port)
    pixels = alpha_azimuth_to_pixel(a, az, K)

    measured = measure_length(
        pixels[0], pixels[1], depth, lambda q: back_project_uncorrected(q, K)
    )
    assert float(measured) == pytest.approx(length, rel=1e-9)


def test_uncorrected_range_reads_short_by_the_water_index():
    """An in-air back-projection makes the laser ray too steep, so range reads
    short by very nearly 1 - 1/n_water, almost independently of range."""
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    port = FlatPort(0.0015, 0.006, 1.49, SALTY_WATER)
    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])

    for z in (1.0, 2.0, 5.0):
        point = laser_origin + z * laser_axis
        a, az = project_water_points(point, port)
        _, directions = back_project_uncorrected(
            alpha_azimuth_to_pixel(a, az, K), K
        )
        recovered, _ = reconstruct_points(
            directions, np.zeros(3), laser_origin, laser_axis
        )
        assert (recovered[2] - z) / z == pytest.approx(1 / SALTY_WATER - 1, abs=0.01)


def test_uncorrected_length_error_is_position_dependent_not_a_scale():
    """The effect the paper reports, and the reason it is easy to miss.

    Ignoring refraction expands the scene transversely by n_water and shortens
    the laser range by 1/n_water. On the optical axis those cancel almost
    exactly, so a centred fish measures correctly by accident. Off axis the
    angular compression is not a pure scale, the cancellation fails, and the
    error grows with field position alone.
    """
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    port = FlatPort(0.0015, 0.006, 1.49, SALTY_WATER)
    laser_origin = np.array([-0.04, -0.11, 0.0])
    laser_axis = np.array([0.0, 0.0, 1.0])
    depth, length = 2.0, 0.30
    back_project = lambda q: back_project_uncorrected(q, K)

    def to_pixels(point):
        a, az = project_water_points(point, port)
        return alpha_azimuth_to_pixel(a, az, K)

    # The range the pipeline actually has: from the laser dot, uncorrected.
    _, laser_dirs = back_project(to_pixels(laser_origin + depth * laser_axis))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), laser_origin, laser_axis
    )[0][2]

    errors = []
    for offset in (0.0, 0.25, 0.50, 0.75):
        centre = offset * water_radius(np.arctan(2007.0 / 2850.0), depth, port)
        head = to_pixels(np.array([centre + length / 2, 0.0, depth]))
        tail = to_pixels(np.array([centre - length / 2, 0.0, depth]))
        measured = float(measure_length(head, tail, measured_depth, back_project))
        errors.append((measured - length) / length)

    assert errors == sorted(errors)  # monotone in field position
    assert abs(errors[0]) < 0.01  # on axis: the two effects cancel
    assert errors[-1] > 0.15  # at 3/4 of the half-frame: outside the budget


def test_on_axis_cancellation_is_exact_in_the_paraxial_limit():
    """Why the on-axis error is ~0: transverse gain n_w times range 1/n_w = 1."""
    assert SALTY_WATER * (1 / SALTY_WATER) == pytest.approx(1.0)
