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
    flat_port_cost,
    flat_port_error_field,
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


# --- the numbers §4.5 quotes -----------------------------------------------
#
# §4.5 ("What the port correction buys") is the paper's one simulated result,
# so unlike every other section it has no CSV pinning it. These pin it to
# `flat_port_cost` instead. The robustness tests matter more than the headline:
# the housing was never opened, so the pane's thickness and index and the
# camera-to-glass spacing are assumed, and the section claims in bold that the
# result does not rest on them.


def _cost_summary(port, fish=0.30, depth=2.0, n_points=200):
    """`(on_axis, edge, edge_deg, budget_crossing_deg, range_pct)` for any port.

    A re-implementation of `flat_port_cost` that takes the port rather than
    constructing it at the optimal spacing, so the assumptions that function
    makes can be varied. Kept here rather than in the module because varying
    them is a test concern; `test_summary_agrees_with_flat_port_cost` holds the
    two together.
    """
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    laser_origin, laser_axis = np.array([-0.04, -0.11, 0.0]), np.array([0.0, 0.0, 1.0])
    back_project = lambda q: back_project_uncorrected(q, K)

    def to_pixels(points):
        a, az = project_water_points(points, port)
        return alpha_azimuth_to_pixel(a, az, K)

    _, laser_dirs = back_project(to_pixels(laser_origin + depth * laser_axis))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), laser_origin, laser_axis
    )[0][2]

    half_frame = water_radius(np.arctan(2007.0 / 2850.0), depth, port)
    offsets = np.linspace(0.0, 0.75 * half_frame, n_points)
    zeros, depths = np.zeros_like(offsets), np.full_like(offsets, depth)
    head = to_pixels(np.stack([offsets + fish / 2, zeros, depths], -1))
    tail = to_pixels(np.stack([offsets - fish / 2, zeros, depths], -1))
    measured = measure_length(head, tail, measured_depth, back_project)

    angles = np.degrees(np.arctan(offsets / depth))
    errors = 100 * (measured - fish) / fish
    return (
        errors[0],
        errors[-1],
        angles[-1],
        float(np.interp(15.0, errors, angles)),
        100 * (measured_depth - depth) / depth,
    )


def _paper_port(d1=0.006, n_glass=1.49, n_water=SALTY_WATER, d0=None):
    if d0 is None:
        half_fov = np.arctan(np.hypot(4014 / 2, 3016 / 2) / 2850.0)
        d0 = optimal_d0(d1, n_glass, n_water, half_fov)[0]
    return FlatPort(d0, d1, n_glass, n_water)


def _error_at_frame_fraction(fraction, fish=0.30, depth=2.0):
    """Percent length error for a target centred `fraction` of the way to the edge."""
    K = np.array([[2850.0, 0, 2007.0], [0, 2850.0, 1508.0], [0, 0, 1.0]])
    laser_origin, laser_axis = np.array([-0.04, -0.11, 0.0]), np.array([0.0, 0.0, 1.0])
    port = _paper_port()
    back_project = lambda q: back_project_uncorrected(q, K)

    def to_pixels(point):
        a, az = project_water_points(point, port)
        return alpha_azimuth_to_pixel(a, az, K)

    _, laser_dirs = back_project(to_pixels(laser_origin + depth * laser_axis))
    measured_depth = reconstruct_points(
        laser_dirs, np.zeros(3), laser_origin, laser_axis
    )[0][2]

    centre = fraction * water_radius(np.arctan(2007.0 / 2850.0), depth, port)
    head = to_pixels(np.array([centre + fish / 2, 0.0, depth]))
    tail = to_pixels(np.array([centre - fish / 2, 0.0, depth]))
    measured = float(measure_length(head, tail, measured_depth, back_project))
    return 100 * (measured - fish) / fish


def test_summary_agrees_with_flat_port_cost():
    """The helper above and the module's own entry point are the same scenario."""
    r = flat_port_cost()
    on_axis, edge, edge_deg, crossing, range_pct = _cost_summary(_paper_port())
    a, e = r["field_angle_deg"], r["length_pct_error"]
    assert on_axis == pytest.approx(e[0], abs=0.01)
    assert edge == pytest.approx(e[-1], abs=0.01)
    assert edge_deg == pytest.approx(a[-1], abs=0.01)
    assert range_pct == pytest.approx(r["range_pct_error"], abs=0.01)
    assert crossing == pytest.approx(np.interp(15.0, e, a), abs=0.05)


def test_section_4_5_headline_numbers():
    """Every figure §4.5 states in prose, and Figure 9's caption."""
    r = flat_port_cost()
    a, e = r["field_angle_deg"], r["length_pct_error"]

    # "+0.1 % for a 300 mm target at 2 m" -- the accidental on-axis agreement.
    assert e[0] == pytest.approx(0.1, abs=0.05)

    # "-25.6 % ... against the -25.5 % the ratio alone predicts"
    assert r["range_pct_error"] == pytest.approx(-25.6, abs=0.1)
    assert r["range_pct_error"] == pytest.approx(
        100 * (1 / SALTY_WATER - 1), abs=0.2
    )

    # "a quarter of the way to the frame edge ... +1.8 %; half way, +7.3 %"
    # Evaluated exactly rather than interpolated off the plotted samples: the
    # curve is convex, so linear interpolation reads low by about 0.05 points.
    assert _error_at_frame_fraction(0.25) == pytest.approx(1.8, abs=0.05)
    assert _error_at_frame_fraction(0.50) == pytest.approx(7.3, abs=0.05)

    # The plotted span is three quarters of the half-frame; recover the fraction
    # from the angles rather than assuming a sample count.
    offsets = np.tan(np.radians(a))
    fraction = 0.75 * offsets / offsets[-1]

    # "crosses the 15 % budget at 18 deg -- seven tenths of the way out"
    crossing = np.interp(15.0, e, a)
    assert crossing == pytest.approx(18.3, abs=0.1)
    assert np.interp(crossing, a, fraction) == pytest.approx(0.70, abs=0.02)

    # "reaches +17.7 % at 19.6 deg, beyond which a 300 mm target no longer fits"
    assert e[-1] == pytest.approx(17.7, abs=0.1)
    assert a[-1] == pytest.approx(19.6, abs=0.1)

    # "larger at the frame edge than the -13.4 % a 30 deg pose costs"
    assert 100 * (np.cos(np.radians(30)) - 1) == pytest.approx(-13.4, abs=0.1)
    assert e[-1] > abs(100 * (np.cos(np.radians(30)) - 1))


def test_headline_survives_the_unmeasured_housing():
    """§4.5's bold claim: the result does not rest on the parameters we guessed.

    Pane thickness, glass index and camera-to-glass spacing were never measured.
    The section claims the edge error stays within +16.7 to +17.7 % and the
    budget crossing within 18.3 to 19.1 deg across every plausible value, because
    a pane displaces a ray sideways but cannot change its direction in water --
    so the answer is set by the index ratio and the field of view, both known.
    """
    edges, crossings = [], []
    for d1 in (0.002, 0.006, 0.010, 0.015, 0.020):
        _, edge, _, crossing, _ = _cost_summary(_paper_port(d1=d1))
        edges.append(edge)
        crossings.append(crossing)
    for n_glass in (1.46, 1.49, 1.52, 1.62):
        _, edge, _, crossing, _ = _cost_summary(_paper_port(n_glass=n_glass))
        edges.append(edge)
        crossings.append(crossing)
    for d0 in (0.002, 0.010, 0.030, 0.050, 0.080):  # 0.080 is absurd, deliberately
        _, edge, _, crossing, _ = _cost_summary(_paper_port(d0=d0))
        edges.append(edge)
        crossings.append(crossing)

    assert min(edges) == pytest.approx(16.7, abs=0.1)
    assert max(edges) == pytest.approx(17.7, abs=0.1)
    assert min(crossings) == pytest.approx(18.3, abs=0.1)
    assert max(crossings) == pytest.approx(19.1, abs=0.1)


def test_headline_survives_fresh_water():
    """"Fresh water in place of salt gives +17.5 % and a crossing at 18.6 deg"."""
    _, edge, _, crossing, _ = _cost_summary(_paper_port(n_water=SWEET_WATER))
    assert edge == pytest.approx(17.5, abs=0.1)
    assert crossing == pytest.approx(18.6, abs=0.1)


class _Stack:
    """A waterproof camera inside a second housing: air, pane, air, pane.

    §4.5 reports this case but the module deliberately carries no `LayeredPort`
    -- the multi-pane treatment belongs to the companion paper. Everything in
    `refraction` reaches the port only through `layers`, `d2`, `n_air` and
    `n_water`, so the stack is expressible here without widening that boundary.
    """

    def __init__(self, d0, t1, n1, gap, t2, n2, n_water, n_air=1.0):
        self.d0, self.n_water, self.n_air = d0, n_water, n_air
        self._panes = ((d0, n_air), (t1, n1), (gap, n_air), (t2, n2))

    @property
    def layers(self):
        return self._panes

    @property
    def d2(self):
        return sum(thickness for thickness, _ in self._panes)


def test_two_pane_stack_does_not_move_the_headline():
    """"two 6 mm panes separated by 5 to 30 mm of air -- +17.4 to +17.7 %"."""
    half_fov = np.arctan(np.hypot(4014 / 2, 3016 / 2) / 2850.0)
    d0 = optimal_d0(0.006, 1.49, SALTY_WATER, half_fov)[0]

    edges, crossings = [], []
    for gap in (0.005, 0.015, 0.030):
        _, edge, _, crossing, _ = _cost_summary(
            _Stack(d0, 0.006, 1.49, gap, 0.006, 1.49, SALTY_WATER)
        )
        edges.append(edge)
        crossings.append(crossing)

    assert min(edges) == pytest.approx(17.4, abs=0.1)
    assert max(edges) == pytest.approx(17.7, abs=0.1)
    assert min(crossings) == pytest.approx(18.3, abs=0.1)
    assert max(crossings) == pytest.approx(18.6, abs=0.1)


def test_flat_port_cost_reports_the_frame_it_is_measured_against():
    """The x axis of Figure 9 is a position in the picture, not a pose, and
    without the frame's own half-width nobody can tell which.

    Section 4.4's figure is also in degrees and means the fish's angle to the
    image plane -- the opposite quantity. These two keys are what let the figure
    carry a second axis reading centre-to-edge, so they are pinned rather than
    left as an incidental part of the return value.
    """
    r = flat_port_cost()
    assert r["depth_m"] == pytest.approx(2.0)
    assert r["half_frame_m"] == pytest.approx(0.950, abs=0.005)

    edge_deg = np.degrees(np.arctan(r["half_frame_m"] / r["depth_m"]))
    assert edge_deg == pytest.approx(25.4, abs=0.1)

    # the plotted span stops short of the frame edge: past three quarters of the
    # half-width a 300 mm target no longer fits, so the curve is not the worst case
    assert r["field_angle_deg"][-1] == pytest.approx(19.6, abs=0.1)
    assert r["field_angle_deg"][-1] < edge_deg

    # and the axis really is a position: its last point is MAX_FRAME_FRACTION
    # of the half-width, in metres, not an angle of incidence
    offset = r["depth_m"] * np.tan(np.radians(r["field_angle_deg"][-1]))
    assert offset / r["half_frame_m"] == pytest.approx(0.75, abs=0.01)


def test_the_error_field_is_not_radially_symmetric():
    """Section 4.5's argument, and the reason Figure 9b exists.

    The port is rotationally symmetric, so a radius is; the target is not a
    point. Held horizontal it lies along a radius at the side of the frame and
    across one at the top, and radial and tangential magnification differ. If
    this test ever passes trivially -- side == top -- the field has collapsed to
    a scale error and the paper's claim that no calibration can absorb it is
    wrong.
    """
    f = flat_port_error_field(cell_px=16.0)
    e = f["error_pct"]
    h, w = e.shape

    centre = float(np.nanmin(np.abs(e)))
    top = float(np.nanmax(e[:, w // 2]))
    side = float(np.nanmax(e[h // 2, :]))

    assert centre == pytest.approx(0.09, abs=0.05)
    assert top == pytest.approx(6.1, abs=0.4)
    assert side == pytest.approx(23.0, abs=0.6)
    assert side > 3 * top, "a horizontal target must read far worse at the side"

    # the corner is worse than anything the 1-D slice of Figure 9 reaches
    assert float(np.nanmax(e)) > flat_port_cost()["length_pct_error"][-1]


def test_the_error_field_samples_square_cells():
    """Why the mask edge is smooth: the fit/no-fit boundary is a curve that can
    only land on a cell edge, so the cells have to be square and small. Equal
    counts on both axes would make them 4:3 and staircase it unevenly."""
    f = flat_port_error_field(cell_px=10.0)
    h, w = f["error_pct"].shape
    W, H = f["image_size_px"]
    assert W / (w - 1) == pytest.approx(10.0, abs=0.2)
    assert H / (h - 1) == pytest.approx(10.0, abs=0.2)

    # NaN only ever means "would not fit", never "off the end of the grid"
    assert np.isfinite(f["error_pct"][h // 2, w // 2])


def test_removing_the_index_step_removes_the_error():
    """Figure 9b's right panel: the air path the corrective optic restores.

    Deliberately close to tautological -- with no water interface there is no
    refraction error to have -- and pinned anyway, because the panel's whole job
    is to carry that magnitude beside the left one. If this ever stops being
    ~zero, the model has grown a term that is not refraction.

    Note what this is NOT: the Pinax correction and the in-water single-viewpoint
    calibration belong to the companion paper and are absent from this module by
    design. Nothing here corrects anything; it only takes the interface away.
    """
    corrected = flat_port_error_field(n_water=1.0, cell_px=16.0)
    assert np.nanmax(corrected["error_pct"]) < 0.1
    assert abs(corrected["range_pct_error"]) < 0.2

    uncorrected = flat_port_error_field(cell_px=16.0)
    assert np.nanmax(uncorrected["error_pct"]) > 25.0


# --- the range error the length figure cancels against -----------------------


def test_the_flat_port_range_error_is_the_snell_factor():
    """§4.5's Figure 9c: an uncorrected flat port reads a quarter short in
    range, and the error is a scale rather than a range-dependent term.

    Pinned because Figure 9's length result depends on it entirely -- the
    transverse expansion cancels this shortening on axis -- so if this stopped
    converging on 1/n_w, the cancellation argument would be wrong and the
    length figure would still look fine.
    """
    from fishsense_imwut import refraction as rf

    s = rf.flat_port_range_error()
    assert s["asymptote_pct"] == pytest.approx(100 * (1 / rf.SALTY_WATER - 1), abs=1e-9)
    # converging on the Snell factor from below, i.e. worse near the port
    assert s["pct_error"][0] < s["pct_error"][-1] < 0
    assert s["pct_error"][-1] == pytest.approx(s["asymptote_pct"], abs=0.05)
    # and past a metre it is a scale: that stretch moves under 0.5 pp
    far = s["depth_m"] >= 1.0
    assert s["pct_error"][far].max() - s["pct_error"][far].min() < 0.5
    ratio = s["measured_m"][far] / s["depth_m"][far]
    assert ratio.std() < 0.002
    # inside a metre it is not, and that is the near-field tail the figure
    # exists to show -- an earlier sweep stepped 0.5 m and drew it as one
    # straight segment
    assert s["pct_error"][0] < -30.0
    assert s["depth_m"][0] == pytest.approx(0.30, abs=0.001)


def test_the_range_error_matches_flat_port_cost_at_the_same_depth():
    """One model, two entry points: the sweep must not drift from the figure
    that reports a single depth."""
    from fishsense_imwut import refraction as rf

    s = rf.flat_port_range_error(depths_m=[rf.MEASUREMENT_DEPTH_M])
    one = rf.flat_port_cost(depth_m=rf.MEASUREMENT_DEPTH_M, n_points=3)
    assert s["pct_error"][0] == pytest.approx(one["range_pct_error"], abs=1e-9)


def test_the_range_curvature_is_the_dot_going_off_axis_not_the_pane():
    """What Figure 9c's second curve is for, and it corrects a draft claim.

    The mounted laser sits 11.7 cm off the optical axis and runs parallel to it,
    so the dot's field angle SHRINKS with range -- 13.2 degrees at 0.5 m, 1.2 at
    5.5 -- and a near-field dot is an off-axis dot. Move the laser nearly
    coaxial and the depth dependence collapses. A draft of the caption blamed
    the pane's thickness against a small standoff; this asserts it is geometry,
    and reconciles the figure with `flat_port_error_field`, which shows the same
    radial term across the frame instead of along the dot's track.
    """
    from fishsense_imwut import refraction as rf

    s = rf.flat_port_range_error()
    mounted_span = s["pct_error"].max() - s["pct_error"].min()
    coaxial_span = s["coaxial_pct_error"].max() - s["coaxial_pct_error"].min()
    assert mounted_span > 4.0, "the mounted sweep should curve"
    assert coaxial_span < 0.5, "the coaxial sweep should be nearly flat"
    assert mounted_span > 20 * coaxial_span

    # the dot really does walk in from the edge of the frame toward the centre
    a = s["dot_field_angle_deg"]
    assert a[0] == pytest.approx(21.3, abs=0.1)
    assert a[-1] == pytest.approx(1.2, abs=0.1)
    assert (np.diff(a) < 0).all()

    # and the module constant is left exactly as it was found
    assert rf.LASER_POSITION_M.tolist() == [-0.04, -0.11, 0.0]


def test_the_range_error_is_a_scale_only_past_a_metre():
    """Figure 9c's two panels have to agree, and they did not.

    Panel (a) is a line and read as a pure scale; panel (b) says it is not one,
    and (b) is right -- a pure scale plots flat there, and the error moves
    4.8 pp over the sweep. The claim that survives is the narrower one the
    shading marks, and this pins both halves of it so a caption cannot quietly
    widen it again.
    """
    from fishsense_imwut import refraction as rf

    s = rf.flat_port_range_error()
    z, pct = s["depth_m"], s["pct_error"]
    ratio = s["measured_m"] / z

    # NOT a scale over the whole sweep
    assert pct.max() - pct.min() > 4.0

    # a scale past a metre, to better than half a point
    far = z >= 1.0
    assert pct[far].max() - pct[far].min() < 0.5
    assert ratio[far].min() == pytest.approx(0.7408, abs=0.001)
    assert ratio[far].max() == pytest.approx(0.7450, abs=0.001)

    # and emphatically not inside one
    near = z <= 1.0
    assert pct[near].max() - pct[near].min() > 4.0
    assert ratio[near].min() == pytest.approx(0.697, abs=0.001)
