"""The corrective optic's counterfactual: calibrate underwater exactly as we did, but
through the bare flat port, and measure with the laser.

`refraction.py` answers a different question -- an in-air calibration carried
underwater with no correction -- and leaves the in-water single-viewpoint (SVP)
calibration to the WUWNet paper, which fits it at one distance and measures at the
true range. This module is what this paper needs instead: the whole FishSense Lite
pipeline run on an SVP calibration of the bare port.

1. Synthetic checkerboard views through the bare port, at varying angles and
   distances (`calibration_views`).
2. A pinhole + radial distortion fit (k1, k2, k3; tangential fixed at zero) to those
   views, the model OpenCV's `calibrateCamera` fits (`calibrate`). No OpenCV: the
   same least-squares problem, solved with scipy, so the repository needs no new
   dependency.
3. The laser fitted under that calibration from slate frames, by production's
   recipe: pose each slate, put the dot's ray on its plane, fit a line (`fit_laser`).
4. A 300 mm target measured with that calibration and beam: range from the dot,
   both endpoints back-projected at the dot's depth (`measure`).

The forward model is `refraction.py`'s, unchanged: Pinax geometry, the housing's
fitted standoff, salt water, the in-air pinhole camera. The corrective optic is
modelled as that module models it, as the air path restored (`n_water = 1`).

Everything is noise-free. Every number therefore isolates what an SVP model cannot
represent about an axial camera, which is the question; calibration noise is a
separate, and WUWNet's, question.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import refraction as rf

BOARD_COLS, BOARD_ROWS, BOARD_PITCH_M = 14, 10, 0.042   # the E4E board, interior corners
N_VIEWS = 55                                            # the manual: 50-60 views
SLATE_GRID = (4, 3)
SLATE_PITCH_M = 0.08
SLATE_RANGES_M = np.linspace(0.8, 3.0, 10)              # field lever arms run 1.0-2.4 m
BUDGET_PCT = 15.0


def make_port(n_water=rf.SALTY_WATER, d0=None, d1=rf.GLASS_THICKNESS_M):
    """`refraction.py`'s housing: the standoff that makes the port most nearly a
    single viewpoint (`optimal_d0`) unless `d0` is given, and the 6 mm placeholder
    pane unless `d1` is."""
    K = rf.camera_intrinsics()
    half_fov = np.arctan(np.hypot(rf.IMAGE_WIDTH_PX / 2, rf.IMAGE_HEIGHT_PX / 2) / rf.FOCAL_LENGTH_PX)
    if d0 is None:
        d0, _, _ = rf.optimal_d0(d1, rf.N_GLASS, rf.SALTY_WATER, half_fov)
    return rf.FlatPort(d0, d1, rf.N_GLASS, n_water), K


def to_pixels(points, port, K):
    a, az = rf.project_water_points(points, port)
    return rf.alpha_azimuth_to_pixel(a, az, K)


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def _rvec(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def board_points(cols=BOARD_COLS, rows=BOARD_ROWS, pitch=BOARD_PITCH_M):
    g = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2).astype(float)
    g = (g - g.mean(0)) * pitch
    return np.c_[g, np.zeros(len(g))]


@dataclass
class View:
    R: np.ndarray
    t: np.ndarray
    pixels: np.ndarray


def calibration_views(port, K, distances, n_views=N_VIEWS, max_tilt_deg=35.0, seed=0):
    """Board views through `port`, every corner inside the frame (a partial board is
    not detected), distance uniform over `distances`, tilts uniform to `max_tilt_deg`,
    the board placed anywhere it fits."""
    rng = np.random.default_rng(seed)
    B = board_points()
    views = []
    while len(views) < n_views:
        z = rng.uniform(*distances)
        R = _rot(*np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2)), np.radians(rng.uniform(-15, 15)))
        half = 0.5 * z * np.array([rf.IMAGE_WIDTH_PX, rf.IMAGE_HEIGHT_PX]) / (K[0, 0] * port.n_water)
        t = np.array([rng.uniform(-half[0], half[0]), rng.uniform(-half[1], half[1]), z])
        P = B @ R.T + t
        if np.any(P[:, 2] <= port.d2 + 0.05):
            continue
        px = to_pixels(P, port, K)
        if np.all((px[:, 0] > 0) & (px[:, 0] < rf.IMAGE_WIDTH_PX) & (px[:, 1] > 0) & (px[:, 1] < rf.IMAGE_HEIGHT_PX)):
            views.append(View(R, t, px))
    return views


# --- the pinhole + radial model (OpenCV's, tangential fixed at zero) ----------------

def distort(xy, k):
    r2 = (xy ** 2).sum(-1, keepdims=True)
    return xy * (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)


def project(P, intr):
    fx, fy, cx, cy, *k = intr
    xy = distort(P[..., :2] / P[..., 2:3], k)
    return np.stack([fx * xy[..., 0] + cx, fy * xy[..., 1] + cy], -1)


def unproject(px, intr, iters=50):
    """Pixel -> normalized ray (x, y, 1), by fixed-point undistortion."""
    fx, fy, cx, cy, *k = intr
    px = np.asarray(px, float)
    d = np.stack([(px[..., 0] - cx) / fx, (px[..., 1] - cy) / fy], -1)
    xy = d.copy()
    for _ in range(iters):
        r2 = (xy ** 2).sum(-1, keepdims=True)
        xy = d / (1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3)
    return np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)


def calibrate(views, n_water):
    """Joint least squares over (fx, fy, cx, cy, k1, k2, k3) and every view's pose,
    the problem `cv2.calibrateCamera` solves. Started from the paraxial focal length
    (f * n_water) and the views' poses, as OpenCV's closed-form start would give."""
    B = board_points()
    nv = len(views)
    x0 = [rf.FOCAL_LENGTH_PX * n_water, rf.FOCAL_LENGTH_PX * n_water,
          rf.IMAGE_WIDTH_PX / 2, rf.IMAGE_HEIGHT_PX / 2, 0.0, 0.0, 0.0]
    for v in views:
        x0 += list(_rvec(v.R)) + list(v.t)
    x0 = np.array(x0, float)
    obs = np.stack([v.pixels for v in views])

    def resid(x):
        intr = x[:7]
        out = []
        for i in range(nv):
            r, t = x[7 + 6 * i: 10 + 6 * i], x[10 + 6 * i: 13 + 6 * i]
            out.append(project(B @ _rodrigues(r).T + t, intr) - obs[i])
        return np.concatenate(out).ravel()

    n_pts = len(B)
    J = lil_matrix((nv * n_pts * 2, len(x0)), dtype=int)
    for i in range(nv):
        rows = slice(i * n_pts * 2, (i + 1) * n_pts * 2)
        J[rows, :7] = 1
        J[rows, 7 + 6 * i: 13 + 6 * i] = 1
    sol = least_squares(resid, x0, jac_sparsity=J, x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x[:7], rms


# --- the laser, fitted under the calibration -------------------------------------------

def slate_points():
    c, r = SLATE_GRID
    g = np.stack(np.meshgrid(np.arange(c), np.arange(r)), -1).reshape(-1, 2).astype(float)
    g = (g - g.mean(0)) * SLATE_PITCH_M
    return np.c_[g, np.zeros(len(g))]


def _pnp(body, px, intr, R0, t0):
    def resid(x):
        return (project(body @ _rodrigues(x[:3]).T + x[3:], intr) - px).ravel()
    x = least_squares(resid, np.r_[_rvec(R0), t0], xtol=1e-14, ftol=1e-14, gtol=1e-14).x
    return _rodrigues(x[:3]), x[3:]


def fit_laser(port, K, intr, ranges=SLATE_RANGES_M, max_tilt_deg=20.0, seed=1):
    """Production's recipe: each slate posed by PnP under the calibration, the dot's
    ray put on its plane, a line fitted through the points. The slate is held with the
    dot on it, so it sits around the beam."""
    rng = np.random.default_rng(seed)
    body = slate_points()
    o, a = rf.LASER_POSITION_M, rf.LASER_DIRECTION / np.linalg.norm(rf.LASER_DIRECTION)
    pts = []
    for z in ranges:
        R = _rot(*np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2)), 0.0)
        centre = o + a * z + np.r_[rng.uniform(-0.05, 0.05, 2), 0.0]
        P = body @ R.T + centre
        n = R[:, 2]
        dot = o + a * ((centre - o) @ n / (a @ n))          # the true dot, on the slate plane
        Rh, th = _pnp(body, to_pixels(P, port, K), intr, R, centre)
        ray = unproject(to_pixels(dot, port, K), intr)
        nh = Rh[:, 2]
        pts.append(ray * (th @ nh) / (ray @ nh))
    pts = np.array(pts)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    d = d if d[2] > 0 else -d
    return c - d * c[2] / d[2], d


# --- the measurement ------------------------------------------------------------------

def measure(port, K, intr, beam, z, dot_at=0.5, length=rf.FISH_LENGTH_M):
    """A horizontal target of `length` at true range `z`, with the dot `dot_at` of the
    way along it from the tail (0.5: dot mid-body; 0 or 1: dot at an end, the furthest
    off axis a target can sit and still carry the dot). Returns (length % error,
    range % error)."""
    o, a = rf.LASER_POSITION_M, rf.LASER_DIRECTION / np.linalg.norm(rf.LASER_DIRECTION)
    dot = o + a * z
    tail = dot - np.array([dot_at * length, 0, 0])
    head = tail + np.array([length, 0, 0])
    px = to_pixels(np.stack([dot, head, tail]), port, K)
    rays = unproject(px, intr)
    p, _ = rf.reconstruct_points(rays[0], np.zeros(3), beam[0], beam[1])
    zhat = p[2]
    h, t = rays[1] * zhat, rays[2] * zhat
    return 100 * (np.linalg.norm(h - t) / length - 1), 100 * (zhat / z - 1)


def run(distances, n_water=rf.SALTY_WATER, ranges=np.linspace(0.5, 5.0, 46), seed=0, d0=None, d1=rf.GLASS_THICKNESS_M):
    """The whole counterfactual for one calibration-distance spread. With
    `n_water=1.0` it is the corrective optic's baseline."""
    port, K = make_port(n_water, d0, d1)
    views = calibration_views(port, K, distances, seed=seed)
    intr, rms = calibrate(views, n_water)
    beam = fit_laser(port, K, intr)
    res = {"ranges": ranges, "intrinsics": intr, "rms_px": rms, "d0": port.d0,
           "view_distances": np.array([v.t[2] for v in views])}
    for name, at in (("mid", 0.5), ("end", 0.0)):
        e = np.array([measure(port, K, intr, beam, z, dot_at=at) for z in ranges])
        res[f"length_{name}"], res[f"range_{name}"] = e[:, 0], e[:, 1]
    return res
