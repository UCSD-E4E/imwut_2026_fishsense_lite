"""How much residual lens distortion could be doing to a length.

The question this answers: a near target subtends more pixels AND reaches
further from the principal point, where the radial polynomial is steeper. Both
effects push the same way, so could residual distortion explain the near-field
droop in Figure 3?

The answer is no, and the interesting part is the bound rather than the verdict.

**Work on the ENDPOINTS, never the laser dot.** `corpus.csv` carries the dot, and
using it as a proxy gives the opposite answer -- it reports a radial effect that
the clicked endpoints show is not there. The reason is mechanical: dot radius
differs from endpoint radius by half the apparent span, and the apparent span is
proportional to 1/range, so partialling range out of the two leaves different
residuals. `sql/extract_head_tail.sql` exports the endpoints.

**Labels are clicked on UNDISTORTED images**, so `distortion_coefficients` is the
model that was already removed. A leverage of 0.5 % means undistortion changed
that span by 0.5 %; if the model is wrong by a fraction eps, the length is wrong
by roughly eps x 0.5 %. So leverage is a ceiling on what distortion can do, and
the residual is a fraction of it.
"""

from __future__ import annotations

import json

import numpy as np


def distort(pixels, camera_matrix, coefficients):
    """Undistorted pixels -> distorted, OpenCV's radial + tangential model.

    The forward direction, because the labels are already undistorted and we
    want to know what undistortion did to them.
    """
    k = np.asarray(coefficients, dtype=float)
    fx, fy = camera_matrix[0][0], camera_matrix[1][1]
    cx, cy = camera_matrix[0][2], camera_matrix[1][2]
    p = np.asarray(pixels, dtype=float)
    x = (p[..., 0] - cx) / fx
    y = (p[..., 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3
    xd = x * radial + 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x)
    yd = y * radial + k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y
    return np.stack([xd * fx + cx, yd * fy + cy], axis=-1)


def load_head_tail(path) -> list[dict]:
    """Read `data/head_tail.csv` as produced by `sql/extract_head_tail.sql`."""
    import csv

    fields = ("dive_id", "measurement_id", "camera_id", "model", "known_length_m",
              "length_m", "depth_m", "head_x", "head_y", "tail_x", "tail_y",
              "km", "dist")
    out = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if len(row) < len(fields) or row[0].startswith("("):
                continue
            r = dict(zip(fields, row))
            try:
                r["dive_id"] = int(r["dive_id"])
                r["camera_id"] = int(r["camera_id"])
                for key in ("known_length_m", "length_m", "depth_m",
                            "head_x", "head_y", "tail_x", "tail_y"):
                    r[key] = float(r[key])
                r["km"] = json.loads(r["km"])
                r["dist"] = json.loads(r["dist"])
            except (ValueError, json.JSONDecodeError):
                continue
            out.append(r)
    return out


def leverage(row) -> dict:
    """What the distortion model did to one frame's measured span.

    `leverage_pct` is the whole modelled effect at the endpoints' ACTUAL
    positions. `centred_pct` places the same span symmetrically about the
    principal point, so the ratio of the two isolates how much the radial
    position compounds the apparent-size effect -- roughly 1.5x at short range.
    """
    km, k = row["km"], row["dist"]
    head = np.array([row["head_x"], row["head_y"]])
    tail = np.array([row["tail_x"], row["tail_y"]])
    centre = np.array([km[0][2], km[1][2]])

    span = float(np.linalg.norm(head - tail))
    distorted = float(np.linalg.norm(distort(head, km, k) - distort(tail, km, k)))

    unit = (head - tail) / span
    ch, ct = centre + unit * span / 2, centre - unit * span / 2
    centred = float(np.linalg.norm(distort(ch, km, k) - distort(ct, km, k)))

    return {
        "span_px": span,
        "r_max_px": float(max(np.linalg.norm(head - centre),
                              np.linalg.norm(tail - centre))),
        "leverage_pct": 100.0 * (span / distorted - 1.0),
        "centred_pct": 100.0 * (span / centred - 1.0),
        "pct_error": 100.0 * (row["length_m"] - row["known_length_m"])
        / row["known_length_m"],
        "depth_m": row["depth_m"],
    }


def leverage_table(rows) -> dict:
    """`leverage` over many rows, as arrays."""
    recs = [leverage(r) for r in rows]
    return {k: np.array([r[k] for r in recs]) for k in recs[0]}
