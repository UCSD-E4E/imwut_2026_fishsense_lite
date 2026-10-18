"""Annotate one measured frame: the laser dot, the clicked snout and tail-fork,
and the span between them. Nothing else is drawn -- no length, range or species;
the caption carries those.

    python annotate_frame.py FRAME.JPG labels.json out.png

`labels.json` holds the labeller's points for that frame as production stores
them -- in UNDISTORTED pinhole pixel coordinates of the raw 4014 x 3016 sensor --
plus the unit's intrinsics and distortion. To draw on the camera's own JPEG the
distortion is re-applied and the (8, 8) sensor crop removed, as in
`fork_render.py`; skipping either puts every mark down and to the right of the
feature it marks.

The output is written without EXIF, so it carries none of the source photograph's
metadata (the camera JPEGs hold GPS).
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

CROP = np.array([8.0, 8.0])   # raw 4014 x 3016 origin -> camera-JPEG 4000 x 3000 origin

DOT_RING = (255, 255, 255)    # the laser dot: a white ring, distinct from the endpoints
ENDPOINT = (255, 196, 0)      # snout and tail-fork: amber
SPAN = (255, 196, 0)


def to_jpeg(points, K, dist):
    pts = np.asarray(points, float)
    n = (np.linalg.inv(K) @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]
    d = cv2.projectPoints(np.c_[n, np.ones(len(n))], np.zeros(3), np.zeros(3),
                          K, dist)[0].reshape(-1, 2)
    return d - CROP


def annotate(jpg, labels, out, margin=(0.35, 0.9), scale=1.0):
    K = np.asarray(labels["K"], float)
    dist = np.asarray(labels["dist"], float)
    (dx, dy), (hx, hy), (tx, ty) = to_jpeg([labels["dot"], labels["snout"], labels["fork"]], K, dist)
    im = Image.open(jpg).convert("RGB")

    span = np.hypot(tx - hx, ty - hy)
    lw = max(3, int(round(span / 200 * scale)))     # line width tracks the fish's size
    r_end = max(6, int(round(span / 70 * scale)))
    r_dot = int(round(r_end * 1.9))

    d = ImageDraw.Draw(im)
    d.line([(hx, hy), (tx, ty)], fill=SPAN, width=lw)
    for x, y in ((hx, hy), (tx, ty)):
        d.ellipse([x - r_end, y - r_end, x + r_end, y + r_end], outline=ENDPOINT, width=lw)
    d.ellipse([dx - r_dot, dy - r_dot, dx + r_dot, dy + r_dot], outline=DOT_RING, width=lw)

    # crop around the fish, wider than tall, inside the frame
    mx, my = margin[0] * span, margin[1] * span * 0.45
    x0, x1 = min(hx, tx) - mx, max(hx, tx) + mx
    cy = (hy + ty + dy) / 3
    y0, y1 = cy - (x1 - x0) * 0.30, cy + (x1 - x0) * 0.30
    box = [int(max(0, x0)), int(max(0, y0)), int(min(im.width, x1)), int(min(im.height, y1))]
    im.crop(box).save(out)          # PIL writes no EXIF unless asked to
    return box


if __name__ == "__main__":
    jpg, labels_path, out = sys.argv[1:4]
    labels = json.load(open(labels_path))
    print(annotate(jpg, labels, out))
