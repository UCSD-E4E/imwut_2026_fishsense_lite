"""Each field slate frame's laser dot as a 3D point, from its slate pose.

Writes data/slate_points.csv: one row per slate frame with a dot, for dives 279, 341,
347, 349, 465 and 471. `X, Y, Z` (metres, camera frame) is where the dot's ray meets
the slate plane, the plane coming from PnP on the slate's human corner labels. These
are the observations a slate calibration is fitted to, so Z is a range that does not
depend on the laser.

The geometry is production's own stage-13 code (`calibration_geometry`), loaded by path
from fishsense-lite through the SenSys repo's E1 harness, which reproduces stored slate
calibrations exactly. It needs that harness's environment, which this repository does
not have (OpenCV and the compiled fishsense_core):

    cd ../cscw-fishsense2027/calibration_analysis/e1_label_free
    PYTHONPATH=<this repo> ../../../wuwnet-fishsense2026/.venv/bin/python \\
        <this repo>/fish_model_analysis/derive_slate_points.py

Inputs: the SenSys repo's `data/db_extracts/slate_dots.psv` (corner labels and dots,
from a production backup) and `data/e1/slates.csv` (slate templates). That extract
omits `diveslatelabel.skipped_points`; the one field label with a skipped point
(image 129259, dive 465, point 5) was read from production and is listed below.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SENSYS = Path(__file__).resolve().parents[2] / "cscw-fishsense2027"
sys.path.insert(0, str(SENSYS / "calibration_analysis" / "e1_label_free"))
import e1  # noqa: E402  (production's calibration_geometry, loaded by path)

from fishsense_imwut import calibration as cal  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
SLATE_OF_DIVE = {279: 7, 341: 8, 347: 11, 349: 8, 465: 8, 471: 8}   # dive.dive_slate_id
SKIPPED = {129259: [5]}                                             # diveslatelabel.skipped_points


def main():
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv").set_index("dive_id")
    K = {int(r["camera_id"]): np.array(json.loads(r["km"])) for r in cal.load_rows(DATA / "corpus.csv")}
    slates = pd.read_csv(SENSYS / "data/e1/slates.csv").set_index("dive_slate_id")
    sd = pd.read_csv(SENSYS / "data/db_extracts/slate_dots.psv", sep="|", header=None,
                     names=["dive_id", "camera_id", "image_id", "pts", "x", "y", "taken"]).drop_duplicates("image_id")
    rows = []
    for dive, sid in SLATE_OF_DIVE.items():
        s = slates.loc[sid]
        template = np.array(json.loads(s.reference_points), float) / float(s.dpi) * e1.INCH_TO_M
        Kc = K[int(fits.loc[dive, "camera_id"])]
        for r in sd[(sd.dive_id == dive) & sd.pts.notna()].itertuples():
            image = np.array(json.loads(r.pts), float)
            body = np.delete(template, SKIPPED.get(r.image_id, []), axis=0)
            if image.shape != body.shape:                       # production's pairing guard
                continue
            plane = e1.geometry.plane_from_correspondences(body, image, Kc)
            p = None if plane is None else e1.geometry.laser_point_on_plane(plane, np.array([r.x, r.y]), Kc)
            if p is not None:
                rows.append(dict(dive_id=dive, image_id=int(r.image_id), taken_utc=r.taken,
                                 dot_x=r.x, dot_y=r.y, X=p[0], Y=p[1], Z=p[2]))
    out = pd.DataFrame(rows).sort_values(["dive_id", "taken_utc", "image_id"])
    out.to_csv(DATA / "slate_points.csv", index=False)
    print(out.groupby("dive_id").size().to_dict())


if __name__ == "__main__":
    main()
