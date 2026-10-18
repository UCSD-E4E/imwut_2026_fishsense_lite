"""Apparent slate size under the laser dot, for the field dives' calibration ends.

Writes data/slate_sizes.csv: one row per slate frame, with the dot used (`dot_source`:
`human` label or automated `detector`), and `size` = sqrt(area) in pixels at the
1600 x 1202 render of the SAM 3.1 mask (text prompt "slate", then "white board") that
CONTAINS the dot, or empty when no slate mask contains it. Size goes as 1/range, so
within a dive `Z * size` is a constant and a size gives a range once one frame with a
pose range (data/slate_points.csv) fixes the constant. This is the size-constancy
method of the SenSys repo (cscw-fishsense2027, calibration_analysis/e1j_size_constancy).

Two sources, both outside this repository:

1. Dives 279, 465, 471 (both ends): every slate frame the slate-presence model scored
   >= 0.5 that carries a human laser dot or an automated laser prediction with
   confidence >= 0.5. Presence and laser predictions from the fishsense-services (v2)
   backup of 2026-10-18 (fishsense_process_work/database_backups_v2/fishsense_services),
   restored read-only into a throwaway Postgres:
       select d.v1_id, c.v1_id, c.captured_at, l.x, l.y, l.confidence
       from captures c join dives d on d.id = c.dive_id
       join slate_presence_predictions p on p.capture_id = c.id and p.probability >= 0.5
       join lateral (select * from laser_predictions l where l.capture_id = c.id
                     order by l.created_at desc limit 1) l on true
       where d.v1_id in (279, 465, 471) and l.confidence >= 0.5;
   Human dots win where both exist. Sized by SenSys's sam_sizes.py (`main("-", frames,
   prefix)`), outputs redirected away from that repo, run from fishsense-lite's venv on
   GPU with production's checkpoint (sam3.1_multiplex.pt).
2. Dives 341, 347, 349 (single bursts, the noise floor): SenSys's existing
   reef_sizes/sam3_text.csv, same method, human dots.

This script only assembles the two into the cache; pass the sam_sizes detail file
from step 1 and SenSys's e1j_size_constancy directory.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def main(frames_csv, sam_detail_csv, sensys_e1j):
    F = pd.read_csv(frames_csv)
    D = pd.read_csv(sam_detail_csv)[["image_id", "text_area"]]
    ends = F.merge(D, on="image_id", how="left").rename(columns={"source": "dot_source"})
    ends["size"] = np.sqrt(ends.text_area)
    ends = ends.drop(columns=["text_area", "cached"], errors="ignore")
    pts = pd.read_csv(DATA / "slate_points.csv")
    reef = pd.read_csv(Path(sensys_e1j) / "reef_sizes" / "sam3_text.csv").merge(
        pts[pts.dive_id.isin([341, 347, 349])][["dive_id", "image_id", "taken_utc", "dot_x", "dot_y"]],
        on="image_id").rename(columns={"taken_utc": "taken"}).assign(dot_source="human")
    out = pd.concat([ends, reef[ends.columns]], ignore_index=True).sort_values(["dive_id", "taken", "image_id"])
    out.to_csv(DATA / "slate_sizes.csv", index=False)
    print(out.groupby("dive_id").agg(frames=("image_id", "size"), sized=("size", "count")).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:4])
