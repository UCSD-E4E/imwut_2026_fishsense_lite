"""Reverse-engineer where Shark's fork label would have to sit to read 605 mm.

Method: each dive's calibration scale is fitted from GROUPER (the near-unbiased
anchor, model effect -0.58 pp), never from Shark. That correction is applied to
the Shark frames, and the tail label is then slid along the head->tail axis
until the back-projected length equals the 605 mm reference. Where it lands is
the fork position the reference implies.

Only WELL-PRESENTED frames are usable: foreshortening is one-sided negative, so
an angled frame legitimately reads short and its implied tail would sit beyond
the real one. Frames are ranked by corrected length within their dive and only
the top decile is kept.
"""
import csv, json, sys
import numpy as np
from scipy.optimize import brentq
from fishsense_imwut import calibration as cal

SHARK_M = 0.605
NAS = "/home/chris/mnt/fishsense_data/REEF/data"


def pt_plane_z(u, inv_k, z):
    """Back-project pixel u onto the plane Z = z. This model reproduces the
    stored length_m to 0.000 mm, so it is the pipeline's, not an approximation."""
    p = inv_k @ np.array([u[0], u[1], 1.0])
    return p / p[2] * z


def load(path):
    rows = [r for r in csv.DictReader(open(path), delimiter="|")
            if r.get("ht") and r.get("dot")]
    by_dive = {}
    for r in rows:
        by_dive.setdefault(int(r["dive_id"]), []).append(r)
    return by_dive


def analyse(by_dive, top_frac=0.10, min_frames=5):
    out = []
    for did in sorted(by_dive):
        rows = by_dive[did]
        grouper = [r for r in rows if r["model"] == "Grouper"]
        shark = [r for r in rows if r["model"] == "Shark"]
        if len(grouper) < min_frames or len(shark) < min_frames:
            continue
        o, a, inv_k_dive, n = cal.dive_geometry(rows)
        phi = cal.fit_phi(grouper, o, a, inv_k_dive, n)   # GROUPER anchors the dive
        axis = cal.rotate(a, n, phi)

        recs = []
        for r in shark:
            K = np.array(json.loads(r["km"]))
            inv_k = np.linalg.inv(K)
            ht = [float(x) for x in r["ht"].split(";")]
            h, t = np.array(ht[:2]), np.array(ht[2:4])
            dot = [float(x) for x in r["dot"].split(";")]
            z = cal.triangulate_depth(dot, o, axis, inv_k)      # corrected depth
            length = np.linalg.norm(pt_plane_z(h, inv_k, z) - pt_plane_z(t, inv_k, z))
            # slide the tail along the head->tail ray until the length is 605 mm
            f = lambda s: np.linalg.norm(
                pt_plane_z(h, inv_k, z) - pt_plane_z(h + s * (t - h), inv_k, z)
            ) - SHARK_M
            s = brentq(f, 0.2, 1.8)
            sep = np.linalg.norm(t - h)
            recs.append(dict(
                dive=did, path=r["path"], K=K, dist=np.array(json.loads(r["dist"])),
                head=h, tail=t, implied=h + s * (t - h), z=z, length=length,
                over_pct=100 * (length / SHARK_M - 1), sep_px=sep,
                shift_px=(1 - s) * sep, shift_frac=(1 - s),
            ))
        recs.sort(key=lambda d: -d["length"])
        keep = max(1, int(round(top_frac * len(recs))))
        for d in recs[:keep]:
            d["rank"] = f"top {keep}/{len(recs)}"
            out.append(d)
    return out


if __name__ == "__main__":
    best = analyse(load(sys.argv[1]))
    print(f"{'dive':>4} {'z(m)':>5} {'corr len':>9} {'over':>7} {'sep px':>7} "
          f"{'shift px':>9} {'shift %':>8}  frame")
    for d in best:
        print(f"{d['dive']:>4} {d['z']:5.2f} {d['length']:9.4f} {d['over_pct']:+6.2f}% "
              f"{d['sep_px']:7.1f} {d['shift_px']:9.1f} {100*d['shift_frac']:7.2f}%  "
              f"{d['path'].split('/')[-1]}")
    A = np.array([[d["over_pct"], d["shift_px"], 100 * d["shift_frac"]] for d in best])
    print(f"\nwell-presented Shark frames: n={len(A)}")
    print(f"  over-read   median {np.median(A[:,0]):+.2f}%  "
          f"range [{A[:,0].min():+.2f}, {A[:,0].max():+.2f}]")
    print(f"  tail shift  median {np.median(A[:,1]):.1f} px = "
          f"{np.median(A[:,2]):.2f}% of body length")
    print(f"  in metres   median {SHARK_M*np.median(A[:,2])/100*1000:.1f} mm of the 605 mm")
