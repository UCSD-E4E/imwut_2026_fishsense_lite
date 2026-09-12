"""Draw current head/tail labels and the implied fork position on the camera JPEG."""
import sys
import numpy as np, cv2
from PIL import Image, ImageDraw
from fork_probe import load, analyse, NAS

CROP = np.array([8.0, 8.0])   # raw 4014x3016 origin -> camera-JPEG 4000x3000 origin


def to_jpeg(pts, K, dist):
    """Labels are stored in UNDISTORTED pinhole coords. To draw on the camera's
    own JPEG we must re-apply distortion and subtract the (8,8) crop -- skipping
    either puts every mark down-and-right of the true feature."""
    n = (np.linalg.inv(K) @ np.c_[pts, np.ones(len(pts))].T).T[:, :2]
    d = cv2.projectPoints(np.c_[n, np.ones(len(n))].astype(np.float64),
                          np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    return d - CROP


def render(d, outdir, pad=190):
    jpg = f"{NAS}/{d['path'].replace('.ORF', '.JPG')}"
    P = to_jpeg(np.array([d["head"], d["tail"], d["implied"]]), d["K"], d["dist"])
    (hx, hy), (tx, ty), (ix, iy) = P
    im = Image.open(jpg).convert("RGB")
    dr = ImageDraw.Draw(im)
    # full head->tail span in grey, the disputed segment in red
    dr.line([(hx, hy), (tx, ty)], fill=(120, 120, 120), width=5)
    dr.line([(ix, iy), (tx, ty)], fill=(255, 40, 40), width=9)
    dr.ellipse([hx - 26, hy - 26, hx + 26, hy + 26], outline=(0, 255, 90), width=8)
    dr.ellipse([tx - 30, ty - 30, tx + 30, ty + 30], outline=(255, 210, 0), width=9)
    dr.ellipse([ix - 30, iy - 30, ix + 30, iy + 30], outline=(0, 220, 255), width=9)

    name = d["path"].split("/")[-1].replace(".ORF", "")
    cx, cy = (tx + ix) / 2, (ty + iy) / 2
    box = (int(cx - pad), int(cy - pad), int(cx + pad), int(cy + pad))
    tail = im.crop(box).resize((620, 620), Image.LANCZOS)
    tail.save(f"{outdir}/tail_d{d['dive']}_{name}_{d['over_pct']:+.1f}pct.png")

    full = im.crop((max(0, int(min(hx, tx, ix)) - 260), max(0, int(min(hy, ty, iy)) - 420),
                    min(im.width, int(max(hx, tx, ix)) + 260),
                    min(im.height, int(max(hy, ty, iy)) + 420)))
    full.thumbnail((1250, 1250))
    full.save(f"{outdir}/full_d{d['dive']}_{name}_{d['over_pct']:+.1f}pct.png")
    return name


if __name__ == "__main__":
    csvp, outdir = sys.argv[1], sys.argv[2]
    best = analyse(load(csvp))
    best.sort(key=lambda d: -d["sep_px"])          # sharpest framing first
    for d in best[:5]:
        print("rendered", render(d, outdir), f"dive {d['dive']}",
              f"over {d['over_pct']:+.2f}%  shift {d['shift_px']:.0f}px")
