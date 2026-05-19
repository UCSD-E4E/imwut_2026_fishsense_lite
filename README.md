# imwut_2026_fishsense_lite

Analysis code and simulations supporting the IMWUT 2026 submission
**"FishCamera: A Camera-Based, Single Laser Hardware Framework with Software
System for In Situ Fish Length Measurement."**

## Layout

```
fishsense_imwut/       # Shared library: camera model, calibration, plotting helpers
figures/               # Paper-bound figures (see figures/README.md)
notebooks/             # Analyses, organized by paper section
```

The paper source lives separately (see the bundled zip). This repo produces
the figures and tables in `figures/`, which map into the paper's
`Images/graphs/` directory.

## Notebook → paper mapping

| Notebook                                                              | Paper section / figure                                       |
| --------------------------------------------------------------------- | ------------------------------------------------------------ |
| `notebooks/max_distance.ipynb`                                        | §3.2 — useful range of the ray-intersection estimator        |
| `notebooks/laser_calibration/baseline.ipynb`                          | §3.3.4 — pairwise-difference direction + z=0 origin (method) |
| `notebooks/laser_calibration/weighted_pca.ipynb`                      | §3.3.4 — weighted PCA comparison                             |
| `notebooks/laser_calibration/pca.ipynb`                               | §3.3.4 — plain PCA comparison                                |
| `notebooks/laser_calibration/ransac.ipynb`                            | §3.3.4 — RANSAC comparison                                   |
| `notebooks/laser_calibration/least_squares.ipynb`                     | §3.3.4 — iterative nonlinear refinement comparison           |
| `notebooks/laser_calibration/least_squares_with_corrections_2d.ipynb` | §3.3.4 — LS variant with 2D corrections                      |
| `notebooks/laser_calibration/corrections_2d.ipynb`                    | §3.3.4 — direct 2D corrections variant                       |
| `notebooks/laser_calibration/baseline_vs_weighted_pca.ipynb`          | §3.3.4 — head-to-head comparison summary                     |
| `notebooks/reconstruction/known_calibration*.ipynb`                   | §3.2 / §3.3.6 — reconstruction with a known laser pose       |
| `notebooks/reconstruction/calculated_calibration.ipynb`               | §3.3.6 — reconstruction using calibration from labeled data  |
| `notebooks/field_reconstruction/laser_spot_path.ipynb`                | §4 Results — laser spot path across field images             |
| `notebooks/field_reconstruction/pixel_sensitivity.ipynb`              | §4 Results — pixel-level sensitivity analysis                |
| `notebooks/field_reconstruction/pixel_tolerance_vs_spot.ipynb`        | §4 Results — pixel tolerance vs. laser spot size             |
| `notebooks/optics_simulation/flat_port_refraction.ipynb`              | §3.3.2 — flat-port refraction error simulation               |
| `notebooks/optics_simulation/ray_geometry.ipynb`                      | §3.3.2 — refracted ray geometry illustration                 |
| `notebooks/laser_labeling/laser_label_analysis.ipynb`                 | §3.3.4 — field laser label QA                                |

## Shared helpers

Notebooks should import common setup from `fishsense_imwut` rather than
re-declaring it:

```python
from fishsense_imwut.camera import make_camera_intrinsics
from fishsense_imwut.constants import IMAGE_WIDTH, IMAGE_HEIGHT, FOCAL_LENGTH_PX
from fishsense_imwut.figures import save_paper_figure

K, K_inv = make_camera_intrinsics()
# ... produce fig ...
save_paper_figure(fig, "laser_spot_path.pdf")
```

See `figures/README.md` for the figure-naming convention.

## Development

Managed with `uv`:

```
uv sync
uv run jupyter lab
```
