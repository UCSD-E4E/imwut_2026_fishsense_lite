# imwut_2026_fishsense_lite

**Companion repository for** *FishSense Lite: A Camera-Based, Single Laser Hardware
Framework with Software System for In Situ Fish Length Measurement* (IMWUT / *Journal of
Ocean Engineering*; preprint [10.5281/zenodo.22882398](https://doi.org/10.5281/zenodo.22882398)).

Everything a reader of the paper might want to check or rerun is here: the measurement
corpus and every number and figure in the Results, the estimator comparison the System
section refers to, and the simulations behind the length-recovery method.

A commercial dive camera with one rigidly mounted laser recovers metric depth from a
single image; head and tail are back-projected at that depth to give a length. This repo
holds the evidence that it works, how well, and what limits it: a 2,927-measurement corpus
of rigid targets of known length over 32 pool sessions, a rule that decides which sessions
are accuracy evidence, a designed foreshortening experiment, and the simulation studies
behind the reconstruction and calibration methods.

**Headline.** Over the 19-session accuracy cohort (995 measurements, 0.28–5.06 m): median
−2.00 %, $p_{90}$ +0.36 %, 80 / 98 / 99 % of frames within 5 / 10 / 15 %.

## Reproducing this without database access

**You do not need our database, credentials, or a C/Rust toolchain.** Every number in
`PAPER.md` comes from the committed CSVs in `fish_model_analysis/data/`, documented file
by file in [`fish_model_analysis/data/MANIFEST.md`](fish_model_analysis/data/MANIFEST.md).

```bash
uv sync                         # no DB client, no compiler
uv run pytest -q tests          # 156 tests; pins the cohort, the polish, the headline numbers
uv run jupyter lab              # fish_model_analysis/fish_model_measurements.ipynb
```

Nothing here reaches the database from Python at all — the cache is re-exported with
`psql` against `fish_model_analysis/sql/*.sql`, and that is the only path needing
credentials.

The one optional dependency is the **simulation** group, which pulls the maturin/pyo3
`fishsense-core` Rust extension for `fishsense_core.laser.calibrate_laser`. Only
`calibration_analysis/` and `reconstruction_analysis/calculated_calibration.ipynb` use it,
to compare estimators against the production laser fit. It is the one path that needs a
compiler (`linker 'cc' not found` without one):

```bash
uv sync --group sim             # simulation notebooks only
```

The flake supplies the toolchain:

```bash
nix run .#default -- -c 'uv run jupyter lab'
nix develop                     # or drop into the FHS shell
```

## Where things are

| path | what |
|---|---|
| `fish_model_analysis/` | **the corpus and the accuracy analysis.** `data/MANIFEST.md` (the cache index), `data/corpus.csv`, the notebook that produces every figure, `FINDINGS.md` (§7 is current), and the August `HANDOFF.md` |
| `post_labeling_analysis/HANDOFF.md` | **read this first.** Ground rules, what is established, what was tried and failed, and the per-session table |
| `PAPER.md` | the Section 4 draft, its LaTeX table, and the recommendation on Figures 4/5 |
| `fishsense_imwut/` | `calibration.py` (geometry, median polish, cohort rule, range trend), `pubfig.py` (figures; `nearest_rank_p90` is the estimator of record), `camera.py`, `plots.py` |
| `reconstruction_analysis/` | simulation: reconstruction under known and fitted calibration — ODR, least squares, 2D/3D corrections, `pixel_sensitivity.ipynb`, and `known_calibration_bundle_adjustment.ipynb` (issue #2) |
| `calibration_analysis/` | simulation: which estimator recovers the laser calibration — PCA, weighted PCA, RANSAC, least squares, 2D corrections |
| `laser_labeling_analysis/` | `laser_labels_cleaned.csv`, 37,811 laser-dot labels over 262 dives |
| `tests/` | pins the cohort so a change to the paper's numbers is a diff; `test_data_cache.py` pins the offline-reproduction contract |

Start with `post_labeling_analysis/HANDOFF.md` §0 and §7 — §0 is the ground rules learned
the hard way, §7 is where a fresh analysis is most likely to repeat a dead end.

## Sibling repos

| repo | paper |
|---|---|
| `../wuwnet-fishsense2026` | flat-port refraction, in-air calibration (WUWNet) |
| `../cscw-fishsense2027` | deployability for citizen scientists; reads this repo's corpus and label export by relative path |
| `../fishsense-lite` | the production pipeline — stage 13 calibration, stage 14 measurement, `range_trend.py` |
