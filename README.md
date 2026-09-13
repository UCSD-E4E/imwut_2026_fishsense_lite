# imwut_2026_fishsense_lite

Length-measurement accuracy for FishSense Lite — the system-characterization paper
(IMWUT / *Journal of Ocean Engineering*).

A commercial dive camera with one rigidly mounted laser recovers metric depth from a
single image; head and tail are back-projected at that depth to give a length. This repo
holds the evidence that it works, how well, and what limits it: a 2,927-measurement corpus
of rigid targets of known length over 32 pool sessions, a rule that decides which sessions
are accuracy evidence, a designed foreshortening experiment, and the simulation studies
behind the reconstruction and calibration methods.

**Headline.** Over the 13-session accuracy cohort (771 measurements, 0.25–4.7 m): median
−2.19 %, $p_{90}$ +0.06 %, 78 / 98 / 99 % of frames within 5 / 10 / 15 %.

```bash
uv run pytest -q tests          # 17 tests; pins the polish, the cohort rule, the range trend
uv run jupyter lab              # fish_model_analysis/
```

`uv sync` needs a C/Rust toolchain, because `fishsense-meta` pulls the maturin/pyo3
`fishsense-core` sdist and the build fails with `linker 'cc' not found` without it. The
flake supplies one:

```bash
nix run .#default -- -c 'uv run jupyter lab'
nix develop                     # or drop into the FHS shell
```

## Where things are

| path | what |
|---|---|
| `fish_model_analysis/` | **the corpus and the accuracy analysis.** `data/corpus.csv`, the notebook that produces every figure, `FINDINGS.md` (§7 is current), and the August `HANDOFF.md` |
| `post_labeling_analysis/HANDOFF.md` | **read this first.** Ground rules, what is established, what was tried and failed, and the per-session table |
| `PAPER.md` | the Section 4 draft, its LaTeX table, and the recommendation on Figures 4/5 |
| `fishsense_imwut/` | `calibration.py` (geometry, median polish, cohort rule, range trend), `pubfig.py` (figures; `nearest_rank_p90` is the estimator of record), `camera.py`, `plots.py` |
| `reconstruction_analysis/` | simulation: reconstruction under known and fitted calibration — ODR, least squares, 2D/3D corrections, `pixel_sensitivity.ipynb`, and `known_calibration_bundle_adjustment.ipynb` (issue #2) |
| `calibration_analysis/` | simulation: which estimator recovers the laser calibration — PCA, weighted PCA, RANSAC, least squares, 2D corrections |
| `laser_labeling_analysis/` | `laser_labels_cleaned.csv`, 37,811 laser-dot labels over 262 dives |
| `tests/` | pins the cohort so a change to the paper's numbers is a diff |

Start with `post_labeling_analysis/HANDOFF.md` §0 and §7 — §0 is the ground rules learned
the hard way, §7 is where a fresh analysis is most likely to repeat a dead end.

## Sibling repos

| repo | paper |
|---|---|
| `../wuwnet-fishsense2026` | flat-port refraction, in-air calibration (WUWNet) |
| `../cscw-fishsense2027` | deployability for citizen scientists; reads this repo's corpus and label export by relative path |
| `../fishsense-lite` | the production pipeline — stage 13 calibration, stage 14 measurement, `range_trend.py` |
