# The data cache — what a reader without database access needs

**Every number in `PAPER.md` is reproducible from the files in this directory alone.**
Nothing in the reproduction path touches the production database, and nothing in
`fishsense_imwut/` imports a database client. This directory *is* the cache; the SQL in
`../sql/` records how it was pulled, so the extraction is auditable even by a reader who
cannot run it.

```bash
uv sync                       # no `db` group: no DB client, no Rust/C toolchain
uv run pytest -q tests        # 251 tests, all against these files
uv run jupyter lab            # fish_model_analysis/fish_model_measurements.ipynb
```

`uv sync` deliberately does **not** install the `sim` group; nothing needed to reproduce
the paper's numbers is in it. Re-exporting these files (see *Refreshing the cache* below)
needs database credentials an outside reader does not have and does not need, but it goes
through `psql` rather than through any Python client — this repository has none.

## What each file is

`loaded` is the row count its canonical loader returns, which is what
`tests/test_data_cache.py` pins. It differs from the line count where a file carries a
header, a trailing psql `(n rows)` footer, or rows the loader drops.

| file | loaded | pulled by | what it backs |
|---|---|---|---|
| `corpus.csv` | 2927 | `sql/extract_corpus.sql` | **the corpus.** Every rigid-target measurement in prod, 31 pool dives. Backs §4.1–§4.3: the cohort rule, the median polish, every accuracy number and Figures 1–3. |
| `angles.csv` | 1428 | `sql/extract_angles.sql` | §4.4, the designed foreshortening experiment — one Snook stepped 0–45° over dives 87/94/103/107/114. |
| `field.csv` | 162 | `sql/extract_field.sql` | §4.1's wild fish: 162 measurements of 73 individuals over seven Florida-reef deployments. **Extracted 2026-10-14 from the fishsense-services (v2) production database**, after PR #72 regrouped three frames (101302 → animal 307, 129275 → 331, 111926 → new 575); every other file here comes from the legacy database, which still carries the old grouping and measurement ids. Lengths, ranges and depths are identical in both. |
| `calibration_fits.csv` | 31 | `sql/extract_calibration_fits.sql` | every accepted laser calibration. Backs §4.3's checkerboard-vs-slate baseline comparison (the calibration **object** per fit is recorded nowhere else) and the laser-drift figure, which needs the full fitted beam (`laser_*`, `axis_*`), the timestamp, and `n_frames` — how many frames the fit rests on. |
| `laser_colour.csv` | 3130 | `sql/extract_laser_colour.sql` | the colour tag (red/green) on the dot every measurement used, one row per measurement. Backs the colour split of each cohort; read only through `calibration.load_laser_colour`. Legacy database: three field frames carry measurement ids that `field.csv` (v2) has replaced, so join the two on `image_id`. |
| `dive_timeline.csv` | 5437 | `sql/extract_dive_timeline.sql` | per-image shot times, dot pixels and species label for every dive that owns a calibration or holds a measurement. Backs the field-runnable checks on a session's own calibration (lever arm in laser range, bracketing, time gap, dot-line residual); read through `calibration.load_dive_timeline`. Board frames are the ones labeled 'E4E Checkerboard'. **No per-frame board pose**: production does not store one, so the pose-derived range of a checkerboard calibration is not recoverable here or in production. |
| `laser_state.csv` | 6108 | `sql/extract_laser_state.sql` | every image of every dive that owns a calibration or holds a measurement, labeled or not: laser label state (dot / empty / none), dot pixels, the automated detector's confidence and gate verdict. Backs the search for laser on/off switches; read through `calibration.load_laser_state`. **Laser state is not logged anywhere**: an empty label means no visible dot, which a switched-off laser and a dot out of frame both produce. |
| `slate_points.csv` | 60 | `derive_slate_points.py` (derived) | each field slate frame's dot as a 3D point on the slate plane (PnP on the human corner labels, production's stage-13 geometry), dives 279, 341, 347, 349, 465, 471. A range that does not depend on the laser. **Derived, not extracted:** regenerating it needs the SenSys repo's E1 harness and its environment (OpenCV, compiled fishsense_core); everything downstream runs here. |
| `slate_sizes.csv` | 90 | `derive_slate_sizes.py` (derived) | apparent slate size under the dot (SAM 3.1 mask containing the dot, sqrt(area) at the 1600-px render) for the calibration ends of 279, 465, 471 and the single bursts of 341, 347, 349, with the dot's source (human label or automated detector). Detector dots and slate-presence scores come from the fishsense-services (v2) backup of 2026-10-18; sizes from the SenSys repo's `sam_sizes.py`. **Derived:** regenerating it needs that backup, the SenSys repo and a GPU. |
| `label_repeats.csv` | 9850 | `sql/extract_label_repeats.sql` | repeat positions of one landmark on one frame: every human laser label on frames carrying two or more, the detector's dot beside the live human dot, every human snout/fork label on frames carrying two or more, the head/tail predictor beside the live human label; with each dive's fitted image line. Backs section 5.7's labeling-noise terms. **Most repeats are not independent** (labels seeded from predictions): see `tests/test_triangulation_noise.py`. |
| `stereo_pairs.csv` | 41 | `sql/extract_stereo_pairs.sql` | our side of the paired stereo day (2023-08-03): 41 frames of the individuals the archive also measured. |
| `head_tail.csv` | 1051 | `sql/extract_head_tail.sql` | per-frame head/tail label pixels. **No header row.** |
| `stereo_reference.csv` | 8 | derived, no SQL | the stereo side of the paired day, lifted from `stereo_archive.csv`. Eight references; seven match one of ours, so `build_pairs` yields seven. |
| `stereo_archive.csv` | 3503 | **external** | the collaborators' `SMILE_Archive_LengthData.csv` verbatim. Not ours, not from our database; the source `stereo_reference.csv` was cut from. |
| `all.csv` | 464 | no committed SQL | the August 2026-08-26 seven-dive handoff export. Backs `HANDOFF.md` and the repair figures. Not a subset of `corpus.csv` — six mislabel corrections differ deliberately. |
| `shark_grouper.csv` | 228 | no committed SQL | Shark and Grouper frames with label pixels and distortion, for the §3.5 fork probe. |
| `ruler.csv` | 28 | no committed SQL | the printed measuring board, read by `measure_ruler_scale.py`. |
| `corpus_20260912.csv` | 2927 | `sql/extract_corpus.sql` | frozen 2026-09-12 export. Kept **only** because the reference-sensitivity result is reproducible against it and nothing else. |
| `corpus_20260916.csv` | 2927 | `sql/extract_corpus.sql` | frozen export at the corrected 0.04217 m grid pitch; `corpus.csv` is its working copy. |

## The two things a reader should know before trusting a number

**The delimiter is `|`, not `,`, for every file pulled from prod.** The geometry columns
are JSON and contain commas. `cal.load_rows` handles this; `pd.read_csv` with defaults
will silently mangle these files.

**`corpus.csv` carries 2,927 rows but the accuracy cohort is 995 measurements over 19
sessions.** The gap is the cohort rule, not missing data — `cal.accuracy_cohort` drops the
design hold-outs and the sessions the scale-free range check rejects, and the held-out
targets come out separately. The rule is in `calibration.py` and pinned in
`tests/test_calibration.py`; it is not a filter anyone applied by hand.

## Refreshing the cache

Only needed when prod changes. Requires credentials.

```bash
psql "$FISHSENSE_DSN" -A -F'|' -f sql/extract_corpus.sql -o data/corpus.csv
```

Re-export writes the psql footer and the pipe delimiter that the loaders expect; do not
hand-edit the result. Display-name fixes (`DISPLAY_NAMES` in `calibration.py`) are applied
at **load** time on purpose, so a re-export stays byte-identical to what the database
returned and the cache can always be diffed against a fresh pull.

After any re-export, run `uv run pytest -q tests`. The suite pins the cohort, the polish
and the headline numbers, so a change in prod shows up as a test diff rather than as a
quietly different figure.
