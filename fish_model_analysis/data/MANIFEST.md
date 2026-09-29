# The data cache — what a reader without database access needs

**Every number in `PAPER.md` is reproducible from the files in this directory alone.**
Nothing in the reproduction path touches the production database, and nothing in
`fishsense_imwut/` imports a database client. This directory *is* the cache; the SQL in
`../sql/` records how it was pulled, so the extraction is auditable even by a reader who
cannot run it.

```bash
uv sync                       # no `db` group: no DB client, no Rust/C toolchain
uv run pytest -q tests        # 156 tests, all against these files
uv run jupyter lab            # fish_model_analysis/fish_model_measurements.ipynb
```

`uv sync` deliberately does **not** install the `db` group. That group exists only to
re-export these files (see *Refreshing the cache* below) and needs credentials an outside
reader does not have and does not need.

## What each file is

`loaded` is the row count its canonical loader returns, which is what
`tests/test_data_cache.py` pins. It differs from the line count where a file carries a
header, a trailing psql `(n rows)` footer, or rows the loader drops.

| file | loaded | pulled by | what it backs |
|---|---|---|---|
| `corpus.csv` | 2927 | `sql/extract_corpus.sql` | **the corpus.** Every rigid-target measurement in prod, 31 pool dives. Backs §4.1–§4.3: the cohort rule, the median polish, every accuracy number and Figures 1–3. |
| `angles.csv` | 1428 | `sql/extract_angles.sql` | §4.4, the designed foreshortening experiment — one Snook stepped 0–45° over dives 87/94/103/107/114. |
| `field.csv` | 162 | `sql/extract_field.sql` | §4.1's wild fish: 162 measurements of 73 individuals over seven Florida-reef deployments. |
| `calibration_fits.csv` | 31 | `sql/extract_calibration_fits.sql` | §4.3's checkerboard-vs-slate baseline comparison. The calibration **object** per fit is recorded nowhere else, which is why this is its own pull. |
| `stereo_pairs.csv` | 41 | `sql/extract_stereo_pairs.sql` | our side of the paired stereo day (2023-08-03): 41 frames of the individuals the archive also measured. |
| `head_tail.csv` | 1051 | `sql/extract_head_tail.sql` | per-frame head/tail label pixels. **No header row.** |
| `stereo_reference.csv` | 8 | derived, no SQL | the stereo side of the paired day, lifted from `stereo_archive.csv`. Eight references; seven match one of ours, so `build_pairs` yields seven. |
| `stereo_archive.csv` | 3503 | **external** | the collaborators' `SMILE_Archive_LengthData.csv` verbatim. Not ours, not from our database; the source `stereo_reference.csv` was cut from. |
| `all.csv` | 464 | no committed SQL | the August 2026-08-26 seven-dive handoff export. Backs `HANDOFF.md` and the repair figures. Not a subset of `corpus.csv` — six mislabel corrections differ deliberately. |
| `shark_grouper.csv` | 228 | no committed SQL | Shark and Grouper frames with label pixels and distortion, for the §3.5 fork probe. |
| `ruler.csv` | 28 | no committed SQL | the printed measuring board, read by `measure_ruler_scale.py`. |
| `corpus_20260912.csv` | 2927 | `sql/extract_corpus.sql` | frozen 2026-09-12 export. Kept **only** because the reference-sensitivity result is reproducible against it and nothing else. |
| `corpus_20260916.csv` | 2927 | `sql/extract_corpus.sql` | frozen export at the corrected 0.04217 m grid pitch; `corpus.csv` is its working copy. |
| `models.csv` | 437 | no committed SQL | **unused.** No code in this repo reads it. Kept for provenance; delete if it is still unreferenced at submission. |

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
uv sync --group db            # pulls fishsense-meta -> fishsense-core (needs cc/rustc)
psql "$FISHSENSE_DSN" -A -F'|' -f sql/extract_corpus.sql -o data/corpus.csv
```

Re-export writes the psql footer and the pipe delimiter that the loaders expect; do not
hand-edit the result. Display-name fixes (`DISPLAY_NAMES` in `calibration.py`) are applied
at **load** time on purpose, so a re-export stays byte-identical to what the database
returned and the cache can always be diffed against a fresh pull.

After any re-export, run `uv run pytest -q tests`. The suite pins the cohort, the polish
and the headline numbers, so a change in prod shows up as a test diff rather than as a
quietly different figure.
