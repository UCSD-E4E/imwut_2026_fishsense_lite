# Post-labeling handoff — the full fish-model corpus, 2026-09-12

Written for: an agent starting a fresh analysis of FishSense Lite's length-measurement
accuracy, with no access to the sessions that produced this. Everything needed is
committed in this repository; nothing depends on a scratchpad or a transcript.

The previous handoff, `../fish_model_analysis/HANDOFF.md` (2026-08-26), covers the seven
original fish-model dives and the calibration-repair physics. It is still correct. This
document covers what happened after: a labeling campaign that grew the corpus from 464 to
2,927 measurements, a rule-derived accuracy cohort, the foreshortening experiment, and a
list of things that were tried and shown not to work. Read §1 and §7 before doing anything;
§7 is where a fresh analysis is most likely to repeat a dead end.

---

## 0. Ground rules that were learned the hard way

- **Known lengths are the validation set, never a calibration source.** Anything that
  fits a calibration to the models and then grades it on the models is circular. The one
  exception in the record is scale-free: "the ruler must read the same length at two
  ranges" spends no reference length (`fish_model_analysis/HANDOFF.md` §4).
- **Aggregate a target's frames at $p_{90}$, never a mean.** Foreshortening is one-sided
  negative. A mean measures the diver's poses.
- **Per-dive error is an angle, not a percentage.** −8 % and −1 % are 0.29° and 0.03° of
  in-plane laser rotation; percent is confounded by range (0.15° is −2.2 % at 0.9 m and
  −5.1 % at 2.2 m, measured on dive 60's geometry).
- **Never "snap" laser labels to a fitted line.** It moves depths arbitrarily. Stage 13
  intersects the slate plane at the labelled pixel; the pixel is the observation.
- **Prod writes are the user's.** The analysis environment cannot write to the production
  database or API (an automated policy blocks it). Write the SQL, hand it over, verify
  afterwards read-only. Read-only queries are fine and expected.
- **A folder name is not evidence.** "the name isn't enough to justify why it's wrong" —
  a dive is excluded on measured grounds or not at all.
- **Do not "fix" the Shark reference** or any other reference from the data. See §6.
- **A median against a known length is not a calibration check.** Two calibrations with
  14 % and 9 % short baselines graded "−1 %" on the median because a flat scale error and
  a compensating angle error cancel at mid-range (§3.1, §7.3). The scale-free range trend
  is the check; it spends no known length.

---

## 1. What is where

### This repository (`imwut_2026_fishsense_lite`, branch `corpus-accuracy-analysis`)

| path | what |
|---|---|
| `fish_model_analysis/data/corpus.csv` | **the corpus**: 2,927 rows, one per `Measurement` of a rigid target, 32 dives, re-pulled 2026-09-12 23:0x UTC after the 490 → 527 split (490 has no rows; 527 has 62). `\|`-delimited (geometry columns are JSON). Columns: `dive_id, calibration_dive_id, camera_id, model, known_length_m, length_m, depth_m, dot (x;y px), leid, pos, ax, km`. |
| `fish_model_analysis/data/all.csv` | the August 464-row export; strict subset of the corpus (tested). |
| `fish_model_analysis/data/angles.csv` | the foreshortening experiment: 1,428 rows, `dive_id,image_id,taken_datetime,fish_angle_degrees,angle_category,length_m,depth_m`. Snook, 455 mm. |
| `fish_model_analysis/sql/extract_corpus.sql`, `extract_angles.sql` | the psql that produced both; re-runnable read-only. `extract_angles.sql` was re-run on 2026-09-12 and reproduces `angles.csv` to the export's rounding. |
| `fishsense_imwut/calibration.py` | geometry (triangulation, φ rotation, implied-yaw floor), `median_polish`, `cell_p90_grid`, `theil_sen`, `range_trend`, `range_trend_flagged_dives`, `accuracy_cohort`, `load_angles`, `binned_angle_error`, and every cohort constant. Read its comments; they carry the reasoning. |
| `fishsense_imwut/pubfig.py` | publication style and every figure function. `nearest_rank_p90` is the estimator of record. |
| `fish_model_analysis/fish_model_measurements.ipynb` | the analysis. Executes clean top to bottom; cell 3 asserts the August handoff numbers, cell 5 asserts the cohort. |
| `tests/test_calibration.py` | `uv run pytest -q tests` — 17 tests; pins the polish, the Theil–Sen interval, the range-trend pre-filter, the rule, and the published 13-dive cohort. |
| `fish_model_analysis/FINDINGS.md` | §1–6 August findings; **§7 the corpus results**. |
| `fish_model_analysis/HANDOFF.md` | August physics and repair method. |
| `PAPER.md` | Section 4 draft and the recommendation to cut Figures 4/5. |
| `fish_model_analysis/figures/` | fig1–3, 6, 8, A on the corpus cohort; fig4, 5, 7 on the August data. |

Run everything with `uv run` from the repository root; the notebook from
`fish_model_analysis/` (`DATA` paths are relative to it).

### The pipeline repository (`../fishsense-lite`)

`CLAUDE.md` there is long and accurate; the sections that matter here are "Laser depth
(`LaserDepth`) + calibration provenance", "Borrowed laser calibration", "Laser-label
validation" (the within-dive line), and "Priority.NONE semantics". Its
`services/fishsense-data-processing-workflow-worker/.../perform_laser_calibration_activity.py`
is stage 13; `measure_fish_activity.py` is stage 14; `laser_geometry.py` is the shared
kernel. The stage-13 plausibility gate (baseline 7.8–14.5 cm) and the count-mismatch guard
fixed in v3.8.1 (PR #887) live there.

Merged to `main` on 2026-09-12, all from this work: `range_trend.py` +
`scripts/audit_length_range_trend.py` (the scale-free range-trend audit, PRs #891/#893),
`scripts/audit_scale_against_checkerboard.py` with a corrected justification (#891), the
`DiveLaserLine` docstrings (#890), and `fishsense_shared/calibration_bounds.py` with the
baseline floor raised 7.8 → 9.7 cm (#894). The data-worker `feat` commits will roll into its
next release. To run the range-trend audit against prod, paste it into the api-worker
container (`docker exec -i fishsense-fishsense-api-workflow-worker-1 /app/.venv/bin/python -`
with the module inlined and `fishsense_api_workflow_worker.config.settings` for the SDK
credentials); that is how it was validated.

### Production data

Postgres in the Incus slot; read-only recipe used throughout:

```
ssh krg-admin@krg-nat.ucsd.edu 'incus exec fishsense --project fishsense -- \
  docker exec -i fishsense-postgres-1 psql -U postgres -d fishsense -c "<sql>"'
```

`-At -F'|'` for machine output. psql `\pset` echo lines are *not* suppressed by `-A`; strip
non-12-field lines before parsing (this bit once: a stray `Field separator is "|".` line
put a quote into the CSV and broke `csv.DictReader`). NAS is mounted read-only at
`~/mnt/fishsense_data`; `Dive.path` values are relative to `REEF/data/` for the REEF dumps
and `__unsorted-data/` for the 2023 pool tests. Processed JPEGs live in Garage; raw `.ORF`
on the NAS (decode with `rawpy`; the 16-bit raw shows red laser dots that the 8-bit JPEG
saturates away on white targets — `rawpy.postprocess(no_auto_bright=True, output_bps=16,
highlight_mode=Ignore, gamma=(1,1))`).

### Data model, as it matters here

- `dive` — `priority` (`HIGH` = in every hourly cohort; `LOW` = not yet; `NONE` = parked on
  purpose, with `notes`), `calibration_dive_id` (borrow), `dive_slate_id`,
  `calibration_target_id` (checkerboard), `calibration_refused_at/_reason/_labels_at`
  (stage 13 refused to persist; the reason text is worth reading).
- `laserextrinsics` — `laser_position` (ℓ), `laser_axis` (α), one row per self-calibrated
  dive. Baseline = ‖ℓ_xy‖, ~10.3–10.4 cm on this fleet.
- `laserlabel` — dot pixel; `completed`, `superseded` (dead-letter, never deleted), `x/y`.
  A dive can carry two live labels on one image.
- `specieslabel.content_of_image` — `"Fish Model, <name>"`, `"Calibration Targets, Box"`,
  `"Calibration Targets, Ruler"`; `fish_angle_degrees` on the angle dives.
- `headtaillabel`, `diveslatelabel` — as named.
- `measurement` — `length_m`, `laser_extrinsics_id` (**provenance**: which calibration
  produced it). `laserdepth` — `depth_m` (Z), `range_m` (slant), `residual_m` (see §7.2).
- `fishmodelreference` — `known_length_m` per model name.

---

## 2. The corpus

32 dives, 2,927 measurements, 8 targets (Box 150 mm, Purple Angel 192, Weasly Fish 310 — a
stylised rainbow trout model; "Ginny"/"George" in folder names,
Ruler 342.9, Grouper 360, Snook 455, Shark 605, and 2 frames of Yellow Anthias). Laser
range 0.25–5.47 m.

Three populations:

1. **2023 slate-borrow dives** (58, 59, 60, 61, 66, 76, 84): fish-only, each borrows a
   companion slate dive's extrinsics shot ~30 min earlier (`BORROW_MAP` in
   `calibration.py`). The August analysis.
2. **2023 angle experiment** (87, 94, 103, 107, 114; parked sixth session 526): one Snook
   at 0–45° in 5° steps; self-calibrated from a slate burst in the same folder. 87/94 at
   ~4 m, 103/107/114 at ~2 m.
3. **2025 pool tests** (480–522, ingested 2026-09-02): Box + Weasly Fish, sometimes Shark;
   **checkerboard** calibration (E4E board, 14×10 interior corners, 42 mm pitch, ±1.2 %),
   self or borrowed from a same-day, same-rig sibling. `camera_id` does **not** track the
   physical rig number. Dive 527 is 490's fish frames (§5). Three of these calibrations
   are now known wrong on a scale-free test and, since PR #894, outside the api's baseline
   bounds: 502 (8.90 cm, borrowed by 503/504) and 498 (9.51 cm).

The per-dive table (n, calibration source, polish dive effect, status) is at the end of
this document (§9) and is regenerated by notebook cell 9.

---

## 3. What is established

### 3.1 The accuracy cohort and its rule (`cal.accuracy_cohort`)

Tukey median polish over the $p_{90}$ percent error of each (dive, model) cell with ≥ 5
frames, fitted on all 30 polishable dives. Cell = overall + dive effect + model effect +
residual; median |residual| **0.15 pp**. Keep |dive effect| ≤ 2.5 pp. Two hold-outs are
applied to the result:

1. **By design** (`DESIGN_EXCLUDED_DIVES`), fixed before any number was computed: the angle
   dives, 60 and 76 (August repairs), 66 (disputed).
2. **By the scale-free range trend** (`range_trend_flagged_dives`, added 2026-09-12; ported
   from fishsense-lite `range_trend.py` and agreeing to the decimal): a rigid object must
   read the same length at every range, so the Theil–Sen slope of length vs laser depth
   (frames ≥ 0.8 m, ≥ 8 frames, ≥ 2× spread) estimates the in-plane calibration error with no
   known length. A dive drops when a cell's whole 95 % interval clears ±2 %/m. Both signs:
   rotated axis → negative; **short fitted baseline → positive**, because the LS fit pairs it
   with a compensating angle, and that kind is invisible to the polish (503: −14 % at 0.8 m,
   +0.4 % beyond 3.5 m, $p_{90}$ −0.75). Flags **76, 491, 492, 494, 503, 504, 509** — all
   known bad — and no sound-baseline dive. 498 (9.51 cm, +2.8 %/m, interval [+2.0, +3.8]) is
   the borderline that stays.

   **Its constants barely matter, and that is the strongest argument that it was not tuned.**
   Any threshold from 2.0 to 4.0 %/m selects the identical 13-dive cohort with identical
   numbers (only ≤ 1.5 %/m moves it, to 12 dives and $p_{90}$ +0.09); `MIN_DEPTH_M` over
   0.6–1.0 and `MIN_FRAMES` over 6–12 change the flag set not at all. Three of the four
   constants are inert on this corpus and the fourth sits on a two-point plateau. Worth
   quoting alongside the polish sweep; the paper now does.

   **Known limitation — it flags a dive when ANY cell flags, and both multi-cell flags here
   are cells their sibling contradicts.** 76: Shark +5.33 [+4.19, +6.77] against Purple Angel
   −2.24 [−3.55, +2.72]. 509: Box −4.11 [−4.52, −3.64] against Weasly +0.06 [−0.79, +1.39].
   A calibration error moves every target on the dive together — that is the premise the
   polish rests on — so a lone flagging cell its sibling contradicts is a model effect, which
   is exactly what §6.1 says about 76's Shark. Requiring agreement among usable cells flags
   (491, 492, 494, 503, 504) and yields the **identical cohort**, because 509 is out on the
   polish band (−2.83) and 76 is design-excluded. So it costs nothing here and the code is
   deliberately left matching fishsense-lite's `range_trend.py` to the decimal. It is not
   free in general: on a dive with a sound calibration and one anomalous target it rejects
   the dive. The guard belongs in `range_trend.py` first, then here.

Two structural facts about the grid, both load-bearing and neither visible in the 0.15 pp:

- **52 of 210 cells are observed, against 36 free parameters, and 20 of the 30 dives carry
  exactly one cell** — on those the dive effect absorbs the cell and the residual is
  identically zero by construction. Over the ten multi-model dives the median |residual| is
  **0.45 pp** (max 4.33). The additive model is well supported; it is not "essentially
  exact", and quoting 0.15 pp as though it were fitted over 30 × 7 cells overstates it.
- **The grid is disconnected.** The 2023 dives span {Grouper, Purple Angel, Ruler, Shark,
  Snook} and the 2025 dives span {Box, Weasly Fish}, with no cell in common. Dive and model
  effects are therefore identified only *within* each component; the polish's global
  centring is what puts the two on one scale, and the rule compares dive effects across
  that seam. Shifting the 2025 block by ±1 pp — which the data cannot rule out — moves
  membership and the cohort $p_{90}$ by roughly as much as the 1.5→3.5 pp threshold sweep
  does. The headline survives every variant, but §6.3's checkerboard-vs-slate gap is this
  unidentifiable quantity, not a measurement.

Result on the corpus (re-pulled after the 490 → 527 split, §5):
**59, 61, 84, 495, 497, 498, 500, 501, 507, 519, 520, 521, 522** — n = 771, median −2.19 %,
$p_{90}$ +0.06 %, mean |err| 3.71 %; 78 % of frames within 5 %, 98 % within 10 %, 99 %
within 15 %. Threshold sweep 1.5→3.5 pp: 8→15 dives, $p_{90}$ −0.42…+0.45 %. 491 (−2.73)
and 527 (+2.95) sit just outside the polish band; 527 reads like the three 2023 slate dives
(n = 207, median −1.17, $p_{90}$ +1.29), ~1.5 pp above the ten checkerboard dives (n = 564,
median −2.67, $p_{90}$ −0.42). That gap is real and unexplained (§6.3).

Ladder ($p_{90}$ per model on the cohort): Box −0.14, Purple Angel +0.85, Weasly −1.89,
Grouper +0.29, Snook −0.94, Shark +4.29.

Model effects (pp): Snook −0.57, Box −0.38, Weasly −0.38, Grouper 0.00, Ruler +0.08,
Purple Angel +1.21, **Shark +3.23**.

Range (Figure 3): binned median flat at −1.8…−2.1 % from 0.8 to 4.7 m. Below 0.8 m the
**Weasly Fish alone** reads −6.8 % (Box −2.3 % at the same ranges); the earlier "corpus-wide
near-range bias" was mostly the short-baseline dives, now filtered. See §6.4.

### 3.2 The foreshortening curve (Figure 8)

Pooled median error by designed angle: 0° −3.8, 5° −3.5, 10° −4.8, 15° −5.8, 20° −8.7,
25° −11.7, **30° −15.0**, 35° −20.5, 40° −25.0, 45° −31.4 %. Tracks $\cos\theta - 1$
offset by ~−3.5 %. Five sessions at two ranges agree within the pooled IQR. Broadside
(0–5°) per session: 87 −3.4, 94 −3.0 (both ~4 m); 103 −5.7, 107 −2.4, 114 −5.7 (~2 m).
Read the angle from `fish_angle_degrees` (the card in frame), never from
`fish_angle_category` — the categories merged 15° with 20° during labeling.

### 3.3 Physics that constrains everything (proofs in `fish_model_analysis/HANDOFF.md` §1)

- The image of the laser line fixes only the plane through the camera centre containing
  it. In-plane rotation φ and the baseline length are **invisible to the dots**
  (~1e-13 px). Scale error lives exactly there.
- A dot offset perpendicular to the epipolar line → residual, harmless. Along the line →
  depth error, residual-silent.
- Length is linear in depth (snout and tail back-project at one depth), so any scale
  hypothesis can be tested by rescaling recorded lengths without relabelling
  (`length_ratios`).
- Foreshortening is one-sided; the implied-yaw floor (10th percentile of the yaw a rigid
  target would need to read its measured length) separates calibration error (lifts every
  target's floor) from pose (one-sided tail). It clips at 0 and is blind to over-reading.

---

## 4. What changed in production during this work

| when | what | why |
|---|---|---|
| 2026-09-11 | stage-13 count-mismatch guard now logs and returns `None` instead of raising (data-worker v3.8.1, PR #887 → #888 → #889) | dive 526 (and any dive with a slate frame lacking a laser label) crashed the hourly parent instead of being skipped |
| 2026-09-11 | 23 superseded laser labels on dive 347 un-superseded (user ran `unsupersede.sql`); 14 usable observations restored | the labels had been superseded by a legacy detector run, not by a labeler |
| 2026-09-11 | dives 518 and 522 self-calibrated (10.35 / 10.29 cm) | drained after the guard fix |
| 2026-09-12 | **dive 526 parked** (`priority = NONE`, note appended) | 17 slate frames in a 13 s single-distance burst; stage 13 refused at 2.00 cm baseline; no other slate frames; a borrow from 107 would be validatable only by the angle experiment itself |
| 2026-09-12 22:5x | **dive 490 split**: its 70 pre-burst frames and 13 PREDICTION clusters became **dive 527** (borrows 489); user ran `split_490.sql` | the laser rotated 0.82° between those frames and 490's own board burst (§5); borrow resolution is single-hop, so re-linking would have stranded 491/492 |
| 2026-09-12 23:0x | `ComputeLaserDepthsParentWorkflow` and `MeasureFishParentWorkflow` fired by hand from inside the api-worker container (`trigger_527.py`; user-authorised); both drained 527 | 62 measurements re-derived under extrinsics 37: median −1.70 %, p90 +1.28 % |
| 2026-09-12 (on next api deploy) | baseline floor 7.8 → 9.7 cm (fishsense-lite #894) | 498 (9.51) and 502 (8.90) are wrong calibrations hidden by a cancelling median; the api will read 498/502/503/504 as uncalibrated and re-enter them into the calibration cohorts, where the same fit is refused again — **they need parking or the robust fit; not done** |

Not applied, superseded by later analysis: `park_calibration_folders.sql` (would have
parked the LaserCalibration folders by name). The Canyonview dataset
(`~/mnt/fishsense_data/2025.05.30.FishSense.Canyonview`: more Box + Weasly frames, Ginny
= the Weasly Fish, checkerboard `LaserCalibration` folders that would borrow) was examined
and deliberately **not** ingested — it adds frames of the two targets already best covered.

---

## 5. Dive 490 — RESOLVED 2026-09-12: the laser moved between its fish frames and its board burst

This section originally read "irreducible contradiction". It is not. The read-only prod
timeline for rig FSL-02D on 2023-08-14 (`image.taken_datetime`) — but see §6.9: this date
contradicts §2's classification of 480–522 as the 2025 pool tests, and one of the two is
wrong. The ordering below is what the argument rests on, and it is unaffected either way:

| when (UTC) | what | dive | extrinsics |
|---|---|---|---|
| 18:33–18:37 | George (Weasly Fish) | 492 | borrows 490 |
| 18:38–18:39 | Box | 491 | borrows 490 |
| 18:59:20–19:00:40 | `LaserCalibration2` board burst | 489 | id 37, own |
| **19:00:49–19:02** | **490's 68 Weasly Fish frames** | 490 | measured under id 39 |
| 19:07:25–19:08:53 | `LaserCalibration3` board burst | 490 | id 39, own |

Extrinsics 37 (489) and 39 (490) differ by **0.823° in-plane**, 0.06° out-of-plane,
baselines 10.33 / 10.24 cm. Re-measuring 490's fish frames under 489's row (nine seconds
older than the first fish frame): median −1.70 %, $p_{90}$ +1.28 %, residual φ +0.02°,
Spearman ρ(depth, error) from −0.985 to +0.50 with the binned medians flat at −5.4 (near)
→ 0.3 (far) — i.e. the corpus-wide near-range bias and nothing else. So the laser rotated
between 19:02 and 19:07: **after** the fish frames and **before** the board burst that
calibrated them. The fish frames belong to the calibration that precedes them.

Why the earlier evidence was not exculpatory: "the board dots reproject correctly" and
"plane-from-corners agrees with laser depth on the board frames" are exactly what a φ error
looks like, because φ is invisible to reprojection (§3.3). The 23-point range trend in the
error (−15 % at 0.5 m → −38 % at 2.9 m) was the tell: a baseline-scale error is range-flat,
a pointing error grows with range at φ/b per metre.

491/492 (shot 25 minutes before either burst) fit **neither** row — 0.21° from 39, 0.65°
from 37 — so the laser was in a third state then. Their contemporaneous burst,
`LaserCalibration1` (dive 488), is parked for pool caustics with its checkerboard species
labels lost when its LS projects were deleted; its 28 laser labels survive.

Consequences:
- 490's ~14 % error is a **pointing** error, fully explained; it is not evidence about the
  checkerboard producer or about labels. It stays out of the accuracy cohort (its stored
  measurements are still wrong) until stage 14 revisits. **Prod fix applied 2026-09-12
  22:5x UTC:** the 70 pre-burst frames and their 13 clusters were split into **dive 527**
  ("LaserCalibration3 fish", borrows 489). A re-link was not possible — borrow resolution
  is single-hop, so deleting row 39 would have stranded 491/492. 527's 62 measurements
  named extrinsics 39 at the time of the split; the depth and stage-14 cohorts re-derive
  on that mismatch. Both parents were fired by hand at 23:0x UTC and drained 527: 62
  measurements now name extrinsics 37, median −1.70 %, $p_{90}$ +1.28 % — the offline
  prediction to the decimal. `corpus.csv` was re-pulled: 490 has 0 rows, 527 has 62. 527
  does **not** enter the cohort (+2.95 pp; §3.1).
- After the re-pull 491 (−2.73) is out of the cohort and 527 (+2.95) is not in it; both are
  within half a point of the band. 491 borrows 490's row across a ~0.2° laser movement.
- The general detector — the range trend of a rigid object's length, scale-free — now
  exists in fishsense-lite as `range_trend.py`, merged to `main` (PRs #891/#893; see §1).
  On the re-pulled corpus it flags **76, 491, 492, 494, 503, 504, 509** and no cohort dive
  (490 no longer has rows; the authoritative statement of the flag set and its sensitivity
  is §3.1). Both signs are signal — the positive side is what caught 503/504/498's short
  baselines — but it is blind to range-flat baseline errors (506/507 via 505), and no
  wild-fish dive in prod has the range spread it needs (§6.8).

Earlier claims from the session that were wrong and should not be repeated: that 490's far
frames had no fish (misread of a downscaled render); that the folder held two laser states
in the sense of the fish frames coming *after* the board (they came before).

## 6. Other open questions

1. **Shark reference.** +3.23 pp model effect ≈ +20 mm on 605 mm. Only the Box and
   Weasly Fish are physically available to re-measure (both already within 0.4 pp — not
   worth the hour). The Shark is 25 of 771 cohort frames; dropping it moves $p_{90}$ from
   +0.06 to −0.24 %. On dive 76 it also reads +5.3 %/m against −2.2 for the Purple Angel on
   the same dive, which one calibration angle cannot do — a Shark-specific range effect
   (its long body? its landmark?) that the range-trend note names as "not a calibration
   finding". Reported as-is. The fork-vs-total-length hypothesis was excluded by
   magnitude (`FINDINGS.md` §3.5).
2. **Boundary dives.** After the re-pull and the pre-filter: 491 (−2.73) and 527 (+2.95)
   sit just outside the polish band; 498 (9.51 cm, range trend +2.8 %/m, interval
   [+2.0, +3.8]) is inside the band and just short of the range-trend flag — the borderline
   case, and since #894 a dive the api no longer counts as calibrated. 58 (+3.14) and 506
   (+3.97) are the positive-side rejects. The rule is defensible; the membership is not
   sacred.
3. **Checkerboard vs slate offset — not estimable as posed.** Inside the cohort the ten 2025
   checkerboard dives (median −2.67, $p_{90}$ −0.42) sit ~1.5 pp below the three 2023
   slate-borrow dives (−1.17, +1.29), and 527 — a checkerboard-calibrated dive — reads like
   the slate ones. But the two groups share no target (§3.1), so calibration method is
   perfectly confounded with target set, and the polish cannot break it: the gap *is* the
   quantity the disconnected grid leaves to a centring convention. Candidates remain the
   board's own ±1.2 % pitch tolerance, a systematic in corner detection, and the target sets
   themselves; the near-range frames are no longer one (§6.4). **The only thing that settles
   it is a shared target**: one dive that photographs a fish model and a Box/Weasly, or a
   slate and a checkerboard (check `dive_slate_id` and `calibration_target_id` both
   non-null). Until then, report the gap as unidentified rather than unexplained.
4. **Near-range bias — explained.** Part was the short-baseline dives (503/504/498, now
   filtered). The rest is **half-thickness parallax**: the laser dot lands on a solid model's
   flank while the snout and fork lie in its midplane, so the length reads short by
   (half thickness / range). Fitting error = a + b/z per target: trout b = −1.8 cm
   [−2.4, −1.3], Snook −2.4, Grouper −5.0, Box −0.4 (≈ 0). A property of the method, small
   for real fish at survey range (−1 % for a 40 cm fish at 2 m). See FINDINGS §7.6.
9. **The trout's reference.** 310 mm is unprovenanced. An SfM scan scaled on the Box's tape
   span gives 315 ± 1 mm snout-to-fork (FINDINGS §7.6, landmarks verified on the cloud and
   on six labelled frames). Against 315 the trout still carries a range-flat −4 pp relative
   to the Box that parallax does not explain: scan scale, mount yaw, or the Box reference.
   A tape on the physical model (snout tip to fork, on its side) is the arbiter. Reference
   left at 310 until then.
5. **60 / 66 / 76.** The August repairs and dispute stand as written; 60's raw dive effect
   on the corpus polish is −0.87 (inside the band) because the corpus median moved, which
   is why it is held out by design rather than by threshold.
6. **526** — parked; could be un-parked if a same-rig calibration ever becomes
   defensible (see §4).
7. **498 / 502 / 503 / 504 after #894.** Their calibrations are wrong (range trend +2.8 to
   +5.5 %/m on 8.90–9.51 cm baselines) and the api will stop honouring them. Refitting
   the same board observations gives the same fit — the unrobust least-squares problem
   CLAUDE.md names as the open follow-up. Until a robust fit exists: park them, and treat
   their stored measurements as wrong (503 and 504 are the −14 to −18 % close-range rows in
   the corpus).
8. **Field applicability of the range-trend check is zero today.** Every 2023 August
   fish-model dive and every wild fish in prod is "insufficient" (no object measured ≥ 8
   times over a ≥ 2× range spread beyond 0.8 m). Only the 2025 pool sessions and the angle
   tests qualify. It becomes a field check only if the protocol adds a rigid reference
   photographed at two ranges after calibrating.
9. **Unresolved: are dives 489–492 from 2023 or 2025?** §5's prod timeline dates them to
   **2023-08-14** on rig FSL-02D; §2 puts 480–522 in the **2025** pool tests, and `PAPER.md`
   §4.1 sells the cohort as "three slate-calibrated 2023 sessions and ten
   checkerboard-calibrated 2025 sessions". `angles.csv` puts the angle dives at 2023-08-31,
   so 2023 dates in the 480s are not absurd. Nothing in the cohort turns on it — 489–492 and
   527 are all excluded — but the paper's era labeling and §6.3's framing do. One read-only
   query settles it and was not run (prod reads were unavailable in the session that
   found this):

   ```sql
   SELECT dive_id, min(taken_datetime)::date, max(taken_datetime)::date, count(*)
   FROM image WHERE dive_id BETWEEN 480 AND 527 GROUP BY 1 ORDER BY 1;
   ```

---

## 7. Things that were tried and do not work — do not re-derive

Each of these was proposed, implemented, tested against the whole corpus at the user's
insistence ("run this check against all the dives before we do anything else with it"),
and discarded.

### 7.1 Line-fingerprint agreement cannot rank calibration borrows

Proposed three separate times; falsified three times. Two dives whose laser lines agree to
0.2° in the image can differ by 3° in φ (dives 383/471: +44 % and +305 % length error).
Same line ⇏ same rig state, and same rig ⇏ same line (the laser rotates *inside* its
clamp, sweeping a cone). The within-dive line is sound for outlier rejection and for
detecting moved predictions; it is never a prior for another dive. The `DiveLaserLine`
docstring in fishsense-lite still claims otherwise — do not build on it.

### 7.2 `LaserDepth.residual_m` does not predict error

Spearman ρ(residual, |pct error|) = −0.026, n = 464; the extremes invert (dive 84: 10× the
residual, best error). Scale error is along the epipolar line, the one direction the
residual cannot see. Record it, never gate on it.

### 7.3 ~~A per-rig baseline gate is a false-positive machine~~ — WITHDRAWN 2026-09-12

This section claimed that 498 (9.51 cm), 502 (8.90) and 107 (12.95) "measure at −1.1…−1.6 %"
and so a tighter baseline gate would false-flag. The −1 % was a *median*, and the range trend
(§3.1) showed it was cancelling a −14 to −18 % close-range error against a +5 %/m ramp: 498
and 502's borrowers are wrong calibrations. The fishsense-lite floor moved from 7.8 to
9.7 cm on that evidence (PR #894). 107 at 12.95 remains untested — its frames span no range.
The surviving lesson: **a median against a known length is not a calibration check.**

### 7.4 The checkerboard "laser-free" scale audit is near-tautological on its own frames

Comparing laser-derived depth to plane-from-corners depth *on the dot-on-board frames*
agrees to ±0.6 % on every dive including 490 — because the calibration was fitted on those
very frames. It cannot see a scale error that shows up on the fish frames. Its one
non-circular use is a frame where a rigid target and the board co-occur with the dot on the
target (7 such frames on 490); that is rare. The script's docstring now says exactly this
(fishsense-lite #891). For the general case use the range-trend audit (§3.1).

### 7.5 Snapping labels to the line, and excluding folders by name

Both rejected by the user on principle (§0). Snapping changes depths arbitrarily; a name
is not evidence.

### 7.6 Session range does not explain session offsets

Angle sessions at ~2 m vs ~4.5 m differ by +0.8 to +4.4 pp in broadside error, in no
consistent direction; range is not the variable.

---

## 8. Suggested starting points for a fresh analysis

Deliberately not conclusions — things the record leaves open and a fresh pair of eyes
could settle with the data already in `data/`.

- **490** is resolved (§5). The open piece is 491/492: neither burst fits them, and their
  own burst (488) has laser labels but no board labels. Re-labelling 488's board frames
  would give the three dives a contemporaneous calibration.
- **The Weasly-Fish close-range under-read** (§6.4): −6.8 % below 0.8 m against −2.3 % for
  the Box at the same ranges, on sound dives. Compare the fish's pixel span × depth / f
  against 310 mm per frame to see whether labelers click inside the outline when the model
  fills the frame, or whether the fork landmark moves.
- **Why do 87 and 114 read positive even broadside-only** (+3.1 / +5.5 %/m on ≤ 5° frames)?
  Single-object angle sessions on sound 10.2–10.3 cm baselines. Pose-by-range was the
  guess and the broadside cut argues against it. Unresolved; they are held out by design.
- **Checkerboard vs slate** (§6.3): the 1.8 pp gap is testable on dives that have both a
  slate and a checkerboard frame set, if any exist (check `dive_slate_id` and
  `calibration_target_id` both non-null).
- **A per-dive φ on the corpus**: `fit_phi_joint` runs on any dive with ≥ 2 targets. Its
  distribution over 30 dives is the honest "how much does the mount move" figure; the
  seven-dive version spans 0.27°.
- **Do not** start from the laser line, the residual, or a median against a known length
  as an accuracy proxy (§7). The baseline IS a diagnostic now, in the other direction: a
  fitted baseline under ~9.7 cm is a wrong calibration (§7.3), and the range trend says so
  without a reference.

---

## 9. Per-dive table

Generated from `corpus.csv` by notebook cell 9 (`polish` = Tukey median polish over the
$p_{90}$ cells; `cal_src` = dive that owns the extrinsics actually used, `self` when own).

| dive | n | cal_src | baseline (cm) | models | range (m) | median % | p90 % | dive effect (pp) | status |
|---|---|---|---|---|---|---|---|---|---|
| 492 | 20 | 490 | 10.24 | Weasly Fish | 0.37–3.15 | -9.10 | -7.48 | -5.83 | out: range trend (calibration) |
| 103 | 197 | 83 | 10.37 | Snook | 1.85–2.03 | -15.63 | -5.51 | -3.68 | held out: angle experiment |
| 494 | 41 | 493 | 10.51 | Box | 0.28–3.45 | -12.72 | -5.12 | -3.47 | out: range trend (calibration) |
| 76 | 91 | 63 | 10.43 | Grouper, Purple Angel, Shark, Snook | 0.86–2.49 | -7.70 | -3.95 | -3.44 | held out: August repair |
| 509 | 162 | self | 10.42 | Box, Weasly Fish | 0.30–3.68 | -7.32 | -3.28 | -2.83 | out: range trend (calibration) |
| 491 | 28 | 490 | 10.24 | Box | 0.32–2.84 | -5.90 | -4.38 | -2.73 | out: range trend (calibration) |
| 66 | 51 | 83 | 10.37 | Grouper, Purple Angel, Ruler, Shark, Snook | 1.08–3.13 | -5.34 | -1.08 | -2.47 | held out: disputed |
| 520 | 40 | 518 | 10.35 | Weasly Fish | 0.39–2.80 | -4.26 | -3.73 | -2.08 | **ACCURACY** |
| 522 | 157 | self | 10.29 | Box, Weasly Fish | 0.36–3.06 | -3.74 | -0.81 | -1.22 | **ACCURACY** |
| 60 | 103 | 65 | 10.45 | Grouper, Purple Angel, Ruler, Shark, Snook | 0.63–2.16 | -3.46 | +0.51 | -1.16 | held out: August repair |
| 94 | 335 | self | 9.87 | Snook | 1.95–4.77 | -9.07 | -2.94 | -1.11 | held out: angle experiment |
| 114 | 341 | self | 10.27 | Snook | 1.85–5.06 | -11.01 | -2.85 | -1.02 | held out: angle experiment |
| 107 | 180 | self | 12.95 | Snook | 1.96–2.06 | -10.77 | -2.54 | -0.70 | held out: angle experiment |
| 501 | 42 | self | 10.12 | Weasly Fish | 0.27–4.35 | -5.67 | -2.21 | -0.56 | **ACCURACY** |
| 87 | 375 | self | 10.2 | Snook | 1.89–5.47 | -10.04 | -1.86 | -0.03 | held out: angle experiment |
| 498 | 35 | self | 9.51 | Weasly Fish | 0.32–4.08 | -6.89 | -1.62 | +0.03 | **ACCURACY** |
| 495 | 28 | 493 | 10.51 | Weasly Fish | 0.40–3.49 | -3.37 | -1.09 | +0.56 | **ACCURACY** |
| 61 | 69 | 80 | 10.44 | Grouper, Purple Angel, Snook | 1.21–2.46 | -1.28 | -0.29 | +0.68 | **ACCURACY** |
| 521 | 79 | self | 10.45 | Box, Weasly Fish | 0.41–4.06 | -2.36 | -0.73 | +0.75 | **ACCURACY** |
| 84 | 52 | 62 | 10.37 | Grouper, Purple Angel, Shark, Snook | 0.89–2.51 | -1.02 | +1.14 | +0.75 | **ACCURACY** |
| 503 | 43 | 502 | 8.9 | Box | 0.25–3.86 | -8.09 | -0.75 | +0.90 | out: range trend (calibration) |
| 519 | 51 | 518 | 10.35 | Box | 0.29–2.67 | -1.21 | -0.39 | +1.26 | **ACCURACY** |
| 500 | 50 | 499 | 10.41 | Box | 0.36–4.71 | -1.34 | +0.09 | +1.74 | **ACCURACY** |
| 507 | 40 | 505 | 10.39 | Weasly Fish | 0.30–3.43 | -2.60 | +0.20 | +1.85 | **ACCURACY** |
| 59 | 86 | 77 | 10.37 | Grouper, Purple Angel, Shark, Snook | 0.89–2.78 | -0.82 | +2.08 | +2.04 | **ACCURACY** |
| 497 | 42 | 496 | 10.36 | Box | 0.38–4.30 | -1.15 | +0.43 | +2.08 | **ACCURACY** |
| 504 | 50 | 502 | 8.9 | Weasly Fish | 0.25–5.30 | -6.67 | +1.06 | +2.71 | out: range trend (calibration) |
| 527 | 62 | 489 | 10.33 | Weasly Fish | 0.59–5.04 | -1.70 | +1.30 | +2.95 | out: |effect| > 2.5 (490's fish frames, borrows 489) |
| 58 | 25 | 71 | 10.49 | Grouper, Purple Angel, Shark | 1.26–2.72 | +2.14 | +4.82 | +3.14 | out: |effect| > 2.5 |
| 506 | 49 | 505 | 10.39 | Box | 0.31–4.75 | +0.10 | +2.32 | +3.97 | out: |effect| > 2.5 |
| 436 | 1 | self | 10.5 | Yellow Anthias | 1.73–1.73 | -8.69 | -8.69 | — | no cell with ≥5 frames |
| 505 | 2 | self | 10.39 | Weasly Fish | 4.41–4.59 | -2.04 | -1.73 | — | no cell with ≥5 frames |
