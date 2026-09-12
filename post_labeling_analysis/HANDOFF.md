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

---

## 1. What is where

### This repository (`imwut_2026_fishsense_lite`, branch `corpus-accuracy-analysis`)

| path | what |
|---|---|
| `fish_model_analysis/data/corpus.csv` | **the corpus**: 2,927 rows, one per `Measurement` of a rigid target, 32 dives, as of 2026-09-12. `\|`-delimited (geometry columns are JSON). Columns: `dive_id, calibration_dive_id, camera_id, model, known_length_m, length_m, depth_m, dot (x;y px), leid, pos, ax, km`. |
| `fish_model_analysis/data/all.csv` | the August 464-row export; strict subset of the corpus (tested). |
| `fish_model_analysis/data/angles.csv` | the foreshortening experiment: 1,428 rows, `dive_id,image_id,taken_datetime,fish_angle_degrees,angle_category,length_m,depth_m`. Snook, 455 mm. |
| `fish_model_analysis/sql/extract_corpus.sql`, `extract_angles.sql` | the psql that produced both; re-runnable read-only. `extract_angles.sql` was re-run on 2026-09-12 and reproduces `angles.csv` to the export's rounding. |
| `fishsense_imwut/calibration.py` | geometry (triangulation, φ rotation, implied-yaw floor), `median_polish`, `cell_p90_grid`, `accuracy_cohort`, `load_angles`, `binned_angle_error`, and every cohort constant. Read its comments; they carry the reasoning. |
| `fishsense_imwut/pubfig.py` | publication style and every figure function. `nearest_rank_p90` is the estimator of record. |
| `fish_model_analysis/fish_model_measurements.ipynb` | the analysis. Executes clean top to bottom; cell 3 asserts the August handoff numbers, cell 5 asserts the cohort. |
| `tests/test_calibration.py` | `uv run pytest -q tests` — 11 tests; pins the polish, the rule, and the published 15-dive cohort. |
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

A worktree `checkerboard-scale-audit` holds an unpushed commit `c0aa0a93`
(`scripts/audit_scale_against_checkerboard.py`). Its docstring's justification is circular
(§7.4); do not build on it without rewriting the justification.

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

32 dives, 2,927 measurements, 8 targets (Box 150 mm, Purple Angel 192, Weasly Fish 310,
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
   physical rig number.

The per-dive table (n, calibration source, polish dive effect, status) is at the end of
this document (§9) and is regenerated by notebook cell 9.

---

## 3. What is established

### 3.1 The accuracy cohort and its rule (`cal.accuracy_cohort`)

Tukey median polish over the $p_{90}$ percent error of each (dive, model) cell with ≥ 5
frames, fitted on all 30 polishable dives. Cell = overall + dive effect + model effect +
residual; median |residual| **0.15 pp** — the additive model is essentially exact. Keep
|dive effect| ≤ 2.5 pp. Held out by design (`DESIGN_EXCLUDED_DIVES`), fixed before any
number was computed: the angle dives (single-object; pose-dominated), 60 and 76 (August
repairs), 66 (disputed). Dive 490 is *not* on that list and falls out on its own (−13.8).

Result: **59, 61, 84, 491, 495, 497, 498, 500, 501, 503, 507, 519, 520, 521, 522** —
n = 842, median −2.37 %, $p_{90}$ +0.02 %, mean |err| 4.04 %; 74 % of frames within 5 %,
95 % within 10 %, 99 % within 15 %. Threshold sweep 1.5→3.5 pp: 8→19 dives, $p_{90}$
−0.45…+0.02 %. Sub-populations inside the cohort: 2023 slate-borrow (3 dives, n = 207)
median −1.17 / $p_{90}$ +1.29; 2025 checkerboard (12 dives, n = 635) median −3.01 /
$p_{90}$ −0.43. That 1.8-point gap between the two calibration methods is real and
unexplained (§6.3).

Ladder ($p_{90}$ per model on the cohort): Box −0.24, Purple Angel +0.85, Weasly −1.89,
Grouper +0.29, Snook −0.94, Shark +4.29.

Model effects (pp): Snook −0.57, Box −0.38, Weasly −0.38, Grouper 0.00, Ruler +0.08,
Purple Angel +1.21, **Shark +3.23**. Snook's August effect (−2.36) shrank once the angle
dives contributed best-presented frames — it was pose. Shark's grew.

Range (Figure 3): binned median flat at −1.9…−2.3 % from 0.8 to 4.7 m; **below 0.8 m the
median is −5.8 %** (108 frames, all Box + Weasly). Unexplained; see §6.4.

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

Not applied, superseded by later analysis: `park_calibration_folders.sql` (would have
parked the LaserCalibration folders by name). The Canyonview dataset
(`~/mnt/fishsense_data/2025.05.30.FishSense.Canyonview`: more Box + Weasly frames, Ginny
= the Weasly Fish, checkerboard `LaserCalibration` folders that would borrow) was examined
and deliberately **not** ingested — it adds frames of the two targets already best covered.

---

## 5. Dive 490 — RESOLVED 2026-09-12: the laser moved between its fish frames and its board burst

This section originally read "irreducible contradiction". It is not. The read-only prod
timeline for rig FSL-02D on 2023-08-14 (`image.taken_datetime`):

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
  on that mismatch. **Re-pull `corpus.csv` after that** (`sql/extract_corpus.sql`): 490
  will drop to 0 measurements, 527 will appear at ~−1.7 % median and is expected to enter
  the accuracy cohort on the rule; re-run the notebook and the pinned cohort test will
  need updating.
- 491 is in the accuracy cohort on the rule (dive effect −2.44) while borrowing across a
  laser movement of ~0.2°. The rule is honest about it; a reviewer may still ask.
- The general detector — the range trend of a rigid object's length, scale-free — now
  exists in fishsense-lite as `range_trend.py` (branch
  `chore/checkerboard-scale-audit-script`). On this corpus it flags exactly 490, 494, 509,
  491, 492 and no cohort cell; only its negative side is signal (§6.4's close-range
  under-read produces positive slopes on good dives); it is blind to range-flat baseline
  errors (506/507 via 505); and no wild-fish dive in prod has the range spread it needs.

Earlier claims from the session that were wrong and should not be repeated: that 490's far
frames had no fish (misread of a downscaled render); that the folder held two laser states
in the sense of the fish frames coming *after* the board (they came before).

## 6. Other open questions

1. **Shark reference.** +3.23 pp model effect ≈ +20 mm on 605 mm. Only the Box and
   Weasly Fish are physically available to re-measure (both already within 0.4 pp — not
   worth the hour). The Shark is 25 of 842 cohort frames; dropping it moves $p_{90}$ from
   +0.02 to −0.28 %. Reported as-is. The fork-vs-total-length hypothesis was excluded by
   magnitude (`FINDINGS.md` §3.5).
2. **Boundary dives.** 509 (−2.53), 491 (−2.44), 520 (−1.79) on the negative side and 504
   (+3.00), 58 (+3.44), 506 (+4.27) on the positive side are within a point of the
   threshold. The rule is defensible; the membership is not sacred.
3. **Checkerboard vs slate offset.** Inside the cohort the 2025 checkerboard dives sit
   1.8 pp below the 2023 slate-borrow dives. Candidates: the board's own ±1.2 % pitch
   tolerance; a systematic in corner detection; target set (Box/Weasly vs the fish models);
   the near-range frames (§6.4) which are all 2025.
4. **Near-range bias.** Below 0.8 m the cohort median is −5.8 % vs −2.1 % beyond, all Box
   and Weasly frames. Hypotheses: label placement on a large, close target; laser-dot
   centroid bias when the dot spans many pixels; the box's 150 mm reference vs the edge a
   labeler clicks. The angle dives don't reach this range, so it cannot be pose.
5. **60 / 66 / 76.** The August repairs and dispute stand as written; 60's raw dive effect
   on the corpus polish is −0.87 (inside the band) because the corpus median moved, which
   is why it is held out by design rather than by threshold.
6. **526** — parked; could be un-parked if a same-rig calibration ever becomes
   defensible (see §4).

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

### 7.3 A per-rig baseline gate is a false-positive machine

The plausibility gate (7.8–14.5 cm) catches degenerate fits (2–5 cm, 22 cm). A tighter
per-rig gate does not: 498 (9.51 cm), 502 (8.90), 107 (12.95) are baseline outliers and
measure at −1.1…−1.6 %. The baseline is a diagnostic of fit degeneracy, not of accuracy.

### 7.4 The checkerboard "laser-free" scale audit is near-tautological

Comparing laser-derived depth to plane-from-corners depth *on the board frames* agrees to
±0.6 % on every dive including 490 — because the calibration was fitted on those very
frames. It cannot see a scale error that shows up on the fish frames. The justification in
commit `c0aa0a93`'s docstring is this circular argument.

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
- **The near-range bin** (§6.4): plot error vs. range per target for Box and Weasly
  separately; check whether the bias is in `depth_m` (laser) or in the pixel span
  (labels) by comparing the box's pixel length × depth / f against 150 mm.
- **Checkerboard vs slate** (§6.3): the 1.8 pp gap is testable on dives that have both a
  slate and a checkerboard frame set, if any exist (check `dive_slate_id` and
  `calibration_target_id` both non-null).
- **A per-dive φ on the corpus**: `fit_phi_joint` runs on any dive with ≥ 2 targets. Its
  distribution over 30 dives is the honest "how much does the mount move" figure; the
  seven-dive version spans 0.27°.
- **Do not** start from the laser line, the residual, or the baseline as an accuracy
  proxy (§7).

---

## 9. Per-dive table

Generated from `corpus.csv` by notebook cell 9 (`polish` = Tukey median polish over the
$p_{90}$ cells; `cal_src` = dive that owns the extrinsics actually used, `self` when own).

| dive | n | cal_src | models | range (m) | median % | p90 % | dive effect (pp) | status |
|---|---|---|---|---|---|---|---|---|
| 490 | 62 | self | Weasly Fish | 0.54–2.93 | -27.88 | -15.74 | -13.80 | out: |effect| > 2.5 |
| 492 | 20 | 490 | Weasly Fish | 0.37–3.15 | -9.10 | -7.48 | -5.54 | out: |effect| > 2.5 |
| 103 | 197 | 83 | Snook | 1.85–2.03 | -15.63 | -5.51 | -3.39 | held out: angle experiment |
| 494 | 41 | 493 | Box | 0.28–3.45 | -12.72 | -5.12 | -3.18 | out: |effect| > 2.5 |
| 76 | 91 | 63 | Grouper, Purple Angel, Shark, Snook | 0.86–2.49 | -7.70 | -3.95 | -3.14 | held out: August repair |
| 509 | 162 | self | Box, Weasly Fish | 0.30–3.68 | -7.32 | -3.28 | -2.53 | out: |effect| > 2.5 |
| 491 | 28 | 490 | Box | 0.32–2.84 | -5.90 | -4.38 | -2.44 | **ACCURACY** |
| 66 | 51 | 83 | Grouper, Purple Angel, Ruler, Shark, Snook | 1.08–3.13 | -5.34 | -1.08 | -2.18 | held out: disputed |
| 520 | 40 | 518 | Weasly Fish | 0.39–2.80 | -4.26 | -3.73 | -1.79 | **ACCURACY** |
| 522 | 157 | self | Box, Weasly Fish | 0.36–3.06 | -3.74 | -0.81 | -0.93 | **ACCURACY** |
| 60 | 103 | 65 | Grouper, Purple Angel, Ruler, Shark, Snook | 0.63–2.16 | -3.46 | +0.51 | -0.87 | held out: August repair |
| 94 | 335 | self | Snook | 1.95–4.77 | -9.07 | -2.94 | -0.81 | held out: angle experiment |
| 114 | 341 | self | Snook | 1.85–5.06 | -11.01 | -2.85 | -0.73 | held out: angle experiment |
| 107 | 180 | self | Snook | 1.96–2.06 | -10.77 | -2.54 | -0.41 | held out: angle experiment |
| 501 | 42 | self | Weasly Fish | 0.27–4.35 | -5.67 | -2.21 | -0.26 | **ACCURACY** |
| 87 | 375 | self | Snook | 1.89–5.47 | -10.04 | -1.86 | +0.26 | held out: angle experiment |
| 498 | 35 | self | Weasly Fish | 0.32–4.08 | -6.89 | -1.62 | +0.32 | **ACCURACY** |
| 495 | 28 | 493 | Weasly Fish | 0.40–3.49 | -3.37 | -1.09 | +0.85 | **ACCURACY** |
| 61 | 69 | 80 | Grouper, Purple Angel, Snook | 1.21–2.46 | -1.28 | -0.29 | +0.97 | **ACCURACY** |
| 521 | 79 | self | Box, Weasly Fish | 0.41–4.06 | -2.36 | -0.73 | +1.04 | **ACCURACY** |
| 84 | 52 | 62 | Grouper, Purple Angel, Shark, Snook | 0.89–2.51 | -1.02 | +1.14 | +1.05 | **ACCURACY** |
| 503 | 43 | 502 | Box | 0.25–3.86 | -8.09 | -0.75 | +1.20 | **ACCURACY** |
| 519 | 51 | 518 | Box | 0.29–2.67 | -1.21 | -0.39 | +1.56 | **ACCURACY** |
| 500 | 50 | 499 | Box | 0.36–4.71 | -1.34 | +0.09 | +2.03 | **ACCURACY** |
| 507 | 40 | 505 | Weasly Fish | 0.30–3.43 | -2.60 | +0.20 | +2.15 | **ACCURACY** |
| 59 | 86 | 77 | Grouper, Purple Angel, Shark, Snook | 0.89–2.78 | -0.82 | +2.08 | +2.33 | **ACCURACY** |
| 497 | 42 | 496 | Box | 0.38–4.30 | -1.15 | +0.43 | +2.37 | **ACCURACY** |
| 504 | 50 | 502 | Weasly Fish | 0.25–5.30 | -6.67 | +1.06 | +3.00 | out: |effect| > 2.5 |
| 58 | 25 | 71 | Grouper, Purple Angel, Shark | 1.26–2.72 | +2.14 | +4.82 | +3.44 | out: |effect| > 2.5 |
| 506 | 49 | 505 | Box | 0.31–4.75 | +0.10 | +2.32 | +4.27 | out: |effect| > 2.5 |
| 436 | 1 | self | Yellow Anthias | 1.73–1.73 | -8.69 | -8.69 | — | no cell with ≥5 frames |
| 505 | 2 | self | Weasly Fish | 4.41–4.59 | -2.04 | -1.73 | — | no cell with ≥5 frames |
