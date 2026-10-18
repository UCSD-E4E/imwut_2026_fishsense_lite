# In-plane laser drift within a dive: what the companion repo can bound

*Results of 2026-10-17. Written for the SenSys paper's agent, which owns what this means for the
full corpus, and for the IMWUT manuscript. Every number is pinned by a test named beside it:
`tests/test_in_plane_drift.py` unless noted, against the cached files in `data/` (see
`data/MANIFEST.md`). Nothing here needs database access to reproduce.*

## The question

FishSense Lite triangulates range from where a laser dot lands on the image, using a fitted laser
beam (the *calibration*). The beam can move in two ways after it is fitted:

- **Out of plane** (out of the plane containing the camera axis and the laser): the dot leaves the
  image line the calibration predicts. That is **visible** in any frame and moves lengths very
  little.
- **In plane**: the dot slides *along* that line, so every range, and therefore every length, is
  rescaled. That is **invisible** in a dive's own frames, because no frame disagrees with the line.
  An in-plane change of δ rescales a length at range z by about 100·δ·z/B percent (B, the
  laser–camera baseline, is about 10.3 cm). At 1.5 m that is about 25 % per degree.

Most dives carry **one calibration**. If the beam moves in plane during the dive, every later
length is wrong by a scale factor nothing in that dive can detect. The question is how large that
exposure is, and what can be claimed about it.

### What is already established (pinned elsewhere)
- Between successive calibrations of one unit, the beam moves a median **0.65° in-plane and
  0.35° out of plane** (18 steps; in-plane the larger in 11). — `test_laser_switching.py::test_the_fleets_steps_on_the_same_basis`
- Within a pool session it can jump: **0.85° in-plane, 0.06° out** in 8.2 min (dives 489→490); and
  **−0.29° in-plane, 0.014° out** across a 10.6-min pause in recording (dive 509). The 509 step
  comes from known lengths, the 489→490 step from fitted beams. —
  `test_laser_switching.py::test_489_to_490_spans_a_pause_in_recording_not_a_dark_run`,
  `::test_509s_trout_steps_at_a_ten_minute_pause_and_the_step_is_in_plane`
- In four of seven field dives, the dots drift off the calibration line at **+0.10 to +0.24° per
  hour** (out of plane), after separating time from range. —
  `test_laser_switching.py::test_field_dots_drift_slowly_off_the_line_in_four_of_seven_dives`

## A. Does visible (out-of-plane) drift predict invisible (in-plane) drift?

**No, on this evidence.**

Method: the same 18 step pairs between successive well-conditioned calibrations (each fit resting
on ≥ 10 frames). Correlate |in-plane| with |out-of-plane| step size; bootstrap the interval
(20,000 resamples, seed 0).

| quantity | value |
|---|---|
| Spearman, magnitudes | **0.23**, 95 % interval **−0.31 to 0.65**, p = 0.36 |
| Pearson, signed | −0.09 |
| regression | \|in\| = 0.84 + 0.10 · \|out\| (degrees) |

The two within-session moves sit apart from any proxy relationship. Both are large in plane
(0.85°, 0.29°) with almost no out-of-plane movement (0.06°, 0.014°). A dot-line check would
have passed both. 489→490 happens to land on the regression's intercept, which is what "no
relationship" predicts: about 0.84° in plane, whatever the out-of-plane movement.

Eighteen pairs are few, and the interval runs from a moderate negative to a strong positive
correlation. The data cannot show that out-of-plane movement is a usable flag, and the two
within-dive events show it failing. — `test_A_out_of_plane_steps_do_not_predict_in_plane_steps`

## B. Do lengths trend with time since calibration?

**Field: no trend is detectable at any rate that matters. Stereo day: no trend detectable either,
at a sharper but still coarse limit.**

**Time since calibration** is measured to the nearest frame the dive's fit rests on: a frame with a
completed slate label and a laser dot. That is the rule `calibration_fits.n_frames` counts by, and
the frame times come from `dive_timeline.csv`. Across the 162 field measurements it runs **0.9 to
51.3 min (median 16.5, 90th percentile 35.5)**. — `test_B_field_times_since_calibration`

**Field, species-normalized.**
- Method: each measurement divided by its species' median per-animal p90 length (percent), then
  regressed on hours since calibration, with a separate intercept per dive.

| | slope (%/h) | 95 % interval | smallest detectable* |
|---|---|---|---|
| all 162, dive intercepts | **+18.8** | **−0.6 to +38.2** | **27.6** |
| all 162, pooled without intercepts | +12.3 | −2.9 to +27.5 | 21.7 |
| without "Other (nontarget)" (152), dive intercepts | +14.5 | −4.9 to +33.8 | — |

\* Smallest detectable slope at 80 % power, two-sided α = 0.05.

- At the cohort's median range (1.48 m), 27.6 %/h is about **1.1° per hour** in plane, four to
  five times the worst out-of-plane field rate. — `test_B_field_length_trend_is_undetectable_at_useful_rates`
- **What this regression mostly measures** is which animals were filmed early or late. Two
  deployments show per-dive slopes whose intervals exclude zero (347: +61 %/h; 465: +55 %/h). Those
  are composition effects, larger or different fish later in the dive, not a credible instrument
  effect.
- **The within-animal version is impossible.** Fish don't grow, but no animal is filmed across the
  dive: its frames are a median **1 second** apart (longest **2.4 min**, 42 animals with ≥ 2
  frames). — `test_B_within_an_animal_there_is_no_time_to_regress_on`

**Stereo day (7 fish against a stereo-video reference, one fish per dive).**
- Method: each fish's FishSense Lite p90 minus its stereo length (percent), regressed on that
  fish's median time since calibration.
- Result: **−6.8 %/h (−26.7 to +13.0), smallest detectable 26.3 %/h**, about 0.9° per hour at the
  frames' median 1.78 m.
- A frame-by-frame regression gives −8.5 %/h (−15.4 to −1.6), an interval that excludes zero. But
  39 frames of 7 fish are not 39 independent samples, and fish-to-fish disagreement (−13.8 % to
  +11.2 %) dominates. Do not quote the frame-level interval. — `test_B_stereo_difference_against_time`

## C. The field cohort's exposure, under assumed rates

For each measurement:
- **Tolerance:** the in-plane angle that would move that length by the 15 % budget at its own
  range, using the existing `pointing_budget_deg` at the frame's z rather than 2 m. The tighter
  side is used, because a drift's sign is unknown.
- **Accumulated drift:** rate × time since calibration, at two **assumed** rates: 0.24° per hour
  (the worst out-of-plane field rate) and 0.48° per hour (twice that, after the fleet's 2:1
  in-plane : out-of-plane step ratio).

Field tolerances run **0.19° to 1.73° (median 0.52°)**.

| | 0.24°/h | 0.48°/h |
|---|---|---|
| field, timed to the nearest fit frame | **162 / 162** inside (worst 0.70 of tolerance) | **157 / 162** (worst 1.40; out: 4 in dive 347, 1 in 383) |
| field, timed to the farthest fit frame | 162 / 162 (worst 0.77) | 147 / 162 (worst 1.54) |
| stereo day (39 frames) | **39 / 39** (worst 0.90) | **29 / 39** (worst 1.80; out: fish of dives 16, 28, 35, 39) |

— `test_C_field_exposure_under_assumed_rates`, `test_C_stereo_day_exposure_and_timing`

**Stereo-day timing** (for the manuscript's "26 minutes to 1 h 42"):
- The day's calibration (dive 32) rests on two slate frames, at **08:58:20 and 08:58:27 UTC**.
- The measured fish run **25.9 to 102.6 min** after the later frame.
- **102.6 min is 1 h 42 min 33 s, which rounds to 1 h 43.** "1 h 42" is the truncated figure.

— `test_C_stereo_day_exposure_and_timing`

**Read these as conditional.**
- The tolerance is the *whole* 15 % budget. Drift consumes it before pose and dot placement add
  their own error.
- The rates are assumptions. **No in-plane rate within a field dive has been measured**, and A
  shows the out-of-plane rate does not predict it.
- The pool's two within-session in-plane moves were **steps** (0.29°, 0.85°), not slow rates. A
  single 0.85° step would exceed the median field tolerance (0.52°).

## D. Second calibrations within a dive

**None can be split into two fits.** Frames a fit can rest on (completed slate label + dot), by
field deployment:

| dive | frames the fit rests on | measurements | design |
|---|---|---|---|
| 341, 347, 349 | one burst each, < 2 min long | after it | opening only |
| 279 | 3 frames, 12:33:55–12:34:01 | 12:07–12:28 | **closing only** |
| 383 | 2 frames, 07:52:25–07:52:37 | 07:01–07:17 | **closing only** |
| 465 | 09:18:08 + 10:09:32, 10:09:36 | 09:19–09:53 | bracketed, 1 frame + 2 frames |
| 471 | 09:16:33 + 10:08:37 | 09:21–10:00 | bracketed, 1 frame each end |

- **465 and 471 already straddle their measurements**, but one end holds a single frame, so neither
  end can be fitted alone. Their stored line is a joint fit across 51–52 min. If the beam moved in
  plane between the ends, that line is a compromise applied to every length, and nothing in the
  fit can reveal it.
- **279's opening burst shows the slate in 13 frames, none labeled for a fit.** Labeling them
  (a production action) would give a separate opening fit to compare with its closing one.
- The stereo day (dive 32) has one burst of two frames.

— `test_D_no_dive_has_two_separately_fittable_calibrations`

**Not computable from the cache:** fitting a burst needs each frame's slate pose and dot as a 3D
observation. Those live in production (slate labels and their fitted observations), not here.

## E. The 496 → 509 step on one basis

- **0.46° in-plane, 1.54° out of plane.** On the vertical/horizontal basis the same step is 0.00°
  vertical, 1.61° horizontal.
- The in-plane figure includes the change in fitted baseline, folded in at 2 m, which is why it is
  not zero although the vertical component is.

— `test_E_the_496_to_509_step_on_both_bases`

## Which claim the evidence supports

- **Bounded:** *not supported as stated.* C keeps ≥ 97 % of field lengths inside tolerance at both
  assumed rates (100 % at 0.24°/h). But B cannot show "no trend" at a useful sensitivity (≈ 1°/h
  in the field, ≈ 0.9°/h on the stereo day), and the rates in C are assumptions. The bound is
  conditional: *if* in-plane drift within a field dive runs no faster than twice the observed
  out-of-plane rate, nearly all field lengths stay within budget.
- **Detectable:** *not supported.* Out-of-plane movement does not predict in-plane movement (A), and
  both observed within-session in-plane moves would have passed a dot-line check.
- **Present:** *not shown.* No length trend is detectable (B), and no dive yields a second fit (D).
  Absence is not shown either.
- **Inconclusive:** *supported.* IMWUT should state that in-plane drift within a dive is not
  measurable on single-calibration dives. It should give C as a conditional bound, note that the
  pool showed in-plane steps of 0.29° and 0.85° within sessions, and recommend a closing
  calibration. Two field dives (465, 471) already have frames at both ends, but too few to fit
  each end separately; the recommendation is a closing **burst**, not a closing frame.

## What holds for the 162-measurement field cohort, and what needs the full corpus

**Holds here:** A–E as above.

**Needs the full corpus (SenSys):**

1. **A with more step pairs.** Every dive with its own calibration gives a step against the unit's
   previous one. The full corpus could take the 18 pairs to hundreds and settle whether any proxy
   relationship exists.
   Query: all `laserextrinsics` with dive time, camera and the fit's frame count (the
   `extract_calibration_fits.sql` columns), then `cal.laser_drift` and `cal.laser_step_directions`
   unchanged.
2. **B with population leverage.** Within-animal leverage does not exist. What scales is
   species-normalized length against time across many dives, with dive intercepts. Its
   detectability improves roughly with the square root of the measurement count; at about 5,000
   measurements the field figure of 1.1°/h would fall towards 0.2°/h. Individuals re-identified
   across a dive (if the `fish` table links frames minutes apart) would give the stronger,
   within-animal test.
3. **C for every measurement.** Per measurement, the query needs:
   - the image's `taken_datetime` and the measurement's `laserdepth.depth_m`;
   - the extrinsics actually used (`measurement.laser_extrinsics_id` → `laserextrinsics.dive_id`)
     and its baseline (hypot of `laser_position` x and y);
   - the times of the frames that fit rests on: on the slate path, images in the calibration dive
     with a completed, live `diveslatelabel` and a live `laserlabel` with x/y; on the checkerboard
     path, laser-labeled images whose species label is 'Calibration Targets, E4E Checkerboard'.

   Then `pointing_budget_deg(B, z)` per measurement, as in `test_C_field_exposure_under_assumed_rates`.
4. **D across the corpus.** Per dive, take the fit-eligible frames above and cluster them by time
   (a new cluster after a > 5 min gap). Count:
   - dives with ≥ 2 clusters;
   - dives where ≥ 2 clusters each hold ≥ 2 frames, i.e. each can be fitted alone;
   - dives whose clusters straddle the dive's measurements.

   Also count frames whose species label is a slate but carry no completed slate label. Those are
   fits waiting on labels, as in dive 279. Fitting each fittable cluster separately needs the
   per-frame slate observations (production), and the in-plane change between the two fits is then
   `cal.laser_beam_change_deg`, as in A.

## F. Measuring in-plane drift across a field dive (added 2026-10-17)

*Tests: `tests/test_in_plane_ends.py`.*

Three field dives carry calibration frames at more than one time (D): 279, 465 and 471. The
question was whether those frames allow a **direct** measurement of in-plane drift across the
dive, without new labels.

### F1. A label-free fit of each end: not possible on these dives

Two label-free routes exist.

1. **Automated slate corners** (`2026-07-31_slate_training`: template search, ECC, a small UNet,
   `solvePnP`). The SenSys repo's E1 (`cscw-fishsense2027/calibration_analysis/e1_label_free`)
   ran production's own calibration code on pool slate dives with predicted corners and detector
   dots. **6 of 12 dives passed production's gates, and one (62) passed while 2.1 points
   wrong.** That predictor was retired for false fits.
2. **Size constancy**, SenSys's reef method (`e1j_size_constancy/reef_labelfree.py`). SAM 3.1
   segments the slate under the dot; its apparent size falls as 1/range along the dive's laser
   line, and the point on that line where the size would reach zero fixes the in-plane angle.
   No corners and no slate size are needed. On reef dives 341, 347, 349 and 436, the SenSys
   repo reports the label-free angle within **0.04–0.18°** of the stored calibration (its
   `reef_labelfree_cal.csv`; SenSys's number, not pinned here). It needs **at least 8 frames
   with the dot on the slate**, spread in range.

Neither can fit an end of these three dives, because the ends do not hold the frames. Frames with
the dot **on** the slate, per end — `test_no_end_of_these_dives_supports_its_own_fit`:

| dive | opening | closing |
|---|---|---|
| 279 | **0**: all 13 opening slate frames were answered "Laser not on slate"; 12 have no dot at all | 3 (the stored fit) |
| 465 | 1 | 2 |
| 471 | 1 | 1 |

So **279's opening burst cannot yield a calibration by any method, labeled or label-free**, and
Step 2 of the request (opening against closing on 279) cannot be done. The field dives with
enough laser-on-slate frames for size constancy (341, 347, 349) each have a single burst.

### F2. The ends tested against each other with the existing labels

The test needs no new labels:
- Each slate frame's dot, placed on the slate plane by PnP on its existing corner labels, is a 3D
  point whose range does not depend on the laser (`data/slate_points.csv`, written by
  `derive_slate_points.py` through production's stage-13 geometry via the SenSys E1 harness).
- The harness reproduces the stored fits of 279, 465 and 471 exactly. For 465 this needs the
  opening label's skipped corner (point 5), read from production, because the SenSys extract
  omits it.
- A frame's laser-derived range, under a line fitted at the **other** end, disagrees with its pose
  range by however far the beam swung in plane, plus pose and dot noise.

**279** — all three fit frames are in the closing burst; nothing to test.

**471** — one frame per end. The stored line passes exactly through both (agreement within
0.01 %, `test_the_slate_points`). Two points leave nothing over to test, so **in-plane drift on 471
is unmeasurable**.

**465** — the one testable dive. `test_465_opening_frame_against_its_closing_line`:
- The two closing frames sit at 1.37 m and 2.51 m, 1.15 m apart, which is enough lever arm to fix
  a line by themselves.
- The opening frame, **51.4 min** earlier at 1.16 m, reads **1.91 % short** against that closing
  line: an in-plane change of **−0.095°**.
- The tolerance at that range (15 % budget, tighter side) is **0.66°**, so the change is 15 % of
  the tolerance.
- Under the dive's single stored line, the two closing frames disagree by +1.08 % and −0.53 %. The
  stored line is a compromise across the 51 minutes.

**Noise floor for this exact design** — `test_the_noise_floor_of_that_design`:
- The same test where nothing can have drifted: inside one burst (341, 347; 349 has no pair 1 m
  apart), fit a line through two frames at least 1 m apart in range and test a third.
- |disagreement|: median **0.57 %**, 90th percentile **1.97 %**, 95th **4.19 %**.
- 13 % of draws reach 465's 1.91 %.
- In degrees, at 465's opening range: median **0.03°**, 95th percentile **0.20°**.

**So, on dive 465 over 51 minutes: no in-plane drift distinguishable from noise.** The test could
have shown a change larger than about 0.2° at that range, a third of the tolerance; anything
smaller is within its noise. It rests on one opening frame.

### F3. Extending it

- **Within the cache:** no other field dive or the stereo day has calibration frames at more than
  one time. 383 has only a closing burst and the stereo day (dive 32) a single two-frame burst.
- **Across the corpus (SenSys):**
  1. Select dives with laser-on-slate frames in ≥ 2 time clusters (D's query, with species label
     'Slate, Laser on slate' or a completed `diveslatelabel` with a live dot).
  2. Pull corner labels **with** `skipped_points` (the SenSys extract lacks the column) and the
     slate template per `dive.dive_slate_id`.
  3. Run `derive_slate_points.py`'s geometry, then this section's test. Where one end has ≥ 8
     laser-on-slate frames, size constancy can fit that end label-free and compare angles
     directly, as in A.
  4. Build the noise floor for each design (frames per end, lever arm), as here, from
     single-burst dives.

### F4. With every frame's automated predictions (v2) and SAM slate size

The fishsense-services (v2) database ran a slate-presence model on **every** frame and
the laser detector on many frames nobody labeled. The data come from its 2026-10-18 backup,
restored read-only. Two pieces of SenSys's size-constancy method make the ends usable without
new labels:
- **SAM 3.1 mask under the dot.** It segments the slate under the dot and returns a size only
  when the slate mask contains the dot. Frames where the dot misses the slate drop out by
  construction.
- **Size as range.** A slate's size goes as 1/range, so one frame with a pose range fixes the
  scale and every other frame's size gives its range.

Inside single reef bursts, size-implied range agrees with pose range to within **0.8–2.3 %**
(coefficient of variation, 341 / 347 / 349). — `test_size_gives_range_to_a_percent_or_two`

Frames per end (slate-positive with a dot / sized, i.e. dot inside the slate mask) —
`test_automated_predictions_reach_frames_nobody_labeled`:

| dive | opening | closing |
|---|---|---|
| 279 | 20 / 3 (19 detector dots) | 7 / 3 |
| 465 | 4 / 4 | 3 / 3 |
| 471 | 3 / 3 | 1 / 1 |

No end reaches the 8 frames a full size-constancy fit wants. But one end's frames can still fix
a line by themselves: pose range where a slate pose exists, size-implied range for the rest,
with the size scale taken from that end only. The other end's pose frames are then tested
against it, as in F2.

**279: the opening detections are not the laser.**
- The detector puts dots on the slate in 78686 and 78688 at y = 903 and 810 px. Under the
  closing fit that is **0.49 and 0.42 m**, while the slate's size puts it at **0.91 and
  0.98 m**.
- One second later (78690), the detector puts the dot **445 px** further along the line, on the
  diver's chest, at about the same range.
- The dive's laser is red and blooms at these ranges (closing frame 78797). These are small red
  specks.

They are not the measuring laser on the slate. The laser was off, or not visible, at the
opening, consistent with the labelers' "no dot". **279 stays unmeasurable.** Detector dots
also need a consistency gate before use, because the SAM gate does not catch this. —
`test_279s_opening_detections_are_not_its_laser`

**465, both directions** — `test_465_both_directions`:
- The opening line (4 frames, 1.16–2.72 m) reads the two closing pose frames **+3.09 % and
  +3.72 %** (**+0.127°, +0.083°**).
- The closing line (3 frames, 1.25–2.51 m) reads the opening pose frame **−1.42 %**
  (**−0.071°**).
- Both directions agree in sign: about **0.1° over 51 minutes**.

**471** — the opening line (3 frames, 1.12–3.51 m) reads the closing pose frame, **51.6 min**
later, **−0.76 % (−0.017°)**. The closing end has one frame and cannot fit a line. —
`test_471_opening_line_against_the_closing_frame`

**Noise floor for this design** — `test_noise_floor_of_the_size_ranged_design`:
- Inside single reef bursts: scale from one pose frame, line through it and two size-ranged
  frames ≥ 1 m apart, test a fourth frame's pose range.
- |disagreement|: median **1.06 %**, 90th percentile **3.19 %**, 95th **4.25 %**.
- In angle at the test frame's range: median **0.031°**, 90th **0.103°**, 95th **0.127°**.
- 465's +3.09 % and +3.72 % sit at the 89th and 93rd percentiles. 465's −1.42 % and 471's
  −0.76 % are ordinary.

### What F supports

- **Dive 465: clean.** Tested both ways, the change across 51 min is about **0.1°**, near the
  90th percentile of the noise floor and well inside tolerance. The tolerance at the test
  frames' ranges is 0.31° at 2.5 m and 0.66° at 1.16 m (15 % budget, tighter side).
- **Dive 471: clean.** −0.017° across 51.6 min, inside noise. Tested one way only.
- **Dive 279: unmeasurable.** Its opening burst has no laser on the slate; the detector's
  opening "dots" are not the laser.

IMWUT can state: *in-plane drift was measured directly on two field dives across about 51
minutes and stayed within about 0.1° — at or below measurement noise and well within the
in-plane tolerance; on the third dive with calibration frames at both ends, the laser was not
visible at the opening.*

**For SenSys:**
- **Scaling.** Section F scales wherever v2 holds slate-presence and laser predictions and a
  dive has slate frames at two times. The queries are the one in `derive_slate_sizes.py`
  (widened to all dives) and D's clustering.
- **SAM sizing** runs on the render cache at about one frame per second on a laptop GPU.
- **A required gate:** detector dots need a consistency check, and the SAM "mask contains the
  dot" test is not enough. Two candidates: agreement of laser range with size-implied range,
  and stability of the dot's along-line position within a burst.

## Reproduce

```bash
uv run pytest -q tests/test_in_plane_drift.py tests/test_laser_switching.py tests/test_in_plane_ends.py
```

Data: `calibration_fits.csv`, `dive_timeline.csv`, `laser_state.csv`, `field.csv`,
`stereo_pairs.csv`, `stereo_reference.csv`, `corpus.csv`, `slate_points.csv` and `slate_sizes.csv` (derived; see
`derive_slate_points.py` and `derive_slate_sizes.py` for what regenerating them needs).
