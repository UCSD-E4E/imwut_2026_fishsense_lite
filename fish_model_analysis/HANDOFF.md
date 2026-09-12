# Fish-model calibration repair — handoff

Working session, 2026-08-26. Goal: decide, per fish-model dive, whether its
**borrowed** laser calibration should be repaired, and with what.

Everything below is measured on the prod DB backup, not inferred.

---

## 1. The physical model (this is the load-bearing idea)

None of the seven fish-model dives self-calibrates. Each is fish-only and
**borrows** a sibling slate dive's `LaserExtrinsics` via `Dive.calibration_dive_id`:
58←71, 59←77, 60←65, 61←80, 66←83, 76←63, 84←62. The slate dives were shot
**~30 minutes earlier on the same rig** (dive 60←65: 35 min; 66←83: 29 min).

Between the two, the laser moves. Because the angles are tiny (tenths of a
degree), any plausible pivot within ~0.5 m of the aperture displaces the origin
by <1.6 mm — which matches the independently measured family stability. So:

> **The borrow error is a rotation about a fixed point, not an arbitrary
> 4-DOF line change.**

Measured on dive 60 (23 ruler frames, 0.79–1.90 m):

| model | params | result | rms | bias |
|---|---|---|---|---|
| A: in-plane φ only, pivot fixed | 1 | φ = −0.1589° | 0.49% | −0.04% |
| B: φ + baseline length | 2 | φ = −0.1459°, baseline +0.39 mm | 0.47% | — |
| C: free 4-DOF line | 4 | origin walks **70.5 mm** | 0.42% | — |

One parameter explains 23 frames over a 2.4× range span to 0.5% rms with no
bias. The 4-DOF fit buys 0.05 pp and sends the origin across the camera — that
is the flat direction, caught in the act.

### DOF bookkeeping

| | DOF |
|---|---|
| a laser ray in general | 4 |
| pivot preserved (baseline held to ≤0.39 mm over 35 min) → change is a rotation | 2 |
| out-of-plane component — free from the dive's own dots, worth ~0.1% of depth | −1 |
| **actually unknown and actually matters** | **1** |

### Why the dot line cannot supply that last DOF

The image of a 3-D line determines exactly one thing: **the plane through the
camera centre containing it.** A plane through a fixed point has 2 DOF — that is
the entire information content of the observed dot line. The line's other 2 DOF
(where it sits *within* that plane) project identically.

Measured on dive 60's real geometry:

| perturbation | line angle | max dot displacement |
|---|---|---|
| **in-plane φ = 0.15°** (the unknown) | 72.48068° | **0.000 px** |
| **in-plane φ = 1.00°** (a −26% length error) | 72.48068° | **0.000 px** |
| **baseline scaled +2%** | 72.48068° | **0.000 px** |
| out-of-plane 0.10° | 72.48068° | 4.943 px |
| baseline azimuth rotated 1° | 73.47344° | 8.599 px |

Machine epsilon, not "weak signal". The line's **angle** encodes the baseline
azimuth; its **offset** encodes the axis's out-of-plane component. Neither is φ,
and neither is baseline length — i.e. **both quantities that set metric scale are
exactly invisible.** This is monocular scale ambiguity.

Corollary: any objective built on dot reprojection has a flat direction along φ.
That is why the earlier scipy reprojection refit gave a *better* fit (1.36→1.24 px)
and an identical measurement error (−7.76→−7.90%), and why a free refit "walks the
origin off-family."

### Two mechanisms, opposite signatures

Baseline is `|o| = 103.3 mm` along `[-0.30, -0.95, 0]` — the laser sits essentially
*above* the camera, so the triangulation plane Π is nearly the vertical plane.

| rotation | out-of-plane (dots see it, depth doesn't) | in-plane (depth error, dots blind) |
|---|---|---|
| cold-shoe **yaw** (about shoe normal ≈ cam Y) | **95.4%** | 29.9% |
| shoe **rock/pitch** (about cam X) | 29.9% | **95.4%** |
| **laser roll in its clamp** (cone of half-angle ε) | sweeps all phases | **up to ε, any phase** |

So the big sloppy DOF (shoe yaw) is benign; the small ones (rock, laser roll) are
what set scale. Observed φ spread across seven dives is 0.36°, implying ε ≥ 0.18° —
an ordinary beam-to-body misalignment.

**This proves the 2-D line fingerprint cannot rank borrows.** Writing the beam
deviation as `ε(cos ψ, sin ψ)`, a matching fingerprint means `sin ψ₁ = sin ψ₂`,
which admits `ψ₂ = ψ₁` (perfect) **or** `ψ₂ = π − ψ₁` (in-plane difference up to
`2ε` — the theoretical maximum). Dive 84 is that second branch: calibration 279
matched the dot line at 3.8 px vs 62's 21 px, implied 1.894 vs 1.477 m for the same
pixel, and graded +27% (0/13) vs −1% (13/13). **Retire the roadmap's
"matching line ⇒ same calibration" claim rather than tuning its tolerance.**

---

## 2. The diagnostic that ended up mattering: implied-yaw floor

Convert each measurement into the out-of-plane yaw it would require, using the
exact single-depth back-projection of a rigid plate (second-order term included —
it matters at 1.1 m):

```
measured = L·cos θ / (1 − ((L/2)·sin θ / z)²)
```

Then take the **10th percentile** over a cell (the best-presented frames). Physics
gives a hard floor: you cannot present better than side-on, so a sound calibration
with at least one good frame must sit at ~0°.

- A **calibration** error lifts every object's floor **uniformly**.
- **Pose** only adds a one-sided tail *above* the floor.

**Limitation, learned the hard way:** the floor **clips at 0 and is blind to
over-correction.** Always read it alongside the signed p90 error — that is what
caught dive 66 (see §3).

---

## 3. Final recommendations — repair two, leave five

| dive | ←cal | verdict | yaw floor before → after | φ | anchor | spent |
|---|---|---|---|---|---|---|
| 58 | 71 | leave | Grouper 0.0, Shark 0.0 | — | — | nothing |
| 59 | 77 | leave | Grouper/Purple/Shark 0.0, Snook 6.5 | — | — | nothing |
| **60** | 65 | **repair** | 12.5 / 11.7 / 12.6 / 17.9 → **all 0.0** | **−0.1458°** | ruler, scale-free | **nothing** |
| 61 | 80 | leave | Purple 1.8, Grouper 6.4, Snook 8.9 | — | — | nothing |
| 66 | 83 | leave, flagged | see below | — | — | nothing |
| **76** | 63 | **repair** | 17.9 / 11.6 / 20.0 / 19.0 → 5.8 / 0.0 / 2.1 / 10.7 | **−0.1505°** | 4 models jointly | models on this dive |
| 84 | 62 | leave | Grouper/Purple/Shark 0.0, Snook 11.1 | — | — | nothing |

Proposed writes are in `sql/apply_corrections.sql` (NOT yet applied to prod).

**Dive 60** is the strong case and costs nothing. Four independent objects sat at a
uniform 11.7–17.9° floor before; all five land at 0.0° after. The φ was fitted
**scale-free** — only "the ruler must read the same length at 0.82 m and 1.85 m",
never its 342.9 mm. That number stays held out and grades the result at **+0.1%**.
Independent check: φ fitted on the 0.8 m cluster predicts the 1.9 m cluster to
**0.35%** (and far→near, 0.54%) — a 2.3× extrapolation with no free parameters left.

**Dive 76** costs the models on that dive. Report via a **1/3 stratified anchor**
drawn from all four models so all four stay represented: φ = −0.1723 ± 0.016,
held-out Grouper +0.1, Purple +2.1, Shark +0.4, Snook −1.1 (mean 0.92%, vs 4.43%
uncorrected). **Apply** the all-frame joint value −0.1505° (standard
split-then-refit). **Exclude the 13 close-range Shark frames (z < 1.4 m)** — their
implied yaw never drops below 18°, so the model was never presented side-on there.

**Dive 66 flips to leave.** Its ruler is 4 frames at one depth (1.78–1.91 m) so the
fit must use the 342.9 mm; it gives φ = −0.1591°. Applying that pushes held-out
models from (Purple −1.0, Shark −0.4, Snook −4.7) to (**+5.0, +4.2**, −0.3) — clear
over-correction. Two independent objects say the calibration is sound to ~1%; four
ruler frames at one depth say it's 4.9% short. Unresolved; do not act.

### For the paper

- **Accuracy claim** → dives 58, 59, 61, 84 (untouched) **+ dive 60** (ruler anchor
  is scale-free *and* a physically different object from the models). Five dives,
  nothing spent.
- **Calibration-model claim** → dives 60 and 76, plus the mount-state distribution
  φ ∈ [−0.19°, +0.11°] across all seven.
- **Dive 76** demonstrates the correction works; it is not accuracy evidence.
- **Dive 66** is an open discrepancy — report it, don't bury it.

Report per-dive error as an **angle**, not a percentage: a −8% dive is not eight
times worse than a −1% dive, it is a mount 0.29° off instead of 0.03°, and the
percentages are confounded by each dive's shooting distance.

---

## 4. Methods that FAILED, and why (do not re-try blind)

| method | result | why |
|---|---|---|
| refine calibration from the 2-D dot line | **impossible** | φ moves the projected dots by 6.8e-13 px — see §1 |
| sub-pixel laser-dot refinement | negative | dots are ~0.5% of the error |
| reprojection-error line refit | negative | better fit (1.24 vs 1.36 px), identical error |
| residual_m as a quality gate | negative | Spearman ρ = −0.026 over 1109 depths; it sees only the harmless DOF |
| line fingerprint to rank borrows | **provably cannot** | two-branch ambiguity, §1 |
| **scale-free ratio on fish models** | **negative** | §4.1 below |

### 4.1 Scale-free ratio (same object at two depths) — works on the ruler, fails on the models

Ruler alone, dive 60, its own two clusters (0.82 vs 1.85 m, 2.3×):
- scale-free φ = **−0.1458°** vs metric φ = −0.1607° → agree to **0.015°** (~0.5% at 2 m)
- leaves the ruler reading 341.6 mm against a true 342.9 mm it never saw (−0.38%)

Fish models, dive 76: φ_free = **+0.034 ± 0.082°** (bootstrap, 300×) against a truth
of **−0.141°** — wrong sign, truth at the CI edge. On the dives known sound (true
φ ≈ 0) it returns **+0.05 to +0.11°**, a systematic worth ~2.5% at 2 m.

**Cause: pose is confounded with depth.** Dive 76's Snook, sorted by range, reads a
flat −18% at 1.07–1.16 m (steeply angled), −5% at 1.28–1.33 m (side-on, and excluded
by the binning), and −6…−27% at 2.07–2.13 m. The method's one assumption — that
apparent length differs between bins only through calibration — is false when a
diver rotates a hand-held model while swimming. The ruler satisfies it by
construction (flat, rigid, ≤4° tilt on all 29 frames; within-bin sd 0.31%/0.20%
versus 2.7–7.1% for dive 76's models).

Also tried and same answer: joint fit of φ with one free length per model over all
depth terciles (dive 76 → +0.048°). Not an estimator problem.

---

## 5. Gotchas that cost real time

1. **Raw/JPEG crop offset — 8 px.** `RawImage` uses `rawpy.postprocess(...)`, which
   returns the **full 4014×3016** array; the camera's own JPEG is the 4000×3000 crop
   starting at **(8, 8)** (`crop_left_margin=8, crop_top_margin=8`). **Labels live in
   4014×3016 raw coordinates.** Drawing them on the camera JPEG without subtracting
   (8, 8) puts every mark down-and-right of the true feature and makes correct labels
   look wrong. `scripts/ann2.py` has the correct mapping.
2. **Lens distortion is a red herring near the axis.** At r ≈ 194 px (r/f ≈ 0.069)
   the radial term contributes 0.12 px and tangential 0.12 px. It cannot explain a
   10 px discrepancy — the crop can.
3. **The laser labels are fine.** Raw red plane at the label: 2107 vs local median
   622 (3.4×) at 1.11 m, centroid within 4 px. At 2.48 m the dot is faint (660 vs
   508) and red-excess is a poor discriminator because the dot lands on a *white*
   glossy flank — labelers saw the CLAHE-stretched JPEG where it is obvious.
4. **Never measure photometry on the pipeline JPEG.** It is rawpy → auto-gamma →
   **CLAHE** → undistort. CLAHE is local adaptive equalisation, so apparent dot size
   is a function of the surroundings.
5. **Aggregate a fish's frames at p90, never a mean.** Foreshortening is one-sided
   negative; a mean measures the pose distribution.
6. **The yaw floor clips at 0** (§2).
7. **Anchor choice needs an a-priori rule, not a post-hoc pick.** Snook's scale-free
   estimate on dive 76 (−0.1485°) happens to match the truth; selecting it on that
   basis is circular, and the principled criteria (flat, large apparent extent) point
   *away* from it — it is the only round model and has the worst scatter (7.1% sd).

---

## 6. Reference data

**Per-model fidelity ("the ladder"), p90 on the untouched dives 59/61/84:**

| model | n | p90 err | median err |
|---|---|---|---|
| Grouper | 68 | **+0.18%** | −1.22% |
| Purple Angel | 49 | +0.91% | −0.51% |
| Snook | 61 | −0.95% | −2.42% |
| **Shark** | 24 | **+4.23%** | +1.99% |

An anchor's own bias transfers into φ at ~0.03° per 1%. Grouper is the near-unbiased
anchor. **Shark reading systematically long is the one open oddity** — nothing in the
error model pushes measurements long, so it points at its 605 mm reference being
short, or labelers including the caudal filament. Worth chasing independently.

**Instance budget:**

| model | total | dives | dive 76's share |
|---|---|---|---|
| Grouper | 131 | 7 | 30 (23%) |
| Snook | 114 | 6 | 21 (18%) |
| Purple Angel | 96 | 7 | 12 (12%) |
| Shark | 96 | 6 | 27 (28%) |
| Ruler | 27 | **2** (60, 66) | — |

**Ruler frames exist on only dives 60 and 66.** Dive 76 has no ruler, no slate, and
zero laser frames without a model in view — every dot lands on a model (6–84 px from
model centre). There is no free metric reference in dive 76.

---

## 7. Operational recommendation (the payoff for the field)

The pool wall at Canyonview is **tiled** — planar, regular pitch, in frame at subject
range. One frame per deployment with the laser **aimed at the tiled wall** is a
complete calibration costing no known length: tile pitch → wall homography → metric
distance at the dot's pixel → a known 3-D point on the laser ray, which with the fixed
pivot pins both remaining DOF. Such frames already occur accidentally — the slate
dives carry 1–6 `Slate, Laser not on slate` frames each.

Needs one thing: **the tile pitch at Canyonview, measured once with a tape.**

Second best, if no tiled surface: **two ruler frames per deployment, near and far,
≥2× apart in range, held the same way both times** — that is scale-free (§4.1), so
the ruler's length is never spent.

Also worth doing, cheap: **index the laser's roll** (flat + set screw, or a scribe
mark) so ψ is repeatable — it is the only malignant DOF. Index it with the beam
misalignment lying *in* the camera–laser plane (ψ = 0 or 180°), where
`du/dψ = −ε·sin ψ = 0` and roll wobble enters only at second order. Indexing at
ψ = 90°, where the dot line looks cleanest, is the **worst** choice.

And **bench-measure ε** (V-block, roll 360°, trace the dot on a wall at known D;
ε = atan(R/D)). It converts the error budget from a seven-dive empirical spread into
an a-priori bound from a component measurement.

---

## 8. Open items

1. **Dive 66** ruler-vs-model disagreement (~4%), unresolved.
2. **Shark's +4.23%** — reference length or landmark convention?
3. **Laser dot size as a range cue** — deprioritised by the user, not dead. Blocker
   is *range-dependent bias*, not noise (±6 cm/frame over ~90 frames averages to
   ±0.3%, plenty; you need the trend unbiased to ~0.7%). **Decisive cheap test:**
   build the estimator on dive 60's raws, fit `d = A + B/z` against the
   ruler-derived true depths, and look at the residual **slope**. Slope-free at
   0.7% → usable; any trend → stop. Must use the raw red subplane (Bayer, no
   demosaic), mask saturated cores and fit the unsaturated wings, normalise by EXIF
   exposure. Confounds are all range-correlated: saturation, demosaic, exposure,
   surface BRDF, forward scatter.
4. **Dive 76 close-range Shark** — implied yaw 29.6° near vs 21.2° far, never below
   18°; recommended exclusion.
5. Nothing has been written to prod.

---

## 9. Reproducing the environment

**Database** (nothing here needs prod access):

```bash
docker run -d --name fs-an -e POSTGRES_PASSWORD=localonly \
  -e POSTGRES_DB=fishsense -p 55433:5432 postgres:17
docker cp /home/chris/mnt/fishsense_process_work/database_backups/fishsense/2026-08-26T03-00-02Z.dump \
  fs-an:/tmp/fs.dump
docker exec fs-an pg_restore -U postgres -d fishsense --no-owner --no-privileges /tmp/fs.dump
docker exec fs-an psql -U postgres -d fishsense -c "<query>"
```

Backups live in `/home/chris/mnt/fishsense_process_work/database_backups/fishsense/`
(nightly, ~24 MB each). The dump itself is **not** in this zip.

**Raw imagery** (NAS):

```
/home/chris/mnt/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/082923_Pool Calibration/
    082923_FishModels_FSL03/   <- dive 76   (94 images)
    082929_FishModels_FSL04/   <- dive 60   (103 images)
    082923_Slate_FSL03/        <- dive 63   (dive 76's calibration source)
    082929_Slate_FSL04/        <- dive 65   (dive 60's calibration source)
```

`Dive.path` names the `2024.06.20.REEF/...` copy. **A second copy of the same
captures exists** at `REEF/data/2023-09-07 REEF Data Dump/082923_Pool Calibration/`
— same file sizes. Read the path `Dive.path` names and verify against
`Image.checksum` (md5 of the whole file) rather than trusting filenames.

**Python:** run `./setup.sh` first — it assembles a flat `work/` dir, because the
scripts `exec()` each other's prologues and read the CSVs from the current
directory. Then `cd work && uv run --with numpy --with scipy --with pillow --with
opencv-python-headless --with rawpy python <script>`.
--with opencv-python-headless --with rawpy python <script>`. Scripts `exec()` each
other's prologues, so run them from the directory they live in.

---

## 10. Files

```
data/     ruler.csv        23 ruler frames, dive 60 + 4 on dive 66 (laser px, head/tail px, length, depth)
          all.csv          every fish-model measurement + geometry ('|' delimited — JSON has commas)
          models.csv       same, models only
          d76_all.csv      all dive-76 laser labels + depth + model
          d60_dots.csv     dive 60 laser dots (103)   d65_dots.csv  dive 65 laser dots (32)
          shk.txt          4 dive-76 Shark frames used for the figures
          shk_all.txt      all 27 dive-76 Shark frames, depth + length

scripts/  pivot.py fit.py fit2.py       -- pivot model, the dive-60 A/B/C fits, dot-line checks
          invar.py gauge.py mech.py     -- the invariance proof and the two-mechanism decomposition
          dof.py extrap.py              -- one-anchor sufficiency, depth extrapolation
          repair.py final.py allphi.py  -- per-dive phi, leave-one-model-out
          scalefree.py boot2.py         -- the scale-free ratio method + its bootstrap (negative result)
          rulerfree.py d60final.py      -- the ruler scale-free fit that works
          why76*.py overid.py joint.py  -- dive-76 diagnosis, over-identification test
          strat.py sharksplit.py        -- stratified anchor keeping all models represented
          yaw.py yaw2.py final_all.py   -- the implied-yaw floor diagnostic (start here)
          d66.py budget.py              -- dive 66, instance budget
          ann2.py dot.py rawdot.py dims.py -- figure rendering (CORRECT crop handling) + dot verification

figures/  fix_*.png         four dive-76 Shark frames, correctly registered overlays
          fix_dot_*.png     +-200 px crops around the laser label
          raw_excess_*.png  raw red-excess patches

sql/      extract_queries.sql   the queries that produced data/
          apply_corrections.sql the two proposed writes -- NOT APPLIED
```

Start with `scripts/final_all.py` (reproduces §3's table) and
`scripts/fit.py` (reproduces §1's three-model comparison).

setup.sh  assembles work/ (scripts + data flat) -- run this first
sql/apply_corrections.sql  the two proposed writes -- NOT APPLIED to prod
