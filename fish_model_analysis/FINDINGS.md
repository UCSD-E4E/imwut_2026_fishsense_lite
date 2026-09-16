# Fish-model accuracy figures — what was built and what it turned up

Companion to `HANDOFF.md` (the 2026-08-26 calibration-repair session, which supplied the
data). This document covers the figure work that followed and the findings that came out
of building it. Nothing here has been written to prod.

---

## 1. What is here

| path | what it is |
|---|---|
| `data/all.csv` | 464 per-frame measurements + per-frame geometry. `\|`-delimited: the geometry columns are JSON and contain commas. Carries a trailing psql `(n rows)` footer, dropped on load. |
| `data/corpus.csv` | **every** rigid-target measurement in prod, re-exported 2026-09-14 (§9.6): 2,927 frames, 32 dives. Same schema. Extracted by `sql/extract_corpus.sql`. It is *not* a superset of `all.csv` — the six mislabel corrections of §10 differ from it deliberately, which is why the superset test was deleted (§11). |
| `data/corpus_20260912.csv` | the frozen 2026-09-12 export. Kept because §9.6's reference-sensitivity result is only reproducible against it, and for nothing else. |
| `data/stereo_pairs.csv` | our 41 measurements of the eight 2023-08-03 morning individuals, exported 2026-09-16. Extracted by `sql/extract_stereo_pairs.sql`. |
| `data/stereo_reference.csv` | the stereo side: seven lengths lifted from the collaborators' `SMILE_Archive_LengthData.csv`, which is not in this repo. |
| `sql/extract_stereo_pairs.sql` | the paired-comparison extraction |
| `../fishsense_imwut/stereo_pairs.py` | the paired analysis: both estimators, the range-trend check, the geometry check |
| `../tests/test_stereo_pairs.py` | pins §12's numbers against the two committed extractions |
| `data/field.csv` | the seven Florida-reef deployments behind PAPER.md §4.5: 162 wild-fish measurements of 73 individuals, exported 2026-09-14. Extracted by `sql/extract_field.sql`. |
| `sql/extract_field.sql` | the field extraction, so §4.5's numbers can be re-pulled rather than re-typed |
| `../fishsense_imwut/repeatability.py` | the within-group CV estimator behind §4.5, with the bootstrap and the nearest-rank p90 |
| `../tests/test_repeatability.py` | pins §4.5's field and pool figures against the two committed extractions |
| `data/angles.csv` | the foreshortening experiment: 1,428 frames of one Snook at 0–45° (dives 87/94/103/107/114) |
| `sql/extract_corpus.sql` | the psql extraction, so the corpus can be re-pulled |
| `../tests/test_calibration.py` | pins the median polish, the cohort rule, and the published cohort |
| `data/shark_grouper.csv` | Shark + Grouper frames with head/tail label pixels and distortion, for §3.5 |
| `fish_model_measurements.ipynb` | the analysis; produces every figure |
| `fork_probe.py`, `fork_render.py` | the §3.5 fork reverse-engineering and its image overlays |
| `figures/` | vector PDF (for `\includegraphics`) + PNG (preview) |
| `../fishsense_imwut/calibration.py` | the geometry, ported from the handoff's scripts |
| `../fishsense_imwut/pubfig.py` | print style, palette, figure functions |

**The port is pinned to the handoff.** Cell 2 asserts all seven dives' implied-yaw floors
against `HANDOFF.md` §3 verbatim and raises if the geometry drifts. It passes — dive 60's
12.5/11.7/12.6/17.9 → all 0.0, dive 76's 17.9/11.6/20.0/19.0 → 5.8/0.0/2.1/10.7. Every
figure is downstream of that assertion, so a silent regression in the geometry cannot
reach a plot.

The data covers **5 objects** (Grouper, Purple Angel, Shark, Snook, Ruler) over **7 dives**
at **0.63–3.13 m**. Gray Anthias, Yellow Anthias and Weasly Fish have reference rows but
zero frames — `Weasly Fish` was seeded as groundwork, so any figure keyed on the reference
table rather than the measurements will show three objects that do not exist.

## 2. Figures

| | figure | cohort |
|---|---|---|
| 1 | measured vs. known length, with the $p_{90}$ estimator | accuracy |
| 2 | error distribution by model, estimator marked | accuracy |
| 3 | error vs. range | accuracy |
| 4 | per-dive mount state $\varphi$ | all seven |
| 5 | the repair, seen through the implied-yaw floor | dives 60 and 76 |
| 6 | the Shark anomaly (emphasis form) | ladder |
| 7 | fork-label scaling test (label bias vs. short reference) | Shark, Grouper-anchored |
| A | all seven dives, percent error | all seven — appendix |

Two conventions from the handoff are enforced in code rather than left to the caller:
per-dive error is reported as an **angle**, never a percentage (percent error is
confounded by each dive's shooting distance); and a fish's frames aggregate at
**$p_{90}$**, never a mean (foreshortening is one-sided negative, so a mean measures the
pose distribution).

Figures carry no titles — the LaTeX caption is the title — and are pre-sized to the
`acmart` column, so they take no `width=`.

---

## 3. Findings

### 3.1 The cohort decides the headline number

| cohort | n | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|
| accuracy — 58/59/60/61/84 | 329 | −1.52% | +1.41% | **3.03%** |
| untouched — 58/59/61/84 | 226 | −0.99% | +2.30% | 2.82% |
| ladder — 59/61/84 | 202 | −1.17% | +1.29% | 2.72% |
| method demo — 76 | 90 | −7.75% | −3.95% | 9.20% |
| disputed — 66 | 45 | −5.53% | −0.86% | 8.11% |
| **all seven pooled** | 464 | −2.70% | +1.07% | **4.72%** |

Dives 76 and 66 drag the pooled figure down by ~1.7 pp. Publishing the pooled number
would understate the system by more than 1.5×, and dive 76 is *supposed* to be bad — it is
the uncorrected borrow the repair demonstrates on, and its models are spent fitting the
correction, so it cannot also grade it.

### 3.2 Shark's over-read is mostly dive coverage, not Shark

A **calibration** error is per-*dive* and moves every model together; a **reference or
landmark** error is per-*model* and follows it everywhere. A Tukey median polish on the
$p_{90}$ of each (dive, model) cell separates them, and the fit is tight — median
|residual| **0.31 pp** — which is itself evidence that these two additive terms are the
whole story.

| model effect (pp) | | dive effect (pp) | |
|---|---|---|---|
| Snook | −2.36 | 76 | −3.50 |
| Grouper | −0.58 | 60 | −2.19 |
| Ruler | 0.00 | 66 | −1.46 |
| Purple Angel | +0.45 | 61 | 0.00 |
| **Shark** | **+1.88** | 84 | +0.70 |
| | | 59 | +2.07 |
| | | 58 | +2.98 |

Shark's frame coverage is unlucky in exactly the wrong direction: **zero frames in dive
61** — the one dive with a near-zero offset — and a third of its frames in **dive 58**, the
dive with the largest positive offset. Decomposing the ladder number: +2.11 (dive part) +
1.88 (model part) ≈ **+3.99**, against the observed +4.29. That accounts for it.

The residual **+1.9 %, or 13 mm on 605 mm**, is real and is a per-model scale offset.
Three things constrain what it can be:

1. **It can only be the reference.** Foreshortening is one-sided negative — nothing in the
   chain pushes a reading *long*. A persistent positive per-model offset means the
   reference is short relative to what labelers click.
2. **The magnitude rules out fork-vs-total-length.** Clicking the upper caudal lobe tip on
   a heterocercal shark tail would give ~+15–25 %, not +2 %. This is ~1 cm of landmark
   ambiguity or a tape error, not a convention mismatch.
3. **Snook is the opposite mechanism.** It reads short (−2.36 pp) because its yaw floor
   never reaches 0 (6.5–11.1° across dives) — it is essentially never presented side-on,
   so even its $p_{90}$ stays foreshortened.

**Caveat:** Shark's yaw floor pins at 0.0 in most dives, but that is a *consequence* of
reading long — the floor clips at zero — not independent evidence it was well presented.
It cannot be used to corroborate any of this.

**Why it bites.** In dive 60 every object read short except Shark:

```
Grouper −2.43%   Purple Angel −2.00%   Ruler −2.20%   Snook −4.59%   Shark +1.23%
```

Dive 60 was genuinely 0.1458° off. Shark's bias nearly cancelled that error, making it the
one object that looked fine on a demonstrably miscalibrated dive. That is a concrete
argument for **Grouper** as the anchor, and a reason to treat Shark as unusable for
validation until its reference is re-measured.

### 3.3 Range-dependence does not survive into the measurement

Figure 3: the binned median sits flat at −1 to −3 % from 0.85 to 2.4 m and the IQR band
does not widen with range. Triangulation conditioning goes as $Z^2$, so this is a positive
result that can be claimed directly rather than argued only from the calibration side.

### 3.4 Seven gross outliers, all Snook, two of them species mislabels

Seven frames read worse than −25 %, and every one is labelled Snook. The two worst (dive
84, −58 %) measure 0.189 and 0.193 m — Purple Angel's 0.192 m to within 1.4 %. Those are
species mislabels and they sit **inside the accuracy cohort**.

| accuracy cohort | n | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|
| as-is | 329 | −1.52% | **+1.41%** | 3.03% |
| minus the 2 suspects | 327 | −1.51% | **+1.41%** | 2.69% |

$p_{90}$ does not move at all; mean \|error\| drops 0.34 pp. A good advertisement for the
estimator — but any mean-based number in the paper should exclude them. They are currently
left in.

---

### 3.5 Reverse-engineering the fork label: the reference is short, not the labels

**Method.** Each dive's calibration is fitted from **Grouper** — the near-unbiased anchor,
never Shark. That correction is applied to the Shark frames, and the tail label is then slid
along the head→tail axis until the back-projected length equals 605 mm. Where it lands is
the fork position the reference implies.

Two things make the inversion trustworthy. The pipeline places head and tail on the plane
$Z = z$, and reproducing that reproduces the stored `length_m` to **0.000 mm**, so the
forward model is exact rather than approximate. And only **well-presented** frames are
used: foreshortening is one-sided negative, so an angled frame legitimately reads short and
its implied tail would land beyond the real one. Frames are ranked by corrected length
within their dive and only the top decile kept.

**Result — Grouper-anchored, Shark still over-reads in every dive:**

| dive | over-read | implied snout-to-fork |
|---|---|---|
| 58 | +2.48% | 620.0 mm |
| 59 | +4.28% | 630.9 mm |
| 60 | +3.08% | 623.6 mm |
| 84 | +2.03% | 617.3 mm |
| 76 *(method demo)* | +0.99% | 611.0 mm |

Median excluding 76: **+2.78%, implying ~622 mm** against the stated 605 mm. The tail label
would have to move ~28 px inboard at typical framing — about **15 mm**, or 2.4% of body
length.

**The overlays show no alternative landmark.** Rendered on the camera JPEG (distortion
re-applied and the 8 px crop subtracted — `fork_render.py`), the implied fork lands
essentially *on top of* the existing label: the two markers overlap within one radius, on
the same caudal-fin junction. There is no second feature there to have confused a labeler
with. Whatever the 15 mm is, it does not correspond to picking a visibly different
anatomical point.

**How the shift scales rules out one explanation, not two.** A click bias in the *image
plane* — a systematic cursor or rendering offset — is a constant number of **pixels**
whatever the apparent size. The frames span a 4x range of apparent size (641–2766 px), so
that hypothesis is testable:

| model | rms residual |
|---|---|
| constant pixels (image-plane click bias) | 20.2 px |
| constant fraction of body | **11.9 px** |

corr(apparent body px, implied shift px) = **+0.83**; fitted slope 128 % of the
pure-fraction slope, intercept −6 px ~ 0. A pixel-domain bias would give slope 0, so that
explanation is dead.

**But constant-fraction does not isolate the reference.** A fixed offset in *object* space —
labelers consistently landing ~15 mm posterior of the true fork — is also a constant
fraction of body length, because the object is a fixed size. This test therefore separates
image-plane error from object-space error, and no further. Both survivors are object-space:

  1. the 605 mm reference is short, or
  2. the labelers' "fork" sits ~15 mm behind the anatomical one, consistently.

The overlays cannot separate them either: the implied position and the current label land
within one marker radius of each other on the same fin junction, so neither is visibly the
"right" one.

*Robustness of the scaling test:* the sharp version leans on three close-range dive-60
frames (z ~ 0.63 m, 2766 px). Those are legitimate — their over-read (+2.0 to +2.8 %) matches
every other frame — but dropping them weakens the correlation to +0.61. Constant-fraction
still beats constant-pixel there (rms 11.96 vs 14.81 px).

**What is settled.** The measurement chain is not at fault. Among well-presented frames the
Shark model's within-cell scatter is **CV 0.49 %**, comparable to the rigid Ruler (0.41 %)
and Grouper (0.46 %) — so the model is rigid and the frames agree with *each other* to half
a percent while disagreeing with the reference by 2–4 %. And the reference itself is
unevidenced: `is_provisional = false` but `notes` is **empty** — no date, no method, no
measurer. It is asserted, not documented, and had never been tested because Shark was never
an anchor. (Contrast `Weasly Fish`, whose note records its interval, its source, and how to
judge it — and which states the governing rule: *"nothing known pushes measurements long,
so [a positive reading] would say the reference is too short."*)

**Why Shark and not the others.** It is the only object here whose posterior landmark is
genuinely ambiguous. Every other reference is a teleost with a clean homocercal fork, or a
literal ruler. A shark's caudal fin is heterocercal — the upper lobe far longer than the
lower, and the "fork" a shallow rounded notch rather than a sharp V — so *snout to fork* on
it is a judgement call worth about a centimetre. That same ambiguity is available to
whoever calipered it and to whoever labels it, which is precisely why the two hypotheses
above are hard to separate.

**Conclusion: ~15 mm of the Shark reference is unaccounted for, in object space.** The
measurement chain is exonerated; the remaining question is whether the 605 mm is short or
the labelers' fork sits 15 mm behind the anatomical one. One session with a caliper settles
both at once: measure snout-to-fork, and separately snout-to-the-point-labelers-click. The
prediction is that one of them reads **618–622 mm**. Whichever it is, 605 mm is not the
distance being measured in these frames.

Reproduce with `fork_probe.py` (analysis) and `fork_render.py` (overlays) against
`data/shark_grouper.csv`; Figure 7 is the scaling test.

## 4. Corrections to `HANDOFF.md`

**§6's ladder cohort is 59/61/84, but §3's accuracy cohort includes 58.** The two are not
interchangeable: dive 58 carries the largest positive offset of any dive (+2.98 pp; +2.14 %
median on its own) over just 24 frames, 12 of them Shark. Folding it into the ladder
inflates precisely the model under investigation, moving Shark +4.29 → +4.80. The cohorts
are now separate constants (`LADDER_DIVES` vs `UNTOUCHED_DIVES`) with the reasoning
recorded at the definition.

**§1's "spread 0.36° ⇒ ε ≥ 0.18°" rests on a spread that includes anchor bias.** The 0.36°
is the spread over *per-anchor* fits, not per-dive mount state. Shark-anchored $\varphi$ is
the highest in every dive it appears in — 58: +0.1621 vs Grouper's +0.0734; 59: +0.1036 vs
+0.0060; 84: +0.0957 vs +0.0199 — which is exactly the +4 % bias transferring at the stated
~0.03°/1 %. The bias-free per-dive spread is **0.272°, so ε ≥ 0.136°**. Same conclusion,
weaker number.

**The $p_{90}$ convention differs between the handoff and the pipeline.** §6 used
`np.percentile` (linear interpolation); `fish_length_estimate` uses nearest-rank
`ceil(0.9n)`. Grouper +0.19 vs +0.28, Purple +0.91 vs +1.14. The ladder table prints both.

---

## 5. Open decisions

1. **Which $p_{90}$ convention the paper cites.** Nearest-rank matches the shipped view;
   interpolated matches the handoff's tables. Pick one before numbers reach the text.
2. **Whether to drop the two dive-84 mislabel suspects.** No effect on $p_{90}$, 0.34 pp on
   mean \|error\|.
3. **Shark's reference length — now narrowed to one measurement.** §3.5 rules out the
   labeling explanation: the implied fork lands on top of the existing label, and the
   offset scales as a fraction of body size rather than as a constant pixel bias. Take a
   caliper to the physical model, snout to caudal fork. Predicted **618–622 mm** against
   the 605 mm on record. If it confirms, update `fishmodelreference` and Shark becomes
   usable as an anchor instead of a trap.
4. **Dive 66** remains the unresolved ruler-vs-model disagreement from `HANDOFF.md` §8.

## 6. Running it

```sh
nix run .#default -- -c 'uv run jupyter lab'
```

`nbconvert` is not currently in the venv, so the notebook has no saved outputs in git;
`uv add --dev nbconvert` if committed outputs are wanted.

---

## 7. Corpus extension (2026-09-12)

Sections 1–6 describe the seven-dive August analysis and are left as written. This section
supersedes §3.1's headline: the accuracy numbers now come from `data/corpus.csv` — every
rigid-target measurement in prod — under one rule applied to all 32 dives.

### 7.1 The rule

Tukey median polish over the $p_{90}$ of each (dive, model) cell with ≥ 5 frames
(`cal.median_polish`, `cal.cell_p90_grid`), fitted on the whole corpus. A dive is accuracy
evidence when |dive effect| ≤ 2.5 pp (`cal.MAX_DIVE_EFFECT_PP`). Held out by design, decided
before any corpus number was looked at (`cal.DESIGN_EXCLUDED_DIVES`): the angle experiment
(87/94/103/107/114 — single-object dives, pose-dominated), the two August repairs (60, 76)
and disputed 66. Dive 490 was never on that list; since the 490 → 527 split (§7.4) it has no
rows in the corpus and does not enter the polish at all.

Median |residual| **0.15 pp**, but read that number with two structural facts (both spelled
out in `post_labeling_analysis/HANDOFF.md` §3.1): only 52 of the 30 × 7 cells are observed
against 36 free parameters, and 20 of the 30 dives carry a single cell whose residual is
zero by construction — over the ten multi-model dives the median is **0.45 pp**. And the
grid is **disconnected** (2023 dives span the fish models, 2025 dives the Box and Weasly,
no shared cell), so dive and model effects are identified only within each component and
the rule compares dive effects across a seam the polish's centring convention sets. The
additive decomposition is well supported; it is not exact, and the checkerboard-vs-slate gap
(`post_labeling_analysis/HANDOFF.md` §6.3) is that unidentifiable quantity rather than a
measurement.

**Scale-free pre-filter (added 2026-09-12, `cal.range_trend_flagged_dives`).** A rigid object
must read the same length at every range, so the Theil–Sen slope of its length against laser
depth (frames ≥ 0.8 m, ≥ 8 frames over ≥ 2× range) checks the calibration with no known
length. A dive is dropped when a cell's whole 95 % interval clears ±2 %/m. Both signs count: a
rotated axis reads negative; a short fitted baseline reads positive, because the fit pairs it
with a compensating angle, and that kind (503/504 at 8.90 cm, 498 at 9.51) is invisible to
the polish — its flat scale error and its ramp cancel where the $p_{90}$ sits (503: −14 % at
0.8 m, −6 % at 2.5 m, +0.4 % beyond 3.5 m; $p_{90}$ −0.75). Flags 76, 491, 492, 494, 503, 504,
509 and no sound-baseline dive; 498 is borderline (interval [+2.0, +3.8]) and stays.

Two things to know before reusing it. It flags a dive when **any** cell flags, and the only
two multi-cell flags here (76's Shark against its Purple Angel, 509's Box against its Weasly)
are cells a sibling on the same extrinsics contradicts — a calibration error must move every
target together, so those two are model effects by the polish's own premise. Requiring
agreement yields the identical cohort here (509 fails the band anyway, 76 is design-excluded),
so the code is left matching fishsense-lite's `range_trend.py` to the decimal; the guard
belongs there first. And `theil_sen`'s Sen interval is one order statistic narrower per side
than `scipy.stats.theilslopes`, which biases very slightly toward flagging; it changes nothing
here, since 498's lower bound (+1.96) clears the 2.0 threshold under either convention.

Result, pinned as `cal.CORPUS_ACCURACY_DIVES` and by `tests/test_calibration.py`:
**59, 61, 84, 495, 497, 498, 500, 501, 507, 519, 520, 521, 522** — three August dives
plus ten 2025 pool dives (checkerboard self-calibration or a same-rig borrow). Re-pulled
2026-09-12 after dive 490's fish frames became dive 527 (§7.4): with 490's −13.8 gone the
corpus median moved ~0.3 pp, so 491 (−2.73) and 527 (+2.95) both sit just outside the band.

### 7.2 Headline

| cohort | dives | n | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| **rule: \|dive effect\| ≤ 2.5 pp, after the pre-filter** | 13 | 771 | −2.19 % | **+0.06 %** | 3.71 % |
| August five (58/59/60/61/84) | 5 | 335 | −1.51 % | +1.41 % | 3.00 % |
| every dive not held out (design + range trend) | 18 | 910 | −1.94 % | +0.68 % | 3.49 % |
| everything except the angle experiment | 27 | 1499 | −3.48 % | +0.16 % | 5.15 % |
| everything | 32 | 2927 | −5.91 % | −0.59 % | 9.20 % |

The two hold-outs do the rejecting and the 2.5 pp band does the tightening. Admitting the
five dives the band alone excludes costs $p_{90}$ (+0.06 → +0.68 %) while leaving the median
and mean |err| marginally *better* — so "every widening makes the numbers worse" is no longer
true of that row, and should not be written; dropping the hold-outs as well runs the mean
|err| from 3.49 % to 5.15 % and then 9.20 %, which is the real cost of not cherry-picking.
$p_{90}$ stays within ±0.7 % of zero in every row.

Sensitivity, both free numbers: the polish band over 1.5 → 3.5 pp moves membership from 8 to
15 dives and $p_{90}$ between −0.42 % and +0.45 %. The range-trend threshold is flatter still
— anything from 2.0 to 4.0 %/m gives the identical 13 dives and identical figures (only
≤ 1.5 %/m moves it, to 12 dives and $p_{90}$ +0.09), and neither `MIN_DEPTH_M` over 0.6–1.0
nor `MIN_FRAMES` over 6–12 changes the flag set at all.

Ladder on the cohort ($p_{90}$): Box −0.14, Purple Angel +0.85, Weasly Fish −1.89, Grouper
+0.29, Snook −0.94, Shark +4.29.

### 7.3 Model effects

| model | effect (pp) |
|---|---|
| Snook | −0.57 |
| Box | −0.38 |
| Weasly Fish | −0.38 |
| Grouper | 0.00 |
| Ruler | +0.08 |
| Purple Angel | +1.21 |
| **Shark** | **+3.23** |

Snook's August effect (−2.36) shrinks to −0.57 once the angle experiment contributes its
best-presented frames — the August number was pose, as §3.2 suspected. Shark's grows. The
Shark is not on hand to caliper (only the Box and Weasly Fish are, and both are already
within 0.4 pp), and it is 25 of 771 cohort frames: dropping it moves the cohort $p_{90}$
from +0.06 % to −0.24 %; re-referencing it at 620 mm gives about −0.2 %. Reported as-is with
the model effect stated.

### 7.4 Dives the rule rejects

| dive | effect (pp) | what it is |
|---|---|---|
| 490 | — | **Resolved 2026-09-12**: the laser rotated 0.82° in-plane between its fish frames (19:00–19:02) and its own board burst (19:07–19:08); under the preceding burst's calibration (dive 489, 9 s earlier) it reads −1.7 % median. A pointing error, not a scale error. Prod fixed and the corpus re-pulled 2026-09-12: its fish frames are now dive 527 borrowing 489, and 490 itself has no rows and no dive effect. See `post_labeling_analysis/HANDOFF.md` §5. |
| 492 | −5.8 | borrows 490's row, shot 25 min before either of that rig's bursts; 0.25° from its true state |
| 494 | −3.5 | borrows 493 |
| 509 | −2.8 | self-calibrated; flagged on its Box cell (−4.1 %/m) while its Weasly cell is flat (+0.06, CI [−0.79, +1.39]) — the two disagree, so the flag is not by itself a calibration finding; the polish band excludes it anyway |
| 491 | −2.7 | borrows 490's row across a 0.2° laser movement; range trend −3.8 %/m (flagged) |
| 527 | +2.9 | 490's fish frames under 489's calibration: −1.7 % median, $p_{90}$ +1.3 %, i.e. it reads like the 2023 slate dives, which sit ~1.8 pp above the checkerboard-dominated corpus median |
| 503 / 504 | +0.9 / +2.7 | borrow 502 (8.90 cm): range trend +5.3 / +5.5 %/m (flagged); the polish alone would have kept 503 |
| 58 / 506 | +3.1 / +4.0 | positive side; 58 is the August dive §3.2 flagged |

### 7.5 Foreshortening (Figure 8)

Pooled median error by designed angle: 0° −3.8 %, 5° −3.5 %, 10° −4.8 %, 15° −5.8 %,
20° −8.7 %, 25° −11.7 %, **30° −15.0 %**, 35° −20.5 %, 40° −25.0 %, 45° −31.4 %. The curve
is $\cos\theta - 1$ offset by a ~−3.5 % broadside bias; five sessions at two ranges agree
within the pooled IQR; the 15 % budget is crossed at 30°.

A sixth session exists and is deliberately absent. Dive 526 is the evening FSL05 burst,
split out of 107 on 2026-09-09: 189 angle frames at 0–45°, fully head/tail labelled, never
measured. Its 17 slate frames are a 13-second burst at a single distance (laser dot within
1–3 px), so the 3-D laser line is unconstrained and stage 13 refused the fit at a 2.00 cm
baseline on 2026-09-12. No other slate frames exist, and borrowing 107's calibration could
only be validated by the Snook's known length — the quantity the experiment measures — so
it was parked (`Priority.NONE`, note on the row) on 2026-09-12 rather than included.

### 7.6 The trout's reference, the scan, and half-thickness parallax (2026-09-12, evening)

The Weasly Fish (a stylised rainbow trout model) reference of 310 mm is a number carried from
the previous version of the paper; nobody can say how it was measured. A dense SfM point cloud
of the model (`~/fish_models/dense_fish_point_cloud.ply`, scaled on the Box's 150 mm tape
corner-to-corner span, factor 14.446761277 → cm) gives, on its principal axes:

| | cm |
|---|---|
| snout tip → fork, 3D chord | 31.60 |
| the same chord projected onto the body midplane | 31.60 (lateral offset 0.09) |
| snout tip → upper caudal lobe tip | 32.63 |
| body: length × depth × thickness | 32.6 × 12.9 × 7.3 |

So the scan says **315 ± 1 mm** for the pipeline's landmarks, 1.6 % above 310 — and since it
is scaled on the same tape span the pipeline's Box reference uses, scan and pipeline are ratios
to one object, and the Box's absolute length cancels from any comparison between them.

Against that reference the pipeline reads the trout short, and a fit of error $= a + b/z$ per
target on the cohort separates two parts:

| target | $a$ (range-flat) | $b$ (cm, the $1/z$ term) | 95 % CI on $b$ |
|---|---|---|---|
| Weasly Fish @ 315 | −5.2 | −1.8 | [−2.4, −1.3] |
| Box | −1.2 | −0.4 | [−0.8, −0.0] |
| Snook | −0.9 | −2.4 | [−4.7, −0.1] |
| Grouper | +1.6 | −5.0 | [−8.0, −2.7] |
| Purple Angel | −0.1 | −0.7 | [−2.7, +1.4] |
| Shark | +4.4 | −5.9 | [−9.3, +0.0] |

**The $1/z$ term is half-thickness parallax and it is a property of the method.** The laser dot
lands on the model's flank; the snout tip and fork lie in the midplane, half a body thickness
further from the camera; stage 14 back-projects the landmarks at the dot's depth, so a solid
model reads short by (offset / range). Every fish model shows it and the Box — a tape patch
on the face the dot hits — does not. The trout's fitted 1.8 cm is about half its 3.6 cm
half-thickness, consistent with the dot landing above the midline where the body is thinner.
This is what the "Weasly-only close-range under-read" (§7 above, handoff §6.4) was. For a
real fish it is small at survey range (a 40 cm fish, 4 cm thick, at 2 m: −1 %), and it is
one-sided, so it belongs in the paper's error budget as a stated bias.

**A second, box-free instrument agrees with the scan.** The FishSense Mobile captures
(`~/Desktop/fishsense-mobile-data/`, iPad and iPhone, May 2025; the June 2025 CCFRP boat
set on the NAS) photograph this model with an iPhone/iPad LiDAR, and the head/tail labels
for them survive in `~/fishsense-mobile-recovery/mobile_headtail_project46.json` with the
`target = George` classification in `project48`. The capture databases are schema v1 and
store no intrinsics (the app only added `intrinsics_bytes` at schema v7), but the
per-device calibration matrices are published in the sibling analysis repo
`UCSD-E4E/fishsense-mobile-oceans-2025`, `scripts/01_process.ipynb`: iPhone
fx = fy = 1375.0719 (cx 968.64, cy 723.05), iPad 1604.2147 (cx 956.58, cy 717.76).

Back-projecting each labelled snout and fork at its LiDAR depth — sampling depth with that
notebook's own flood-fill, which snaps each landmark onto the object's connected depth
component so a click near the silhouette cannot read the table behind it — gives:

| capture set | camera | n | snout→fork |
|---|---|---|---|
| iPad, May 2025 | f = 1604.21 | 30 | 32.09 ± 1.05 cm |
| iPhone, May 2025 | f = 1375.07 | 29 | 31.54 ± 0.82 cm |
| iPhone, June 2025 (CCFRP) | f = 1375.07 | 7 | 32.89 ± 0.88 cm |
| **pooled** | | **66** | **31.94 ± 0.13** (sem) |

Two cameras whose focal lengths differ by 17 % agree to 0.55 cm — that agreement, not the
pooled sem, is the evidence the intrinsics and the method are right. The LiDAR takes its
scale from the sensor's own metric depth, so unlike the scan it owes nothing to the box.
It therefore corroborates the scan against the reference rather than against the box: both
instruments put the model longer than 310 mm, and both exclude the ~301 mm that the
pipeline's box-relative reading would imply. The session-to-session spread (31.5 / 32.1 /
32.9) is larger than either instrument's internal precision, which is why neither number
is adopted.

**Decision: the reference stays at 310 mm and the uncertainty is reported.** Neither 315
nor 319 is a direct measurement, they disagree by more than their stated precisions, and
the physical model can be tape-measured. `PAPER.md` §4.1 states the ±2 % and §4.2 quotes
the consequence: at 315 mm the cohort reads median −2.88 %, $p_{90}$ −0.45 % over 615
frames (three sessions re-sort out of the cohort: 59, 497, 500, 520, 527); at 319 mm,
−3.41 % and −0.60 %. Worth about half a point of headline, in the pessimistic direction,
changing no conclusion. **The arbiter is a tape on the physical model** — snout tip to
fork, lying on its side, twice — and it supersedes both reconstructions.

**A bug worth reporting upstream.** `compute_fish_length` in that OCEANS notebook does
`points3d[idx, :] *= depth_pixel` over a 3×2 array whose *columns* are the two points, so
it scales row 0 (both X components) by the snout's depth and row 1 (both Y components) by
the fork's depth, and never scales row 2 — the two points keep Z = 1, so the ΔZ term drops
out of the norm entirely. It coincides with the correct chord only when the two depths are
equal. On these frames |ΔZ| is ~2 cm and the effect is 0.1–0.3 cm (the difference between
the 3-D and fronto-parallel columns above), but the same notebook computes a per-frame
`angle`, so tilted frames are in its scope and there the error grows.

**What is still open is the range-flat part.** At 315 the trout's $a$ is −5.2 % against the
Box's −1.2 % on the same dives — a 4 pp gap no parallax explains; at 310 it would be 2.5 pp.
Two candidates remain. The scan's scale is no longer one of them: the span it was scaled on
is confirmed to be the same box tape corner-to-corner 150 mm the reference names, and the
LiDAR — which never touches the box — agrees with the scan to within its session spread. So
either the model carries a consistent yaw on its rod-and-wire mount (16° would do it; the
six frames inspected are near-broadside but not measurably so), or the box's own 150 mm is
long. The second is testable: the box and the trout appear together on dives 521, 522 and
509, so a per-frame comparison of the two on the same frames isolates it. The head/tail
clicks themselves were inspected on six frames from 0.9 to 3.0 m and sit on the snout tip and
the fork notch; the landmarks are not the problem.

---

## 8. The SMILE stereo-video archive — what a field comparison can and cannot say (2026-09-13)

`~/Downloads/SMILE_Archive_LengthData.csv` is our collaborators' EventMeasure export: 1,471
lengths from the Florida Keys, 2023–2024, 16 sites. It is the SOTA the paper wants to
compare against, and it is the only external measurement of the same wild fish we shoot.

**Use the stereo rows only.** 1,343 rows are `camtype = SV` (stereo video). The other 128
are `camtype = FSL` and carry `Length` but no `Range`, `Precision` or `Direction` — they are
second-hand entries of unknown provenance, not pipeline output, and the user confirmed they
should not be trusted. Our side must come from prod.

**Our field corpus is small.** Every real-fish measurement in prod: 154 frames, **73
individuals, 7 dives, 6 camera rigs, one reef** (Alligator), median 2 frames per fish. The
stereo has 1,120 rows at the same reef over 10 named sites. Our laser range there is median
1.48 m (5–95 % 0.71–3.32); the stereo's is 1.97 m (1.05–3.64).

**Both sides are fork length** (confirmed with the collaborators), so the medians are
comparable in principle. Per species, ours against the stereo's Alligator subset, per fish:

| species | our fish | rigs | stereo n | offset | bootstrap 95 % |
|---|---|---|---|---|---|
| Hogfish | 34 | 7 | 96 | −10.3 % | [−19.8, −0.6] |
| Stoplight Parrotfish | 16 | 6 | 345 | −17.0 % | [−26.3, +8.2] |
| Rainbow Parrotfish | 5 | 3 | 55 | −20.7 % | [−34.7, −14.7] |
| Nassau Grouper | 8 | 4 | 7 | +2.2 % | [−14.0, +37.6] |
| Black Grouper | 6 | 2 | 50 | +9.4 % | [−8.1, +24.3] |

The direction splits by caudal shape — the three lunate/forked-tail species read short, the
two rounded-tail groupers read long — which is the shape a fork-vs-total convention
mismatch would make. That explanation is excluded by the convention being shared, and two
more were excluded directly: the fish are labelled `No Curve` (Hogfish 55 of 68, Stoplight
25 of 28), and every one of the nine dive-calibration pairs has a baseline of 9.72–11.80 cm,
inside the tightened gate and mostly at the fleet's 10.2–10.5.

**But the comparison cannot carry a conclusion, and this is the finding.** Split by rig, the
same species at the same reef gives: Hogfish −19.1, −17.4, −13.6, −13.0, +0.8, +1.3, +7.0 %
(2–10 fish per rig); Stoplight −43.7, −31.1, −22.3, +4.8, +9.3, +25.2 %. The between-rig
scatter is as large as the offset it is supposed to support — and it is **not distinguishable
from small-sample noise**: for Hogfish the observed between-rig sd of offsets is 10.5 %
against ~10.2 % expected from sampling 2–10 fish at a per-fish CV of 18.7 % (Kruskal–Wallis
across rigs with ≥3 fish, H = 6.34, p = 0.18). Stoplight is marginal (p = 0.018) on 1–4 fish
per rig. So the per-species offsets above are ~2 SE effects resting on the assumption that
our fish and theirs are drawn from one population, which different dates do not guarantee.

> **Addendum (2026-09-16): one estimator, and this section predates it.** Everything
> above reduces an animal's frames with a median, and Figure 13 used a mean. The paper
> now reports **p90 by nearest rank everywhere a set of frames becomes a length**, so
> Figures 13, 14 and 15 all use it and the numbers in this section are superseded by the
> notebook's. What moves: the per-species offsets go from −21 %…+9 % to −21 %…+12 %
> (only Black Grouper meaningfully, +9.4 → +11.7 %), the Hogfish per-camera ANOVA from
> F(5,27) = 1.17, p = 0.35 to F(5,27) = 1.10, p = 0.38, and the between-fish CV on
> Hogfish from 18.7 % to 19.1 % — which happens to be exactly the stereo's, so "the two
> systems' spreads are the same" becomes literal rather than approximate. **No
> conclusion in this section changes.** Note also that no wild animal here carries more
> than 8 frames and ceil(0.9n) is n for n <= 10, so every field p90 is that animal's
> longest frame; that is intended with a one-sided error, but it is why a field p90 does
> not carry the estimator precision a 26-frame pool cell does.

**Two things it does settle.**

The draft's SOTA sentence — "we see a narrower spread of results for FishCamera in the same
environment on the same fish" — is not supported. Between-fish CV is 18.7 % for us and
19.1 % for the stereo, and in both cases that is mostly the real size spread of the fish
encountered rather than measurement noise. Cut or rewrite it regardless of how the offset
resolves.

And the precision figures are not comparable as stated. Ours is empirical: 24 individuals
with ≥3 frames, within-fish CV median 3.28 %, p90 14.2 %. The stereo's is EventMeasure's
propagated click-error, median 1.04 % of length, p90 2.98 % — and with essentially one
measurement per individual (1,260 groups over 1,343 rows) there is no way to check it
against repeats. Quoting "1 % versus 3.3 %" compares a formal estimate against a measured
one.

**The landmark check ran and came back negative.** The hypothesis was that our labelers click
short of the fork on a lunate tail (reading short) and at the trailing edge on a rounded one
(reading long), which would explain both signs. The processed JPEGs for these dives are gone
from Garage (`NoSuchKey` under all three prefixes — the retention question), so the frames
were re-rectified from the NAS raws locally: `rawpy.postprocess` → `cv2.undistort` with the
dive camera's stored intrinsics, which is what `RectifiedImage` does, so stored label pixels
land where the labeler put them. **Alignment was verified on the laser dot**: where the dot
is visible, the stored `laserlabel` sits 3–19 px from the brightest red-excess pixel
(img 101415: 3 px; img 101368: 19 px, inside the bloom). On the frames inspected the fork
clicks are *noisy* — at the trailing edge, sometimes past it onto background — not
systematically forward of the fork. Six frames is weak evidence, but it points away from a
systematic landmark bias and toward the same conclusion as the rig scatter.

**What would make this comparison work.** More reef fish per rig — tens, not 2–10 — and a
rigid reference in the water on reef dives. The scale-free range-trend check (§7, `range_trend.py`)
needs one rigid object measured ≥8 times over a ≥2× range spread; no wild fish in prod
satisfies that (the five with ≥5 measurements span at most 1.5×), so field calibration state
is currently unvalidatable. A diver-carried reference would supply it, and it is the same
recommendation §4.2 of the paper already makes from the pool data.

### 8.1 Calibration conditioning — `MIN_LASER_POINTS` counts labels, not observations

The reef dives all carry their own slate and their own stage-13 calibration (no borrows), so
the obvious question was whether the per-rig scatter above is poor field calibration. The
answer is no, but asking it exposed a defect.

Stage 13 fits the laser ray from 3-D points, each one the camera ray through a labelled dot
intersected with the slate plane. What determines the fitted *direction* is the lever arm —
the depth separation of those points — against the label noise. One pixel of dot-label noise
at range z is z/f metres of lateral error, so it rotates the fitted ray by about
(z/f)/lever radians, and the session-measured sensitivity is ~30 % of length per degree at
2 m. That gives a per-dive conditioning number, **percent of length per pixel of laser-label
noise**, computable from the fit's own observations with no reference length.

Over all 32 stored calibrations whose observations are recoverable (19 slate, 13
checkerboard), **29 are well conditioned** at under 1 % per pixel. Three are not:

| dive | source | obs frames | labels | distinct dot pixels | depth range | lever | % length / px |
|---|---|---|---|---|---|---|---|
| **347** | slate | **1** | 2 | **1** | 1.20 | 0.00 m | degenerate |
| **349** | slate | 2 | 2 | 2 | 2.36–2.62 | 0.26 m | 5.8 % |
| **107** | slate | 16 | 16 | 12 | 1.96–2.02 | 0.06 m | 5.5 % |

Dive 347's calibration is fitted from **one frame carrying a duplicate laser label at the
identical pixel** (image 102732, both at 1961, 1222). Two coincident points cannot determine
a line. It passed `MIN_LASER_POINTS = 2` because that gate counts label *rows*. Duplicate
labels are widespread — 28 of the 32 calibrations have more labels than distinct dot pixels
(dive 341: 60 labels over 30 frames, 29 distinct pixels), which is the known duplicate-LS-task
issue and is harmless wherever there are many genuine observations. On 347 it was the whole
fit.

Dives 349 and 107 are the *single-distance burst* geometry that wedged dive 526: plenty of
observations in 107's case, but all within 6 cm of one range, so the lever arm is nil. This
also explains an anomaly CLAUDE.md records as unexplained — dive 107's 12.95 cm baseline, the
highest in the fleet and previously filed under "healthy extreme, measures correctly". It is
not healthy; it is an ill-conditioned fit whose baseline happened to land inside the
plausible range. Dive 526's identical geometry collapsed to 2.00 cm and was refused. Same
defect, different luck, and neither the baseline gate nor `check_fit_self_consistency` can
tell them apart.

**It does not explain the field offsets.** The opposite, in fact: dive 341 is the
best-conditioned calibration in the corpus (30 observations over a 1.55 m lever, 0.1 % per
pixel) and carries the *largest* negative hogfish offset (−17.4 %), while 347, the degenerate
one, reads +0.8 %. Conditioning and field offset are unrelated here, which strengthens the
§8 conclusion that the offsets are sampling noise on 2–10 fish per rig.

**Recommended pipeline change** (not yet made): replace `MIN_LASER_POINTS`'s label count with
a requirement on the observation geometry — at least two *distinct* dot pixels from at least
two *distinct* images, plus a minimum lever arm or, better, a conditioning bound of the form
above. It flags exactly 347, 349 and 107 and passes the other 29, it needs no reference
length, and it would have caught dive 526 for the right reason rather than by the baseline
coming out absurd. Remediation for the three is the usual: delete the row so the dive
re-enters the cohort, and park it if the observations cannot be improved.

### 8.2 What the field set can actually support (2026-09-13)

No more field data is coming — re-shooting these dives is a full operation — so the question
is what the 154 existing real-fish measurements establish. Three things, and one of them is
strong.

**Repeatability (strong, and self-contained).** For fish captured in ≥3 frames, the
within-fish CV needs nothing external: a calibration error is common to a fish's frames and
cancels in a relative spread, so do reference and convention errors, and no comparison
population is involved.

| | cells / fish | CV median | bootstrap 95 % | $p_{90}$ |
|---|---|---|---|---|
| pool, repeat frames of one target in one session | 23 cells | **1.36 %** | [1.24, 2.16] | 4.49 % |
| field, repeat frames of one wild fish | 24 fish | **3.28 %** | [2.36, 4.25] | 14.19 % |
| field, recomputed 2026-09-14 after the recovery and the orphan fix | 25 fish | **2.91 %** | [1.63, 4.06] | 9.11 % |

The 2026-09-13 intervals did not overlap. **On the corrected 2026-09-14 figures they do**
(field 1.63-4.06 % against pool 1.24-2.16 %), so the separation claim does not survive: it
is a ~2x difference in point estimate limited by 25 field individuals, not a clean split.
Two causes, both recorded elsewhere: dives 347 and 349 were recovered (§9), and two orphaned
measurement rows left by re-clustering were removed from the count (§9.4) -- one of them
gave fish 176 a duplicated 193.4 mm frame beside frames up to 298 mm, which was the old
p90's main driver.
That is the measured cost of the field — the animal moves, the water is turbid, the landmarks
are harder — and it is consistent with §7.5 (a few degrees of pose change between frames is
worth a few percent). At ≥2 frames the field figure is 3.14 % median over 40 fish. By species
(≥3 frames): Black Grouper 1.23 % (n=3), Stoplight 2.36 % (n=3), Hogfish 3.22 % (n=11).

**An upper bound on unit-to-unit calibration spread.** Observed between-rig variance =
calibration variance + sampling variance, so a one-way variance-components fit on log length
bounds the first. Hogfish (7 rigs, 34 fish): point estimate 4.9 %, **95 % upper limit 9.2 %**,
within-rig sd 18.6 % — i.e. not distinguishable from sampling, but bounded. Stoplight (4 rigs,
14 fish) has a point estimate of 24.6 % against a 6.6 % H0 limit, so it does show a rig effect,
but on 14 fish across four dives at different dates it is equally consistent with genuine
population differences. Report the hogfish bound and the stoplight caveat together.

**A deployment demonstration.** All 70 individuals, 8 species, plausible lengths — this is
Figure 6 and it stands as a demonstration, not an accuracy claim.

**What it cannot support, stated once:** absolute field accuracy. No in-water reference means
the range check of §7 cannot grade these calibrations, and 4–18 fish per rig against a 19 %
between-fish size spread puts the standard error on a rig median at 7–14 %. The stereo
comparison (§8) is a consistency check at best. PAPER.md §4.5 is written to these limits.



## 9. Conditioning, measured against an independent range standard (2026-09-14)

Everything in §7 grades a calibration with a *known length*. This section grades one
without, using the calibration target's own pose, and the result is the text PAPER.md §4.2
now carries.

**The standard.** A calibration frame carries a target of known geometry, so `solvePnP`
gives a per-frame distance that does not involve the laser. Comparing laser-triangulated
range against it, frame by frame, is the only check in the toolkit with access to an
*independent* depth — which is why it is the only one that can see scale. Reprojection
residual cannot (§7.x epipolar blindness), and `LaserDepth.residual_m` was measured not to
(Spearman −0.026, n=464).

**Control on the estimator, run before any claim was made.** An offline harness replicating
stage 13 (`plane_from_correspondences` → `laser_point_on_plane` → Atanasov fit) was run
against the 11 dives whose slate-frame laser labels are *all* live, so the fit's inputs are
provably still what the database holds. It reproduces prod's stored `laser_position` and
`laser_axis` on all 11: worst baseline disagreement **1e-5 cm**, worst axis disagreement
**0.0000°**, including dive 107 at 12.954 cm. Prediction made from it and then confirmed in
prod: dive 347 would refit to 10.11 cm and pass all four gates — prod returned **10.101 cm**
and no refusal.

**Dive 107: locally accurate, catastrophically wrong 2 m away.** 16 observations spanning
0.03 m of range.

| calibration | baseline | vs PnP range at 2.0 m (n=16) | at 4.2 m (n=15) |
|---|---|---|---|
| 107's own (stored) | 12.954 cm | −0.12 % | **−17.25 %** (out of sample) |
| 526's own | 1.995 cm | −127.58 % (out of sample) | +2.87 % |
| joint 107+526 | 10.865 cm | −0.22 % | −1.10 % |

Leave-one-out on the joint fit: held-out 526 frames median **−1.19 %** (worst single frame
10.6 %, one weak observation), held-out 107 frames median **−0.23 %**, max 1.91 %. So the
two bursts — same camera, same slate, same NAS folder, 5.5 h apart — are consistent with
one mount state, and the joint fit is the better calibration for both.

All 180 of dive 107's measured frames sit at 1.96–2.06 m, inside its anchor range, so **no
published 107 number is affected**; its stored fit is wrong only where nothing measures.
That is also why 526 could not simply borrow it.

**The negative result: leave-one-out is not a screen. Do not build it.** Refitting N times
leaving one slate observation out and predicting the held-out depth gives, per dive:

    sound dives (n=11)   lever 1.02–2.32 m   LOO median |depth err| 0.52–1.48 %
    dive 107             lever 0.03 m        0.56 %   <- better than most sound dives
    dive 526             lever 0.07 m        4.42 %

Dive 107 scores 0.56 % while being 17 % wrong at 4.2 m. The reason is structural, not a
threshold: a frame held out of a single-distance burst is predicted at the distance the
remaining frames already anchor, so it carries no information about the ray's direction.
**Every in-distribution check is blind to conditioning** — known-length medians at the
working distance, reprojection residual, and cross-validation alike. Only the observation
geometry (the lever arm) or an evaluation at a genuinely different distance sees it.

**Two independent scale standards agree to ~1 %.** The pipeline's metric scale comes from
the calibration target, so it *does* need a size reference — a printed one. Slate templates
(scanned DPI) and the E4E checkerboard (4.2 cm square) are independent standards, and the
baseline is a rig constant, so any camera calibrated both ways should agree. Over the **5
cameras** carrying both (19 fits, **5 distinct slate templates**), mean checkerboard−slate
difference **−0.027 cm = −0.27 %** of baseline, sd 0.104 cm, |max| 0.154 cm — *below* the
within-standard same-camera noise floor (10 groups, mean sd 0.121 cm). 95 % CI on the
relative scale error: **−1.14 % to +0.61 %**. Gate-refused fits (107, 347-old, 349-old, 498,
502) excluded and named.

Consequence for the error budget: the dominant uncertainty in the accuracy number is the
**fish-model reference** (±2 % on the 310 mm Weasly), not the calibration chain. Cannot see:
a common-mode error in both standards, or in the intrinsics — both rescale the PnP depth and
the laser depth together. A per-camera focal error would show as a per-camera baseline
offset and is bounded by the 7-camera agreement; a fleet-wide focal bias is invisible to all
of it.

**Recovery of dives 347 and 349 (the two §4.5 exclusions).** Both were left with 1 and 2
usable observations because the 3σ per-dive validator superseded their genuine slate dots:
347's thirteen sit 0.34–8.48 px off a line its 319 fish dots define to 1.25 px median, and
349's twelve sit 2.87–5.98 px off a 0.94 px line, against thresholds of 3.56 and 3.00 px.
Fleet-wide, the largest live calibration-frame offset is 8.33 px and the nearest real
mislabel is 45.57 px, so a 20 px absolute bound on calibration frames separates them; it
protects exactly the 21 revived labels and changes no other live label in the fleet
(verified over all 19 dives with calibration frames). Shipped as
`COARSE_CALIBRATION_TOLERANCE_PX`; the revived labels survived the next validator pass in
prod. Refits: 347 → 10.101 cm (predicted 10.11), 349 pending at the next stage-13 firing.
Recovers **73 measurements of 24 fish**, taking §4.5 from 91/50 to the full seven
deployments.


### 9.1 Screening the rest of the corpus: nothing else is recoverable (2026-09-14)

Eight dives carry superseded slate-frame laser dots. Each was dry-run through the validated
harness: fit the dive line on its live dots, revive a superseded slate dot only if it lies
within `COARSE_CALIBRATION_TOLERANCE_PX` of that line, refit, run all four gates.

| dive | superseded slate dots, offset from the live line | verdict |
|---|---|---|
| 466 | 22 dots in **two clusters, 42.6–46.2 and 66.5–72.9 px** | labels are CORRECT; the laser moved twice. Unrecoverable — see §9.2 |
| 103 | none — its fit spans **0.02 m** of range, baseline 6.25 cm | single-distance burst, refused by geom+baseline |
| 427 | none — **0.05 m**, baseline 16.01 cm | same |
| 279 | 6 dots, four at 0.7–2.8 px, two at 10.1 | refit gives **29.6 cm** baseline → refused. Leave |
| 383 | 9 dots, one at 2.4, rest 18.1–32.4 px | refit **fails describes-the-dive**. Leave |
| 465 | 14 dots at 3.6–4.6 px | the one real contest, resolved below. Leave |
| 471 | 4 at 4.2–5.6 px, one at 38.2 | refit moves the baseline 0.45 %. No benefit |
| 77 | 4 at 3.4–8.6 px, 34 at ~35 px | the reflection dive. Marginal. Leave |

**No dive gains a calibration it does not have.** 103 and 427 are the dive-107 disease and
neither has a same-folder, same-camera sibling holding slate observations at another range
(103 is camera 10 among cameras 1/4/5/6; 427's only folder-mate has no slate labels), so the
107+526 joint-fit remedy is unavailable to them.

**Dive 465, judged against the slate PnP range.** Its 3 live observations already span
1.16–2.51 m, so the 7 revivable dots add density, not range.

    3-obs fit (stored, 10.217 cm)    in-sample 0.53 %   out-of-sample 3.15 % (n=7)
    10-obs fit (revived, 9.927 cm)   in-sample 1.72 %   leave-one-out 2.14 % (max 5.01)

The revived fit predicts unseen ranges modestly better (2.14 % against 3.15 %), but it moves
the baseline to 9.927 cm — *outside* the fleet IQR of 9.99–10.45 where the stored 10.217 sits
inside — on a 10-sample statistic with no interval, for a dive whose 31 measurements of 18
fish are in the paper's field set. A ~3 % shift in published lengths is not justified by a
one-point difference in a 10-sample median. Left alone.

**The finding that matters for reuse: the 20 px tolerance is NOT a revival criterion.** It was
calibrated for one job — stop the validator superseding genuine calibration dots *going
forward* — and the fleet check showed it changes no live label except protecting 347's and
349's. Used instead as a filter for *reviving* already-dead dots it admits ones that
demonstrably wreck a fit: 279 (→ 29.6 cm baseline) and 383 (→ describes-the-dive refuses).
The two tasks have different error costs, so a revival must be validated per dive by the four
gates and, where a second range exists, by the independent PnP range test. Dives 347 and 349
clear that bar; nothing else in the corpus does.

### 9.2 Correction: dive 466's labels are right, and that is why it is unrecoverable

§9.1 as first written called dive 466's 22 superseded slate dots specular-reflection
mislabels, by analogy with dive 77 and on the strength of a ~45 px offset from the dive's
fish-frame line. **That was wrong, and looking at the frames is what showed it.**

Rectified with the dive's stored intrinsics and zoomed 4x, every stored label sits on a
real, compact, saturated dot on the slate, and the position the fish-frame line predicts
sits on blank slate. (The red-excess test that worked on the Florida head/tail frames is
uninformative here: max red excess is −2 and 0 at the labels, because a laser dot on a white
slate at close range saturates all three channels rather than staying red-dominant.)

The offsets are not one cluster but **two**, and they partition exactly by burst:

    burst 1, 09:17:16-39, 6 frames (12 rows with duplicates)   42.6-46.2 px
    burst 2, 10:09:37-51, 5 frames (10 rows with duplicates)   66.5-72.9 px
    fish frames, 71 live dots                                  own line, MAD 0.64 px

Three distinguishable laser states in one dive: each slate burst is internally tight, they
disagree with each other, and neither lies on the line the fish frames define. So the labels
need no fixing — **relabelling would achieve nothing** — and no fit from these bursts can
describe the frames 466 would measure. `check_calibration_describes_dive` refuses them
correctly and for the right reason.

This is also the strongest single piece of evidence in the corpus for PAPER.md §4.2's claim
that a session is not a safe unit: stronger than the dive-490 example (two calibrations 7
minutes apart differing by 0.82°), because here one dive contains three states separated by
52 minutes, each self-consistent, with the disagreement measured against the dive's own
fish dots rather than against another fit.

**Method note, and it is the third time today the same error shape appeared.** Reasoning
from an offset *magnitude* to a *cause* — 45 px therefore reflection — failed here, exactly
as extrapolating a length change from a baseline ratio failed on dive 107 (the axis co-varies)
and as leave-one-out failed as a conditioning screen (the held-out sample carries no new
information). In all three the number was real and the mechanism inferred from it was wrong.
The check that settles a mechanism question is the one that looks at the mechanism: the
frames, the kernel, or an evaluation at a different range.

**Recovery status of the corpus, final:** dives 347 and 349 are recovered and measured. No
other dive is recoverable from existing data, and no relabelling would change that.


### 9.3 Dive 498 repaired — and what that means for §4.3's cohort (2026-09-14)

Dive 498's own calibration (baseline 9.51 cm from a 0.34 m lever) was retired in prod.
Because `Dive.calibration_dive_id = 496`, deleting it did **not** leave the dive
uncalibrated: it fell back to dive 496's row, and 496 is the *same camera, same day, same
rig folder* — `ED-00/FSL-04D/LaserCalibration` against 498's `ED-00/FSL-04D/George` — i.e.
that session's own calibration burst. 496's fit passes the baseline gate (10.362 cm) and
describes 498's 40 live dots, so the borrow is the calibration 498 should have been using.

Re-measured under it, with the old lengths snapshotted first:

| | n | median | median % error vs 310 mm | range trend |
|---|---|---|---|---|
| own 9.51 cm fit | 35 | 288.6 mm | **−6.89 %** | −14…−18 % at 0.8 m, ramping to ~0 at 4 m |
| 496's 10.36 cm fit | 35 | 300.1 mm | **−3.20 %** | **+0.81 %/m, 95 % CI [−0.08, +1.79]** |

The range trend is the scale-free check, and it is now flat with an interval spanning zero,
inside the ±1 %/m band sound calibrations occupy. The residual −3.2 % is ordinary against
the cohort median of −2.19 %, half-thickness parallax on a solid model, and ±2 % on the
reference.

**The consequence for the paper, stated because it is a reproducibility problem and not a
numbers problem.** §4.3's cohort is derived by a *rule*, and `data/corpus.csv` is frozen at
2026-09-12. 498 was excluded by the range-trend pre-filter on a trend that no longer exists,
so **a fresh export would admit it and the published cohort would become 14 dives, not 13**.
The pinned tests stay green because they run against the frozen CSV, which is the intended
behaviour but also means they cannot notice this.

Two honest options, and the first is what the submission should do:

1. **Keep the frozen export and say so.** The corpus is a stated snapshot; 498's repair
   postdates it. Cite the export date in §4.3 and note that one excluded session was
   subsequently repaired in the pipeline. Nothing in §4.3 changes.
2. **Re-export and re-run.** Correct but not free: the cohort, Table 1, the threshold sweep
   and every figure derived from `corpus.csv` move together, and 498 is a −3.2 % dive that
   would pull the cohort median slightly.

Do not do the third thing — quietly re-running the rule against new data while quoting the
old cohort — which is how a stated rule and a stated result come apart.


### 9.4 Two orphaned measurement rows, and the paper numbers they moved (2026-09-14)

Reconciling prod after the recovery turned up two measurement rows that no cohort and no
activity will ever revisit, and they had already reached a draft.

    dive 341  image 101302   fish 176 @ NULL provenance  AND  fish 305 @ ext 10   both 193.4 mm
    dive 383  image 111926   fish 283 @ NULL             AND  fish 323 @ ext 16   both 484.9 mm

Both frames were re-assigned between Fish rows and the old binding stayed. `measure_fish_activity`
has a self-heal for exactly this, but it was guarded by `if model_name is not None` -- a
model's Fish is knowable from its name up front, while a real fish's comes from its
LABEL_STUDIO cluster, resolved *after* the "already measured with the current calibration"
skip. So the corrected row satisfies the skip on every later run and the loop never reaches
the point where it could notice the old one. `post_measurement` upserts on
(image_id, fish_id), so the correction was added alongside rather than replacing. Fixed in
fishsense-lite PR #905 by reading the expected binding off the cluster's own `fish_id`.

**What they cost.** Fish 176 keeps three other current frames, so it is a real individual;
fish 283's only frame moved away, leaving it as an individual with nothing current. Hence
removing two rows drops the individual count by one, not two:

| | measurements | individuals | within-fish CV, >=3 frames | $p_{90}$ |
|---|---|---|---|---|
| as counted | 164 | 74 | 3.19 % [2.03, 4.25] | 13.80 % |
| corrected | **162** | **73** | **2.91 % [1.63, 4.06]** | **9.11 %** |

The p90 moved most because fish 176's set carried the duplicated 193.4 mm beside frames up
to 298 mm, inflating its CV.

**And it costs a claim.** §4.5 said the field and pool repeatability intervals do not
overlap. On the corrected figures they do -- field 1.63-4.06 % against pool 1.24-2.16 % --
so the honest statement is a ~2x difference in point estimate (2.9 % against 1.4 %) limited
by 25 field individuals, not a clean separation. PAPER.md §4.5 now says that. This is a
weakening of a result caused by fixing a data defect, which is the right direction for the
error to have been found in, but it is a weakening and it is not buried.

**Still to do in prod:** the two rows predate the fix and the activity will not revisit those
frames, so they need an operator delete -- `scratchpad/delete_orphaned_measurements.sql`.
Until then the live database still reports 164/74 while the paper reports 162/73, and the
discrepancy is exactly those two rows.


### 9.5 Half-thickness parallax is not testable on this corpus (2026-09-14, null)

Two of the targets are solid — the Weasly Fish, 58.69 mm across the mid-body (calipered
2026-08-20), and the Snook, whose thickness has not been measured — while the rest are flat
plates. Parallax predicts a solid target reads short by a term in $1/z$, since its near
flank sits closer to the camera than the plane the length is measured in. Two tests, both
null.

**Per-target fit of `pct_error = a + b/z` over the cohort** (b in percent·metres):

| target | | b | 95 % CI |
|---|---|---|---|
| Weasly Fish | solid, 58.7 mm | −1.87 | [−2.34, −1.42] |
| Snook | solid | −2.65 | [−5.13, +0.16] |
| Grouper | flat | **−3.71** | [−5.65, −1.08] |
| Box | flat | −0.59 | [−0.90, −0.25] |
| Purple Angel | flat | +0.70 | [−5.49, +4.10] |

A *flat* target carries the largest significant term, and the trout's −1.87 is **five times
smaller** than the −9.4 its own thickness predicts. Each target is also measured on its own
sessions, so b mixes parallax with whatever range dependence those calibrations retain.

**Within-session differential**, which removes that confound: inside one session the
calibration is common to every target, so the difference in b between a solid and a flat
target isolates the target effect, and it needs no thickness value. Sessions 59, 60, 61,
66, 76, 84 pair the Snook with flats; 509, 521 and 522 pair the trout with the Box. (An
earlier note here said the trout had no within-session comparator — wrong, it has three.)

    solid - flat   n=17 pairs   median +1.79 %·m   <- WRONG SIGN for parallax
    flat  - flat   n=11 pairs   median -0.70 %·m   spanning -14.6 to +16.6

**The control is the result.** Flat-versus-flat pairs scatter as widely as
solid-versus-flat, so the instrument has no resolving power here: individual b estimates
carry intervals like [−73.8, +82.5] on 10–30 frames over a 1.4–2× range spread. The three
trout-vs-box pairs are −7.9, −2.0 and +1.8, inconsistent in sign.

So the hypothesis is neither supported nor refuted, and PAPER.md §4.3 is written to that —
it reports the per-target offsets as target effects, which is what the polish estimates
them as, and says what they are *not*. What would settle it is not more of this corpus: it
is one session that photographs a solid and a flat target together over a wide range spread
with enough frames on each, or the Snook's thickness plus a session pairing both solids
(their lengths differ 1.5×, so the same mechanism predicts different percentages).


### 9.6 Corpus refreshed to 2026-09-14, and why the numbers came back (2026-09-14)

§9.3 posed the export question and this resolves it: `data/corpus.csv` is now a fresh
export, and the 2026-09-12 one is kept beside it as `data/corpus_20260912.csv` because the
reference-sensitivity result below is only reproducible against it.

Four prod changes land together, all of them improvements and none of them the reference
alone:

    the Weasly reference is now 313 mm, measured (§9.x, prod row 8 updated)
    dive 498 re-measured under dive 496's calibration -- its own session's board
        burst -- after its 0.34 m-lever fit was retired (§9.3)
    dive 502 REFITTED BY THE PIPELINE, 8.90 -> 10.354 cm, once the shipped baseline
        floor made the old fit read as uncalibrated; 503 and 504 borrow it
    two orphaned measurement rows deleted (§9.4)

**Result: 13 dives, n = 793, median −2.16 %, p90 +0.12 %**, cohort
(59, 61, 84, 495, 497, 498, 500, 501, 507, 519, 521, 522, 527).

**The trap in that result.** It is within 0.03 points of the originally published
−2.19 % / +0.06 %, and anyone diffing the two would conclude nothing changed. The
membership proves otherwise — 520 left, 527 joined — and the mechanism is two independent
corrections pushing opposite ways:

    reference 310 -> 312.7 alone      13 dives -> 11, median -2.19 -> -2.63 %
    plus 498/502 recalibration        11 dives -> 13, median -2.63 -> -2.16 %

Report the coincidence as a coincidence. It is not corroboration of the old number.

**Better news on the rule.** On the fresh data the bound is not knife-edge: 2.25, 2.50 and
2.75 pp all select the same 13 dives and the same 793 frames, so the chosen 2.5 sits mid
plateau (below 2.0 it drops to 10; at 3.0 it admits an 11th). Four dives still sit within
0.5 pp of the cut — 520 out at −2.75, and 497, 527, 59 in at about +2.2 — so a member can
be marginal even where the membership is stable. Both facts are now in PAPER.md §4.3.

**The range-trend filter still earns its place, and on different dives.** It now removes
503 and 504, which borrow the *refitted* 10.354 cm calibration. A plausible baseline is
therefore not sufficient: the scale-free trend still says their lengths vary with range.
That is the cleanest demonstration yet that the two checks are independent.

**One estimator note worth keeping.** The paper defines $p_{90}$ as nearest-rank
($\lceil 0.9n \rceil$), which is what the pipeline reports, and `pubfig.nearest_rank_p90`
is what the notebook uses. Interpolated `np.percentile(...,90)` agrees to 0.00 pp at
n = 793 but differs by up to 0.08 pp on the per-target figures, where n is 25-313 — so the
per-target numbers in §4.3 are the nearest-rank ones. Do not mix them.

Figures 1, 2, 3, 6, 8 and A were regenerated from the fresh export with the same `pubfig`
calls the notebook makes.


## 10. Six mislabelled frames, found by the implied pixel separation (2026-09-14)

A frame's measured length is geometry; its label is what the length is *graded against*. So
a frame whose length fits another target present in the same session far better than its own
label is a labelling error, not a measurement error. The decisive statistic is the head/tail
**pixel separation** the measurement implies, $\text{px} = L f / z$, which depends on neither
the length convention nor the reference table:

| session | frames | labelled | actually | implied span | nearest correct | nearest labelled |
|---|---|---|---|---|---|---|
| 84 | 2 | Snook | **Purple Angel** | 460, 369 px | 456, 372 | 1065, 951 |
| 521 | 4 | Box | **Weasly Fish** | 1411, 1289, 871, 595 px | 1353, 1261, 920, 611 | 752, 662, 421, 286 |

Dive 84's landmarks are 2.3× too close together to be on a snook and match the angelfish
within a percent; the diver was alternating targets at the same range (purple angel at
1.200/1.209/1.211 m, snook at 1.193/1.208/1.210 m), which is how the confusion happens.
Dive 521's four read +95 to +102 % as a 150 mm box, which no pose or calibration error
produces. Corrected in prod 2026-09-14 (`specieslabel.content_of_image`).

**Detector reliability, measured before any flag was trusted.** Of 2,927 frames only 875
have another target in the same session to compare against at all; over those, the own label
fits better by 45.8 pp at the median, and only 13 frames have an alternative better by
20 pp and 7 by 40 pp. The flags are the extreme tail, not noise. Threshold used: own error
≥ 30 % and some same-session alternative ≤ 10 %.

**One flag deliberately not acted on.** Dive 66 has a frame at 315.93 mm labelled Snook: the
ruler fits it to 7.9 % and the snook label to −30.6 %, and *both* are inside the
foreshortening range (−31.4 % at 45°, §4.4). It sits at the very bottom of that session's
continuous snook range, and the two candidates' lengths are too close for the pixel test to
separate them. Ambiguous, and design-excluded anyway.

**What the method cannot see**, so this is not a clean bill of health: a confusion between
targets of similar length (purple angel 192 vs gray anthias 195 mm), or a whole session
mislabelled — the comparison is within-session. No detector built on length finds either.

**Effect.** The lengths do not change, only what they are graded against. The cohort stays
13 sessions and 793 frames:

    median      -2.18 -> -2.19 %
    p90         +0.09 -> +0.06 %
    mean |err|   3.71 ->  3.09 %      the six frames carried 57-102 % errors
    within 15 %    99 ->   100 %

The corrected median and p90 coincide with the originally published −2.19 % / +0.06 % — the
third such coincidence in this analysis and, like the others, arithmetic rather than
corroboration: the cohort membership and the frame count both differ from the original.


## 11. Section 4.5 re-derived from the database (2026-09-14)

Every number in §4.5 had been computed in a notebook cell against an export that no longer
existed on disk, which is the one condition under which a paper number cannot be checked.
They are now computed by `fishsense_imwut/repeatability.py` from `data/field.csv`
(`sql/extract_field.sql`, pulled from prod today) and pinned by
`tests/test_repeatability.py`. Five of them were wrong.

**The headline figure survived exactly.** 162 measurements of 73 individuals; 25 with three
or more frames; within-fish CV median **2.91 %**, bootstrap 95 % CI **[1.63, 4.06]**. That
is the same to two decimals as the figure §9.4 arrived at, and the orphaned-row delete has
held: no measurement in the field set carries NULL provenance, and dives 347 and 349 are
measured under extrinsics 54 and 55 — their refits, not the retired fits.

**What was wrong:**

1. **The $p_{90}$ was interpolated, not nearest-rank: 9.1 % where the paper's own
   convention gives 11.4 %.** §9.6 closes with "Do not mix them", and §4.5 was mixing them
   — because §4.5's figure came down a computation path with no `pubfig` call in it, so the
   convention was never applied. A convention stated in prose does not propagate; only a
   shared function does, which is the reason the estimator is now a module.
2. **The pool comparison was computed from the frozen August export.** On
   `corpus_20260912.csv` it is median 1.48 %, p90 4.56 % — the draft's "1.4 %, p90 4.5 %".
   On the current corpus it is **1.33 %, p90 3.03 %**. The pool repeatability improved
   because the six mislabelled frames of §10 were corrected: a frame labelled as the wrong
   target lands in the wrong (session, target) cell and inflates that cell's spread.
   Consequence for the claim: the field/pool ratio is 2.2× on the median and 3.8× on the
   tail, not "about twice" on both.
3. **The laser range was reported as a span when it is a 5th-to-95th percentile.** "0.71–
   3.14 m" is p5–p95; the actual span is **0.46–3.90 m**, median 1.49 m. Both are true
   statements, but only one is what the sentence said.
4. **The two repaired baselines were said to be inside "the 9.99–10.45 cm the fleet
   occupies", and 10.52 cm is not.** 9.99–10.45 was the fleet IQR at the time, quoted as
   though it were the range. The fleet now holds 31 fits spanning 9.87–12.95 cm; 30 of them
   lie in 9.87–10.54 and both repairs are inside that. The single outlier is dive 107 at
   12.95 cm, whose observations span 2.8 cm along the ray — rejected by the conditioning
   criterion without reference to its baseline, which is the useful thing to say about it.
5. **The unit-to-unit bound of "≤ 9 % (95 % upper limit, 34 hogfish)" does not reproduce,
   and is replaced by a weaker, correct statement.** It is 33 hogfish now, and no standard
   construction gives 9 %: the $F$-based interval is degenerate (the point estimate is
   1.5 % between cameras with $F = 1.03$, $p = 0.42$, so the lower bound on the variance
   component is zero and the upper is unbounded at 95 %), a $\chi^2$-on-MSB upper limit
   gives 15 %, and a nonparametric bootstrap over individuals gives 13 % by camera and
   16 % by deployment. So there is **no detectable unit effect**, and the sample cannot
   bound one below the between-fish spread it sits in. That is a real weakening: the draft
   claimed field lengths agree across units to 9 %, and the data does not support a number
   that small in either direction.

Two smaller corrections went with them: the within-fish range spread ("no wild fish spans
more than 1.5×" — the best is 1.9× over all 73 individuals, 1.6× among the 25 with three
frames) and the six-vs-seven units question. There are **six cameras across seven
deployments** — camera 3 shot both 383 and 471 — so §8's per-rig hogfish split, which lists
seven groups, is split by *deployment*. PAPER.md §4.5 says "six camera units" and is right;
it now reports both groupings for the variance fit rather than picking one silently.

**One thing §4.5 gained.** The paragraph asserting that the field figure needs nothing
external now has a test behind it rather than a before/after anecdote:
`test_rescaling_one_dives_lengths_leaves_the_field_figure_alone` multiplies dive 347's
lengths by 1.18 and asserts the median, the p90 and the group count are unmoved. That is
the invariance the claim rests on, stated as an executable property. It also explains the
anecdote the paragraph used to lean on: the repair moved the figure from 3.3 % to 2.9 %
rather than not at all, because a refit turns the axis as well as changing the baseline, so
its correction factor varies with range instead of being exactly common to a fish's frames.

**A sixth error, found on the way past.** The limitations paragraph ("Sessions the rule
rejects") says six of the nine rejected sessions borrow their calibration from another
session. **Eight of the nine do** — every one but 509, which fitted its own. That makes
borrowing the strongest single predictor of rejection in the corpus rather than one factor
among several, which strengthens §4.2's recommendation instead of weakening it, and it is
now pinned by `test_the_rejected_sessions_are_what_the_limitations_paragraph_reports`
along with the other two counts in that sentence (nine rejected, six range-trend flagged,
baselines 10.24–10.51 cm — all three of which do hold).

**And one limit §4.5 was missing**, now written into the paper: the species attributions are
unverified. The mislabel detector of §10 compares a frame's implied head-to-tail pixel
separation, $Lf/z$, against what the named species would require, and it needs a known
length — no wild fish has one. So a systematic confusion between two similarly-shaped
species would appear as a per-species offset indistinguishable from a measurement bias,
which is a second reason §8's stereo comparison cannot carry a conclusion. The
repeatability is the one field result immune to it, being computed within an individual
whatever that individual turns out to be.


## 12. A paired field measurement — 2023-08-03, seven fish, two instruments (2026-09-16)

§4.5 says field accuracy is unmeasurable, and for the seven established
deployments that is exactly right: no known-length reference was ever in the
water. This is the exception the archive turned out to contain — one day where
the same individual animals were measured by FishSense Lite **and** by a
calibrated stereo-video rig.

**The pairing is by named individual, not by population.** Each dive folder is
one fish (`Hogfish01_MolHITW_0926_080323`) and the archive numbers the same
fish the same way; both sides' hogfish numbering at ConchReef runs 1–12 and
**both skip 10**, which is a shared registry rather than two independent
counts. That is a far cleaner design than §8's population-median comparison,
and it lands in a different place.

### What it took to get there

Nothing about this was available on 2026-09-15. Four pipeline defects had to
ship first, and each one is written up in `CLAUDE.md`:

* `content_of_image` took whichever taxonomy path the labeler clicked first, so
  34 rows across 6 dives lost their laser answer. Dive 22 lost all ten frames.
* stage 1 dropped HDBSCAN noise points, so an evenly-spaced dive returned `[]`
  and head-of-line blocked the cohort forever (dive 8, selected 8 times in a
  row).
* the calibration refusal message described an hourly churn that the refusal
  record had already made impossible, and I read my own stale sentence back as
  a live diagnosis.
* sentinel species rows were read as labeler answers by stage 6.1, which split
  one hogfish into two `Fish` rows carrying 4 and 7 measurements.

The day's calibration is **dive 32** (`H Slate Dive 1`, 09:11), fitted at
**10.551 cm** — inside the 9.87–10.54 cm band 30 of the fleet's 31 fits
occupy. The other two H-Slate captures were refused on the lever arm: dive 22
at 0.25 m over ten observations, dive 23 at **0.59 m** against a 0.60 m gate
over five. Only the 09:11 burst varied range enough to be determinable, which
is a fact about the capture protocol and not about the pipeline.

All eight morning dives borrow dive 32 via `Dive.calibration_dive_id`. The
afternoon individuals (ConchReef, SnappersLedge) are **excluded**: 5–7 hours
from the only sound fit, with both afternoon slates refused, so nothing checks
whether the laser moved during the day.

### The result, reported at p90

**p90 is the reported estimator** — the paper's only one, wherever frames become
a length. The median and maximum are kept here as context, because the spread
between them is what justifies the choice.

| individual | n | our median | **our p90** | our max | stereo | median diff | **p90 diff** | max diff |
|---|---|---|---|---|---|---|---|---|
| Hogfish 1, MolHITW | 11 | 304.8 | **326.7** | 345.6 | 315.8 | −3.5 % | **+3.5 %** | +9.4 % |
| Hogfish 1, MolPeLe | 3 | 285.6 | **288.0** | 288.0 | 265.2 | +7.7 % | **+8.6 %** | +8.6 % |
| Hogfish 2, MolHITW | 3 | 275.7 | **277.9** | 277.9 | 322.3 | −14.5 % | −13.8 % | −13.8 % |
| Blue Parrotfish 1, MolHITW | 10 | 460.5 | **472.1** | 479.0 | 426.5 | +8.0 % | **+10.7 %** | +12.3 % |
| Stoplight Parrotfish 1, MolHITW | 6 | 190.5 | **220.0** | 220.0 | 197.9 | −3.7 % | **+11.2 %** | +11.2 % |
| Grey Snapper 1, MolPeLe | 3 | 233.7 | **273.9** | 273.9 | 298.2 | −21.6 % | −8.1 % | −8.1 % |
| Blue Parrotfish 1, MolPeLe | 3 | 338.6 | **353.5** | 353.5 | 377.5 | −10.3 % | −6.4 % | −6.4 % |

    per-fish p90    -> median +3.5 %, mean +0.8 % +- 10.1 %, 4/7 within 10 %   <- REPORTED
    per-fish median -> median -3.7 %, mean -5.4 % +- 11.0 %, 4/7 within 10 %
    per-fish max    -> median +8.6 %, mean +1.9 % +- 10.9 %, 4/7 within 10 %

Note the p90 column against the max column: **identical for five of the seven**.
Nearest rank is ceil(0.9n)-th of n, which is n for every n <= 10, and only the
two best-sampled fish (11 and 10 frames) separate them. p90 here is a high-order
statistic, not a tail estimate. The same holds for every wild animal in §8 — none
has more than 8 frames — so a field p90 is always a longest frame. Intended,
since with a one-sided error the longest frame is the least pose-corrupted, but
it does not carry the estimator precision a 26-frame pool cell does.

**The median is the wrong summary, and using it inverts the reading.** Stage 14
back-projects head and tail at a single laser-derived depth, so an out-of-plane
fish can only read SHORT. That is not an assumption here: over the 16 frames
carrying a laser depth, `length / (head_tail_px · range / f)` runs
**0.9972–0.9997** — self-consistent to a fraction of a percent and never above
1.0. The out-of-plane correction only ever removes length.

So the maximum over frames of one animal is a **lower bound on its true
length**, and the median is biased down by however much the pose varied.
Switching estimator moves the centre from −3.7 % to +3.5 % (p90) or +8.6 %
(max) while leaving the scatter at ~10–11 %: it is a question about the
estimator, not about noise. This is the same reasoning §4.3 already uses when it
reports p90 rather than a mean, and it was not applied here until the frames
forced it.

**Four of seven individuals exceed the stereo at p90**, by 3.5–11.2 % — the same
four that exceed it at the maximum. Our own pose loss cannot explain those,
because pose only subtracts.

### What is ruled out, measured rather than argued

* **Not the calibration.** All 41 measurements resolve through one fit, so a
  scale error is common-mode: it can move the centre and cannot make two fish
  disagree with each other. And it is not even moving the centre — the
  scale-free range check on dive 5 (11 frames over a 1.55× range spread) gives
  **+0.8 %/m, r = 0.03**, flat.
* **Not the projection.** 16 of 16 frames self-consistent, above.
* **Not landmark over-reach**, which was my first hypothesis and the frames
  refuted it. The two Blue Parrotfish are the same species on the same day with
  opposite signs; inspecting the tail clicks, dive 25 (+8.0 %) lands **in the
  fork notch** and dive 39 (−10.3 %) lands **at the caudal fin tip**. Clicking
  the tip lengthens a measurement, so over-reach predicts the opposite of what
  is observed. Recorded because the hypothesis was plausible, cheap to check by
  looking, and wrong.

### What remains unattributable

Either we read 4–11 % long on four of seven fish, or the stereo reads short, or
the pairing is wrong on those individuals. **Seven pairs and no third
instrument cannot separate those**, and the mean-difference interval spans zero
under every estimator (at p90, −6.7 % to +8.3 %). This is the same wall §8 hit from the population side,
reached from a much better design.

Two specific follow-ups worth more than more statistics:

* **Dive 20** is the cleanest anomaly: three frames agreeing to **0.6 % CV**,
  the tightest in the set, sitting 14.5 % from the stereo. Internal consistency
  that good with a large offset is the signature of a scale disagreement, which
  is precisely what reprojection residual cannot see. One dive, not a
  statistical question.
* **The two Blue Parrotfish** disagree by 18 points in opposite directions on
  one species, one day, six frames between them. Either the labelling differs
  or one pairing is wrong.

### What this does and does not license in the paper

It does **not** license changing §4.5's conclusion. Seven pairs at ±8 points is
exactly the arithmetic that paragraph already states, and the honest sentence
is still that field accuracy is unmeasurable from this corpus.

What it does license is a narrower, true claim: on one day, seven wild fish
measured by two instruments agree to within about 10 % per individual with no
detectable systematic offset — and the first paired field dataset this project
has, with the provenance recorded well enough to extend.

Species attribution across all of it remains the labeler's and unverified: the
mislabel detector needs a known length and no wild fish has one. Frame 1378 of
dive 25 was labelled Rainbow Parrotfish among nine Blue Parrotfish and was
caught only because a human looked.
