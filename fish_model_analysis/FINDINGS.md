# Fish-model accuracy figures — what was built and what it turned up

Companion to `HANDOFF.md` (the 2026-08-26 calibration-repair session, which supplied the
data). This document covers the figure work that followed and the findings that came out
of building it. Nothing here has been written to prod.

---

## 1. What is here

| path | what it is |
|---|---|
| `data/all.csv` | 464 per-frame measurements + per-frame geometry. `\|`-delimited: the geometry columns are JSON and contain commas. Carries a trailing psql `(n rows)` footer, dropped on load. |
| `data/corpus.csv` | **every** rigid-target measurement in prod as of 2026-09-12: 2,927 frames, 32 dives. Same schema; strict superset of `all.csv` (pinned by a test). Extracted by `sql/extract_corpus.sql`. |
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
and disputed 66. Dive 490 is *not* on that list; it falls out on its own effect (−13.8 pp).

The polish fits tightly — median |residual| **0.15 pp** over 30 dives × 7 models — so the
two additive terms remain the whole story at corpus scale.

Result, pinned as `cal.CORPUS_ACCURACY_DIVES` and by `tests/test_calibration.py`:
**59, 61, 84, 491, 495, 497, 498, 500, 501, 503, 507, 519, 520, 521, 522** — three August
dives plus twelve 2025 pool dives (checkerboard self-calibration or a same-rig borrow).

### 7.2 Headline

| cohort | dives | n | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| **rule: \|dive effect\| ≤ 2.5 pp** | 15 | 842 | −2.37 % | **+0.02 %** | 4.04 % |
| August five (58/59/60/61/84) | 5 | 335 | −1.51 % | +1.41 % | 3.00 % |
| every dive not held out by design | 23 | 1192 | −3.15 % | +0.19 % | 5.02 % |
| everything except the angle experiment | 27 | 1499 | −3.84 % | +0.04 % | 6.18 % |
| everything | 32 | 2927 | −6.24 % | −0.68 % | 9.72 % |

Every widening beyond the rule makes the numbers worse — that is the cost of not
cherry-picking — and $p_{90}$ stays within ±0.7 % of zero in every row. Threshold
sensitivity: 1.5 → 3.5 pp moves membership from 8 to 19 dives and $p_{90}$ between −0.45 %
and +0.02 %.

Ladder on the cohort ($p_{90}$): Box −0.24, Purple Angel +0.85, Weasly Fish −1.89, Grouper
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
within 0.4 pp), and it is 25 of 842 cohort frames: dropping it moves the cohort $p_{90}$
from +0.02 % to −0.28 %; re-referencing it at 620 mm gives −0.19 %. Reported as-is with
the model effect stated.

### 7.4 Dives the rule rejects

| dive | effect (pp) | what it is |
|---|---|---|
| 490 | −13.8 | checkerboard self-calibrated; board dots reproject correctly; labels verified on raw pixels; every model ~14 % short. **Unresolved.** |
| 492 | −5.5 | borrows 490 |
| 494 | −3.2 | borrows 493 |
| 509 | −2.5 | self-calibrated, just outside |
| 504 / 58 / 506 | +3.0 / +3.4 / +4.3 | positive side; 58 is the August dive §3.2 flagged |

### 7.5 Foreshortening (Figure 8)

Pooled median error by designed angle: 0° −3.8 %, 5° −3.5 %, 10° −4.8 %, 15° −5.8 %,
20° −8.7 %, 25° −11.7 %, **30° −15.0 %**, 35° −20.5 %, 40° −25.0 %, 45° −31.4 %. The curve
is $\cos\theta - 1$ offset by a ~−3.5 % broadside bias; five sessions at two ranges agree
within the pooled IQR; the 15 % budget is crossed at 30°.

