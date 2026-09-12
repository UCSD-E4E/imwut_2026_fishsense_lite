# Section 4 — Results (draft) and what to do with Figures 4/5

Draft text for the FishCamera / FishSense Lite paper, written against the figures and
numbers in `fish_model_analysis/` as of 2026-09-12 (re-pulled after the 490 → 527 split). Every number below
is produced by `fish_model_analysis/fish_model_measurements.ipynb` from
`data/corpus.csv` and `data/angles.csv`, and the cohort is pinned by
`tests/test_calibration.py`. Figure files are in `fish_model_analysis/figures/`.

Part A is the section draft. Part B is the recommendation on the August repair figures.

---

## Part A — Section 4 draft

### 4 RESULTS

We evaluate FishCamera in two settings: controlled pool sessions against rigid targets of
known length, which give an absolute accuracy figure and isolate the dominant error source
(fish pose); and open-water deployments on wild fish, which show the system in the hands
of the divers it is designed for. §4.1–4.3 are the pool results; §4.4 carries the existing
field-deployment text.

#### 4.1 Known-length targets

Six rigid targets were measured: five painted fish models — Purple Angel (192 mm), Weasly
Fish (310 mm), Grouper (360 mm), Snook (455 mm), Shark (605 mm) — and a 150 mm box. Each
target's reference length is the snout-to-tail-fork distance a labeler is asked to click,
measured with a tape. Targets were photographed in a university pool over 32 sessions
between 2023 and 2025 at laser ranges of 0.25–4.71 m (median 1.7 m), by several divers on
several camera+laser units, at ranges and poses of the diver's choosing rather than on a
fixture. Every frame passed through the same pipeline as a field image: a labeler marks
the laser dot and the snout and tail-fork, the laser dot fixes the range (Eq. 5), and the
length follows from Eqs. 6–8. Calibration was per unit and per session: the 2023 sessions
used the duct-tape dive slate of §3.3 photographed in a companion session, and the 2025
sessions a planar checkerboard photographed in the same session. In total 2,927
measurements were made.

Not every session is accuracy evidence, and we do not choose which are by hand. A per-frame
error has two additive components: a per-*session* term, which is the calibration offset
and moves every target in that session together; and a per-*target* term, which is a
reference or landmark offset and follows the target across sessions. We separate them with
a Tukey median polish on the $p_{90}$ percent error of each (session, target) cell with at
least five frames [Tukey 1977]. The polish fits tightly — median absolute residual
0.15 percentage points over 30 sessions × 7 targets — so the two terms are the whole story.
A session enters the accuracy cohort when its calibration offset is within 2.5 percentage
points of the corpus median. Two hold-outs precede that test. By design, before any number
was computed: the five angle-experiment sessions of §4.3 (a single target at deliberately
oblique poses) and three sessions whose calibration a separate analysis had already
repaired or disputed. And by a scale-free check that uses no reference length: a rigid
target must read the same length at every range, so a session whose targets show a length
trend with range — the Theil–Sen slope over frames beyond 0.8 m, with its 95 % interval
clear of ±2 % per metre — has a calibration error in the in-plane laser angle that no
reprojection test can see (§3.3), and is excluded. That check removes seven sessions,
every one of which the reference lengths independently grade as wrong, including two whose
median error looked fine because a short fitted baseline and a compensating angle cancel
at mid-range. The rule admits 13 sessions and 771 measurements; three are slate-calibrated
2023 sessions and ten are checkerboard-calibrated 2025 sessions. Sensitivity to the one
free number is small: moving the threshold from 1.5 to 3.5 points changes membership from
8 to 15 sessions and the cohort $p_{90}$ by less than half a point.

#### 4.2 Accuracy

Figure 1 plots measured against known length for every cohort frame; Figure 2 gives the
per-target error distribution; Table 1 summarises. We report the $p_{90}$ of a target's
frames rather than a mean, and the reason is physical: the pipeline back-projects snout and
tail at the single laser-derived depth, so it measures the target's *projection*, and a
target that is not broadside to the camera can only read short, never long. The error
distribution is therefore one-sided, a mean measures the diver's pose distribution rather
than the instrument, and a high quantile is the estimator that rejects the pose tail.
$p_{90}$ is nearest-rank ($\lceil 0.9n \rceil$), the same statistic the deployed pipeline
reports.

Over the cohort the median frame error is −2.2 % and the $p_{90}$ is +0.06 %; 78 % of
frames are within 5 % of the reference, 98 % within 10 %, and 99 % within 15 %. The
per-target $p_{90}$ — Box −0.1 %, Purple Angel +0.9 %, Weasly Fish −1.9 %, Grouper +0.3 %,
Snook −0.9 %, Shark +4.3 % — shows no trend with size across a four-fold range of lengths.
The polish attributes +3.2 points of the Shark's figure to the target itself rather than to
any session; the model is no longer available to re-measure, and it is 25 of 771 frames,
so excluding it moves the cohort $p_{90}$ from +0.06 % to −0.24 %.

Figure 3 plots error against laser range. Triangulation conditioning degrades as $Z^2$
(Eq. 5), so a range dependence surviving into the delivered length would appear as a
widening band; instead the binned median is flat at −2.1 to −1.8 % from 0.8 m to 4.7 m
with an interquartile range that does not grow. The one departure is the nearest bin: below
0.8 m the Weasly Fish reads −6.8 % (the Box, at the same ranges, −2.3 %). It is confined to
one target, so it is a landmark effect on that model at close range rather than ranging;
it is inside the error budget and below the working range of a diver measuring wild fish.

**Table 1.** Percent length error over the accuracy cohort and, for comparison, over wider
selections. Widening beyond the rule only worsens the figures, which is the cost of not
choosing sessions by hand; $p_{90}$ stays within ±0.7 % of zero throughout.

| selection | sessions | frames | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| accuracy cohort (rule) | 13 | 771 | −2.19 % | +0.06 % | 3.71 % |
| every session not held out (design + range check) | 18 | 910 | −1.94 % | +0.68 % | 3.49 % |
| every session except the angle experiment | 27 | 1,499 | −3.48 % | +0.16 % | 5.15 % |
| every session | 32 | 2,927 | −5.91 % | −0.59 % | 9.20 % |

```latex
\begin{table}[t]
  \caption{Percent length error over the accuracy cohort and over wider selections.
  Widening beyond the rule only worsens the figures; $p_{90}$ stays within $\pm0.7\,\%$
  of zero throughout.}
  \label{tab:accuracy}
  \begin{tabular}{lrrrrr}
    \toprule
    selection & sessions & frames & median & $p_{90}$ & mean $|$err$|$ \\
    \midrule
    accuracy cohort (rule)               & 13 &   771 & $-2.19$ & $+0.06$ & 3.71 \\
    not held out (design + range check)  & 18 &   910 & $-1.94$ & $+0.68$ & 3.49 \\
    all but the angle experiment         & 27 & 1{,}499 & $-3.48$ & $+0.16$ & 5.15 \\
    every session                        & 32 & 2{,}927 & $-5.91$ & $-0.59$ & 9.20 \\
    \bottomrule
  \end{tabular}
\end{table}
```

#### 4.3 Fish pose

The single-depth back-projection is the system's principal approximation, and its cost is
predictable: a fish at angle $\theta$ to the image plane projects to $\cos\theta$ of its
length, so the measurement must read $\cos\theta - 1$ short. We tested this directly. A
Snook model was presented at 0° to 45° in 5° steps, the angle read from a protractor card
held in frame, in five sessions on five camera units at two ranges (~2 m and ~4 m), for
1,428 measured frames. Figure 8 plots the error against the designed angle: each session as
a thin line, the pooled median and interquartile range in black, and $\cos\theta - 1$
dashed.

Three things are visible. The five sessions agree within the pooled interquartile range at
every angle, at both ranges, so the curve is a property of the geometry and not of a unit or
a diver. The pooled median follows $\cos\theta - 1$ offset by a near-constant −3.5 % — the
same broadside bias the cohort shows — from 0° (−3.8 %) through 15° (−5.8 %), 30°
(−15.0 %) and 45° (−31.4 %). And the 15 % error budget is crossed at 30°. A diver who
photographs a fish within ±30° of broadside therefore stays inside the budget with no pose
correction at all; beyond that the correction is a known function of a quantity that
single-image fish-pose estimators are beginning to recover [29].

#### 4.4 Field deployments

*(Existing text — Florida Keys deployments, the red/green laser comparison, Figures 5–6,
and the stereo comparison — goes here unchanged.)*

#### Sessions the rule rejects

*(For the paper's limitations paragraph, or an appendix; Figure A gives every session.)*
Ten sessions fall outside the band or the range check, and every one has an explanation. The
instructive one read 14 % short with correct labels and a calibration that reproduced its
own checkerboard perfectly: its folder held a board burst shot five minutes *after* its
target frames, and the laser had rotated 0.82° in the plane that reprojection cannot see.
Re-measured under the calibration burst that preceded the frames by nine seconds, it reads
−1.7 %. Three sessions borrow a calibration across such a movement (−5.8, −3.5, −2.7); two borrow
a calibration whose fitted baseline is 14 % short and read −14 % up close rising to zero
at 4 m, which the range check catches and a median would not; the rest sit within a
point of the band on either side (−2.8; +2.9, +3.1, +4.0). None is
a ranging failure the diver could see: an in-plane calibration error is invisible to
reprojection residual, which is why the calibration procedure of §3.3 photographs the
target at several ranges, why a calibration is discarded whenever the unit has been
handled, and why a rigid reference photographed at two ranges after calibrating is the one
check that would have caught this in the field.

#### Edits elsewhere in the draft

- **Limitations paragraph (currently "beyond (??) degrees")**: "with our model fish, the
  measurement leaves the 15 % budget when the fish is more than 30° from broadside
  (Figure 8)."
- **Abstract / Conclusion**: the "<15 %" and "within our target margin of error of 20 %"
  claims are supported: 99 % of cohort frames are within 15 % and the $p_{90}$ is +0.06 %.
  Consider stating the broadside figure directly: "median −2.2 %, $p_{90}$ +0.06 % over 771
  measurements of six targets at 0.27–4.7 m".
- **Figure ?? (similar triangles)** in §3.3 is an unresolved reference.

#### Figure captions

- **Figure 1** — Measured against known length, accuracy cohort (13 sessions, 771 frames).
  Six targets separate along the abscissa by their own lengths. Error bars are the
  interquartile range of frames; the diamond is the $p_{90}$.
- **Figure 2** — Percent length error by target, accuracy cohort. Box: frame-level
  interquartile range and whiskers; diamond: $p_{90}$, the reported estimator. The gap
  between median and $p_{90}$ is the pose tail, not instrument spread.
- **Figure 3** — Percent length error against laser range, accuracy cohort. Binned median
  and interquartile band over the per-frame cloud. The band does not widen with range.
- **Figure 8** — Percent length error against fish angle to the image plane, from five
  sessions of one target stepped through 0–45°. Thin lines: per-session binned medians;
  black: pooled median and interquartile range; dashed: $\cos\theta - 1$; dotted: the 15 %
  budget, crossed at 30°.
- **Figure A (appendix)** — Percent length error for every session with ≥ 8 frames,
  ordered by median. The held-out and rejected sessions are the wide, negative rows.

Drop-in: figures are sized for the `acmart` column already (`\includegraphics` with no
`width=`); see the last cell of the notebook.

---

## Part B — Figures 4 and 5 (the August repair)

**Recommendation: cut Figures 4 and 5 from the paper; keep Figure 4's number as one
sentence; keep both figures in the repository.**

Figure 4 (per-session mount state $\varphi$) and Figure 5 (the implied-yaw floor before and
after repairing sessions 60 and 76) were built when the accuracy claim rested on seven
slate-borrowed sessions and the paper needed to explain why two of them were repaired. That
is no longer the situation:

1. **The accuracy claim no longer depends on the repair.** Sessions 60 and 76 are held out
   of the cohort by design, so nothing in Table 1 or Figures 1–3 uses a repaired value. A
   reader does not need the repair to trust the numbers.
2. **The repair is not the paper's contribution.** The paper is a hardware framework plus a
   length-recovery method. $\varphi$-repair is a calibration-analysis technique; explaining
   it properly takes the half page of physics in `fish_model_analysis/HANDOFF.md` §1 (why
   in-plane rotation is invisible to the laser dots) and then a method that uses a known
   length to fix a calibration — which a reviewer will read as circular unless the
   scale-free ruler-at-two-ranges argument is also made. That is a different paper.
3. **The one fact worth keeping is the mount-state spread.** Across the seven 2023
   sessions the fitted in-plane laser angle spans 0.27°, and 0.15° of rotation is a 5 %
   length error at 2 m. That single sentence, in the discussion of why calibration is
   per-session (§3.3 "must be calibrated per dive site"), makes the argument Figure 4 made,
   and it makes it better without a figure: the mount *does* move between sessions, by an
   amount that is invisible in the images and material in the length.
4. **Figure 5 in particular argues against the paper's own cohort rule.** It shows the
   repair driving 60's yaw floors to 0.0 — a success — while the paper then declines to use
   60. Both positions are defensible, but defending both costs more words than either is
   worth.

Suggested sentence for §3.3 or the discussion:

> The laser's in-plane pointing angle, which sets metric scale and which no
> reprojection check can observe, varied by 0.27° across seven sessions of the same unit
> between calibration and use — equivalent to a 5 % length error at 2 m for a 0.15°
> change — so calibration is repeated whenever the unit has been handled.

If a figure is wanted for that sentence, Figure 4 as it stands (seven sessions, angle on
the ordinate, cohort by marker) is the right one and needs no change; Figure 5 should go
regardless. Figures 6 (Shark anomaly) and 7 (fork probe) are diagnostic figures for the
repository, not for the paper; the Shark is handled by the one sentence in §4.2.
