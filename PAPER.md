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
of the divers it is designed for. §4.1–4.4 are the pool results; §4.5 carries the existing
field-deployment text.

#### 4.1 Known-length targets

Six rigid targets were measured: five painted fish models — a purple angelfish (192 mm), a
stylised rainbow trout (310 mm; "Weasly Fish" in the data), a grouper (360 mm), a snook
(455 mm), a shark (605 mm) — and a 150 mm box. Each
target's reference length is the snout-to-tail-fork distance a labeler is asked to click,
measured with a tape.

One of those references was inherited rather than measured, and we measured it. The
trout's **310 mm** came from a prior publication and was recorded as provisional: its true
fork length was known only to lie in 300–310 mm, with 310 the top of that interval. Two
independent reconstructions of the physical model both placed it longer — a dense
photogrammetric point cloud scaled on the box's 150 mm span gives a snout-to-fork chord of
315 ± 1 mm, and 66 frames from two LiDAR phones, with different cameras, independently
published intrinsics and scale taken from the sensors' own metric depth so sharing no
scale chain with the box, give 319 ± 2 mm — but neither is a direct measurement. A tape
now settles it: **312.7 ± 0.5 mm**, from two independent readings of the same landmarks
the labelers click (12 5/16 in on an imperial tape, 312–313 mm on a metric one). We adopt
that value, which both moves the reference 2.7 mm above the top of the assumed interval
and retires the one-sided 0…−3.2 % band that interval implied. It also adjudicates the
reconstructions: both read long, by 2.3 and 6.3 mm, in the direction a straight chord
between two surface points must err when the snout and the fork do not lie in one plane. The **shark's 605 mm** is likewise undocumented and the
model is no longer available; §4.3 reports its offset rather than correcting it.

It is worth being explicit about which half of the measurement chain those uncertainties
sit in, because the other half is checkable and checks out. The pipeline's metric scale
does not come from the targets: it comes from the calibration object — a scanned dive slate
or a printed checkerboard — and a wrong one would rescale every length invisibly, since
reprojection residual cannot see scale (§3.3). Those two objects are independent standards,
and the laser baseline is a property of the rig rather than the dive, so any camera
calibrated both ways must report the same baseline either way. Over the five units carrying
both (19 calibrations, five distinct slate scans) the mean checkerboard-minus-slate
difference is **−0.27 % of baseline (95 % CI −1.1 % to +0.6 %)**, below the scatter between
two calibrations of one unit under one standard (sd 0.12 cm). The scale chain is therefore
verified to about a percent without reference to any fish, which places the dominant
uncertainty in §4.3's accuracy figures on the target lengths above — not on the
calibration. What neither check can see is an error common to both standards, or one in the
camera intrinsics, which rescale the calibration object and the laser together.

Targets were photographed in a university pool over 32 sessions
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
least five frames [Tukey 1977]. Fifty-two cells are observed over 30 sessions and 7 targets,
and the additive fit accounts for them closely: the median absolute residual is 0.15
percentage points, and 0.45 points over the ten sessions that carry more than one target and
so can exhibit a residual at all. One caveat follows from which targets went in the water
when: the 2023 sessions photographed the fish models and the 2025 sessions the box and one
model, with no target in common, so the session and target terms are anchored separately
within each group and the polish's centring is what places the two groups on a single scale.
A reference error common to one group's targets would therefore shift that group's session
offsets bodily against the other's; within a group it could not.

A session enters the accuracy cohort when its calibration offset is within 2.5 percentage
points of the corpus median. Two hold-outs precede that test. By design, before any number
was computed: the five angle-experiment sessions of §4.4 (a single target at deliberately
oblique poses) and three sessions whose calibration a separate analysis had already
repaired or disputed. And by a scale-free check that uses no reference length: a rigid
target must read the same length at every range, so a session whose targets show a length
trend with range — the Theil–Sen slope over frames beyond 0.8 m, with its 95 % interval
clear of ±2 % per metre — has a calibration error in the in-plane laser angle that no
reprojection test can see (§3.3), and is excluded. That check removes seven sessions,
including two whose median error looked fine because a short fitted baseline and a
compensating angle cancel at mid-range. Five of the seven carry a single target and the
reference lengths independently grade every one of them as wrong; the remaining two are
rejected on one target while a second target in the same session shows no trend, and both
are already outside the cohort for other reasons, so nothing in the reported figures turns
on them. The rule admits 13 sessions and 771 measurements; three are slate-calibrated
2023 sessions and ten are checkerboard-calibrated 2025 sessions.

Neither free number is delicate. Moving the polish threshold from 1.5 to 3.5 points changes
membership from 8 to 15 sessions and the cohort $p_{90}$ by less than half a point. The
range check is insensitive over a wider band still: any trend threshold from 2 to 4 % per
metre selects the identical 13 sessions and the identical figures, and neither the 0.8 m
floor (0.6–1.0 m) nor the eight-frame minimum (6–12) changes which sessions it rejects.

#### 4.2 Calibration stability

Both hold-outs above reject sessions on the state of one parameter, and it is worth saying
plainly why that parameter needs watching. The laser's in-plane pointing angle sets metric
scale and is invisible to any reprojection check (§3.3): rotating the axis within the
camera–laser plane moves the projected dot by less than $10^{-12}$ px, so a calibration can
be wrong by an amount no fit residual can reveal. Figure 4 plots that angle, fitted per
session, for seven sessions of one unit.

It is also not stable. Across those seven sessions the angle spans 0.27°, against a
sensitivity of $-2.0\,\%$ in length per 0.15° at 0.9 m and $-4.5\,\%$ at 2.0 m — so the
between-session spread alone exceeds the accuracy we report below. Nor is a session a safe
unit. In one session two calibrations of the same rig, taken seven minutes apart, differ by
0.82°; the target frames shot between them agree with the earlier one, while frames from
25 minutes before agree with neither, placing the mount in a third state.

One field deployment shows the same thing without needing a second calibration to compare
against, and it is the cleanest instance we have. The dive holds two bursts of calibration
frames, 52 minutes apart, and 71 laser dots on its measurement frames. Those 71 define a
line to 0.64 px; the first burst's dots sit 42.6–46.2 px off that line and the second's sit
66.5–72.9 px, each burst internally tight. Three distinguishable laser states in one dive,
measured against the dive's own frames rather than against another fit — and the dots are
correctly placed, which we verified by inspecting the frames at the pixel level after the
offsets first suggested mislabelling. A calibration is valid for the frames it was taken
with and not reliably beyond them, and in this dive no calibration is valid for the frames
that would be measured: it yields no lengths at all.

Three consequences follow, and they shape the rest of the paper. The pipeline re-fits the
laser per dive rather than trusting a stored calibration. A per-session validation target
remains good practice, because the failure is silent — this is what the range check of §4.1
supplies, and it needs no reference length. And calibration state is the dominant reason a
session is not accuracy evidence: of the 25 sessions here with enough frames to judge, 12
fail on one of the two criteria in §4.1. An instrument whose scale parameter drifts by more
than its own measurement error between uses is one that must be calibrated, and validated,
at the point of use.

*(Added 2026-09-14 from the conditioning measurements. This is the text §4.5 refers to
as "the conditioning criterion of §4.2"; it is written to slot in after the paragraph above
and before §4.3. Numbers and their controls are in `fish_model_analysis/FINDINGS.md` §9.)*

A second failure mode of the same parameter is that a session's own fit can fail to
determine it. What fixes the laser's direction is the spread of the calibration
observations *along* the ray — the lever arm — against the noise in locating the dot, not
their number: one pixel of dot-label noise at range $z$ is $z/f$ metres of lateral error,
so it rotates the fitted axis by about $(z/f)/\ell$ radians for a lever $\ell$. Two
observations a metre apart therefore determine the axis far better than sixteen at one
distance, and a burst shot at a single distance does not determine it at all.

The consequence is not a large error everywhere. It is a fit that is accurate where it was
taken and wrong away from it, which makes it invisible to a validation performed at the
working distance. The calibration frames themselves supply the check: the target's pose
gives a per-frame distance that does not involve the laser at all, so laser-triangulated
range can be compared against it frame by frame. For one session of 16 observations
spanning 0.03 m of range, the stored calibration reproduces its own working distance to
$-0.12\,\%$ and a distance 2.2 m further out to $-17.25\,\%$ ($n=16$ and $n=15$ frames);
its fitted baseline, 12.95 cm, is the widest in a fleet whose sound calibrations lie
within 9.99–10.45 cm.

Three checks miss it, and the pattern in how they miss is the point. The known-length
targets in that session all sit within 0.1 m of its calibration distance and read
$-2.4\,\%$ at $0°$ ($n=15$) — ordinary. Reprojection residual cannot see it, for the
reason given above. And leave-one-out cross-validation over the calibration observations
reports 0.56 % median error on the held-out frame, *better* than the eleven sound sessions
(0.52–1.48 %), because a frame held out of a single-distance burst is predicted at the
distance the remaining frames already anchor. Every check evaluated where the data already
lies is blind to conditioning; only the geometry of the observations, or an evaluation at a
different distance, reveals it.

We therefore admit a session as accuracy evidence only if its calibration observations
span at least 0.6 m of range. The criterion needs no threshold search and no reference
length: sound sessions span 1.02–2.32 m, the two rejected span 0.03 and 0.07 m, and the
bound is a statement about the data's geometry rather than about any measured error, so it
applies before a validation target exists.

Where two bursts of one mount state exist at different distances, the remedy is to fit
them jointly. Combining the session above with a burst 5.5 h later at 4.2 m gives a 2.25 m
lever and a 10.87 cm baseline that reproduces both distances to $-0.22\,\%$ and
$-1.10\,\%$, with held-out medians of $-0.23\,\%$ and $-1.19\,\%$; neither burst alone
predicts the other's distance (to $-17.25\,\%$ and $-127.6\,\%$). That the two agree is
also the one direct piece of evidence here that a mount can hold for hours — the failure
documented above is not that it always moves, but that nothing in the data tells you
whether it did without a check at a second distance.

#### 4.3 Accuracy

Figure 1 plots measured against known length for every cohort frame; Figure 2 gives the
per-target error distribution; Table 1 summarises. We report the $p_{90}$ of a target's
frames rather than a mean, and the reason is physical: the pipeline back-projects snout and
tail at the single laser-derived depth, so it measures the target's *projection*, and a
target that is not broadside to the camera can only read short, never long. The error
distribution is therefore one-sided, a mean measures the diver's pose distribution rather
than the instrument, and a high quantile is the estimator that rejects the pose tail.
$p_{90}$ is nearest-rank ($\lceil 0.9n \rceil$), the same statistic the deployed pipeline
reports.

Over the cohort the median frame error is −2.2 % and the $p_{90}$ is +0.12 %; 79 % of
frames are within 5 % of the reference, 97 % within 10 %, and 99 % within 15 %. The
per-target $p_{90}$ — Box −0.1 %, Grouper +0.3 %, Purple Angel +0.9 %, Snook −0.9 %,
Weasly Fish −1.3 %, Shark +4.3 % — shows no trend with size across a four-fold range of
lengths. The polish attributes +3.2 points of the Shark's figure to the target itself
rather than to any session; the model is no longer available to re-measure, and it is 25 of
793 frames, so excluding it moves the cohort $p_{90}$ from +0.12 % to −0.14 %.

How sensitive those figures are to the rule's one free parameter is worth stating, because
the cohort is selected rather than chosen. Moving the bound from 2.25 to 2.75 points
selects the same thirteen sessions and the same 793 frames; below 2.0 it drops to ten and
at 3.0 it admits an eleventh. The chosen 2.5 therefore sits in the middle of a plateau, not
on a knife edge — though four sessions do lie within half a point of it, three inside and
one out, so a cohort member can be marginal even when the membership is not.

The reference matters as much as the bound, and by a route worth recording. On the earlier
export, against the inherited 310 mm, this rule selected thirteen sessions reading median
−2.19 %; correcting the reference alone moved it to eleven sessions and −2.63 %, dropping
two sessions that do not measure the trout at all. The grid is unbalanced — eight of the
thirty-two sessions measure only the trout — so correcting one target's reference shifts
every session effect by about half a point, which is enough to move a marginal member
across the bound. The current figures use the measured reference *and* a re-calibration of
three sessions the pipeline's own gates rejected in the interim, and those two corrections
happen to push the decomposition in opposite directions, landing within 0.03 points of the
original median. That is a coincidence of two independent fixes and not evidence that the
original was right: the cohort membership differs.

The trout's own median of −4.4 % is the most negative of any target, and we can say what it is
*not* rather than what it is. Two of the targets are solid — the trout, 58.7 mm across the
mid-body, and the snook — while the rest are flat plates, so a natural explanation is
parallax: the near flank sits closer to the camera than the plane the length is measured
in, which would read short by a term in $1/z$. Fitting $a + b/z$ per target over the cohort
gives the trout $b = -1.9$ %·m (95 % CI $-2.3$ to $-1.4$), real but about five times
smaller than the $-9.4$ its thickness alone predicts, while a *flat* target, the grouper,
shows a larger $b = -3.7$ ($-5.7$ to $-1.1$). Each target is measured on its own sessions,
so $b$ cannot be separated from whatever range dependence those sessions' calibrations
retain. We therefore report the per-target offsets as target effects — which is what the
polish estimates them as, separately from session effects — and do not attribute them to
thickness on this evidence.

Figure 3 plots error against laser range. Triangulation conditioning degrades as $Z^2$
(Eq. 5), so a range dependence surviving into the delivered length would appear as a
widening band; instead the binned median is flat at −2.1 to −1.8 % from 0.8 m to 4.7 m
with an interquartile range that does not grow. The one systematic departure is a
$1/Z$ term on the solid models and not on the Box: fitting error $= a + b/Z$ per target
gives $b = -1.8$ cm for the trout (95 % CI $-2.4$ to $-1.3$), $-2.4$ for the snook, $-5.0$ for the
grouper, and $-0.4$ (indistinguishable from zero) for the Box. The mechanism is geometric:
the laser dot lands on the model's flank while the snout and fork lie in its midplane, half
a body thickness further from the camera, so the back-projection at the dot's depth reads
the length short by that offset divided by range — ~4 % at 0.9 m, ~1 % at 3 m for a 7 cm
thick model. The Box's tape patch is on the face the dot hits and shows none. The same term
applies to a real fish and is small at survey range: a 40 cm fish 4 cm thick at 2 m reads
1 % short.

**Table 1.** Percent length error over the accuracy cohort and, for comparison, over wider
selections. The two hold-outs do the rejecting and the 2.5-point band does the tightening:
admitting the two sessions the range trend alone excludes costs the median (−2.16 % to −2.37 %) while
leaving the median and mean marginally better, whereas dropping the hold-outs as well runs
the mean error from 3.5 % to 5.2 % and then 9.2 %. $p_{90}$ stays within ±0.7 % of zero
throughout.

| selection | sessions | frames | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| accuracy cohort (rule) | 13 | 793 | −2.16 % | +0.12 % | 8.25 % |
| every session not held out (design + range check) | 18 | 910 | −1.94 % | +0.68 % | 3.49 % |
| every session except the angle experiment | 27 | 1,499 | −3.48 % | +0.16 % | 5.15 % |
| every session | 32 | 2,927 | −5.91 % | −0.59 % | 9.20 % |

```latex
\begin{table}[t]
  \caption{Percent length error over the accuracy cohort and over wider selections.
  The hold-outs do the rejecting and the $2.5$-point band the tightening: admitting the
  sessions the band alone excludes costs $p_{90}$ while leaving the median marginally
  better, whereas dropping the hold-outs runs the mean error from $3.5\,\%$ to $9.2\,\%$.
  $p_{90}$ stays within $\pm0.7\,\%$ of zero throughout.}
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

#### 4.4 Fish pose

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

#### 4.5 Field deployments

*(The existing narrative text — Florida Keys deployments, the red/green laser comparison,
Figures 5–6, the mount failures and the in-field recalibration procedure — goes here
unchanged. What follows replaces the stereo-comparison paragraph, which the data does not
support; see the note at the end of this subsection.)*

Volunteer divers measured wild fish on seven deployments at one Florida reef, yielding 162
measurements of 73 individuals on six camera units, across nine named species and one
nontarget category. Two of the seven calibrations initially failed the conditioning
criterion of §4.2 — one fitted from a single frame, the other from two dots 26 cm apart in
range — and both were repaired rather than excluded. The frames they needed existed but had
been removed by the per-dive outlier filter, which judged the calibration frames against a
line its measurement frames dominate; reinstating them gives lever arms of 2.41 m and 1.30 m
and baselines of 10.10 and 10.52 cm, inside the 9.99–10.45 cm the fleet occupies. All seven
deployments therefore carry calibrations that pass every check in §4.1–4.2. Laser range was
0.71–3.14 m (median 1.48 m), closer than the pool median.

**Repeatability transfers to the field with a measurable penalty.** Where a diver captured
the same individual in three or more frames we can measure the system's repeatability
directly, and this is the one field figure that needs nothing external: a calibration error
is common to all frames of one fish and cancels in a relative spread, as does any error in
the length convention, and no comparison population is involved. (It is therefore the one
result the two repaired calibrations could not have changed, and indeed did not: it stands at
3.3 % before and 2.9 % after.) Over 25 such individuals the within-fish coefficient of
variation is **2.9 % (median; bootstrap 95 % CI 1.6–4.1 %), with a $p_{90}$ of 9.1 %**. The
identical statistic on the pool cohort — repeat frames of one target in one session, 23
cells — is **1.4 % (95 % CI 1.2–2.2 %), $p_{90}$ 4.5 %**. So a
repeat measurement of a wild fish varies about twice as much as a repeat measurement of a
posed model, and its tail about twice as badly. The two bootstrap intervals overlap at the
margin (1.6–4.1 % against 1.2–2.2 %), so this is a difference in point estimate rather than
a cleanly separated one; with 25 field individuals it is the sample size and not the effect
that limits the claim. That penalty is what the field adds: the
animal moves between frames, the water is turbid, and the snout and fork are harder to place.
It is consistent with §4.4 — a few degrees of pose change between frames is worth a few
percent of length — and it is the number a survey designer should use when deciding how many
frames per fish to require.

**What this sample cannot do is measure field accuracy**, and we state the limit rather than
work around it. No known-length reference was in the water on these dives, so the scale-free
range check of §4.1 cannot be applied to them: it needs one rigid object measured repeatedly
across a wide range spread, and no wild fish in the set spans more than 1.5× in range. The
per-unit sample is 4 to 18 fish, and with the observed 19 % between-fish size spread the
standard error on a unit's median length is 7 % at ten fish and 14 % at three — larger than
any bias worth detecting. A variance-components fit across the seven units bounds
unit-to-unit variation in field lengths at **≤ 9 % (95 % upper limit, 34 hogfish)**, with a
point estimate of 5 % that is not distinguishable from sampling.

**A comparison against stereo video is available and comes out inconclusive.** Our
collaborators measured the same species at the same reef in the same seasons with a
calibrated stereo-video rig, 1,120 lengths over ten sites, to the same fork-length
convention. Per species our medians differ from theirs by −21 % to +9 %, with bootstrap
intervals spanning zero for three of the five species that have enough of our fish to
compare, and the offsets vary by as much between our own units as they do against the
stereo. We therefore report it as a consistency check — our field lengths sit within the
sampling limits of the independent archive — and not as a bias measurement. Two further
cautions belong with it: the two systems' spreads are the same (between-fish CV 18.7 %
against 19.1 %), so no claim of a narrower distribution is supported; and the stereo's
quoted precision is a propagated click-error (1.0 % of length) rather than a measured
repeatability, with essentially one measurement per individual in the archive, so it is not
comparable to the 2.9 % above.

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
  claims are supported: 99 % of cohort frames are within 15 % and the $p_{90}$ is +0.12 %.
  Consider stating the broadside figure directly: "median −2.2 %, $p_{90}$ +0.12 % over 793
  measurements of six targets at 0.27–5.0 m".
- **Figure ?? (similar triangles)** in §3.3 is an unresolved reference.
- **§3's promise about refraction is still unmet.** The hardware paragraph says of the
  M52 air lens "we quantify the distortion this corrects in Section 4", and no section here
  does. Figure 9 and `reconstruction_analysis/flat_port_cost.py` supply the number — an
  uncorrected flat port expands the scene by $n_w$ and shortens the range by $1/n_w$, which
  cancel on axis and reach +17.7 % at 20° off axis, crossing the 15 % budget at 18°. It
  wants a short subsection of its own, which would push §4.2–4.5 down one. Not written
  here, because the placement is a structural call: it is the only simulation result in an
  otherwise empirical section, and the WUWNet paper must not lose it as motivation.

#### Figure captions

- **Figure 1** — Measured against known length, accuracy cohort (13 sessions, 771 frames).
  Six targets separate along the abscissa by their own lengths. Error bars are the
  interquartile range of frames; the diamond is the $p_{90}$.
- **Figure 2** — Percent length error by target, accuracy cohort. Box: frame-level
  interquartile range and whiskers; diamond: $p_{90}$, the reported estimator. The gap
  between median and $p_{90}$ is the pose tail, not instrument spread.
- **Figure 3** — Percent length error against laser range, accuracy cohort. Binned median
  and interquartile band over the per-frame cloud. The band does not widen with range.
- **Figure 4** — Fitted in-plane laser angle for seven sessions of one unit, the parameter
  that sets metric scale and that no reprojection check can observe. The 0.27° spread is
  worth −4.5 % in length at 2 m, so the mount moves between uses by more than the accuracy
  the instrument otherwise achieves. Marker distinguishes cohort membership.
- **Figure 9** *(no section references it yet — see the note under "Edits elsewhere")* —
  Length error against position in the frame with no refraction correction,
  simulated for this camera and housing. Ignoring the water's index expands the scene
  transversely by $n_w$ and shortens the laser range by $1/n_w$; on the optical axis these
  cancel to +0.1 %, so the error appears only off axis and cannot be averaged away. Dotted:
  the 15 % budget, crossed at 18°.
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

**Recommendation: keep Figure 4 and cut Figure 5. Figure 4 is now the figure for §4.2;
Figure 5 stays in the repository only.**

*Revised 2026-09-13. This section previously recommended cutting both and keeping Figure 4's
number as a single sentence in §3.3. Two things changed. The 490 → 527 split established
that the mount can move* within *a session — two calibrations seven minutes apart differing
by 0.82° — which is a qualitatively stronger claim than between-session drift and cannot be
carried by one sentence. And the WUWNet analysis independently measured 0.378° across one
session boundary from two direct beam fits, so the drift is now corroborated by a second
observable rather than inferred from length errors alone. Calibration stability has become a
result, §4.2 states it, and Figure 4 is its figure.*

Figure 4 (per-session mount state $\varphi$) and Figure 5 (the implied-yaw floor before and
after repairing sessions 60 and 76) were built when the accuracy claim rested on seven
slate-borrowed sessions and the paper needed to explain why two of them were repaired. The
*repair* is no longer the situation:

1. **The accuracy claim no longer depends on the repair.** Sessions 60 and 76 are held out
   of the cohort by design, so nothing in Table 1 or Figures 1–3 uses a repaired value. A
   reader does not need the repair to trust the numbers.
2. **The repair is not the paper's contribution.** The paper is a hardware framework plus a
   length-recovery method. $\varphi$-repair is a calibration-analysis technique; explaining
   it properly takes the half page of physics in `fish_model_analysis/HANDOFF.md` §1 (why
   in-plane rotation is invisible to the laser dots) and then a method that uses a known
   length to fix a calibration — which a reviewer will read as circular unless the
   scale-free ruler-at-two-ranges argument is also made. That is a different paper.
3. **But the mount state is no longer a passing remark.** §4.2 now reports calibration
   stability as a result: the angle spans 0.27° across seven sessions, 0.82° between two
   calibrations seven minutes apart, and calibration state is the dominant reason a session
   fails to be accuracy evidence (12 of 25). That is three quantities and a mechanism, and
   it is what justifies the cohort rule, the range check, and re-fitting the laser per dive.
   A sentence cannot carry it. Figure 4 as it stands — seven sessions, angle on the
   ordinate, cohort by marker — is exactly the right figure and needs no change.
4. **Figure 5 in particular argues against the paper's own cohort rule.** It shows the
   repair driving 60's yaw floors to 0.0 — a success — while the paper then declines to use
   60. Both positions are defensible, but defending both costs more words than either is
   worth. Cut it.

§3.3 ("must be calibrated per dive site") should now forward-reference §4.2 rather than
carry its own version of the number, so the claim is made once:

> The laser's in-plane pointing angle sets metric scale and no reprojection check can
> observe it; §4.2 measures how far it moves between uses and what that costs.

Figures 6 (shark anomaly) and 7 (fork probe) remain diagnostic figures for the repository,
not for the paper; the shark is handled by the one sentence in §4.3.
