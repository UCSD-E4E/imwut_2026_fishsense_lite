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

Five rigid targets were measured: four painted fish models — a purple angelfish (192 mm), a
stylised rainbow trout (313 mm), a grouper (360 mm) and a snook (455 mm) — and a 150 mm box.
Each target's reference length is the snout-to-tail-fork distance a labeler is asked to
click, measured with a tape.

Targets were photographed in two pools over 31 sessions spanning seventeen days in August
2023, at laser ranges of 0.25–5.47 m (median 2.0 m), by several divers on several
camera+laser units, at ranges and poses of the diver's choosing rather than on a fixture.
Every frame passed through the same pipeline as a field image: a labeler marks the laser
dot and the snout and tail-fork, the laser dot fixes the range (Eq. 5), and the length
follows from Eqs. 6–8. In total 2,828 measurements were made.

Calibration was per unit and per session throughout, but the calibration *object* changed
partway through, and the direction of that change is the point. The first twenty sessions
(14–18 August) used a planar checkerboard photographed in the same session — the standard
tool, and one that has to be kept flat, dry and undamaged. The remaining twelve (29–31
August) used the duct-tape dive slate of §3.3, photographed in a companion session. We
moved *away* from the checkerboard deliberately: a slate is something a volunteer diver
already carries and can photograph in situ, and a system meant to be operated by citizen
scientists cannot depend on a printed board surviving a dive bag. §4.2 reports what that
substitution costs.

Spreading the corpus over many sessions, sites and weather is only useful if no single
unit dominates it, so we check that directly. Seven camera+laser units contributed, six of
them at both pools, and Figure 10 plots each session's calibration offset grouped
by unit — the session term of the polish below, which is comparable across units that
photographed different targets in a way a raw per-unit mean is not. The five
angle-experiment sessions of §4.4 are left out: a single target at deliberately oblique
poses gives the polish no way to separate that session from that target, so what it returns
for them is pose rather than a calibration offset. No unit stands apart among the 25 that
remain. A one-way fit puts the between-unit variance component at zero against a
within-unit spread of 2.6 points ($F(6,18) = 0.74$, $p = 0.63$): the units differ from one
another by less than sessions of the same unit differ among themselves. The sessions that
fail the checks below are likewise spread across five of the seven units rather than
concentrated on one, which is what a bad unit, as opposed to a bad session, would look
like.

Not every session is accuracy evidence, and we do not choose which are by hand. A per-frame
error has two additive components: a per-*session* term, which is the calibration offset
and moves every target in that session together; and a per-*target* term, which is a
reference or landmark offset and follows the target across sessions. We separate them with
a Tukey median polish on the $p_{90}$ percent error of each (session, target) cell with at
least five frames [Tukey 1977]. Fifty-two cells are observed over 30 sessions and 7 targets,
and the additive fit accounts for them closely: the median absolute residual is 0.15
percentage points, and 0.45 points over the ten sessions that carry more than one target and
so can exhibit a residual at all. One caveat follows from which targets went in the water
where: the checkerboard sessions photographed the box and the trout, and the slate sessions
the fish models, with no target in common, so the session and target terms are anchored
separately within each group and the polish's centring is what places the two groups on a
single scale.
A reference error common to one group's targets would therefore shift that group's session
offsets bodily against the other's; within a group it could not.

A session is accuracy evidence unless something about *how it was produced* disqualifies
it. There is deliberately no threshold on how large its error came out: a rule that drops
sessions for disagreeing with the references, and then reports the survivors' agreement
with the references, selects on its own outcome. Two hold-outs precede that test. By design, before any number
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
on them. The rule admits 17 sessions and 908 measurements; twelve are checkerboard-calibrated and
five slate-calibrated. Every session it rejects is rejected by the scale-free check alone.

The one threshold this leaves is not delicate. Any trend bound from 2 to 4 % per metre
selects the identical 17 sessions and the identical figures, and neither the 0.8 m floor
(0.6–1.0 m) nor the eight-frame minimum (6–12) changes which sessions it rejects.

#### 4.2 Calibration stability

Two questions sit under every number in §4.1, and they have opposite answers. Is the
object the calibration is fitted *from* sound? That is checkable, and it checks out. Is
the parameter it fits stable between uses? It is not, and by more than the accuracy we
report.

Replacing the checkerboard with a dive slate is only defensible if the substitute carries
the same scale, and that is checkable without reference to any fish. The pipeline's metric
scale does not come from the targets: it comes from the calibration object, and a wrong one
would rescale every length invisibly, since reprojection residual cannot see scale (§3.3).
The laser baseline, though, is a property of the rig rather than the dive, so any unit
calibrated both ways must report the same baseline either way. Over the five units carrying
both (19 calibrations, five distinct slate scans) the mean checkerboard-minus-slate
difference is **−0.27 % of baseline (95 % CI −1.1 % to +0.6 %)**, below the scatter between
two calibrations of one unit under one standard (sd 0.12 cm). The session calibration
offsets agree as closely: a median **+1.20 pp** over the twelve checkerboard sessions of the
cohort against **+1.09 pp** over the five slate sessions. The deployable object therefore
reproduces the standard one to about a percent, which is what licenses the substitution,
and it places the dominant uncertainty in §4.3's accuracy figures on the target lengths
rather than on the calibration.

Two limits belong with that. Only five of the seventeen cohort sessions are
slate-calibrated, so the deployable path carries the smaller share of the accuracy
evidence and its offset spread (sd 2.10 pp against the checkerboard's 1.94) rests on five
points. And neither check can see an error common to both objects, or one in the camera
intrinsics, which rescale the calibration object and the laser together. We do not compare
the two groups' raw errors, because they photographed disjoint target sets (§4.1) and any
difference would be the targets rather than the calibration.

A sound calibration object is not a sound calibration, and §4.1 rejects every one of its
seven excluded sessions on the state of a single parameter. It is worth saying plainly why
that parameter needs watching. The laser's in-plane pointing angle sets metric scale and
is invisible to any reprojection check (§3.3): rotating the axis within the camera–laser
plane moves the projected dot by less than $10^{-12}$ px, so a calibration can be wrong by
an amount no fit residual can reveal. Figure 4 plots that angle, fitted per session, for
seven sessions of one unit.

It is also not stable. Across those seven sessions the angle spans 0.27°, against a
sensitivity of $-2.0\,\%$ in length per 0.15° at 0.9 m and $-4.5\,\%$ at 2.0 m — so the
between-session spread alone exceeds the accuracy we report below. The whole corpus spans
seventeen days, so this is drift within a fortnight of ordinary handling rather than
ageing over years. Nor is a session a safe
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
session is not accuracy evidence: of the 30 sessions here with enough frames to judge, 8
fail a calibration criterion in §4.1 — seven flagged by the scale-free range check, and
one more held out on the same evidence. An instrument whose scale parameter drifts by more
than its own measurement error between uses is one that must be calibrated, and validated,
at the point of use.

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
its fitted baseline, 12.95 cm, is the widest in the fleet by a wide margin — the other
thirty fits lie within 9.87–10.54 cm.

Three checks miss it, and the pattern in how they miss is the point. The known-length
targets in that session all sit within 0.1 m of its calibration distance and read
$-2.4\,\%$ at $0°$ ($n=15$) — ordinary. Reprojection residual cannot see it, for the
reason given above. And leave-one-out cross-validation over the calibration observations
reports 0.56 % median error on the held-out frame, *better* than the eleven sound sessions
(0.52–1.48 %), because a frame held out of a single-distance burst is predicted at the
distance the remaining frames already anchor. Every check evaluated where the data already
lies is blind to conditioning; only the geometry of the observations, or an evaluation at a
different distance, reveals it.

The pipeline therefore refuses to store a calibration whose observations span less than
0.6 m of range, so such a session cannot reach the accuracy analysis at all. The bound
needs no threshold search and no reference length — it is a statement about the geometry of
the observations, not about any measured error, so it applies before a validation target
exists — and it separates the sessions we can check cleanly: of the eleven whose stored
observations we can recover, the sound ones span 1.02–2.32 m against 0.03 and 0.07 m for
the two refused. That caveat is real and worth stating: the slate path stores the
per-frame observations it fitted, while the checkerboard path does not, so for
checkerboard-calibrated sessions the lever arm is verifiable only going forward, as a gate
at the moment of fitting, and not retrospectively.

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
than the instrument, and a high quantile is robust to that tail in a way a mean is not.
$p_{90}$ is nearest-rank ($\lceil 0.9n \rceil$), the same statistic the deployed pipeline
reports.

It is an upper quantile, and we are careful not to read it as a recovery of the true
length. A quantile sits above the centre of whatever distribution it is given, pose or no
pose: on Gaussian noise at the within-cell spread we measure (1.3 %), the expected
$p_{90}$ is $+1.6\,\%$ with no foreshortening present at all. The angle experiment makes
the same point with real frames — at a *measured* $0°$ the median is $-3.8\,\%$ and the
$p_{90}$ is $+0.6\,\%$, a gap of 4.4 points containing no foreshortening whatever. So a
$p_{90}$ near zero does not establish that the instrument is unbiased at broadside; it
says the pose tail and the measurement spread are of comparable size. What the statistic
is good for is comparison — between targets, between sessions, and against the same
statistic computed by the deployed pipeline — not as an absolute accuracy.

Over the cohort the median frame error is −2.4 % and the $p_{90}$ is +0.09 %; 75 % of
frames are within 5 % of the reference, 97 % within 10 % and 99 % within 15 %. The
per-target $p_{90}$ — Box +0.4 %, Purple Angelfish +0.8 %, Rainbow Trout −1.7 %, Ruler
−3.8 %, Grouper +0.9 %, Snook −1.2 % — shows no trend with size across a three-fold range
of lengths. The median is the more informative number about the divers: −2.4 % is what a
typical pose of 12.5° costs, so the corpus corroborates the 15° presentation guidance of
§4.4 from the other direction — the divers were inside it. The ruler is one session and six
frames, so its entry carries little weight; it is the only target whose reference is
traceable to a printed scale rather than a tape.

The nine frames beyond 15 % are the reason the reported estimator is a high quantile and
not a mean. They are concentrated in two of the cohort's targets rather than spread across
it, and their magnitudes correspond to poses of 32° to 46° — well beyond the 15° §4.4 asks
divers for, and past the 25° at which that experiment sees its first breach. We cannot
verify that independently — the corpus frames carry no measured angle, so the implied pose
is inferred from the error it is meant to explain — which is precisely why the accuracy
claim is stated at $p_{90}$ and the pose claim is made in §4.4, where the angle was read
off a card in frame.

The rule has no free parameter to tune, which is the point of stating it that way. Its
one threshold sits in the scale-free range check, and §4.1 reports that the selection is
unchanged for any trend bound from 2 to 4 % per metre. A stronger property follows from
having no error-magnitude criterion at all: the cohort does not depend on the reference
lengths. Re-running the selection against the earlier, inherited trout reference — 3 mm
different, and enough under the old rule to move two sessions in or out — returns the
identical seventeen sessions. Membership is decided entirely by the angle experiment, one
named hold-out and a check that spends no known length, so no session is in this cohort
because it agreed with a reference.

The trout's own median of −4.5 % is the most negative of any target, and we can say what it is
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
widening band; instead the interquartile range is 2.5 to 3.6 points in every bin beyond
0.85 m and does not grow with distance, and the binned median is flat between −2.4 % and
−1.7 % from 0.85 m out to 5.0 m. The one departure is the nearest bin, where the median is
−5.2 %. Only two targets reach those ranges and they disagree — over the same frames the
trout reads −7.5 % and the box −2.2 % — so the departure follows the target, not the range,
and falls to the paragraph above. It is also below the working range of a diver measuring
wild fish.

**Table 1.** Percent length error over the accuracy cohort and, for comparison, over wider
selections. The scale-free range check does all of the rejecting: dropping it, together
with the angle experiment, runs the mean error from 3.4 % to 5.1 % and then 9.3 %.
$p_{90}$ stays within ±0.9 % of zero throughout.

| selection | sessions | frames | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| accuracy cohort (rule) | 17 | 908 | −2.38 % | +0.09 % | 3.43 % |
| every session except the angle experiment | 26 | 1,400 | −3.98 % | −0.32 % | 5.09 % |
| every session | 31 | 2,828 | −6.22 % | −0.88 % | 9.31 % |

```latex
\begin{table}[t]
  \caption{Percent length error over the accuracy cohort and over wider selections.
  The scale-free range check does all of the rejecting: dropping it, with the angle
  experiment, runs the mean error from $3.4\,\%$ to $9.3\,\%$. $p_{90}$ stays within
  $\pm0.9\,\%$ of zero throughout.}
  \label{tab:accuracy}
  \begin{tabular}{lrrrrr}
    \toprule
    selection & sessions & frames & median & $p_{90}$ & mean $|$err$|$ \\
    \midrule
    accuracy cohort (rule)               & 17 &   908 & $-2.38$ & $+0.09$ & 3.43 \\
    all but the angle experiment         & 26 & 1{,}400 & $-3.98$ & $-0.32$ & 5.09 \\
    every session                        & 31 & 2{,}828 & $-6.22$ & $-0.88$ & 9.31 \\
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
every angle, at both ranges, so the curve is a property of the geometry and not of a unit
or a diver. The pooled median follows $\cos\theta - 1$ offset by a near-constant −2.4
points, from 0° (−3.8 %) through 15° (−5.8 %), 30° (−15.0 %) and 45° (−31.4 %). That 0°
figure is the only measurement in this paper taken at a *known* pose, which makes it the
one place the instrument's reading is separated from the diver's aim by design rather than
by estimator. It should not be read as the accuracy a user can expect. It was obtained by
holding a rigid model against a protractor card, and no such condition exists on a wild
animal: a diver can choose *when* to release the shutter, which is what the guidance below
asks, but cannot ask the fish to hold a pose. It is also one target in five sessions,
carrying their calibrations with it. Treat it as a floor — what remains when pose is
removed entirely — and §4.3's cohort, with the pose real divers achieved, as the
expectation. And the guidance that follows is stated with margin rather than at the
boundary, and is about shutter timing rather than about arranging the animal: **release
the shutter when the fish is within 15° of broadside.** Over the 542 frames inside that
limit the median error is −4.7 %, the $p_{90}$ is −0.8 %, and not one frame falls short of
the 15 % budget — the worst reads −10.7 %. The margin is real rather than nominal. The
limit could be relaxed to 20° before any frame breaches (684 frames, worst −14.2 %); the
first breaches appear at 25° (32 of 833); and the pooled median does not cross 15 % until
30°. Beyond that the correction is a known function of a quantity single-image fish-pose
estimators are beginning to recover [29].

Two of those 542 frames do exceed 15 %, at +28.9 % and +22.2 %. Foreshortening cannot read
long, so a large positive error is a labelling or calibration fault rather than a pose one,
which is why the guidance is stated on the short side.

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
and baselines of 10.10 and 10.52 cm. Thirty of the fleet's thirty-one fitted baselines lie
in 9.87–10.54 cm and both repairs land inside that band; the single exception is a pool
session whose observations span 2.8 cm along the ray, which the conditioning criterion of
§4.2 rejects without reference to its baseline at all. All seven deployments therefore carry
calibrations that pass every check in §4.1–4.2. Laser range spans 0.46–3.90 m with a median
of 1.49 m — closer than the pool median — and 90 % of the frames fall between 0.72 and
3.16 m.

**Repeatability transfers to the field with a measurable penalty.** Where a diver captured the
same individual in three or more frames we can measure the system's repeatability directly, and
this is the one field figure that needs nothing external: a calibration error is common to all
frames of one fish and cancels in a relative spread, as does any error in the length
convention, and no comparison population is involved. (The cancellation is exact for a pure
scale error, which is why the two repaired calibrations barely moved it: 3.3 % before and 2.9 %
after. It is only *barely* rather than not at all because a refit turns the laser axis as well
as changing the baseline, so the correction factor varies a little with range instead of being
exactly common to a fish's frames.) Over 25 such individuals — 97 of the 162 measurements — the
within-fish coefficient of variation is **2.9 % (median; bootstrap 95 % CI 1.6–4.1 %), with a
$p_{90}$ of 11.4 %**. The identical statistic on the pool cohort — repeat frames of one target
in one session, 29 cells — is **1.3 % (95 % CI 1.2–2.0 %), $p_{90}$ 3.3 %**. So a repeat
measurement of a wild fish varies about 2.2× as much as a repeat measurement of a posed model,
and its tail nearly 4× as badly. The two bootstrap intervals overlap at the margin (1.6–4.1 %
against 1.2–1.9 %), so this is a difference in point estimate rather than a cleanly separated
one; with 25 field individuals it is the sample size and not the effect that limits the claim.
That penalty is what the field adds: the animal moves between frames, the water is turbid, and
the snout and fork are harder to place. It is consistent with §4.4 — a few degrees of pose
change between frames is worth a few percent of length — and it is the number a survey designer
should use when deciding how many frames per fish to require.

**The species attributions are unverified, and the per-species comparison below inherits
that.** A labeler names the animal from the frame, and nothing in the field data can check
the name: the one detector we have for a mislabel compares the head-to-tail pixel separation
a frame implies, $L f / z$, against what the named species' length would require — which
found six mislabelled frames in the pool corpus, and needs a known length to run at all. No
wild fish has one. So a species column here is a labeler's judgement, not a measurement, and
a systematic confusion between two similarly-shaped species would appear as a per-species
offset we would have no way to distinguish from a measurement bias. Only the repeatability
above is immune, because it is computed within one individual whatever that individual is.

**What this sample cannot do is measure field accuracy**, and we state the limit rather than
work around it. No known-length reference was in the water on these dives, so the scale-free
range check of §4.1 cannot be applied to them: it needs one rigid object measured repeatedly
across a wide range spread, and the best-spanned wild individual in the set covers only
1.9× — 1.6× among those with the three frames the repeatability needs, and 1.04× for the one
fish with eight. The per-unit sample is 4 to 18 fish, and with the 18 % between-fish size
spread observed among hogfish the standard error on a unit's median length is 7 % at ten
fish and 13 % at three — larger than any bias worth detecting. A one-way variance-components
fit on 33 hogfish finds **no detectable unit effect** ($F = 1.03$ across the six camera
units, $p = 0.42$; $F = 1.55$, $p = 0.20$ if the seven deployments are taken separately),
with a point estimate of 1.5 % between cameras against 18 % between fish. What it cannot do
is bound that tightly: a nonparametric bootstrap over individuals puts the 95 % upper limit
at 13 % between cameras and 16 % between deployments, and the $F$-based interval is
degenerate at these sample sizes. A unit-to-unit bias of order the fish spread itself is
therefore not excluded, which is a statement about the sample and not about the instrument.

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

**One day does better than that, and it is worth reporting for the design rather
than the number.** On 2023-08-03 at two of the sites, the archive and our own
capture record the *same named individuals* — each of our dive folders that day
holds one fish, and the archive numbers the same fish the same way (both sides'
hogfish numbering at one site runs 1–12 and both skip 10, so it is a shared
registry rather than two independent counts). Seven individuals were measured by
both instruments. This is a separate day from the seven deployments above and is
not part of the 162 measurements; its lengths rest on a calibration fitted from
that morning's slate burst (10.55 cm, inside the 9.87–10.54 cm band 30 of the
fleet's 31 fits occupy) and borrowed by each fish dive, which is the first time a
borrowed calibration has produced field lengths checked against anything
external.

**The estimator, not the statistic, carries this result.** Because head and tail
are back-projected at a single laser-derived depth, an out-of-plane fish can only
read *short* — on these frames the measured length is 0.9972–0.9997 of the flat
in-plane span its clicked pixels subtend at that range, never above 1.0. With a
one-sided error the per-fish median is biased downward by however much the pose
varied, and the per-fish *maximum* is a lower bound on the animal. Summarising
the seven pairs by median gives a difference of **−3.7 %**; by maximum,
**+8.6 %**, with the scatter unchanged at ~11 % in both cases. Four of the seven
exceed the stereo even at their maximum, by 8.6–12.3 %, which our own
foreshortening cannot account for. We report both, because reporting the median
alone would read as a small negative bias and would be the wrong summary of a
one-sided error — the same reason $p_{90}$ rather than a mean appears in §4.3.

**What it excludes, and what it cannot.** The disagreement is not a calibration
error: all seven fish resolve through one borrowed fit, so a scale error is
common-mode and cannot separate individuals, and the scale-free range check of
§4.1 applied to the best-sampled fish (11 frames over a 1.55× range spread) is
flat at $+0.8\,\%$ per metre ($r=0.03$). It is not the projection, which is
self-consistent to a fraction of a percent above. And it is not a landmark
convention: the two individuals of one species on this day disagree in *opposite*
directions, and their tail landmarks sit at the fork on the one reading long and
at the fin tip on the one reading short — the reverse of what a systematic
over-reach would produce. What remains is a residual we cannot attribute: either
we read long on four of seven fish, or the stereo reads short, or the pairing is
wrong on those individuals, and with seven pairs and no third instrument those
are not separable. The mean difference spans zero under both estimators.

**So this does not change the conclusion of the preceding paragraphs.** Seven
paired individuals at $\pm 8$ points is the same arithmetic that makes a
per-unit median uninformative, and field accuracy remains unmeasured. What the
day supports is narrower and worth stating exactly: on one day, seven wild fish
measured by two independent instruments agreed to within about 10 % per
individual with no detectable systematic offset. It is also a template — the
same-individual design, and a slate burst shot at two clearly different
standoffs, is what would make a future deployment answer the question this
corpus cannot.

#### Sessions the rule rejects

*(For the paper's limitations paragraph, or an appendix; Figure A gives every session.)*
Seven sessions are rejected, and **every one of them by the scale-free range check** —
none is excluded for reading far from a reference. All seven carry a plausible baseline
(10.24 to 10.51 cm, inside the fleet's range), because the implausible fits that once
dominated this list have since been refitted or retired. What is left is the parameter no
reprojection test can see, with implied in-plane errors of 0.10° to 0.37°. **Six of the
seven borrow** their calibration from another session — every one but 509, which fitted
its own. Borrowing is therefore the single strongest predictor of rejection in the corpus,
and §4.2's recommendation follows from it directly. Dive 60 is held out on the same
evidence, its two independently-measured targets agreeing on −2.2 and −2.3 % per metre,
where the flag rule asks the whole interval to clear ±2 and these reach −1.2.

The instructive case is one the rule no longer rejects, and it is worth following because
the same movement still rejects two others. One session read 14 % short with correct labels
and a calibration that reproduced its own checkerboard perfectly: its folder held a board
burst shot five minutes *after* its target frames, and the laser had rotated 0.82° in the
plane reprojection cannot see. Re-measured under the burst that preceded the frames by nine
seconds it reads −1.7 %, and it is now a cohort member. But two sessions shot minutes
earlier still borrow the *later* burst — the calibration from the far side of the movement
— and read −5.9 % and −9.9 %, with range trends of −3.8 and −2.8 % per metre implying
0.17° and 0.22°. The repair fixed the session that owned the frames; it could not fix the
neighbours that borrowed across the same event.

The extreme case shows what a cross-epoch borrow costs when nothing constrains it: one
session borrows a calibration whose implied in-plane error is 8°, and its lengths vary by
134 % per metre of range. The remaining three fail the session-effect bound with flat
trends — a constant offset rather than a range-dependent one. None of the nine is a ranging
failure the diver could see, which is the point: an in-plane calibration error is invisible
to reprojection residual, which is why the calibration procedure of §3.3 photographs the
target at several ranges, why a calibration is discarded whenever the unit has been
handled, and why a rigid reference photographed at two ranges after calibrating is the one
check that would have caught any of this in the field.

#### Edits elsewhere in the draft

- **Limitations paragraph (currently "beyond (??) degrees")**: "with our model fish, no
  measurement taken within 20° of broadside leaves the 15 % budget, and the guidance we give
  divers is 15° (Figure 8); the pooled median crosses the budget at 30°."
- **Abstract / Conclusion**: the "<15 %" and "within our target margin of error of 20 %"
  claims hold in the form §4.4 establishes them: no measurement within 20° of broadside
  leaves the 15 % budget, and the guidance given to divers is 15°. Over the cohort the $p_{90}$ is +0.09 % and 99 % of frames are within
  15 %. Consider stating the broadside figure directly: "median −2.4 %, $p_{90}$ +0.09 %
  over 908 measurements of six targets at 0.27–5.0 m".
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

- **Figure 1** — Measured against known length, accuracy cohort (17 sessions, 908 frames).
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
- **Figure 10** — Session calibration offset by camera unit, for the 25 sessions outside
  the angle experiment. Each marker is one session's offset (the session term of the median
  polish, which removes the per-target term and so is comparable across units that
  photographed different targets); the bar is the unit's median and the band is ±1
  within-unit standard deviation. Every unit's median falls inside that band, the
  between-unit variance component is zero, and the sessions the rule rejects are spread
  across five of the seven units. Unit 1 contributed a single non-angle session, so its
  marker and median coincide.
- **Figure 10b** *(repository only; no section references it)* — The frame-level companion
  to Figure 10: percent error per camera unit over the cohort, each target centred on its own
  median so units that photographed different targets are comparable. One hue throughout —
  the question is whether the units differ and the answer is no. Kept for the record rather
  than the paper, alongside Figures 5, 6 and 7.
- **Figure 9** *(no section references it yet — see the note under "Edits elsewhere")* —
  Length error against position in the frame with no refraction correction,
  simulated for this camera and housing. Ignoring the water's index expands the scene
  transversely by $n_w$ and shortens the laser range by $1/n_w$; on the optical axis these
  cancel to +0.1 %, so the error appears only off axis and cannot be averaged away. Dotted:
  the 15 % budget, crossed at 18°.
- **Figure 11** — Within-individual repeatability, wild fish against posed models. Each
  point is one group: one wild individual (≥ 3 frames), or one (session, target) cell in
  the pool cohort. Bar: the median, with its bootstrap interval. This is the only §4.5
  figure that measures the system rather than the sample — a calibration error is common to
  every frame of one animal and cancels in a relative spread — and the intervals overlap at
  the margin, which the figure shows rather than hides.
- **Figure 12** — Measured fork length by species over the seven deployments, one point per
  measurement, bar at the median. **Descriptive only.** The species is a labeler's
  judgement that nothing in the field data can check, so a genuine per-species offset and a
  systematic misidentification would look identical here. Species with a single individual
  are pooled.
- **Figure 13** — Hogfish fork length by camera unit, **one point per animal**, with the
  between-fish interquartile range shaded behind. Drawn to show that a unit effect is not
  resolvable on this sample, not that there is none: 2 to 10 fish per unit against an 18 %
  between-fish spread. Plotting all 74 measurements instead of the 33 animals returns
  $F(5,68) = 4.05$, $p = 0.004$ — an apparently significant unit effect that is entirely
  pseudo-replication.
- **Figure 14** — Per-species median length, FishSense Lite against the independent
  stereo-video archive, with the 1:1 line. Bars are bootstrap intervals on each median; the
  vertical ones are computed per *animal*, not per frame. **Read this differently from
  Figure 1.** There, one object is compared with its own known length, so a departure from
  1:1 is error. Here the two axes are different animals — ours and theirs, same reef and
  season but never the same individual — so a departure is error *or* a difference in which
  fish each encountered, and nothing in the data separates them. The arms differ by 12×
  across these five species because the samples do, and that asymmetry is the result: Nassau
  Grouper's median sits ±23.5 cm on a 49 cm fish (7 of our animals against 8 of theirs),
  while Stoplight Parrotfish's stereo median is pinned to ±2.0 cm by 391. Five species with
  at least five of our fish: Hogfish (33 vs 119), Stoplight Parrotfish (16 vs 391), Nassau Grouper
  (7 vs 8), Black Grouper (6 vs 51), Rainbow Parrotfish (5 vs 59). Medians differ by
  −21 % to +9 %.
- **Figure 8** — Percent length error against fish angle to the image plane, from five
  sessions of one target stepped through 0–45°. Thin lines: per-session binned medians;
  black: pooled median and interquartile range; dashed: $\cos\theta - 1$; dotted: the 15 %
  budget. No frame within 20° of broadside breaches it; the pooled median crosses at 30°.
  The guidance in the text is 15°.
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

Figures 6 and 7 remain diagnostic figures for the repository, not for the paper.
