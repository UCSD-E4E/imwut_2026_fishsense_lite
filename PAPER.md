# Section 4 — Results (draft) and what to do with Figures 4/5

Draft text for the FishCamera / FishSense Lite paper, written against the figures and
numbers in `fish_model_analysis/` as of **2026-09-16**, re-pulled after the calibration
board's grid pitch was corrected 0.042 → 0.04217 m and all twelve checkerboard
calibrations refitted. That moved the accuracy cohort from seventeen sessions to
**nineteen** (503 and 504 entered), so §4.2, §4.3, Table 1 and §4.1's field
repeatability all carry different numbers than the 2026-09-12 draft did. The earlier
export is kept as `data/corpus_20260912.csv`; two tests use it to hold the
reference-independence claim, which the live export can no longer exercise on its own.

Every number below is produced by `fish_model_analysis/fish_model_measurements.ipynb` from
`data/corpus.csv` and `data/angles.csv`, and the cohort is pinned by
`tests/test_calibration.py`. Figure files are in `fish_model_analysis/figures/`.

Part A is the section draft. Part B is the recommendation on the August repair figures.

---

## Part A — Section 4 draft

### 4 RESULTS

FishCamera was used by volunteer divers on seven open-water deployments and produced 162
measurements of 73 wild fish. §4.1 reports what that yielded and how repeatable it was,
because it is the result the system exists for. What those numbers are worth rests on the
pool cohort: §4.2 gives the absolute accuracy against rigid targets of known length, §4.3
the calibration stability that dominates it, §4.4 the fish pose that dominates what is
left, and §4.5 the one simulated result — what the port's corrective optic prevents.
Appendix A carries the rule that decides which pool sessions are accuracy evidence; it is
a rule rather than a finding, and nothing in it reads a reference length.

#### 4.1 Field deployments

*(The existing narrative text — Florida Keys deployments, the red/green laser comparison,
Figures 5–6, the mount failures and the in-field recalibration procedure — goes here
unchanged. What follows replaces the stereo-comparison paragraph, which the data does not
support; see the note at the end of this subsection.)*

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
in one session, 30 cells — is **1.4 % (95 % CI 1.2–2.1 %), $p_{90}$ 3.2 %**. So a repeat
measurement of a wild fish varies about 2.1× as much as a repeat measurement of a posed model,
and its tail about 3.6× as badly. The two bootstrap intervals overlap at the margin (1.6–4.1 %
against 1.2–2.0 %), so this is a difference in point estimate rather than a cleanly separated
one; with 25 field individuals it is the sample size and not the effect that limits the claim.
That penalty is what the field adds: the animal moves between frames, the water is turbid, and
the snout and fork are harder to place. It is consistent with §4.4 — a few degrees of pose
change between frames is worth a few percent of length — and it is the number a survey designer
should use when deciding how many frames per fish to require.

**Where the measurements came from, and why all seven deployments count.** Volunteer
divers measured wild fish on seven deployments at one Florida reef, yielding 162
measurements of 73 individuals on six camera units, across nine named species and one
nontarget category. Two of the seven calibrations initially failed the conditioning
criterion of §4.3 — one fitted from a single frame, the other from two dots 26 cm apart in
range — and both were repaired rather than excluded. The frames they needed existed but had
been removed by the per-dive outlier filter, which judged the calibration frames against a
line its measurement frames dominate; reinstating them gives lever arms of 2.41 m and 1.30 m
and baselines of 10.10 and 10.52 cm. Thirty of the fleet's thirty-one fitted baselines lie
in 9.87–10.54 cm and both repairs land inside that band; the single exception is a pool
session whose observations span 2.8 cm along the ray, which the conditioning criterion of
§4.3 rejects without reference to its baseline at all. All seven deployments therefore carry
calibrations that pass every check in §4.3 and Appendix A. Laser range spans
0.46–3.90 m with a median
of 1.49 m — closer than the pool median — and 90 % of the frames fall between 0.72 and
3.16 m.

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
range check of Appendix A cannot be applied to them: it needs one rigid object
measured repeatedly
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

**Two independent comparisons against a stereo-video archive are available, and
they agree with each other: neither detects a systematic offset, and neither can
rule one out.** Our collaborators measured the same species at the same reef in
the same seasons with a calibrated stereo rig — 1,120 lengths over ten sites, to
the same fork-length convention — and on one day they measured the *same
individual animals* we did. The two comparisons differ in design by a lot, so we
give both and let the weaker one be superseded rather than dropped.

**The population comparison cannot carry a conclusion, and the reason is
instructive.** Each animal reduces to one length at $p_{90}$ as above, and the
species is then summarised by the median across animals — a population statistic,
where $p_{90}$ would name the ninetieth-percentile *fish* rather than a
measurement. On that footing our species medians differ from theirs by −21 % to
+12 %, with bootstrap intervals spanning zero for three of the five species that
have enough of our fish to compare. But split by camera unit, the same species at the same
reef scatters by as much between our own units as it does against the stereo, and
that scatter is not distinguishable from sampling 2–10 fish at a per-fish spread
of 19 %. Two further cautions belong with it. The two systems' spreads are the
same — between-fish CV 19.1 % against 19.1 % on the best-sampled species — so no
claim of a narrower distribution is supported. And the stereo's quoted precision is a
propagated click-error (1.0 % of length) rather than a measured repeatability, with
essentially one measurement per individual in the archive, so it is not
comparable to the 2.9 % above.

**The paired comparison is the better design, and it is what the archive
unexpectedly permits.** On 2023-08-03 at two sites, each of our dive folders
holds one fish and the archive numbers the same fish the same way — both sides'
hogfish numbering at one site runs 1–12 and both skip 10, so it is a shared
registry rather than two independent counts. Seven individuals were measured by
both instruments. This is a separate day from the seven deployments above and is
not part of the 162 measurements; its lengths rest on a calibration fitted from
that morning's slate burst (10.55 cm, inside the 9.87–10.54 cm band 30 of the
fleet's 31 fits occupy) and borrowed by each fish dive, which is the first time a
borrowed calibration has produced field lengths checked against anything
external.

**The comparison is reported at $p_{90}$, the estimator used everywhere else in
this paper**, and the reason it is not a median is visible in this data. Because
head and tail are back-projected at a single laser-derived depth, an out-of-plane
fish can only read *short* — on these frames the measured length is 0.9972–0.9997
of the flat in-plane span its clicked pixels subtend at that range, never above
1.0. A median over frames therefore inherits however much the pose varied, and a
high-order statistic rejects it; that is the same argument §4.2 makes against a
mean. Over the seven pairs $p_{90}$ gives a difference of **+3.5 %** (mean
+0.8 %, sd 10.1 %, Figure 15), and **four of the seven read longer than the
stereo, by 3.5 to 11.2 %**, which our own foreshortening cannot account for in
either direction. Read as agreement rather than as a spread, the seven $p_{90}$
estimates sit on the 1:1 line with $R^2 = 0.807$ (residual RMS 3.0 cm). That is
$R^2$ *about identity* — no fitted slope or offset to absorb a bias — and it is
the number to quote rather than a correlation: over fish spanning 20 to 43 cm,
Pearson $r^2$ is 0.838 and would stay near 1.0 under a uniform scale error of any
size, so it would certify only that the seven animals are different lengths.

The estimator does not manufacture that result: the same four
individuals exceed the stereo under any summary that is not the median, and the
median itself would report −3.7 % — a small negative bias that is an artefact of
summarising a one-sided error at its centre.

One property of nearest rank has to be stated here rather than left for a reader
to find. $p_{90}$ is the $\lceil 0.9n \rceil$-th of $n$ frames, which is $n$
itself for every $n \le 10$, and these fish carry 3 to 11 frames each. For five
of the seven, $p_{90}$ is therefore that animal's longest frame, and only the two
best-sampled separate the two at all. It is a high-order statistic on this day,
not a tail estimate. The same is true throughout §4.1: no wild animal here has
more than eight frames, so every field $p_{90}$ is a longest frame. This is
intended rather than tolerated — with a one-sided error the longest frame is the
one least corrupted by pose — but it is why the field figures should not be read
as carrying the same estimator precision the pool figures do, where a cell holds
26 frames at the median.

**What the paired day excludes, and what it cannot.** The disagreement is not a
calibration error, and the reason is the fitted baseline rather than the range
trend. All seven fish resolve through a single slate calibration shot the same
morning at 08:58, between 26 minutes and 1 h 42 before the fish themselves — the
point-of-use practice §4.3 recommends, not a borrow. Its baseline is 10.55 cm,
within the 9.87–10.55 cm every other sound fit in the fleet returns, and
reproducing a $+8.6\,\%$ offset by scale alone would need 11.5 cm. The scale-free
range check of Appendix A applied to the best-sampled fish (11 frames over a 1.55× range
spread) is flat at $+0.8\,\%$ per metre ($r=0.03$), which excludes an in-plane
angle error but not a scale error — a wrong baseline is flat with range, so it is
the baseline's value, not the trend, that rules that out. It is not the
projection, which is self-consistent to a fraction of a percent above. And it is
not a landmark convention: the two individuals of one species on this day disagree
in *opposite* directions, and their tail landmarks sit at the fork on the one
reading long and at the fin tip on the one reading short — the reverse of what a
systematic over-reach would produce. What remains is a residual we cannot
attribute: either we read long on four of seven fish, or the stereo reads short,
or the pairing is wrong on those individuals, and with seven pairs and no third
instrument those are not separable. The mean difference spans zero — −6.7 % to
+8.3 % at 95 % — as it does in the population comparison.

**The two comparisons disagree about the sign of our bias, and that is worth
stating numerically because it bounds what any correction could do.** Over the
four species with enough of our animals, our medians sit **−15.4 %** against the
archive; over the seven paired individuals we read **+3.5 %** long. A candidate
range correction fitted on the pool cohort — the pool's own near-field trend,
$-0.56 - 2.40/z$ per cent, which flattens that trend from a 1.40 pp spread of
binned medians to 0.50 — adds between 1.2 % and 6.2 % at field ranges, median
2.2 %. Applied to our side it moves the population comparison from a median
absolute offset of 15.4 % to 14.4 %, and the paired comparison from 8.6 % to
10.9 %: it helps the one that says we read short and hurts the one that says we
read long, because it only ever adds length. Two things follow. The correction is
**a tenth of the disagreement it would have to explain**, so neither result
licenses it and it is not applied. And a population comparison that a
same-individual comparison contradicts by nineteen points is not a weak
measurement of a bias, it is a measurement of something else — which is the
caution the preceding paragraphs give, here with a number on it.

**So the two comparisons reach the same place from opposite ends, and neither
changes the conclusion of the preceding paragraphs.** The population comparison
is limited by between-unit scatter on 2–10 fish per unit; the paired comparison
is limited by seven individuals at $\pm 8$ points. Field accuracy remains
unmeasured. What the paired day supports is narrower and worth stating exactly:
on one day, seven wild fish measured by two independent instruments agreed to
within about 10 % per individual with no detectable systematic offset. It is also
a template — the same-individual design, and a slate burst shot at two clearly
different standoffs, is what would make a future deployment answer the question
this corpus cannot.

#### 4.2 Accuracy against known lengths

Five rigid targets were measured: four painted fish models — a purple angelfish (192 mm), a
stylised rainbow trout (313 mm), a grouper (360 mm) and a snook (455 mm) — and a 150 mm box.
Each target's reference length is the snout-to-tail-fork distance a labeler is asked to
click, measured with a tape.

Targets were photographed in two pools over 31 sessions spanning seventeen days in August
2023, at laser ranges of 0.25–5.47 m (median 2.0 m), by several divers on several
camera+laser units, at ranges and poses of the diver's choosing rather than on a fixture.
Every frame passed through the same pipeline as a field image: a labeler marks the laser
dot and the snout and tail-fork, the laser dot fixes the range (Eq. 5), and the length
follows from Eqs. 6–8. In total 2,799 measurements were made.

Calibration was per unit and per session throughout, but the calibration *object* changed
partway through, and the direction of that change is the point. The first twenty sessions
(14–18 August) used a planar checkerboard photographed in the same session — the standard
tool, and one that has to be kept flat, dry and undamaged. The remaining twelve (29–31
August) used the duct-tape dive slate of §3.3, photographed in a companion session. We
moved *away* from the checkerboard deliberately: a slate is something a volunteer diver
already carries and can photograph in situ, and a system meant to be operated by citizen
scientists cannot depend on a printed board surviving a dive bag. §4.3 reports what that
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

Figure 1 plots measured against known length for every cohort frame; Figure 2 gives the
per-target error distribution; Table 1 summarises. We report the $p_{90}$ of a target's
frames rather than a mean, and the reason is physical: the pipeline back-projects snout and
tail at the single laser-derived depth, so it measures the target's *projection*, and a
target that is not broadside to the camera can only read short, never long. The error
distribution is therefore one-sided, a mean measures the diver's pose distribution rather
than the instrument, and a high quantile is robust to that tail in a way a mean is not.
$p_{90}$ is nearest-rank ($\lceil 0.9n \rceil$), the same statistic the deployed pipeline
reports.

**This is the paper's only estimator**, used wherever a set of frames becomes one length:
the pool targets here, and in §4.1 each wild animal, in Figures 13, 14 and 15 alike. One
consequence of nearest rank is worth stating once, because the two settings differ. A pool
cell holds 26 frames at the median, where $\lceil 0.9n \rceil$ is a genuine high quantile;
a wild animal here holds at most eight, and $\lceil 0.9n \rceil$ is $n$ for every
$n \le 10$, so **every field $p_{90}$ is that animal's longest frame**. With a one-sided
error that is the intended reading — the longest frame is the one least corrupted by
pose — but the field estimates should not be credited with the precision the pool
estimates carry. Where a *population* is summarised rather than a measurement, §4.1 takes
the median across animals: a $p_{90}$ there would name the ninetieth-percentile fish,
which is a different quantity.

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

Over the cohort the median frame error is −2.0 % and the $p_{90}$ is +0.36 %; 80 % of
frames are within 5 % of the reference, 98 % within 10 % and 99 % within 15 %. The
per-target $p_{90}$ — Box +0.7 %, Purple Angelfish +0.9 %, Rainbow Trout −1.0 %,
Grouper +0.9 %, Snook −1.2 % — shows no trend with size across a three-fold range of
lengths. The median is the more informative number about the divers: −2.0 % is what a
typical pose of 11.5° costs, so the corpus corroborates the 15° presentation guidance of
§4.4 from the other direction — the divers were inside it.

A sixth target, a printed measuring board, is **held out**, and why is worth one sentence
because it is the estimator's one failure mode. Its six frames were all shot with the
board 14.7–20.0° off square — a flat rigid plate foreshortens by $\cos\theta$ exactly as a
fish does — and that pose is unremarkable, the trout's median being worse. What
disqualifies it is that nearest rank is $\lceil 0.9n \rceil$, which is $n$ itself for
$n \le 10$, so over six frames $p_{90}$ is the single best frame, and the best frame is
still 14.7° off. Its residual −3.3 % is $\cos 14.7° - 1$ and nothing else. Every reported
target carries 66 to 407 frames and its $p_{90}$ lands on a pose between 0.0 and 8.7°. The
board is the only target in the corpus with fewer than ten frames, so this bites once.

The nine frames beyond 15 % are the reason the reported estimator is a high quantile and
not a mean. They are concentrated in two of the cohort's targets rather than spread across
it, and their magnitudes correspond to poses of 32° to 46° — well beyond the 15° §4.4 asks
divers for, and past the 25° at which that experiment sees its first breach. We cannot
verify that independently — the corpus frames carry no measured angle, so the implied pose
is inferred from the error it is meant to explain — which is precisely why the accuracy
claim is stated at $p_{90}$ and the pose claim is made in §4.4, where the angle was read
off a card in frame.

The rule has no free parameter fitted to an outcome, which is the point of stating it that
way. Its one threshold sits in the scale-free range check, and Appendix A reports what moves
with it: three sessions enter as the bound widens from 2 to 4 % per metre, while the
reported $p_{90}$ holds within 0.2 points. A stronger property follows from having no
error-magnitude criterion at all: the cohort does not depend on the reference lengths.
Re-running the selection against the earlier, inherited trout reference — 3 mm different,
and enough under the old rule to move two sessions in or out — returns the identical
nineteen sessions. Membership is decided entirely by the angle experiment, one named
hold-out and a check that spends no known length, so no session is in this cohort because
it agreed with a reference.

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
widening band; instead the interquartile range is 3.0 to 4.2 points in every bin beyond
0.81 m and does not grow with distance, and the binned median is flat between −2.4 % and
−1.4 % from 0.81 m out to 5.1 m. The one departure is the nearest bin, where the median is
−5.3 %. Only two targets reach those ranges and they disagree — over the same frames the
trout reads −7.0 % and the box −1.8 % — so the departure follows the target, not the range,
and falls to the paragraph above. It is also below the working range of a diver measuring
wild fish.

**Table 1.** Percent length error over the accuracy cohort and, for comparison, over wider
selections. The scale-free range check does all of the rejecting: dropping it, together
with the angle experiment, runs the mean error from 3.1 % to 4.4 % and then 9.0 %.
$p_{90}$ stays within ±0.6 % of zero throughout.

| selection | sessions | frames | median | $p_{90}$ | mean \|err\| |
|---|---|---|---|---|---|
| accuracy cohort (rule) | 19 | 995 | −2.00 % | +0.36 % | 3.08 % |
| every session except the angle experiment | 26 | 1,371 | −3.30 % | +0.08 % | 4.42 % |
| every session | 31 | 2,799 | −5.84 % | −0.56 % | 9.03 % |

```latex
\begin{table}[t]
  \caption{Percent length error over the accuracy cohort and over wider selections.
  The scale-free range check does all of the rejecting: dropping it, with the angle
  experiment, runs the mean error from $3.1\,\%$ to $9.0\,\%$. $p_{90}$ stays within
  $\pm0.6\,\%$ of zero throughout.}
  \label{tab:accuracy}
  \begin{tabular}{lrrrrr}
    \toprule
    selection & sessions & frames & median & $p_{90}$ & mean $|$err$|$ \\
    \midrule
    accuracy cohort (rule)               & 19 & 995     & $-2.00$ & $+0.36$ & 3.08 \\
    all but the angle experiment         & 26 & 1{,}371 & $-3.30$ & $+0.08$ & 4.42 \\
    every session                        & 31 & 2{,}799 & $-5.84$ & $-0.56$ & 9.03 \\
    \bottomrule
  \end{tabular}
\end{table}
```

#### 4.3 Calibration stability

Two questions sit under every number in §4.2, and they have opposite answers. Is the
object the calibration is fitted *from* sound? That is checkable, and it checks out. Is
the parameter it fits stable between uses? It is not, and by more than the accuracy we
report.

Replacing the checkerboard with a dive slate is only defensible if the substitute carries
the same scale, and that is checkable without reference to any fish. The pipeline's metric
scale does not come from the targets: it comes from the calibration object, and a wrong one
would rescale every length invisibly, since reprojection residual cannot see scale (§3.3).
The laser baseline, though, is a property of the rig rather than the dive, so any unit
calibrated both ways must report the same baseline either way. Over the six units carrying
both in the pool corpus (19 calibration *fits* — 11 checkerboard, 8 slate — which is a
different 19 from the session count, and coincidental) the mean
checkerboard-minus-slate difference is **+0.66 % of baseline (95 % CI −0.23 % to +1.69 %)**,
an interval that spans zero and is of the same order as the scatter between two
calibrations of one unit under one standard (sd 0.135 cm). One unit dominates the spread:
its slate side includes the fleet's shortest fit at 9.87 cm, which alone carries that unit
to +2.9 %.

The sign of that figure changed with the grid-pitch correction and is worth recording,
because it is the substitution's cost being measured rather than asserted. Before the
correction the checkerboard read **−0.27 %** against the slate; refitting the twelve
checkerboard calibrations at the tape-measured pitch moved them up by +0.39 % to +0.79 %
each, and the comparison crossed zero. The two objects agreed within the scatter before and
agree within it now, on either side of it — which is the claim, and it did not depend on
which side the central value happened to fall.

The session calibration offsets, which are reproducible from the committed export, agree
to about half a point: a median **+1.35 pp** over the fourteen checkerboard
sessions of the cohort against **+0.77 pp** over the five slate sessions. The deployable
object therefore reproduces the standard one to about a percent, which is what licenses
the substitution, and it places the dominant uncertainty in §4.2's accuracy figures on the
target lengths rather than on the calibration.

Two limits belong with that. Only five of the nineteen cohort sessions are
slate-calibrated, so the deployable path carries the smaller share of the accuracy
evidence and its offset spread (sd 2.10 pp against the checkerboard's 1.81) rests on five
points. And neither check can see an error common to both objects, or one in the camera
intrinsics, which rescale the calibration object and the laser together. We do not compare
the two groups' raw errors, because they photographed disjoint target sets (§4.2) and any
difference would be the targets rather than the calibration.

A sound calibration object is not a sound calibration, and Appendix A excludes all six
of its
calibration-rejected sessions on the state of a single parameter. It is worth saying plainly why
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
remains good practice, because the failure is silent — this is what the range check of
Appendix A
supplies, and it needs no reference length. And calibration state is the dominant reason a
session is not accuracy evidence: of the 30 sessions here with enough frames to judge, 6
fail a calibration criterion in Appendix A — five flagged by the scale-free range check, and
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
removed entirely — and §4.2's cohort, with the pose real divers achieved, as the
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

#### 4.5 What the port correction buys

§3 says of the M52 air lens at the housing port that we quantify the distortion it
corrects here. This is the one subsection built on simulation rather than on photographs,
and its scope is deliberately narrow: it establishes what a measurement would read if an
in-air calibration were carried underwater with no refraction correction at all. That is a
statement about the failure the optic prevents, not about the accuracy any correction
achieves. The corrections themselves — the Pinax model [??] and the in-water
single-viewpoint calibration it is compared against — are the subject of the companion
flat-port paper [??], and neither they nor the code implementing them appears here. The
forward model used below is shared verbatim with that work, and the tests carried
alongside it pin it against the Pinax paper's own Table 1.

Ignoring the water's index does two things at once, and they very nearly cancel.
Transversely the scene is expanded by $n_w$: every off-axis point images further from the
principal point than a pinhole would place it, so a target spans more pixels than it
should. Along the axis the laser triangulates short by close to $1/n_w$ — $-25.6\,\%$ in
the simulated geometry, against the $-25.5\,\%$ the ratio alone predicts — and the length
is that pixel span scaled by the range. The two factors are reciprocal, so on the optical
axis the measurement comes out right: $+0.1\,\%$ for a 300 mm target at 2 m. A centred
target measures correctly by accident.

Off axis the cancellation fails, because the angular compression is not a pure scale.
Figure 9 plots the length error against where in the frame the target falls, for this
camera's intrinsics and a 300 mm target at 2 m. A quarter of the way to the frame edge the
error is $+1.8\,\%$; half way, $+7.3\,\%$; it crosses the 15 % budget at 18° off axis —
seven tenths of the way out — and reaches $+17.7\,\%$ at 19.6°, beyond which a target of
that length no longer fits in frame.

The shape is what makes it dangerous, more than the size. It is exactly zero where a
careful person would check it, so centring the target on a known length is the one test
that cannot detect it. It is not a scale error, so no calibration of the kind §4.3 reports
can absorb it — a single multiplier cannot be right at the centre and at the edge at once.
And it is a function of where the fish happened to fall in the frame, which is not
recorded, is not under the diver's control, and has no reason to be balanced within a
session, so it would not average away: it would enter §4.2's per-session spread as an
uncontrolled term larger at the frame edge than the $-13.4\,\%$ a 30° pose costs, and
without any of the visual cues a posed fish gives the labeler.

Two inputs to this are not measured. The housing was not opened or gauged, so the pane's
thickness and index and the camera-to-glass spacing are assumed rather than known. **The
result does not rest on them.** Across panes of 2–20 mm, glass indices of 1.46 to 1.62,
and camera-to-glass spacings from the optimal 0.7 mm out to an implausible 80 mm, the
error at the frame edge moves only from $+17.7\,\%$ to $+16.7\,\%$ and the budget crossing
from 18.3° to 19.1°. That insensitivity is structural rather than lucky: a pane shifts a
ray sideways but cannot change its final direction in water, so the figure is set by the
air-to-water index ratio and the field of view, both of which are known. Fresh water in
place of salt gives $+17.5\,\%$ and a crossing at 18.6°, so the choice of water does not
carry it either. The headline figure is for a single pane, where a waterproof camera
inside a second housing really forms a two-pane stack. Simulating that stack directly —
two 6 mm panes separated by 5 to 30 mm of air — gives $+17.4$ to $+17.7\,\%$ at the edge
and a crossing at 18.3° to 18.6°. The general case is treated in [??].

**The optic removes the refraction, not the lens.** The in-air calibration still carries a
radial distortion model, and what it leaves behind is worth one paragraph because it points
at the same guidance §4.4 gives for a different reason. Distortion acts on where the clicked
endpoints fall, and two things move them outward at once: a near target subtends more
pixels, and so reaches further from the principal point, where the polynomial is steeper.
Measuring both together on the cohort's own labels — how much undistortion actually changed
each measured span — the effect runs from **0.02 % of length** at the far end to **0.76 %**
inside 0.8 m, of which the radial reach contributes about half again on top of the size
(a centred span of the same length would give 0.35 %). At the single most extreme frame in
the corpus, a target spanning 2,939 px with an endpoint 1,709 px out, it reaches **4.1 %**.

Those are ceilings on the whole model, not residuals: the labels are clicked on undistorted
images, so a residual fraction $\epsilon$ of the model costs $\epsilon$ times those numbers.
That is why distortion cannot be the near-field trend of Figure 3 — explaining a 3.7-point
droop from a 0.74-point ceiling needs the model wrong by five times itself, and the sign is
backwards, since undistortion *lengthens* a span and the near field reads short. But it is
also why the guidance to centre the fish is an optical instruction as well as a pose one:
the same frame carries forty times more distortion sensitivity at the edge, close in, than
it does centred and far.

Everything in §4.1–§4.4 is downstream of this optic, and none of those sections can
demonstrate it: an error the port correction has already removed leaves no trace in the
photographs. That is the reason the figure is simulated, and the reason it is one figure
rather than a section.

#### Appendix A — how the accuracy cohort is selected

*(Moved out of the results proper. This is the rule rather than a finding, and a reader
after the headline numbers should not have to pass through it to reach them. §4.2's
cohort is whatever this rule returns, and nothing in it reads a reference length.)*

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
with the references, selects on its own outcome. Two things disqualify a session, and
neither looks at the size of its error. The five angle-experiment sessions of §4.4 are set
aside because they are a different experiment — one target driven through 0–45° on
purpose, reported there rather than here. And a scale-free check that uses no reference
length: a rigid target must read the same length at every range, so a session whose
targets show a length trend with range — the Theil–Sen slope over frames beyond 0.8 m,
with its 95 % interval clear of ±2 % per metre — has a calibration error in the in-plane
laser angle that no reprojection test can see (§3.3), and is excluded. That check removes
five sessions. Three of the five carry a single target and the reference lengths
independently grade every one of them as wrong; the remaining two are rejected on one
target while a second target in the same session shows no trend, and one of those two is
already held out as an August repair, so little in the reported figures turns on them. One
further session is held out on the same scale-free evidence, where the automatic threshold
narrowly misses it: two of its targets show consistent trends of −2.2 and −2.3 % per
metre, implying −0.13°, but each interval reaches −1.2 and so fails the requirement that
the whole interval clear ±2 %. It is named rather than loosening the threshold until it is
caught, which would be the same error in another costume. Of the thirty sessions with a
polish cell, that leaves 19 admitted and 995 measurements; fourteen are
checkerboard-calibrated and five slate-calibrated. Nothing is excluded for reading far
from a reference.

The check has also now caught something the known lengths could not, which is the
strongest argument for keeping it. Two sessions used to be rejected for a trend of about
+5 % per metre while their median error looked unremarkable, the signature of a short
fitted baseline paired with a compensating angle that cancel at mid-range. Re-measuring
the calibration board's grid pitch and refitting moved their shared calibration from
8.90 to 10.39 cm, their trends fell to +0.6 and +0.8 % per metre, and they entered the
cohort. A check that spends no known length agreed with a tape measurement — and it did so
on sessions whose medians had given no warning.

Of the three parameters this leaves, two are not delicate: neither the 0.8 m floor
(0.6–1.0 m) nor the eight-frame minimum (6–12) changes which sessions are rejected. The
trend bound does. Widening it from 2 to 4 % per metre admits three more sessions, 19 to
22, and readers should take the stability claim to be about the reported statistic rather
than about cohort membership: $p_{90}$ holds between +0.36 % and +0.15 % across that
range, while the median drifts from −2.06 % to −2.89 %.

##### Sessions the rule rejects

*(For the paper's limitations paragraph, or an appendix; Figure A gives every session.)*
Five sessions are rejected, and **every one of them by the scale-free range check** —
none is excluded for reading far from a reference. All five carry a plausible baseline
(10.29 to 10.55 cm, inside the fleet's range), because the implausible fits that once
dominated this list have since been refitted or retired. What is left is the parameter no
reprojection test can see, with implied in-plane errors of 0.17° to 0.37° on the flagging
cells. **Four of the five borrow** their calibration from another session — every one but
509, which fitted its own. Borrowing is therefore the single strongest predictor of
rejection in the corpus, and §4.3's recommendation follows from it directly. Dive 60 is
held out on the same evidence, its two independently-measured targets agreeing on −2.2 and
−2.3 % per metre, where the flag rule asks the whole interval to clear ±2 and these reach
−1.2.

Two sessions left this list on 2026-09-16 rather than being argued out of it, and the
mechanism is worth recording because it is the check working as intended. Both borrowed a
calibration whose fitted baseline was 8.90 cm — plausible enough to pass a baseline bound,
and their medians were unremarkable — while their lengths grew about +5 % per metre of
range. Re-measuring the calibration board's grid pitch moved that shared fit to 10.39 cm
and their trends to +0.6 and +0.8 % per metre. A short baseline paired with a compensating
angle is exactly the failure the range check exists to catch, and it caught it in
sessions no reference-based test had flagged.

The instructive case is one the rule no longer rejects, and it is worth following because
the same movement still rejects two others. One session read 14 % short with correct labels
and a calibration that reproduced its own checkerboard perfectly: its folder held a board
burst shot five minutes *after* its target frames, and the laser had rotated 0.82° in the
plane reprojection cannot see. Re-measured under the burst that preceded the frames by nine
seconds it reads −1.7 %, and it is now a cohort member. But two sessions shot minutes
earlier still borrow the *later* burst — the calibration from the far side of the movement
— and read −5.5 % and −9.6 %, with range trends of −3.8 and −2.8 % per metre implying
0.22° and 0.17°. The repair fixed the session that owned the frames; it could not fix the
neighbours that borrowed across the same event.

What is no longer on this list is as informative as what is. Earlier drafts led with an
extreme case — a session whose borrowed calibration implied an 8° in-plane error and whose
lengths varied by 134 % per metre. No such session survives: refitting retired it, and the
largest trend anywhere in the corpus is now −6.1 % per metre on dive 494, implying 0.37°.
The list has become uniform, which is what a list of one failure mode should look like.

So all six sessions excluded for calibration — the five above and dive 60 — are excluded
for the same reason, and none is excluded for the size of its error. Every one carries a
range trend its own rigid targets reveal; not one is rejected for a flat offset, because
nothing in the rule can reject a session for that. And none is a ranging failure the diver
could see, which is the point: an in-plane calibration error is invisible to reprojection
residual, which is why the calibration procedure of §3.3 photographs the target at several
ranges, why a calibration is discarded whenever the unit has been handled, and why a rigid
reference photographed at two ranges after calibrating is the one check that would have
caught any of this in the field.

#### Edits elsewhere in the draft

- **Limitations paragraph (currently "beyond (??) degrees")**: "with our model fish, no
  measurement taken within 20° of broadside leaves the 15 % budget, and the guidance we give
  divers is 15° (Figure 8); the pooled median crosses the budget at 30°."
- **Abstract / Conclusion**: the "<15 %" and "within our target margin of error of 20 %"
  claims hold in the form §4.4 establishes them: no measurement within 20° of broadside
  leaves the 15 % budget, and the guidance given to divers is 15°. Over the cohort the $p_{90}$ is +0.36 % and 99 % of frames are within
  15 %. Consider stating the broadside figure directly: "median −2.1 %, $p_{90}$ +0.36 %
  over 995 measurements of five targets at 0.27–5.0 m".
- **Figure ?? (similar triangles)** in §3.3 is an unresolved reference.
- **Future work worth a line in the limitations paragraph.** Every pose statement about the
  cohort is currently inferred from the error it explains, because those frames carry no
  measured angle. A segmentation mask would break that: yaw foreshortens length by
  $\cos\theta$ and leaves dorsal-ventral height unchanged, so the observed aspect ratio
  gives $\theta$ scale-free — no range, no calibration, no known length, and no dependence
  on the measurement. The angle experiment is a ready-made validation set for it (one rigid
  target, 1,428 frames, designed angles with a card in frame). It would license a
  pose-conditioned accuracy figure reported *beside* the unconditioned one, which is a claim
  about diver compliance rather than a trimmed tail. Not attempted here; FINDINGS §7e has
  the physics, the design and the failure modes (roll and fin state first).
- **§3's promise about refraction is now met by §4.5**, written above as "What the port
  correction buys". It is placed after §4.4 rather than earlier so the empirical run
  §4.2–§4.4 is not interrupted by the one simulated result. It is scoped to what an
  *uncorrected* port costs and stops there,
  so the WUWNet submission keeps the correction itself — Pinax, and the in-water
  single-viewpoint calibration — as its own contribution and loses no motivation to this
  paper. Every number comes from `fishsense_imwut/refraction.py` (`flat_port_cost`) and is
  pinned by `tests/test_refraction.py`, including the robustness sweeps, since §4.5 is the
  one subsection with no CSV behind it. **Two citations are unresolved in it**: the Pinax
  paper (Łuczyński, Pfingsthorn & Birk, *Ocean Engineering* 133, 2017, 9–22) and the
  WUWNet submission, both written as `[??]`.

#### Figure captions

- **Figure 1** — Measured against known length, accuracy cohort (19 sessions, 995 frames).
  Five targets separate along the abscissa by their own lengths; the diamond is the
  $p_{90}$. $R^2 = 0.999$ about the 1:1 line on the five $p_{90}$ estimates, 0.976 on the
  995 individual frames. That statistic and not Pearson $r^2$: it has no free slope or
  offset, so a scale error lowers it, where $r^2$ over targets spanning a factor of four
  in length would mostly certify that the targets are different sizes.
- **Figure 2** — Percent length error by target, accuracy cohort. Box: frame-level
  interquartile range and whiskers; dots: every frame past a whisker; diamond: $p_{90}$,
  the reported estimator. The gap between median and $p_{90}$ is the pose tail, not
  instrument spread. **Nothing is clipped and nothing is omitted** — all 995 frames are on
  the axis, out to the worst at −30.6 %. That tail, one-sided and worst on the two largest
  models, is the whole reason the reported estimator is a high quantile rather than a mean,
  so a narrower axis would have hidden the figure's own argument.
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
- **Figure 9** *(§4.5)* —
  Length error against **where in the frame the target sits** with no refraction
  correction, simulated for this camera at 2 m against a 300 mm target. **The abscissa is
  position, not pose** — Figure 8's is also in degrees and means the opposite thing, so the
  upper axis repeats it in picture terms, centre of frame through to edge. The two are
  independent: a fish held perfectly broadside in the corner has 0° of pose and 20° of
  frame position, and would read +17.7 % long. Ignoring the water's index expands the scene
  transversely by $n_w$ and shortens the laser range by $1/n_w$; on the optical axis these
  cancel to +0.1 %, so the error appears only off axis and cannot be averaged away. Dotted:
  the 15 % budget, crossed at 18° — seven tenths of the way out. The plotted range stops at
  three quarters of the half-frame (19.6°) because past it a 300 mm target no longer fits;
  the frame itself reaches 25.4°, so the curve does not show the worst the port can do.
- **Figure 9b** *(§4.5, and the better of the two — needs `figure*`, full width)* — The
  flat-port cost drawn on the frame it happens in, beside what the corrective optic leaves,
  on one colour scale. Each panel **is** the 4014 × 3016 image, so "where in the picture"
  needs no translation into degrees and cannot be misread as the fish's pose. White contour:
  the 15 % budget. Grey: a 300 mm target centred there would not fit in frame.
  **The left field is not radially symmetric, and that is the argument.** The port is
  rotationally symmetric but the target is not a point — held horizontal it lies *along* a
  radius at the left and right edges and *across* one at the top and bottom, and radial and
  tangential magnification differ. The same fish at the same distance from the centre reads
  **+23 % at the side and +6 % at the top**, +0.1 % at the centre, and **+29.5 %** in the
  corner. No single scale factor is right at all three, which is why a calibration cannot
  absorb this.
  **What the right panel is, and is not.** It is the same model with the index step removed
  — the air path the M52 lens restores at the port — and *not* a refraction correction; the
  Pinax model and the in-water single-viewpoint calibration are the companion paper's
  contribution and appear nowhere here. So it is close to tautological: take the water
  interface away and there is no refraction error to have. It earns its half of the figure
  by putting the magnitude of what the optic removes on a scale the eye can compare, which
  the text cannot. A column-width single-panel variant exists
  (`pubfig.fig_flat_port_error_field`) if the full width is not affordable.
- **Figure 11** — Within-individual repeatability, wild fish against posed models. Each
  point is one group: one wild individual (≥ 3 frames), or one (session, target) cell in
  the pool cohort. Bar: the median, with its bootstrap interval. This is the only §4.1
  figure that measures the system rather than the sample — a calibration error is common to
  every frame of one animal and cancels in a relative spread — and the intervals overlap at
  the margin, which the figure shows rather than hides.
- **Figure 12** — Measured fork length by species over the seven deployments, one point per
  measurement, bar at the median. **Descriptive only.** The species is a labeler's
  judgement that nothing in the field data can check, so a genuine per-species offset and a
  systematic misidentification would look identical here. Species with a single individual
  are pooled.
- **Figure 13** — Hogfish fork length by camera unit, **one point per animal** at
  $p_{90}$, with the between-fish interquartile range shaded behind. Drawn to show that a
  unit effect is not resolvable on this sample, not that there is none: 2 to 10 fish per
  unit against an 18 % between-fish spread, $F(5,27) = 1.10$, $p = 0.38$. Plotting all 74
  measurements instead of the 33 animals returns $F(5,68) = 4.05$, $p = 0.004$ — an
  apparently significant unit effect that is entirely pseudo-replication.
- **Figure 14** — Per-species median length, FishSense Lite against the independent
  stereo-video archive, with the 1:1 line. Bars are bootstrap intervals on each median; the
  vertical ones are computed per *animal*, not per frame, each animal reduced at $p_{90}$
  as everywhere else; the median across animals is a population statistic, where a
  $p_{90}$ would name the ninetieth-percentile fish rather than a measurement. **Read this
  differently from
  Figure 1.** There, one object is compared with its own known length, so a departure from
  1:1 is error. Here the two axes are **different animals** — ours and theirs, same reef and
  season, and for the five species plotted never the same individual — so a departure is
  error *or* a difference in which fish each encountered, and nothing in this figure
  separates them. The seven paired individuals of §4.1 are the exception and are
  deliberately not shown here: pooling a same-individual comparison into a
  population-median plot would hide the very distinction this caption draws. The arms differ by 12×
  across these five species because the samples do, and that asymmetry is the result: Nassau
  Grouper's median sits ±23.5 cm on a 49 cm fish (7 of our animals against 8 of theirs),
  while Stoplight Parrotfish's stereo median is pinned to ±2.0 cm by 391. Five species with
  at least five of our fish: Hogfish (33 vs 119), Stoplight Parrotfish (16 vs 391),
  Nassau Grouper (7 vs 8), Black Grouper (6 vs 51), Rainbow Parrotfish (5 vs 59). Medians
  differ by −21 % to +12 %.
- **Figure 15** *(§4.1)* — The paired day: our per-frame lengths against the stereo
  length of the **same individual**, seven fish over 39 frames, with the $p_{90}$ estimator
  and a 1:1 datum. **Figure 1's form and Figure 1's estimator**, so the two read the same
  way — but the 1:1 line here is *agreement*, not truth, because both axes are instruments
  and neither is a known length. $R^2 = 0.807$ about that line on the seven $p_{90}$
  estimates (0.786 per frame), which is an agreement statistic and not a correlation:
  Pearson $r^2$ is 0.838 and would sit near 1.0 under a uniform scale error. Contrast
  Figure 14, where the two axes are different animals. Four of the seven read longer than
  the stereo at $p_{90}$, by 3.5 to 11.2 %, and pose loss cannot produce a positive; the
  median difference is +3.5 % and the mean +0.8 %. One caution about
  the estimator at these sample sizes: nearest rank is `ceil(0.9n)`, which is $n$ itself for
  $n \le 10$, so with 3 to 11 frames per fish $p_{90}$ selects the top sample for five of
  the seven. It is a high-order statistic here rather than a tail estimate, and only the
  two best-sampled fish separate it from a per-fish maximum at all. The two hogfish near
  32 cm are worth following: nearly the same stereo length, disagreeing in opposite
  directions, which is the landmark-convention point the text makes.
- **Figure 16** *(§4.2)* — How many frames $p_{90}$ needs. Rarefaction: draw $n$ frames
  without replacement from each of the fifteen cohort cells holding ≥ 30, take the
  $p_{90}$ of percent length error, and compare it with that cell's full-sample $p_{90}$;
  1,000 draws per cell. **Two statistics, at two levels.** Per fish the estimate is
  $p_{90}$, the paper's estimator throughout, because one fish's frames are a one-sided
  pose-corrupted distribution and a high quantile rejects that tail. Across draws and cells
  the summary is a **median** — the same choice §4.1 makes across animals — because sampling
  error is not one-sided and has no tail that needs rejecting. Band: central 80 % of draws,
  kept signed because the bias changes sign. **The step at $n = 10$ is arithmetic, not
  noise**: nearest rank is $\lceil 0.9n \rceil$, which equals $n$ for every $n \le 10$, so
  below ten frames $p_{90}$ is the sample maximum — the noisiest order statistic there is.
  Crossing that boundary narrows the 80 % interval from 2.08 pp at $n = 9$ to 1.35 at
  $n = 10$ on no extra information, purely because the estimator stops taking the extreme.
  **Two thresholds follow**: ten frames to make $p_{90}$ a quantile at all, and thirteen for
  it to land within 1 pp of its own limit 90 % of the time (20 frames for 92 %, 30 for
  97 %). The sawtooth at $n = 20$ and $30$ is the same arithmetic on a smaller scale. Note
  also that the median error *changes sign* over the small-$n$ range, −0.70 pp at $n = 2$
  through +0.21 at $n = 9$: the maximum of two draws sits below the true 90th percentile and
  the maximum of nine sits above it, so a small sample is not conservative in either
  direction.
  **What this figure does not cover**, and the ruler is the cautionary case: it measures
  sampling spread on cells whose diver *did* vary pose. More frames help only if the animal
  is presented broadside at some point — six frames all taken at 15° return a stable,
  confident, wrong answer, and no sample size fixes that.
- **Figure 8** — Percent length error against fish angle to the image plane, from five
  sessions of one target stepped through 0–45°. Thin lines: per-session binned medians;
  black: pooled median and interquartile range; dashed: $\cos\theta - 1$; dotted: the 15 %
  budget. No frame within 20° of broadside breaches it; the pooled median crosses at 30°.
  The guidance in the text is 15°.
- **Figure A (appendix)** — Percent length error for every session with ≥ 8 frames,
  ordered by median, held-out targets included. The held-out and rejected sessions are the
  wide, negative rows. This one **is** clipped, to −35–15 %, and unlike Figure 2 it has to
  be: a single mis-clicked box frame in one rejected session reads +77.8 %, and letting one
  frame in 2,926 set the scale would flatten every row. The forty frames outside are drawn
  as carets on the boundary and the note carries the extreme.

Drop-in: figures are sized for the `acmart` column already (`\includegraphics` with no
`width=`); see the last cell of the notebook.

---

## Part B — Figures 4 and 5 (the August repair)

**Recommendation: keep Figure 4 and cut Figure 5. Figure 4 is now the figure for §4.3;
Figure 5 stays in the repository only.**

*Revised 2026-09-13. This section previously recommended cutting both and keeping Figure 4's
number as a single sentence in §3.3. Two things changed. The 490 → 527 split established
that the mount can move* within *a session — two calibrations seven minutes apart differing
by 0.82° — which is a qualitatively stronger claim than between-session drift and cannot be
carried by one sentence. And the WUWNet analysis independently measured 0.378° across one
session boundary from two direct beam fits, so the drift is now corroborated by a second
observable rather than inferred from length errors alone. Calibration stability has become a
result, §4.3 states it, and Figure 4 is its figure.*

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
3. **But the mount state is no longer a passing remark.** §4.3 now reports calibration
   stability as a result: the angle spans 0.27° across seven sessions, 0.82° between two
   calibrations seven minutes apart, and calibration state is the dominant reason a session
   fails to be accuracy evidence (6 of 30). That is three quantities and a mechanism, and
   it is what justifies the cohort rule, the range check, and re-fitting the laser per dive.
   A sentence cannot carry it. Figure 4 as it stands — seven sessions, angle on the
   ordinate, cohort by marker — is exactly the right figure and needs no change.
4. **Figure 5 in particular argues against the paper's own cohort rule.** It shows the
   repair driving 60's yaw floors to 0.0 — a success — while the paper then declines to use
   60. Both positions are defensible, but defending both costs more words than either is
   worth. Cut it.

§3.3 ("must be calibrated per dive site") should now forward-reference §4.3 rather than
carry its own version of the number, so the claim is made once:

> The laser's in-plane pointing angle sets metric scale and no reprojection check can
> observe it; §4.3 measures how far it moves between uses and what that costs.

Figures 6 and 7 remain diagnostic figures for the repository, not for the paper.
