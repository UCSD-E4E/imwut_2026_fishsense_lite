# Handoff to P2 — pose, diver compliance, and what P1 could not measure

**Staged in P1 because P2's repo is not on this machine.** Move it into P2 (the
citizen-science / deployability paper, `../cscw-fishsense2027` per this repo's README)
and delete it here. Written 2026-09-17.

P1 keeps the instrument claim: what FishCamera measures against known lengths, and what
its calibration does. Everything below is the part P1 kept running into and could not
answer, because answering it is about **what divers do**, not about what the rig does —
which is P2's subject.

---

## 1. The one measurement P1 never had

Every pose statement about P1's accuracy cohort is inferred from `arccos(1 + e/100)` —
computed from the error it is meant to explain. §4.3 of P1 says so outright. That
circularity blocked, in one afternoon, all of:

* excluding visibly oblique frames from a figure,
* conditioning the accuracy figure on the 15° guidance,
* attributing the error tail to pose rather than to the instrument.

**Do not reach for `angles.csv`'s `angle_category` to fix this.** It is a deterministic
binning of the protractor reading — 1,428 frames, zero disagreement — so it measures the
card, not a reviewer. If P1 or P2 ever claims "trained photo reviewers can identify angle
bins", that claim currently has nothing behind it in either repo and needs a source.

## 2. The proposal: a segmentation mask gives pose, scale-free

Yaw foreshortens **length** by cos θ and leaves **dorsal-ventral height** unchanged, so for
a broadside aspect ratio `a0`:

    h_obs / l_obs = a0 / cos(theta)      ->      theta = arccos(a0 * l_obs / h_obs)

No range, no calibration, no known length, and **no dependence on the measured error**.
That last property is the whole point: it is the independent pose label.

`a0` is exact and free for P1's rigid models. For wild fish it varies with individual, sex,
condition and fin state — which is the hard part, and a P2 problem rather than a P1 one.

**Validation set already exists, in P1's corpus**: the angle experiment, one rigid Snook,
1,428 frames, designed angles 0–45° in 5° steps, five sessions, two ranges, protractor in
frame. Segment those, fit `h/l` against `cos θ`, and the method arrives with its own error
bars and no new fieldwork. Full physics, design and failure modes in P1's
`fish_model_analysis/FINDINGS.md` §7e.

**Failure modes, in the order they will bite.** Roll first — height is invariant under yaw
*only*, so a fish rotated about its long axis loses apparent height and the method infers a
negative angle. Wild fish roll; a rigid model on a diver's hand mostly does not, so **the
validation set will not exercise this and the method will look better there than in the
field.** Then fin state (dorsal erect vs folded moves `h` a lot on a live animal), body flex,
and mask boundary quality. A colour threshold is not sufficient: tried on four Snook frames
2026-09-17, one failed, one captured the diver and returned a 68° pose on a −5.5 % frame,
one behaved.

## 3. Why this is P2's contribution and not P1's

The valuable output is **not** a corrected or trimmed accuracy number. P1 tested that: a
uniform >15° exclusion drops **343 of 995 frames** to move the reported p90 by **0.38 pp**,
and the tail it removes is P1's own stated reason for reporting p90 rather than a mean.

The valuable output is a **pose-conditioned figure reported beside the unconditioned one** —
what the instrument does at the poses divers actually achieved, and what it does when the
15° guidance is followed. That gap is a measurement of **compliance**, which is a
deployability claim: it says what a citizen-science programme gets from its volunteers, and
what training or UI would be worth.

P1 already supplies the other half of that comparison. §4.4 has the controlled curve — a
Snook stepped 0–45° against a protractor, five sessions, two ranges, 1,428 frames — so P2
does not need to re-derive the physics, only to measure what divers did.

## 4. Concrete things P2 can take from P1 today

| what | where in P1 |
|---|---|
| the controlled foreshortening curve, 1,428 frames with real angles | `fish_model_analysis/data/angles.csv`, §4.4, Figure 8 |
| head/tail label pixels + full intrinsics for the cohort, 995 frames | `fish_model_analysis/data/head_tail.csv`, `sql/extract_head_tail.sql` |
| the estimator's small-sample behaviour (p90 needs ~10–13 frames) | `repeatability.p90_rarefaction`, Figure 16, FINDINGS §7c |
| why a range correction was NOT applied, with the external test | `calibration.fit_depth_offset`, Figure D1, FINDINGS §7d |
| the distortion bound (residual cannot explain the near-field trend) | `fishsense_imwut/distortion.py`, FINDINGS §7d |

## 5. One P1 finding P2 should not re-learn the hard way

**Sample size, not tail size, is what breaks p90.** Nearest rank is ceil(0.9n), which equals
n for every n ≤ 10 — so below ten frames "p90" is the sample maximum and inherits whatever
the diver did. P1's worst-posed cell (17 frames, 15 pp of IQR, a −30.6 % frame) still
recovered a p90 within 1 pp of its own session; the ruler (6 frames, all 14.7–20° off) did
not, and is held out.

**The field corpus is squarely in the bad regime**: median 2 frames per animal, maximum 8.
So every field p90 is a longest frame. That is not a reason to distrust field medians, but
it is a reason never to quote a field p90 as if it carried a pool p90's precision — and it
is a direct, actionable deployability finding: *ask volunteers for ten or more frames per
animal.* P1's Figure 16 gives the curve behind that number.
