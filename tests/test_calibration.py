"""Pins for the corpus-cohort machinery in `fishsense_imwut.calibration`.

The August analysis hand-picked its accuracy dives; the corpus analysis derives
them from a rule. These tests pin the rule and the decomposition behind it so
a later edit cannot silently change which dives back the paper's numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"


# --- median polish ----------------------------------------------------------


def test_median_polish_recovers_an_exactly_additive_grid():
    dives = [1, 2, 3, 4]
    models = ["A", "B", "C"]
    # Effects are identifiable only up to their median, so build them centred.
    dive_eff = pd.Series([-2.0, -1.0, 1.0, 3.0], index=dives)
    model_eff = pd.Series([-1.0, 0.0, 2.0], index=models)
    grid = pd.DataFrame(
        {m: 0.5 + dive_eff + model_eff[m] for m in models}, index=dives
    )

    fit = cal.median_polish(grid)

    assert fit.overall == pytest.approx(0.5)
    pd.testing.assert_series_equal(fit.dive_effect, dive_eff, check_names=False)
    pd.testing.assert_series_equal(fit.model_effect, model_eff, check_names=False)
    assert fit.residual.abs().max().max() == pytest.approx(0.0)


def test_median_polish_is_robust_to_a_single_outlying_cell():
    dives = [1, 2, 3, 4, 5]
    models = ["A", "B", "C"]
    grid = pd.DataFrame(0.0, index=dives, columns=models)
    grid.loc[3, "B"] = 40.0  # one wild cell must not move any effect

    fit = cal.median_polish(grid)

    assert fit.dive_effect.abs().max() == pytest.approx(0.0)
    assert fit.model_effect.abs().max() == pytest.approx(0.0)
    assert fit.residual.loc[3, "B"] == pytest.approx(40.0)


def test_median_polish_tolerates_missing_cells():
    grid = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3]
    )
    fit = cal.median_polish(grid)
    # Model B sits exactly 1 above model A wherever both are observed.
    assert fit.model_effect["B"] - fit.model_effect["A"] == pytest.approx(1.0)
    assert np.isnan(fit.residual.loc[3, "A"])


# --- cohort rule ------------------------------------------------------------


def _frame(cells):
    """(dive, model, [pct_errors]) -> tidy frame like `cal.to_frame`."""
    rows = []
    for dive, model, errors in cells:
        for e in errors:
            rows.append(
                {
                    "dive_id": dive,
                    "model_name": model,
                    "known_length_m": 0.3,
                    "length_m": 0.3 * (1 + e / 100),
                    "pct_error": e,
                    "depth_m": 1.0,  # one range: the range-trend filter abstains
                    "baseline_m": 0.103,
                }
            )
    return pd.DataFrame(rows)


def test_the_cohort_has_no_error_magnitude_criterion():
    """The rule must never drop a dive for reading far from the references.

    Selecting sessions by their agreement with the references and then
    reporting agreement with the references is selection on the outcome. A
    dive whose every measurement is 30 % short is still admitted here; only
    how a session was PRODUCED can exclude it."""
    good = [0.0] * 6
    df = _frame(
        [
            (1, "A", good), (1, "B", good),
            (2, "A", good), (2, "B", good),
            (3, "A", [-30.0] * 6), (3, "B", [-30.0] * 6),   # hugely wrong, kept
        ]
    )
    assert cal.accuracy_cohort(df, exclude=(), range_trend_filter=False) == (1, 2, 3)
    assert not hasattr(cal, "MAX_DIVE_EFFECT_PP")
    # ... and a held-out dive's cells still inform the model effects.
    assert cal.accuracy_cohort(df, exclude=(3,), range_trend_filter=False) == (1, 2)
    assert 3 in cal.median_polish(cal.cell_p90_grid(df)).dive_effect.index


def test_design_exclusions_are_fixed_before_any_corpus_number():
    assert set(cal.ANGLE_TEST_DIVES) <= set(cal.DESIGN_EXCLUDED_DIVES)
    assert 60 in cal.DESIGN_EXCLUDED_DIVES      # scale-free range trend, filter near-miss
    assert 76 not in cal.DESIGN_EXCLUDED_DIVES  # the filter catches it unaided
    assert 66 not in cal.DESIGN_EXCLUDED_DIVES  # nothing non-circular excludes it
    assert 490 not in cal.DESIGN_EXCLUDED_DIVES  # was never a design call
    assert cal.SPLIT_DIVES == {490: 527}


def test_accuracy_cohort_ignores_cells_below_min_frames():
    good = [0.0] * 6
    df = _frame(
        [
            (1, "A", good), (1, "B", good),
            (2, "A", good), (2, "B", good),
            (3, "A", good), (3, "B", [-30.0] * 2),  # 2 frames: not a cell
        ]
    )
    # min_frames decides which cells EXIST, which is what the polish sees.
    assert "B" not in cal.cell_p90_grid(df, min_frames=5).loc[3].dropna().index
    assert "B" in cal.cell_p90_grid(df, min_frames=2).loc[3].dropna().index
    assert cal.accuracy_cohort(df, min_frames=5, exclude=(), range_trend_filter=False) == (1, 2, 3)


def test_accuracy_cohort_on_the_real_corpus_is_the_published_set():
    """The paper's cohort, pinned. If this changes, Section 4 changes.

    Nineteen as of the 2026-09-16 export. 503 and 504 joined when the
    checkerboard pitch was corrected to 0.04217 m: that took their borrowed
    fit (dive 502) from 8.897 to 10.394 cm and their range trends from +5.31
    and +5.54 %/m to +0.60 and +0.80."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES
    assert cal.CORPUS_ACCURACY_DIVES == (
        58, 59, 61, 66, 84, 495, 497, 498, 500, 501, 503, 504, 506, 507,
        519, 520, 521, 522, 527,
    )


def test_prods_target_names_never_reach_a_figure():
    """`fishmodelreference.name` reached the y-axis of Figures 2 and 2b
    straight from the data, and two of the names should not have: the trout's
    is an internal joke, and "Purple Angel" disagreed with the prose.

    Both halves are pinned. The CSVs must still carry prod's spelling -- they
    are verbatim `sql/extract_corpus.sql` output and a re-export must stay
    byte-comparable -- and nothing that survives `load_rows` may. If prod is
    ever renamed properly, the raw assertion is the one that fails, and the
    map becomes dead weight rather than wrong."""
    for name in ("corpus.csv", "corpus_20260912.csv"):
        raw = (DATA / name).read_text()
        models = {r["model"] for r in cal.load_rows(DATA / name)}
        for prod_name, shown in cal.DISPLAY_NAMES.items():
            assert f"|{prod_name}|" in raw, f"{name} no longer matches the export"
            assert prod_name not in models
            assert shown in models


def test_the_reef_frame_is_not_part_of_the_pool_corpus():
    """Dive 436 is one frame of a fish model held at Alligator Reef on
    2024-10-16, fourteen months after every other session and not in a pool.

    Pinned on both sides: the export still carries it, because corpus.csv is
    verbatim sql/extract_corpus.sql output, and it must stay a single frame --
    if it ever gains frames it becomes a real field session and the decision to
    drop it has to be revisited rather than inherited."""
    rows = cal.load_rows(DATA / "corpus.csv")
    assert cal.NON_POOL_DIVES == (436,)
    reef = [r for r in rows if int(r["dive_id"]) in cal.NON_POOL_DIVES]
    assert len(reef) == 1, f"dive 436 now has {len(reef)} frames; revisit the exclusion"
    assert reef[0]["model"] == "Yellow Anthias"
    pool = [r for r in rows if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(pool)
    assert df.dive_id.nunique() == 31
    assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES


def test_holding_out_the_shark_removes_frames_not_sessions():
    """Neither held-out target is a validation target. Dropping them must
    remove frames and no sessions -- the shark because its reference is wrong,
    the ruler because six frames at 15 deg measure the grip, not the rig."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.HELD_OUT_MODELS == ("Shark", "Ruler")
    cohort = df[df.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)]
    assert len(cohort) == 1051
    assert len(cohort[cohort.model_name == "Shark"]) == 50
    assert len(cohort[cohort.model_name == "Ruler"]) == 6
    assert len(cohort[~cohort.model_name.isin(cal.HELD_OUT_MODELS)]) == 995
    for dive, g in cohort.groupby("dive_id"):          # no dive measures it alone
        assert set(g.model_name) - set(cal.HELD_OUT_MODELS), dive


def test_the_range_trend_filter_must_see_the_held_out_targets():
    """INVARIANT, and load-bearing: the range-trend check spends no reference
    length, so a target whose reference is wrong is still a valid witness. The
    Shark is the only cell that flags dive 76. Filter HELD_OUT_MODELS out
    before calling accuracy_cohort and dive 76 silently rejoins the cohort --
    which is exactly the tidy-up someone will eventually attempt."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES
    wrong = cal.accuracy_cohort(df[~df.model_name.isin(cal.HELD_OUT_MODELS)])
    assert 76 in wrong and 76 not in cal.CORPUS_ACCURACY_DIVES
    assert set(wrong) - set(cal.CORPUS_ACCURACY_DIVES) == {76}


def test_the_cohort_does_not_depend_on_the_reference_lengths():
    """This used to be the analysis's headline sensitivity: correcting the
    trout from 0.310 to 0.313 m moved two dives across the 2.5 pp band.

    Removing the band removed the mechanism. Membership is now decided by the
    angle experiment, one named scale-free hold-out and the range-trend filter,
    none of which reads a reference length -- so the cohort is invariant to
    them, which is the strongest available answer to the charge that sessions
    were selected for agreeing with the references.

    The claim is invariance WITHIN a file, and that distinction became
    load-bearing on 2026-09-16. This test used to assert the frozen file's
    cohort also equalled `CORPUS_ACCURACY_DIVES`, which held only while the two
    exports happened to select the same sessions. The checkerboard pitch
    correction moved the live cohort 17 -> 19 and broke that incidental
    equality -- a real change in the calibrations, not a reference sensitivity.
    So each file is now checked against itself.

    The frozen file is what keeps this testable at all: it still carries the
    inherited 0.310 m trout, while every live export since has carried the
    0.313 m tape value, making `corrected_references` a no-op on `corpus.csv`.
    """
    frozen = DATA / "corpus_20260912.csv"
    raw = cal.to_frame(cal.load_rows(frozen), corrected_references=False)
    corrected = cal.to_frame(cal.load_rows(frozen))
    assert cal.accuracy_cohort(raw) == cal.accuracy_cohort(corrected)

    rows = cal.load_rows(DATA / "corpus.csv")
    live_raw = cal.to_frame(rows, corrected_references=False)
    live_corrected = cal.to_frame(rows)
    assert cal.accuracy_cohort(live_raw) == cal.accuracy_cohort(live_corrected)
    assert cal.accuracy_cohort(live_corrected) == cal.CORPUS_ACCURACY_DIVES


def test_the_measured_reference_is_the_tape_value():
    """Two independent tape readings, snout to fork: 12 5/16 in = 312.74 mm and
    312-313 mm metric, so 312.7 +- 0.5 mm. Adopted 313 mm -- three significant
    figures like every other reference, which the +-0.5 mm supports and a
    tenth-millimetre digit does not. The cohort is the same for any value in
    that spread, so the rounding costs nothing."""
    assert cal.MEASURED_REFERENCES_M["Rainbow Trout"] == 0.313
    rows = cal.load_rows(DATA / "corpus.csv")
    for metres in (0.3125, 0.3127, 0.3127375, 0.313):
        df = cal.to_frame(rows, corrected_references=False)
        mask = df.model_name == "Rainbow Trout"
        df.loc[mask, "known_length_m"] = metres
        df["pct_error"] = 100 * (df.length_m / df.known_length_m - 1)
        assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES


def test_the_ruler_reference_is_the_printed_scale_value():
    """The Wildco board's clicked span, read off its OWN printed inch ticks.

    Nine near-range frames of dive 60 give 340.4, 340.8, 341.1, 340.5, 340.2,
    340.4, 341.2, 340.7 and 340.7 mm -- median 340.7, sd 0.5. Adopted 341 mm on
    the same three-significant-figure rule the trout uses.

    Admissible under HANDOFF section 0 in a way the shark's re-determination was
    not: the span is a ratio of pixels to pixels inside one frame, so range,
    focal length and the laser calibration all cancel and no part of this
    instrument enters. The shark's ran through the rig as a comparator, which is
    why it stays held out instead.
    """
    assert cal.MEASURED_REFERENCES_M["Ruler"] == 0.341

    rows = cal.load_rows(DATA / "corpus.csv")
    raw = cal.to_frame(rows, corrected_references=False)
    fixed = cal.to_frame(rows)
    assert (raw.loc[raw.model_name == "Ruler", "known_length_m"] == 0.3429).all()
    assert (fixed.loc[fixed.model_name == "Ruler", "known_length_m"] == 0.341).all()

    # The ruler is now held out of the reported set as well, so the correction
    # survives only as the record of what the board actually measures.
    assert cal.accuracy_cohort(fixed) == cal.CORPUS_ACCURACY_DIVES
    r = fixed[fixed.model_name == "Ruler"]
    r = r[r.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)]
    assert len(r) == 6
    assert np.median(r.pct_error) == pytest.approx(-3.66, abs=0.05)


def test_the_ruler_is_held_out_because_p90_cannot_see_past_its_pose():
    """Six frames, all 14.7-20.0 deg off square, and nearest rank cannot help.

    The board's median pose is ordinary -- the trout's is worse. What
    disqualifies it is that ceil(0.9n) is n for every n <= 10, so with six
    frames the ruler's p90 is its single best frame, and that frame is still
    14.7 deg off. cos(14.7 deg) - 1 = -3.3 %, which is the whole of its residual
    error. Every other target's p90 lands on a frame at 0.0-8.7 deg.
    """
    rows = cal.load_rows(DATA / "corpus.csv")
    df = cal.to_frame([r for r in rows
                       if int(r["dive_id"]) not in cal.NON_POOL_DIVES])
    a = df[df.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)]
    pose = np.degrees(np.arccos(np.clip(a.length_m / a.known_length_m, -1, 1)))
    a = a.assign(pose=pose)

    ruler = a[a.model_name == "Ruler"]
    assert len(ruler) == 6
    assert ruler.pose.min() == pytest.approx(14.7, abs=0.2)
    assert ruler.pose.max() == pytest.approx(20.0, abs=0.2)

    # p90 = the best frame, because ceil(0.9 * 6) == 6.
    assert int(np.ceil(0.9 * len(ruler))) == len(ruler)
    best = ruler.pct_error.max()
    assert best == pytest.approx(100 * (np.cos(np.radians(14.7)) - 1), abs=0.4)

    # Every reported target reaches a frame the ruler's sample never contains.
    reported = a[~a.model_name.isin(cal.HELD_OUT_MODELS)]
    for model, g in reported.groupby("model_name"):
        k = int(np.ceil(0.9 * len(g)))
        chosen = g.iloc[np.argsort(g.pct_error.values)[k - 1]]
        assert chosen.pose < 14.0, (model, chosen.pose)


def test_holding_out_the_ruler_leaves_the_cohort_and_p90_alone():
    """It removes frames, never sessions: nothing in the selection rule reads a
    held-out target, and dive 66 keeps four other targets in the polish grid."""
    rows = cal.load_rows(DATA / "corpus.csv")
    df = cal.to_frame([r for r in rows
                       if int(r["dive_id"]) not in cal.NON_POOL_DIVES])
    assert cal.accuracy_cohort(df) == cal.CORPUS_ACCURACY_DIVES
    assert cal.accuracy_cohort(df[df.model_name != "Ruler"]) \
        == cal.CORPUS_ACCURACY_DIVES

    a = df[df.dive_id.isin(cal.CORPUS_ACCURACY_DIVES)
           & ~df.model_name.isin(cal.HELD_OUT_MODELS)]
    assert len(a) == 995
    assert sorted(a.model_name.unique()) == [
        "Box", "Grouper", "Purple Angelfish", "Rainbow Trout", "Snook"]
    assert np.median(a.pct_error) == pytest.approx(-2.00, abs=0.01)
    assert np.percentile(a.pct_error, 90) == pytest.approx(0.36, abs=0.01)


def test_the_shark_is_held_out_rather_than_corrected():
    """The distinction the ruler correction must not blur.

    Both references are wrong. The ruler's replacement is measured without the
    rig, so it is adopted; the shark's candidates all run through the rig, so
    adopting one would let the instrument set its own validation target. It is
    held out instead, and nothing in MEASURED_REFERENCES_M may quietly fix it.
    """
    assert "Shark" not in cal.MEASURED_REFERENCES_M
    assert "Shark" in cal.HELD_OUT_MODELS


def test_the_range_trend_filter_does_not_depend_on_any_reference():
    """It compares a rigid target against ITSELF across range, so no reference
    length enters. Pinned because it is what makes the pre-filter trustworthy
    when a reference moves."""
    frozen = DATA / "corpus_20260912.csv"
    raw = cal.to_frame(cal.load_rows(frozen), corrected_references=False)
    corrected = cal.to_frame(cal.load_rows(frozen))
    assert cal.range_trend_flagged_dives(raw) == cal.range_trend_flagged_dives(corrected)


# --- range-trend pre-filter ------------------------------------------------


def test_theil_sen_is_exact_on_a_line_and_robust_to_one_outlier():
    x = np.linspace(1, 4, 20)
    y = 2.0 + 0.5 * x
    y[3] += 30.0
    slope, lo, hi = cal.theil_sen(x, y)
    assert slope == pytest.approx(0.5, abs=1e-9)
    assert lo <= slope <= hi


def test_range_trend_recovers_an_injected_angle_without_a_known_length():
    z = np.linspace(0.9, 3.0, 40)
    eps = np.radians(-0.25)
    lengths = 0.31 * (1 + eps * z / 0.103)
    trend = cal.range_trend(z, lengths, baseline_m=0.103)
    assert trend["eps_deg"] == pytest.approx(-0.25, abs=0.01)
    assert trend["flagged"]


def test_range_trend_needs_frames_beyond_0_8_m_over_a_2x_spread():
    z = np.linspace(0.3, 0.7, 30)
    assert cal.range_trend(z, np.full(30, 0.31), 0.103) is None
    z = np.linspace(1.0, 1.5, 30)
    assert cal.range_trend(z, np.full(30, 0.31), 0.103) is None


def test_to_frame_carries_the_dive_baseline():
    """The column the range-trend filter needs, on both exports.

    Dive 503 is the interesting one: it borrows dive 502's calibration, which
    the 2026-09-12 export caught at 8.90 cm — below the 9.7 cm floor — and
    which prod refitted to 10.35 cm on 2026-09-14 once the shipped gates made
    the old fit read as uncalibrated. Both values are pinned so a future edit
    cannot silently swap one export for the other."""
    fresh = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    frozen = cal.to_frame(cal.load_rows(DATA / "corpus_20260912.csv"))
    assert "baseline_m" in fresh.columns
    assert frozen[frozen.dive_id == 503].baseline_m.iloc[0] == pytest.approx(0.0890, abs=5e-4)
    assert fresh[fresh.dive_id == 503].baseline_m.iloc[0] == pytest.approx(0.1035, abs=5e-4)
    assert fresh[fresh.dive_id == 500].baseline_m.iloc[0] == pytest.approx(0.1041, abs=5e-4)


def test_range_trend_flagged_dives_on_the_corpus():
    """Both signs, whole interval beyond +-2 %/m, angle dives excluded. The
    flagged set is every dive already known bad from the known lengths plus
    the short-baseline dives the median had hidden -- and no sound one.

    Five since the 0.04217 m pitch correction. 503 and 504 left this set
    because the fit they borrow was wrong, not because the filter moved: dive
    502 went 8.897 -> 10.394 cm and their trends fell from +5.31 and +5.54 %/m
    to +0.60 and +0.80, well inside the +-2 band. The filter is unchanged, and
    that is the point -- a scale-free check responding to a pitch measured with
    a tape is independent corroboration of it."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    assert cal.range_trend_flagged_dives(df) == (76, 491, 492, 494, 509)


def test_the_pre_filter_is_what_removes_491_492_and_494():
    """All three clear the dive-effect bound and are removed by the range trend
    alone: they borrow across the mid-session laser movement of §4.2, reading
    -3.78, -2.83 and -6.09 %/m.

    This test was called `..._removes_491_503_and_504` and its reasoning was
    the opposite of what it now records. It argued that 503 and 504 showed "a
    plausible baseline is not sufficient", because prod had refitted dive 502
    to 10.35 cm and their trends still flagged. The 0.04217 m pitch correction
    refitted 502 again, to 10.394 cm, and their trends collapsed to +0.60 and
    +0.80 %/m -- so the earlier 10.35 cm fit was itself still wrong, and the
    plausible-baseline band had simply admitted it. The lesson stands in
    general (the band cannot see an in-plane error) but 503 and 504 are no
    longer the example of it."""
    df = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    with_filter = cal.accuracy_cohort(df)
    without = cal.accuracy_cohort(df, range_trend_filter=False)
    assert set(without) - set(with_filter) == {76, 491, 492, 494, 509}


def test_the_dive_84_relabels_are_in_the_corpus():
    """The two corrected frames, pinned by the quantity that identified them.

    A purple angelfish reads ~192 mm; as a snook it would read -58 %. Both
    frames now sit in dive 84's purple-angel population (180.9-194.2 mm) and
    nowhere near its snook population (434.9-451.4 mm), and the session reads
    16 purple angel to 13 snook rather than 14 to 15."""
    corpus = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    d84 = corpus[corpus.dive_id == 84]
    counts = d84.model_name.value_counts().to_dict()
    assert counts["Purple Angelfish"] == 16
    assert counts["Snook"] == 13
    assert d84[d84.model_name == "Snook"].length_m.min() > 0.40
    assert d84[d84.model_name == "Purple Angelfish"].length_m.max() < 0.20


def test_the_dive_521_relabels_are_in_the_corpus():
    """Four frames labelled as the box are the trout: 292-304 mm, which is
    +95 to +102 % as a 150 mm box and within a few percent of the trout. Their
    implied head/tail spans sit 16-58 px from the nearest trout frame and
    309-659 px from the nearest box."""
    corpus = cal.to_frame(cal.load_rows(DATA / "corpus.csv"))
    d521 = corpus[corpus.dive_id == 521]
    counts = d521.model_name.value_counts().to_dict()
    assert counts["Box"] == 42
    assert counts["Rainbow Trout"] == 37
    assert d521[d521.model_name == "Box"].length_m.max() < 0.20


# --- angle experiment -------------------------------------------------------


def test_load_angles_reads_the_designed_experiment():
    ang = cal.load_angles(DATA / "angles.csv")
    assert set(ang.dive_id.unique()) == set(cal.ANGLE_TEST_DIVES)
    assert ang.fish_angle_degrees.between(0, 45).all()
    assert "pct_error" in ang.columns
    assert ang.pct_error.notna().all()


def test_binned_angle_error_uses_symmetric_bins_and_min_count():
    ang = pd.DataFrame(
        {
            "fish_angle_degrees": [-2, -1, 0, 1, 2, 18, 19, 20, 21, 22, 45],
            "pct_error": [-1, -2, -3, -4, -5, -8, -9, -10, -11, -12, -30],
        }
    )
    binned = cal.binned_angle_error(ang, bins=(0, 20, 45), half_width=2.5, min_count=5)
    assert binned.loc[0, "median"] == pytest.approx(-3.0)
    assert binned.loc[20, "median"] == pytest.approx(-10.0)
    assert np.isnan(binned.loc[45, "median"])  # 1 frame < min_count


def test_the_rejected_sessions_are_what_the_limitations_paragraph_reports():
    """PAPER.md's "Sessions the rule rejects" states three counts about the same
    set of dives. A draft said six of them borrow their calibration; eight did,
    which makes borrowing the strongest predictor of rejection in the corpus
    rather than a co-equal one. Pin all three so the next edit cannot drift.

    Five since the 0.04217 m pitch correction, down from seven: 503 and 504
    left by being *fixed*, not by a threshold moving. Four of the five borrow,
    so the borrowing claim strengthened rather than weakened -- the only
    self-fitted rejection is still 509."""
    rows = cal.load_rows(DATA / "corpus.csv")
    df = cal.to_frame(rows, corrected_references=True)
    polished = cal.median_polish(cal.cell_p90_grid(df, cal.POLISH_MIN_FRAMES))
    considered = {int(d) for d in polished.dive_effect.index}
    rejected = sorted(
        considered - set(cal.accuracy_cohort(df)) - set(cal.DESIGN_EXCLUDED_DIVES)
    )
    assert rejected == [76, 491, 492, 494, 509]

    flagged = set(cal.range_trend_flagged_dives(df))
    assert set(rejected) <= flagged, "every rejection is now scale-free"

    source = df.groupby("dive_id").calibration_dive_id.first()
    borrowed = [d for d in rejected if int(source[d]) != d]
    assert len(borrowed) == 4
    assert [d for d in rejected if d not in borrowed] == [509]

    baselines = df.groupby("dive_id").baseline_m.first() * 100
    assert baselines[rejected].min() == pytest.approx(10.29, abs=0.01)
    assert baselines[rejected].max() == pytest.approx(10.55, abs=0.01)


# --- the notebook is the only figure generator ------------------------------


def _notebook_saved_figures() -> set[str]:
    """Figure stems the analysis notebook writes, parsed from its source."""
    import json
    import re

    nb = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "fish_model_analysis"
            / "fish_model_measurements.ipynb"
        ).read_text()
    )
    source = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    return set(re.findall(r'save_figure\(\s*\w+\s*,\s*"([^"]+)"', source))


def test_every_committed_figure_is_generated_by_the_notebook():
    """No figure may outlive the cell that draws it.

    Figures 7, 9, 10 and 10b were each produced by a standalone script at some
    point, and 10/10b went stale against a corrected export because the
    regeneration pass ran the notebook and the scripts were not in it. One
    generator, checked here, is the fix: if a figure is in `figures/` and the
    notebook does not save it, either the cell was lost or the figure is an
    orphan, and both are bugs.
    """
    figures = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "figures"
    # The fork-probe frame renders are one-off overlays drawn from NAS imagery by
    # `fork_render.py`, not plots of the corpus; they have no cell and need none.
    on_disk = {
        p.stem
        for p in figures.glob("*.pdf")
        if not p.stem.startswith(("full_", "tail_", "wide_"))
    }
    orphans = on_disk - _notebook_saved_figures()
    assert not orphans, f"figures with no generating cell: {sorted(orphans)}"


def test_the_notebook_draws_every_figure_the_paper_captions():
    """Every `**Figure N**` caption in PAPER.md has a cell behind it."""
    import re

    paper = (Path(__file__).resolve().parents[1] / "PAPER.md").read_text()
    captioned = set(re.findall(r"^- \*\*Figure (\w+)\*\*", paper, re.M))
    saved = _notebook_saved_figures()
    # "fig10b_error_by_camera" -> "10b"; "figA_all_dives_percent" -> "A".
    drawn = {re.match(r"fig([0-9A-Za-z]+?)_", s).group(1) for s in saved}
    assert captioned <= drawn, f"captioned but not drawn: {sorted(captioned - drawn)}"
