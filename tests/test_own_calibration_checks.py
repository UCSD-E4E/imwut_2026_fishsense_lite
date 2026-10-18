"""Can a session's exclusion be seen from its own frames, without known lengths?

Dive 509 is the one range-check exclusion that fitted its own calibration. The
field-runnable checks -- lever arm, bracketing, time gap, dot-line residual --
are computed here for it and for every other session the accuracy analysis
judges (the 19-session cohort, the four other range-check exclusions and dive
60), so that any rule proposed from them is tested on both sides at once.

The pose-derived range of a checkerboard calibration is NOT computed: the
checkerboard path stores no per-frame board pose, in this cache or in
production. The lever arm below is in laser-triangulated range, under the
session's own fitted beam.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsense_imwut import calibration as cal

DATA = Path(__file__).resolve().parents[1] / "fish_model_analysis" / "data"
EXCLUDED = (60, 76, 491, 492, 494, 509)       # five by the range check, plus dive 60


@pytest.fixture(scope="module")
def sessions():
    rows = [r for r in cal.load_rows(DATA / "corpus.csv")
            if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    cohort = cal.accuracy_cohort(df)
    K = {int(r["camera_id"]): np.array(eval(r["km"])) for r in rows}
    camera = {int(r["dive_id"]): int(r["camera_id"]) for r in rows}
    calib = df.groupby("dive_id").calibration_dive_id.first().astype(int)
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv").set_index("dive_id")
    t = cal.load_dive_timeline(DATA / "dive_timeline.csv")

    out = {}
    for d in sorted(set(cohort) | set(EXCLUDED)):
        c = int(calib[d])
        beam = cal._beam(fits.loc[c])
        k = t[(t.dive_id == c) & t.calib].sort_values("taken")
        m = t[(t.dive_id == d) & t.measured & t.has_dot]
        kz = cal.beam_depth_m(k[["dot_x", "dot_y"]].values, K[camera[d]], *beam)
        to_cal = np.abs(m.taken.values[:, None] - k.taken.values[None, :]).min(1) / np.timedelta64(1, "s") / 60
        rk = cal.dot_line_residual_px(k[["dot_x", "dot_y"]].values, K[camera[d]], *beam)
        rm = cal.dot_line_residual_px(m[["dot_x", "dot_y"]].values, K[camera[d]], *beam)
        out[d] = dict(cohort=d in cohort, calibration=c, standard=fits.loc[c, "standard"],
                      n_calib=len(k), bursts=int((k.taken.diff() > pd.Timedelta(minutes=5)).sum() + 1),
                      lever_m=float(np.ptp(kz)), calib_z=(float(kz.min()), float(kz.max())),
                      burst=(k.taken.min(), k.taken.max()), measured=(m.taken.min(), m.taken.max()),
                      n_measured=len(m), max_min_to_calib=float(to_cal.max()),
                      calib_residual_px=float(np.median(rk)), measured_residual_px=float(np.median(rm)))
    return out


def test_the_sessions_judged(sessions):
    assert len(sessions) == 25
    assert sum(s["cohort"] for s in sessions.values()) == 19
    assert sorted(d for d, s in sessions.items() if not s["cohort"]) == list(EXCLUDED)


def test_509_is_the_only_excluded_session_that_fitted_its_own_calibration(sessions):
    borrowed = {d: sessions[d]["calibration"] for d in EXCLUDED if sessions[d]["calibration"] != d}
    assert borrowed == {60: 65, 76: 63, 491: 490, 492: 490, 494: 493}
    assert sessions[509]["calibration"] == 509
    assert sorted(d for d, s in sessions.items() if s["cohort"] and s["calibration"] == d) == [521, 522]


def test_509s_lever_arm(sessions):
    """49 board frames with a dot, spanning 0.56-4.44 m of laser range: 3.88 m,
    far past the 0.6 m refusal bound. A checkerboard fit, not a slate one."""
    s = sessions[509]
    assert s["standard"] == "checkerboard" and s["n_calib"] == 49
    assert [round(z, 2) for z in s["calib_z"]] == [0.56, 4.44]
    assert round(s["lever_m"], 2) == 3.88


def test_509_has_one_burst_and_measures_for_forty_minutes_after_it(sessions):
    s = sessions[509]
    assert s["bursts"] == 1
    assert [f"{x:%H:%M:%S}" for x in s["burst"]] == ["09:08:06", "09:13:17"]
    assert s["n_measured"] == 162
    assert [f"{x:%H:%M:%S}" for x in s["measured"]] == ["09:05:32", "09:53:06"]
    assert round((s["burst"][0] - s["measured"][0]).total_seconds() / 60, 1) == 2.6
    assert round((s["measured"][1] - s["burst"][1]).total_seconds() / 60, 1) == 39.8


def test_509s_dot_line_residual_is_ordinary(sessions):
    s = sessions[509]
    assert round(s["calib_residual_px"], 1) == -0.2
    assert round(s["measured_residual_px"], 1) == 2.5


def test_no_field_runnable_check_separates_509_from_the_cohort(sessions):
    """Each check, as a threshold rule: none flags 509 while passing all 19."""
    coh = {d: s for d, s in sessions.items() if s["cohort"]}
    s = sessions[509]
    # lever arm: every session clears 0.6 m, 509 among the longest
    assert min(x["lever_m"] for x in sessions.values()) > 1.0
    assert round(min(x["lever_m"] for x in coh.values()), 2) == 1.07          # dive 84
    # bracketing: no session in the corpus has a second calibration burst
    assert all(x["bursts"] == 1 for x in sessions.values())
    # time gap: any bound that flags 509 (39.8 min) also flags dive 84, whose
    # calibration is from the previous day; 58 and 59 sit within 6 min of 509
    assert sorted(d for d, x in coh.items() if x["max_min_to_calib"] > 30) == [58, 59, 84]
    assert round(coh[58]["max_min_to_calib"], 1) == 34.0
    assert round(coh[84]["max_min_to_calib"]) == 1334
    assert round(coh[59]["max_min_to_calib"], 1) == 39.4 < round(s["max_min_to_calib"], 1) == 39.8
    # dot-line offset of measured frames from the calibration's own: 509 is mid-pack
    off = {d: x["measured_residual_px"] - x["calib_residual_px"] for d, x in sessions.items()}
    assert round(off[509], 1) == 2.7
    assert round(max(off[d] for d in coh), 1) == 21.1                          # dive 84


def test_509s_trout_reads_shorter_the_later_it_was_shot():
    """Diagnostic only -- it uses the known lengths, so it is not field-runnable.
    In the 20 minutes after the burst the trout reads -4.2 and -4.6 %; 30-40
    minutes after, -9.6 %. The beam moved in plane during the dive, after its
    only calibration, which no dot-line residual can see."""
    rows = [r for r in cal.load_rows(DATA / "corpus.csv") if r["dive_id"] == "509"]
    t = cal.load_dive_timeline(DATA / "dive_timeline.csv")
    t = t[(t.dive_id == 509) & t.measured & t.has_dot]
    when = {(round(x, 4), round(y, 4)): w for x, y, w in zip(t.dot_x, t.dot_y, t.taken)}
    end = pd.Timestamp("2023-08-18 09:13:17+00:00")
    trout = pd.DataFrame([
        {"mins": (when[tuple(round(float(v), 4) for v in r["dot"].split(";"))] - end).total_seconds() / 60,
         "pct": 100 * (float(r["length_m"]) / float(r["known_length_m"]) - 1)}
        for r in rows if r["model"] == "Rainbow Trout"])
    assert len(trout) == 96
    med = trout.groupby(pd.cut(trout.mins, [0, 10, 20, 30, 40]), observed=True).pct.median().round(1)
    assert med.tolist() == [-4.2, -4.6, -9.6]
