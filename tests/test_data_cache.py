"""The cache contract: everything PAPER.md reports comes from committed CSVs.

The point of these tests is a reader who has the repository and no database.
`data/MANIFEST.md` promises that reader a specific set of files with specific
row counts and no database client anywhere in the path. A promise in a markdown
table is not worth much, so it is parsed and checked here -- the manifest is the
fixture, which means the document cannot drift from the data it describes.
"""

import importlib
import re
from pathlib import Path

import pandas as pd
import pytest

from fishsense_imwut import calibration as cal
from fishsense_imwut import repeatability as rep
from fishsense_imwut import stereo_pairs as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fish_model_analysis" / "data"
SQL = ROOT / "fish_model_analysis" / "sql"
MANIFEST = DATA / "MANIFEST.md"

# The canonical loader for each cached file -- the one the notebook and the
# package use. Row counts are pinned as each loader returns them, not as raw
# lines, because the loaders drop a psql footer and (for the corpus) rows with
# no laser dot, and it is the loaded count that any downstream number sees.
LOADERS = {
    "corpus.csv": lambda p: len(cal.load_rows(p)),
    "corpus_20260916.csv": lambda p: len(cal.load_rows(p)),
    "corpus_20260912.csv": lambda p: len(cal.load_rows(p)),
    "all.csv": lambda p: len(cal.load_rows(p)),
    "models.csv": lambda p: len(cal.load_rows(p)),
    "angles.csv": lambda p: len(cal.load_angles(p)),
    "field.csv": lambda p: len(rep.load_field(p)),
    "calibration_fits.csv": lambda p: len(cal.load_calibration_fits(p)),
    "stereo_pairs.csv": lambda p: len(sp.load_ours(p)),
    "stereo_reference.csv": lambda p: len(sp.load_stereo(p)),
    "stereo_archive.csv": lambda p: len(pd.read_csv(p)),
    "ruler.csv": lambda p: len(pd.read_csv(p)),
    "shark_grouper.csv": lambda p: len(pd.read_csv(p, sep="|")),
    "head_tail.csv": lambda p: len(pd.read_csv(p, sep="|", header=None)),
}


def manifest_rows():
    """`{filename: loaded_count}` as MANIFEST.md's table declares them."""
    out = {}
    for line in MANIFEST.read_text().splitlines():
        m = re.match(r"\|\s*`([^`]+\.csv)`\s*\|\s*(\d+)\s*\|", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def test_the_manifest_lists_every_cached_file():
    """A file that exists but is undocumented is the failure mode here: a
    reader cannot tell whether it is evidence or a leftover."""
    listed = set(manifest_rows())
    present = {p.name for p in DATA.glob("*.csv")}
    assert present - listed == set(), f"undocumented cache files: {present - listed}"
    assert listed - present == set(), f"manifest lists missing files: {listed - present}"


@pytest.mark.parametrize("name,expected", sorted(manifest_rows().items()))
def test_each_cached_file_has_the_row_count_the_manifest_promises(name, expected):
    assert LOADERS[name](DATA / name) == expected


def test_every_committed_extraction_has_its_cached_output():
    """The SQL is in the repo so the pull is auditable by someone who cannot
    run it. An extraction with no committed CSV would be exactly that reader's
    dead end."""
    produced = {
        "extract_corpus.sql": "corpus.csv",
        "extract_angles.sql": "angles.csv",
        "extract_field.sql": "field.csv",
        "extract_calibration_fits.sql": "calibration_fits.csv",
        "extract_stereo_pairs.sql": "stereo_pairs.csv",
        "extract_head_tail.sql": "head_tail.csv",
    }
    assert {p.name for p in SQL.glob("*.sql")} == set(produced)
    for sql, csv in produced.items():
        assert (DATA / csv).exists(), f"{sql} has no cached output"


def test_nothing_on_the_reproduction_path_needs_the_compiled_extension():
    """`fishsense-meta` -> `fishsense-core` is a maturin/pyo3 Rust extension and
    the only dependency here needing a compiler. It sits in the `sim` group so a
    plain `uv sync` skips it; the simulation notebooks use it, the paper does
    not. If an analysis module reached for it, that separation would be a lie
    and an outside reader's install would fail at import.

    The database is checked here too, but only for completeness: this repository
    never speaks to it from Python. The cache is re-exported with `psql`."""
    for mod in ("calibration", "pubfig", "refraction", "repeatability",
                "stereo_pairs", "camera", "rig"):
        m = importlib.import_module(f"fishsense_imwut.{mod}")
        src = Path(m.__file__).read_text()
        assert "fishsense_meta" not in src and "fishsense_core" not in src, mod
        assert "psycopg" not in src and "sqlalchemy" not in src, mod


def test_the_headline_numbers_come_out_of_the_cache_alone():
    """PAPER.md's abstract, recomputed from the committed corpus with no
    database and no notebook state."""
    rows = [r for r in cal.load_rows(DATA / "corpus.csv")
            if int(r["dive_id"]) not in cal.NON_POOL_DIVES]
    df = cal.to_frame(rows)
    cohort = cal.accuracy_cohort(df)
    acc = df[df.dive_id.isin(cohort) & ~df.model_name.isin(cal.HELD_OUT_MODELS)]

    assert len(cohort) == 19
    assert len(acc) == 995
    assert acc.pct_error.median() == pytest.approx(-2.00, abs=0.01)

    # and section 4.3's baseline comparison, which needs the one extraction
    # that carries the calibration object
    fits = cal.load_calibration_fits(DATA / "calibration_fits.csv")
    aug = cal.to_frame(cal.load_rows(DATA / "all.csv"))
    pool = (set(aug.calibration_dive_id) | set(aug.dive_id)
            | set(df.calibration_dive_id) | set(df.dive_id))
    cbs = cal.checkerboard_slate_difference(cal.baseline_by_standard(fits, pool))
    assert cbs["n_units"] == 6
    assert cbs["n_checkerboard"] + cbs["n_slate"] == 19
    assert cbs["mean_pct"] == pytest.approx(0.66, abs=0.01)
