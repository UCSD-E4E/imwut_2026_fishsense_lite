"""Double-blind submission: no paper figure may identify the system or the group.

Every figure file the paper uses is scanned, drawn text and metadata both:
- PDF: the document-information metadata, the text pypdf extracts from every page
  (all of the paper's PDFs carry a ToUnicode map or named glyphs, so drawn labels
  come back as text), and the raw bytes of every decompressed stream;
- PNG: every text chunk (tEXt, iTXt, zTXt) and any EXIF.
A PNG's drawn text is checked through its PDF twin. The one PNG with no twin, the
annotated parrotfish frame, draws only lines and circles (`annotate_frame.py`).

The paper's PDFs must also carry no drawn title: the caption carries it.
"""

import re
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfReader

from fishsense_imwut import pubfig

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "fish_model_analysis" / "figures"
PAPER_FILES = [FIGS / f"{n}.{ext}" for n in sorted(pubfig.PAPER_FIGURES) for ext in ("pdf", "png")] + [
    ROOT / "reconstruction_analysis" / "figures" / "laser_spot_path.pdf",
    FIGS / "frame_stereo_parrotfish_P8030086.png",
]

#: "FishSense" in any spacing or case, the short name, the institution and lab, the
#: four authors and their ORCIDs, and anything that looks like a path, URL or host.
IDENTIFYING = re.compile(
    r"fish\s*-?\s*sense|\bFSL\b|UCSD|\bE4E\b|Crutchfield|Semmens|Schurgers|Kastner|ccrutchf"
    r"|0000-000[0-9]-\d{4}-\d{3}[\dX]|/home/|https?://|github|\.edu\b|krg-",
    re.IGNORECASE)


def _pdf_streams(path):
    raw = path.read_bytes()
    out = [raw]
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            out.append(zlib.decompress(m.group(1)))
        except zlib.error:
            pass
    return b"\n".join(out).decode("latin-1")


def _png_text_chunks(path):
    raw = path.read_bytes()
    pos, chunks = 8, []
    while pos < len(raw):
        n, kind = struct.unpack(">I4s", raw[pos:pos + 8])
        data = raw[pos + 8:pos + 8 + n]
        if kind in (b"tEXt", b"iTXt", b"eXIf"):
            chunks.append((kind.decode(), data.decode("latin-1")))
        elif kind == b"zTXt":
            key, _, rest = data.partition(b"\0")
            chunks.append(("zTXt", key.decode("latin-1") + zlib.decompress(rest[1:]).decode("latin-1")))
        pos += 12 + n
    return chunks


def test_every_paper_figure_file_exists():
    missing = [str(p.relative_to(ROOT)) for p in PAPER_FILES if not p.exists()]
    assert not missing


@pytest.mark.parametrize("path", [p for p in PAPER_FILES if p.suffix == ".pdf"], ids=lambda p: p.name)
def test_pdf_identifies_nobody(path):
    r = PdfReader(path)
    meta = {k: str(v) for k, v in (r.metadata or {}).items()}
    assert not meta, f"metadata left in {path.name}: {meta}"
    text = " ".join(p.extract_text() or "" for p in r.pages)
    assert text.strip(), f"no text extracted from {path.name}; the scan would be blind"
    hits = IDENTIFYING.findall(text) + IDENTIFYING.findall(_pdf_streams(path))
    assert not hits, f"{path.name}: {sorted(set(hits))}"


@pytest.mark.parametrize("path", [p for p in PAPER_FILES if p.suffix == ".png"], ids=lambda p: p.name)
def test_png_carries_no_text_or_exif(path):
    assert not _png_text_chunks(path), f"{path.name}: {_png_text_chunks(path)}"
    assert not dict(Image.open(path).getexif())


@pytest.mark.parametrize("path", [p for p in PAPER_FILES if p.suffix == ".pdf"], ids=lambda p: p.name)
def test_paper_pdfs_carry_no_drawn_title(path):
    text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    title = pubfig.FIGURE_TITLES.get(path.stem, "Laser Spot Size and Location on Image Plane")
    # four words of the title after the system's name, whatever name a stale
    # render used (a long title wraps, so whitespace is normalized)
    words = title.replace(pubfig.SYSTEM_NAME, "").split()[:4]
    assert " ".join(words) not in " ".join(text.split())


def test_the_system_name_is_one_setting():
    assert pubfig.SYSTEM_NAME == "FishCamera"
    src = (ROOT / "fishsense_imwut" / "pubfig.py").read_text()
    drawn = re.findall(r'(?:set_[xy]?label|set_title|text|annotate)\([^)]*FishSense', src)
    assert not drawn
