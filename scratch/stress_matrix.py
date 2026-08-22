"""
Structural stress matrix for the universal extraction pipeline.

Each case authors a PDF whose question ids are known by construction, then runs
the REAL production ingest over it and diffs accepted ids against ground truth.
Because the ids are authored rather than guessed, "missing" here means the
pipeline genuinely lost a question that exists in the document.

These are structural probes, not proof of real-world universality: they vary
layout, numbering style, sub-letter depth and question count, not subject matter.

Usage:
    python scratch/stress_matrix.py            # all cases
    python scratch/stress_matrix.py a_f roman  # only matching case names
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, r"D:\pyqrag")

# Isolate from production data before importing anything that touches storage.
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_stress_matrix")

import fitz  # noqa: E402

from rag.dynamic_ingest import DynamicIngestPipeline  # noqa: E402
from rag.workspace_db import WorkspaceDB  # noqa: E402

OUT_DIR = r"D:\pyqrag\scratch\stress_pdfs"

BODIES = [
    "Explain the layered reference model and justify each layer with a diagram.",
    "Describe the scheduling policy and derive its average waiting time.",
    "Differentiate between the two normalisation forms with a worked example.",
    "Discuss the error control mechanism and analyse its overhead.",
    "Derive the expression for throughput and state all assumptions clearly.",
    "Compare the two indexing structures and evaluate their lookup complexity.",
    "Explain the deadlock avoidance algorithm and apply it to a sample matrix.",
    "Illustrate the encoding scheme and compute the resulting bandwidth.",
    "Analyse the concurrency protocol and prove its serialisability.",
    "Describe the routing metric and calculate the shortest path for the graph.",
]


def body(i: int) -> str:
    return BODIES[i % len(BODIES)]


def write_pdf(path: str, pages: list[list[str]], *, scanned: bool = False,
              columns: int = 1) -> str:
    """Render text pages. scanned=True rasterises so only OCR can read them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        if columns == 2:
            half = (len(lines) + 1) // 2
            for col, chunk in enumerate((lines[:half], lines[half:])):
                x = 45 + col * 275
                y = 60
                for ln in chunk:
                    page.insert_textbox(fitz.Rect(x, y, x + 260, y + 46), ln, fontsize=8)
                    y += 46
        else:
            y = 55
            for ln in lines:
                # Paginate, otherwise long papers silently fall off the page and
                # the harness would blame the pipeline for its own clipping.
                if y > page.rect.height - 60:
                    page = doc.new_page()
                    y = 55
                page.insert_textbox(fitz.Rect(45, y, 550, y + 30), ln, fontsize=9)
                y += 30
    if scanned:
        raster = fitz.open()
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            rp = raster.new_page(width=page.rect.width, height=page.rect.height)
            rp.insert_image(rp.rect, stream=pix.tobytes("png"))
        doc.close()
        doc = raster
    doc.save(path)
    doc.close()
    return path


HEADER = [
    "NATIONAL INSTITUTE OF ENGINEERING",
    "END SEMESTER EXAMINATION - MAY 2027",
    "Duration: 3 Hours                                   Max Marks: 80",
    "Instructions: Attempt any four questions. Figures to the right indicate full marks.",
]


def case_a_f():
    """Parent 1 carries six subquestions; later parents carry a (c)."""
    lines = list(HEADER)
    truth = []
    for i, s in enumerate("abcdef"):
        lines.append(f"Q1({s}) {body(i)} [5]")
        truth.append(f"Q1({s})")
    for p in (2, 3, 4):
        for i, s in enumerate("abc"):
            lines.append(f"Q{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "a_f_and_c_subs", [lines], truth, {}


def case_deep_subs():
    """Sub-letters running past (f) to (j) under a single parent."""
    lines = list(HEADER)
    truth = []
    for i, s in enumerate("abcdefghij"):
        lines.append(f"Q1({s}) {body(i)} [4]")
        truth.append(f"Q1({s})")
    for i, s in enumerate("abcd"):
        lines.append(f"Q2({s}) {body(i + 3)} [6]")
        truth.append(f"Q2({s})")
    return "deep_subs_a_to_j", [lines], truth, {}


def _count_case(total: int):
    """Arbitrary totals spread over parents of uneven width."""
    lines = list(HEADER)
    truth = []
    widths, made, p = [], 0, 1
    while made < total:
        w = min((p * 2 + 1) % 5 + 2, total - made)
        widths.append(w)
        made += w
        p += 1
    for pi, w in enumerate(widths, start=1):
        for si in range(w):
            s = "abcdefghij"[si]
            lines.append(f"Q{pi}({s}) {body(pi + si)} [5]")
            truth.append(f"Q{pi}({s})")
    return f"count_{total}", [lines], truth, {}


def case_no_q_prefix():
    """1(a) style with no Q prefix."""
    lines = list(HEADER)
    truth = []
    for p in (1, 2, 3):
        for i, s in enumerate("abc"):
            lines.append(f"{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "no_q_prefix", [lines], truth, {}


def case_space_paren():
    """'1 a)' style with a space between parent and sub."""
    lines = list(HEADER)
    truth = []
    for p in (1, 2, 3):
        for i, s in enumerate("abc"):
            lines.append(f"{p} {s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "space_paren", [lines], truth, {}


def case_roman():
    """Roman numeral subquestions."""
    lines = list(HEADER)
    truth = []
    for p in (1, 2, 3):
        for i, s in enumerate(["i", "ii", "iii"]):
            lines.append(f"Q{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "roman_subs", [lines], truth, {}


def case_cross_page():
    """A question body continues onto the next page."""
    p1 = list(HEADER)
    truth = []
    for i, s in enumerate("abc"):
        p1.append(f"Q1({s}) {body(i)} [6]")
        truth.append(f"Q1({s})")
    p1.append("Q2(a) Explain the transaction recovery procedure and describe how the")
    truth.append("Q2(a)")
    p2 = ["system restores a consistent checkpoint after an unexpected failure. [8]"]
    for i, s in enumerate("bc"):
        p2.append(f"Q2({s}) {body(i + 4)} [6]")
        truth.append(f"Q2({s})")
    return "cross_page", [p1, p2], truth, {}


def case_two_column():
    lines = list(HEADER)
    truth = []
    for p in (1, 2, 3, 4):
        for i, s in enumerate("ab"):
            lines.append(f"Q{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "two_column", [lines], truth, {"columns": 2}


def case_scanned():
    lines = list(HEADER)
    truth = []
    for p in (1, 2, 3):
        for i, s in enumerate("abc"):
            lines.append(f"Q{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    return "scanned_ocr", [lines], truth, {"scanned": True}


def case_noise_markers():
    """Real questions plus decoys that must NOT become questions."""
    lines = list(HEADER) + [
        "Note: B.E. (Sem 7) candidates must answer in the ruled answer book.",
        "Refer to figure 5*5 and matrix 32*32*3 where applicable.",
        "Page 1 of 2",
    ]
    truth = []
    for p in (1, 2, 3):
        for i, s in enumerate("ab"):
            lines.append(f"Q{p}({s}) {body(p + i)} [6]")
            truth.append(f"Q{p}({s})")
    lines.append("2024")
    lines.append("10")
    return "noise_markers", [lines], truth, {}


def case_flat_numbered():
    """Flat numbered questions with no subquestions at all."""
    lines = list(HEADER)
    truth = []
    for p in range(1, 8):
        lines.append(f"Q{p}. {body(p)} [10]")
        truth.append(f"Q{p}")
    return "flat_numbered_7", [lines], truth, {}


CASES = [
    case_a_f,
    case_deep_subs,
    lambda: _count_case(7),
    lambda: _count_case(13),
    lambda: _count_case(18),
    lambda: _count_case(31),
    case_no_q_prefix,
    case_space_paren,
    case_roman,
    case_cross_page,
    case_two_column,
    case_scanned,
    case_noise_markers,
    case_flat_numbered,
]


def run():
    wanted = [a.lower() for a in sys.argv[1:]]
    pipe = DynamicIngestPipeline()
    ws_db = WorkspaceDB()

    rows = []
    for factory in CASES:
        name, pages, truth, opts = factory()
        if wanted and not any(w in name.lower() for w in wanted):
            continue

        path = os.path.join(OUT_DIR, f"stress_{name}.pdf")
        write_pdf(path, pages, **opts)

        ws_id = f"ws-stress-{name}"
        try:
            ws_db.delete_workspace(ws_id)
        except Exception:
            pass
        ws = ws_db.get_or_create(ws_id, subject="Stress Subject")

        try:
            metas = pipe.parse_pyq_pdf(path, ws)
        except Exception as exc:
            rows.append((name, len(truth), 0, ["<exception>"], [], f"ERROR {exc}"[:60]))
            continue

        got = [m.get("question_id") for m in metas if m.get("question_id")]
        quality = (metas[0].get("extraction_quality") if metas else "NONE") or "NONE"
        missing = [t for t in truth if t not in got]
        extra = [g for g in got if g not in truth]
        rows.append((name, len(truth), len(got), missing, extra, quality))

        try:
            ws_db.delete_workspace(ws_id)
        except Exception:
            pass

    print("\n" + "=" * 96)
    print(f"{'case':22s} {'want':>5s} {'got':>5s} {'quality':10s}  missing / extra")
    print("=" * 96)
    failures = 0
    for name, want, got, missing, extra, quality in rows:
        flag = "OK " if (not missing and not extra) else "BAD"
        if flag == "BAD":
            failures += 1
        detail = ""
        if missing:
            detail += f"MISSING={','.join(missing[:10])}"
        if extra:
            detail += f"  EXTRA={','.join(extra[:10])}"
        print(f"{flag} {name:22s} {want:5d} {got:5d} {quality:10s}  {detail}")
    print("=" * 96)
    print(f"{len(rows) - failures}/{len(rows)} structural cases clean")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
