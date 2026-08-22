"""
Per-stage extraction tracer.

Answers one question for ANY pdf: at which stage does a given question id stop
existing? It reuses the production functions rather than reimplementing them, so
what it prints is what the pipeline actually does.

Stages traced per page:
  native            -> raw PyMuPDF text
  filtered_native   -> after glyph/noise filtering
  ocr_text          -> flat tesseract OCR
  ocr_layout        -> coordinate-aware reconstruction
For each representation: markers detected, candidates accepted, candidates
rejected (with reason), and which marker ids are present in the text but absent
from the accepted set.

Finally the document-level reconciliation is shown: the marker set the quality
gate compares against, the accepted ids, and the source line behind every
"missing" id so a genuine question can be told apart from a noise marker.

Usage:
    python scratch/trace_pipeline.py <pdf> [<pdf> ...]
    python scratch/trace_pipeline.py --all
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, r"D:\pyqrag")

import fitz  # PyMuPDF

from rag.dynamic_ingest import (
    DynamicIngestPipeline,
    filter_noise_lines,
    perform_ocr_page,
)
from rag.hybrid_question_extraction import (
    compute_extraction_quality,
    detect_source_question_markers,
    hybrid_extract_document,
)
from rag.ocr_layout import ocr_layout_text
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
    question_structure_score,
)

WS = "ws-trace-diagnostic"


def marker_context(blob: str, marker: str, width: int = 110) -> str:
    """Show the source line a marker id came from, so noise is visible."""
    m = re.search(r"Q?(\d+)\D+([a-z])?", marker, re.I)
    if not m:
        return ""
    parent, sub = m.group(1), (m.group(2) or "")
    pat = rf"(?:^|[\n\r])[^\n\r]*?(?:Q\.?|Question)\s*{parent}\s*[\.\):\-]?\s*\(?{sub}\)?[^\n\r]*"
    hit = re.search(pat, blob, re.I)
    if not hit:
        return "<marker text not locatable>"
    return " ".join(hit.group(0).split())[:width]


def trace_pdf(path: str) -> dict:
    name = os.path.basename(path)
    print("\n" + "=" * 78)
    print(f"PDF: {name}")
    print("=" * 78)

    pipe = DynamicIngestPipeline()
    doc = fitz.open(path)
    print(f"pages: {doc.page_count}")

    pages_payload = []
    for i, page in enumerate(doc):
        raw_native = page.get_text() or ""
        filtered_native = filter_noise_lines(raw_native)
        is_valid, _reason, _metrics = pipe.validate_text_quality(filtered_native)

        reps = {"native": raw_native, "filtered_native": filtered_native}

        ocr_raw = ""
        try:
            ocr_raw = perform_ocr_page(page, dpi=150) or ""
        except Exception as exc:
            print(f"  [page {i+1}] OCR unavailable: {exc}")
        if ocr_raw:
            reps["ocr_text"] = filter_noise_lines(ocr_raw)
        try:
            lay = ocr_layout_text(page, dpi=150)
        except Exception as exc:
            lay = ""
            print(f"  [page {i+1}] layout OCR unavailable: {exc}")
        if lay:
            reps["ocr_layout"] = lay

        print(f"\n--- page {i+1} (native_valid={is_valid}) ---")
        best_name, best_text, best_acc = None, "", -1
        for rep_name, text in reps.items():
            if not (text or "").strip():
                print(f"  {rep_name:16s} : empty")
                continue
            prepared = prepare_page_text_for_extraction(text)
            acc, rej = extract_questions_from_page_text(
                prepared, i + 1, name, WS, subject="", year=0
            )
            acc_ids = [q["question_id"] for q in acc]
            marks = detect_source_question_markers(prepared)
            lost = sorted(set(marks) - set(acc_ids))
            score = question_structure_score(acc_ids)
            print(
                f"  {rep_name:16s} : chars={len(text):6d} markers={len(marks):3d} "
                f"accepted={len(acc):3d} rejected={len(rej):3d} structure={score:.2f}"
            )
            if acc_ids:
                print(f"      accepted ids : {', '.join(acc_ids)}")
            if lost:
                print(f"      marker w/o accepted id: {', '.join(lost)}")
            for r in rej[:6]:
                rid = r.get("question_id") or "?"
                why = r.get("reason") or r.get("rejection_reason")
                body_txt = r.get("raw_text") or r.get("text") or ""
                print(f"      rejected {rid:10s} <- {why} | {body_txt[:60]}")
            if len(acc) > best_acc:
                best_name, best_text, best_acc = rep_name, text, len(acc)

        pages_payload.append(
            {
                "page": i + 1,
                "raw_native_text": raw_native,
                "raw_ocr_text": ocr_raw,
                "reconstructed_text": prepare_page_text_for_extraction(best_text) if best_text else "",
                "ocr_used": best_name in ("ocr_text", "ocr_layout"),
            }
        )
        # Diagnostic only. Highest yield is NOT the production rule and is not
        # evidence of correctness: a mis-segmenting OCR pass can invent extra
        # ids (e.g. reading "b." as "b_" and shifting a parent), scoring higher
        # than the representation that is actually faithful to the page.
        print(f"  => highest-yield representation (diagnostic, not production's "
              f"choice): {best_name} ({best_acc} accepted)")

    doc.close()

    # Document-level reconciliation, exactly as the quality gate sees it.
    hybrid = hybrid_extract_document(
        pages_payload, filename=name, workspace_id=WS, subject="Subject", year=0
    )
    accepted = hybrid["accepted_questions"]
    acc_ids = [q["question_id"] for q in accepted]
    markers = hybrid["source_markers"]
    quality = hybrid["quality"]

    print("\n" + "-" * 78)
    print("DOCUMENT RECONCILIATION")
    print("-" * 78)
    print(f"markers detected : {len(markers)} -> {', '.join(sorted(set(markers)))}")
    print(f"accepted         : {len(acc_ids)} -> {', '.join(acc_ids)}")
    print(f"quality          : {quality['extraction_quality']} "
          f"(confidence {quality['confidence']})")

    missing = quality.get("missing_questions") or []
    if missing:
        blob = "\n".join(
            (p.get("reconstructed_text") or p.get("raw_ocr_text") or p.get("raw_native_text") or "")
            for p in pages_payload
        )
        print(f"\nMISSING ({len(missing)}) — source text behind each marker:")
        for mid in missing:
            print(f"  {mid:10s} :: {marker_context(blob, mid)}")

    rej = hybrid.get("rejected_candidates") or []
    if rej:
        print(f"\nrejected candidates ({len(rej)}):")
        for r in rej[:20]:
            why = r.get("reason") or r.get("rejection_reason")
            body_txt = r.get("raw_text") or r.get("text") or ""
            print(f"  {(r.get('question_id') or '?'):10s} <- {why} | {body_txt[:70]}")

    return {
        "pdf": name,
        "pages": len(pages_payload),
        "markers": sorted(set(markers)),
        "accepted": acc_ids,
        "missing": missing,
        "quality": quality["extraction_quality"],
    }


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args[0] == "--all":
        paths = sorted(glob.glob(r"D:\pyqrag\data\pyq\**\*.pdf", recursive=True))
    else:
        paths = args

    results = [trace_pdf(p) for p in paths]

    out = r"D:\pyqrag\scratch\extraction_audits\pipeline_trace.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"{r['quality']:9s} {len(r['accepted']):3d} accepted / "
              f"{len(r['markers']):3d} markers  missing={len(r['missing']):2d}  {r['pdf'][:52]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
