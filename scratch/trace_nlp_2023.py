"""Trace the 2023 NLP PDF through native / OCR / layout."""
from __future__ import annotations

import os
import sys

import fitz

sys.path.insert(0, r"D:\pyqrag")

from rag.dynamic_ingest import DynamicIngestPipeline, filter_noise_lines, perform_ocr_page
from rag.ocr_layout import _classify, ocr_page_lines, reconstruct_questions_from_layout
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
    question_structure_score,
)
from rag.hybrid_question_extraction import detect_source_question_markers, hybrid_extract_document

PDF = (
    r"D:\pyqrag\data\uploads\ws-nlp-31567ba5"
    r"\be_computer-engineering_semester-7_2023_december_"
    r"dloc-iii-natural-language-processing-rev-2019-c-scheme.pdf"
)


def eval_repr(name, text, page_no):
    if not (text or "").strip():
        print(f"  {name}: EMPTY")
        return
    prepared = prepare_page_text_for_extraction(text)
    acc, rej = extract_questions_from_page_text(prepared, page_no, os.path.basename(PDF), "ws", year=2023)
    marks = detect_source_question_markers(prepared)
    print(
        f"  {name:16s} chars={len(text):5d} markers={len(marks):2d} acc={len(acc):2d} rej={len(rej):2d} "
        f"struct={question_structure_score([q['question_id'] for q in acc]):.3f} "
        f"ids={[q['question_id'] for q in acc]}"
    )
    if marks and set(marks) - {q["question_id"] for q in acc}:
        print(f"      marker-not-accepted: {sorted(set(marks) - {q['question_id'] for q in acc})}")
    for r in rej[:8]:
        print(f"      rejected {r.get('question_id')} <- {r.get('reason')} | {(r.get('raw_text') or '')[:120]}")
    return prepared


def main():
    pipe = DynamicIngestPipeline()
    doc = fitz.open(PDF)
    print("pages", doc.page_count, os.path.basename(PDF))
    pages_payload = []
    for i, page in enumerate(doc):
        page_no = i + 1
        native = page.get_text() or ""
        filtered = filter_noise_lines(native)
        valid, reason, metrics = pipe.validate_text_quality(filtered)
        ocr = perform_ocr_page(page, dpi=150) or ""
        layout_lines = ocr_page_lines(page, dpi=150)
        layout = reconstruct_questions_from_layout(layout_lines)
        print(f"\n=== PAGE {page_no} native_valid={valid} reason={reason} ===")
        print("quality", metrics)
        print("native head:", (native[:400] or "").replace("\n", " | "))
        print("ocr head:", (ocr[:600] or "").replace("\n", " | "))
        print("--- classified ---")
        for ln in layout_lines:
            kind, parent, sub, rest = _classify(ln["text"])
            print(f"  x0={ln['x0']:4d} {kind:12s} p={str(parent):4s} s={str(sub):4s} {ln['text'][:130]}")
        print("--- reconstructed ---")
        print(layout or "<empty>")
        eval_repr("native", native, page_no)
        eval_repr("filtered", filtered, page_no)
        eval_repr("ocr", filter_noise_lines(ocr), page_no)
        eval_repr("layout", layout, page_no)
        best = layout or filter_noise_lines(ocr) or filtered
        pages_payload.append({
            "page": page_no,
            "raw_native_text": native,
            "raw_ocr_text": ocr,
            "reconstructed_text": prepare_page_text_for_extraction(best),
            "ocr_used": True,
        })
    doc.close()
    hybrid = hybrid_extract_document(pages_payload, filename=os.path.basename(PDF), workspace_id="ws", year=2023)
    print("\n=== HYBRID ===")
    print("markers", hybrid.get("source_markers"))
    print("accepted", [q["question_id"] for q in hybrid["accepted_questions"]])
    print("quality", hybrid["quality"])
    print("rejected", [(r.get("question_id"), r.get("reason")) for r in (hybrid.get("rejected_candidates") or [])][:20])


if __name__ == "__main__":
    main()
