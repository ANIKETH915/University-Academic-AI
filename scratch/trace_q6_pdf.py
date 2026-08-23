import glob
import os
import sys
import json
import fitz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_question_extraction import (
    hybrid_extract_document,
    classify_genuine_markers,
    detect_markers_in_text,
)
from rag.question_extractor import (
    prepare_page_text_for_extraction,
    extract_questions_from_page_text,
)
from rag.ocr_layout import extract_layout_lines, reconstruct_questions_from_layout

pdf_files = []
for root in ["data/pyq", "data/uploads"]:
    if os.path.exists(root):
        for path in glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True):
            if "Copy" not in path:
                pdf_files.append(path)

print(f"Scanning {len(pdf_files)} PDFs for Q6 structure...\n")

for pdf_path in pdf_files:
    fname = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
        pages_payload = []
        all_layout_lines = []
        for p_num, page in enumerate(doc):
            raw_native = page.get_text() or ""
            layout_lines = extract_layout_lines(page, p_num + 1)
            all_layout_lines.extend(layout_lines)
            recon = prepare_page_text_for_extraction(raw_native)
            pages_payload.append({
                "page": p_num + 1,
                "raw_native_text": raw_native,
                "raw_ocr_text": "",
                "raw_ocr_hd_text": "",
                "reconstructed_text": recon,
                "ocr_used": False,
                "raw_ocr_layout": layout_lines,
            })
        doc.close()

        if len(pages_payload) > 1 and all_layout_lines:
            doc_layout = reconstruct_questions_from_layout(all_layout_lines)
            if doc_layout:
                pages_payload[0]["reconstructed_text"] = prepare_page_text_for_extraction(doc_layout)

        res = hybrid_extract_document(
            pages_payload,
            filename=fname,
            workspace_id="ws-diag",
            subject="Audit Subject",
            year=2024,
        )
        quality = res.get("quality") or {}
        missing = quality.get("missing_questions") or []
        extracted_qids = [q["question_id"] for q in res.get("accepted_questions") or []]

        # Check if Q6 has unrecovered markers or if Q6(a) absorbed Q6(b..e)
        q6_extracted = [q for q in extracted_qids if q.startswith("Q6")]
        q6_missing = [m for m in missing if m.startswith("Q6")]

        if q6_missing or (len(q6_extracted) == 1 and "Q6(a)" in q6_extracted):
            print("="*70)
            print(f"TARGET PDF DIAGNOSED: {pdf_path}")
            print(f"  Extracted Q6 questions: {q6_extracted}")
            print(f"  Missing Q6 questions:   {q6_missing}")
            print(f"  Total Extracted:         {len(extracted_qids)}")
            print(f"  Quality summary:         {quality}")
            print("="*70)

            print("\n--- 1. NATIVE PDF TEXT AROUND Q6 ---")
            for p in pages_payload:
                txt = p["raw_native_text"]
                if "Q6" in txt or "Question 6" in txt or "Q.6" in txt or "6." in txt:
                    print(f"[Page {p['page']}]")
                    for line in txt.splitlines():
                        if any(k in line for k in ("Q6", "Q.6", "Question 6", "6.", "(a)", "(b)", "(c)", "(d)", "(e)")):
                            print("  ", line)

            print("\n--- 3. LAYOUT RECONSTRUCTED TEXT AROUND Q6 ---")
            for p in pages_payload:
                txt = p["reconstructed_text"]
                if "Q6" in txt or "Question 6" in txt or "Q.6" in txt:
                    print(f"[Page {p['page']}]")
                    for line in txt.splitlines():
                        if "Q6" in line or "Q.6" in line or line.strip().startswith(("a)", "b)", "c)", "d)", "e)", "(a)", "(b)", "(c)", "(d)", "(e)")):
                            print("  ", line)

            print("\n--- 4. DETECTED MARKER CANDIDATES FOR Q6 ---")
            ev = res.get("evidence")
            if ev and hasattr(ev, "marker_candidates"):
                q6_mc = [mc for mc in ev.marker_candidates if str(mc.get("marker_id")).startswith("Q6")]
                print("  ", q6_mc)

            print("\n--- 7. ACCEPTED Q6 QUESTIONS ---")
            for q in res.get("accepted_questions") or []:
                if q["question_id"].startswith("Q6"):
                    print(f"  ID: {q['question_id']} | Method: {q.get('extraction_method')} | Text: {q['exact_text'][:120]}...")

            print("\n--- 8. REJECTED CANDIDATES AROUND Q6 ---")
            for r in res.get("rejected_candidates") or []:
                if str(r.get("question_id")).startswith("Q6"):
                    print(f"  ID: {r.get('question_id')} | Reason: {r.get('reason')} | Text: {str(r.get('exact_text'))[:100]}")

            print("\n" + "="*70 + "\n")

    except Exception as e:
        print(f"Error checking {fname}: {e}")
