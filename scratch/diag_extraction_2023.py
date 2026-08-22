"""Diagnose raw extraction on real MU Deep Learning 2023 PDF (not synthetic)."""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fitz
from rag.dynamic_ingest import DynamicIngestPipeline, filter_noise_lines, perform_ocr_page
from rag.question_extractor import extract_questions_from_page_text

paths = sorted(glob.glob(r"D:\pyqrag\data\pyq\deep-learning\*2023*december*.pdf"))
# Prefer non-Copy
pdf = next((p for p in paths if "Copy" not in p), paths[0] if paths else None)
print("PDF:", pdf)
if not pdf:
    raise SystemExit(1)

pipe = DynamicIngestPipeline.__new__(DynamicIngestPipeline)
doc = fitz.open(pdf)
print("pages", len(doc))
all_acc = []
all_rej = []
for i in range(len(doc)):
    page = doc[i]
    raw = page.get_text() or ""
    is_valid, reason, metrics = DynamicIngestPipeline.validate_text_quality(pipe, raw)
    ocr_used = False
    final = filter_noise_lines(raw)
    print("=" * 60)
    print(f"PAGE {i+1}")
    print(f"PyMuPDF chars = {len(raw)}")
    print(f"quality_valid = {is_valid} reason = {reason} metrics = {metrics}")
    if not is_valid:
        ocr_raw = perform_ocr_page(page, dpi=150)
        print(f"OCR chars = {len(ocr_raw or '')}")
        if ocr_raw:
            ocr_text = filter_noise_lines(ocr_raw)
            ocr_valid, ocr_reason, ocr_metrics = DynamicIngestPipeline.validate_text_quality(pipe, ocr_text)
            print(f"OCR quality_valid = {ocr_valid} reason = {ocr_reason}")
            if ocr_valid:
                final = ocr_text
                ocr_used = True
                is_valid = True
    print(f"OCR = {'USED' if ocr_used else 'NOT USED'}")
    print(f"final chars = {len(final)}")
    print("HEAD:", repr(final[:400]))
    print("TAIL:", repr(final[-300:]))
    if not is_valid:
        print("PAGE REJECTED — no extraction")
        continue
    acc, rej = extract_questions_from_page_text(final, i + 1, os.path.basename(pdf), "ws-diag", year=2023)
    print(f"accepted={len(acc)} rejected={len(rej)} ids={[q['question_id'] for q in acc]}")
    for q in acc:
        print(f"  VALID {q['question_id']}: {q['exact_text'][:160]}")
    for r in rej:
        print(
            f"  REJECT {r.get('question_id')} reason={r.get('rejection_reason') or r.get('reason')} "
            f"text={(r.get('exact_text') or r.get('raw_text') or '')[:120]}"
        )
    all_acc.extend(acc)
    all_rej.extend(rej)
doc.close()
print("=" * 60)
print(f"TOTAL valid={len(all_acc)} rejected={len(all_rej)}")
