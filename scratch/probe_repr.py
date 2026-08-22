"""Compare native / plain-OCR / layout-OCR representations per page."""
import glob
import sys

import fitz

sys.path.insert(0, r"D:\pyqrag")
from rag.dynamic_ingest import filter_noise_lines, perform_ocr_page
from rag.ocr_layout import ocr_layout_text
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
)

for pat in sys.argv[1:] or ["*2023*december*"]:
    for pdf in glob.glob(r"D:\pyqrag\data\pyq\deep-learning\\" + pat):
        doc = fitz.open(pdf)
        for pno, page in enumerate(doc):
            native = filter_noise_lines(page.get_text() or "")
            ocr_plain = filter_noise_lines(perform_ocr_page(page, dpi=150) or "")
            layout = ocr_layout_text(page, dpi=150)
            for name, text in (
                ("native", native),
                ("ocr_text", ocr_plain),
                ("ocr_layout", layout),
            ):
                prep = prepare_page_text_for_extraction(text) if text else ""
                if not prep.strip():
                    print(f"{pat} p{pno+1} {name}: empty")
                    continue
                acc, rej = extract_questions_from_page_text(
                    prep, pno + 1, "f", "w", year=0
                )
                print(f"--- {pat} p{pno+1} {name}: n={len(acc)} rej={len(rej)}")
                for q in acc:
                    print(f"    {q['question_id']:8} {q['exact_text'][:88]}")
        doc.close()
