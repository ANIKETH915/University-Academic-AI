"""Dump layout classification and extracted IDs for one PDF."""
import glob
import sys

import fitz

from rag.ocr_layout import _classify, ocr_page_lines, reconstruct_questions_from_layout
from rag.question_extractor import (
    extract_questions_from_page_text,
    prepare_page_text_for_extraction,
)

pat = sys.argv[1] if len(sys.argv) > 1 else "*2024*december*"
pdfs = glob.glob(rf"D:\pyqrag\data\pyq\deep-learning\{pat}")
if not pdfs:
    raise SystemExit(f"no pdf for {pat}")
pdf = pdfs[0]
print("PDF", pdf)

doc = fitz.open(pdf)
page = doc[0]
lines = ocr_page_lines(page, dpi=150)
print("LAYOUT LINES", len(lines))
print("--- classified ---")
for ln in lines:
    kind, parent, sub, rest = _classify(ln["text"])
    snippet = ln["text"][:90].replace("|", "/")
    print(
        f"x0={ln['x0']:4d} {kind:12s} p={str(parent):4s} s={str(sub):4s}  {snippet}"
    )
print("--- reconstructed ---")
text = reconstruct_questions_from_layout(lines)
print(text)
print("--- extracted ---")
acc, rej = extract_questions_from_page_text(
    prepare_page_text_for_extraction(text), 1, "f", "w", year=0
)
print("accepted", [q["question_id"] for q in acc])
print("rejected", [(r.get("question_id"), r.get("reason")) for r in rej])
for q in acc:
    print(f"\nID {q['question_id']} MARKS {q['marks']} LEN {len(q['exact_text'])}")
    print(q["exact_text"])
doc.close()
